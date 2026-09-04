"""
posetrank.data.citation
=========================

Loader for citation networks in LINQS raw format (Cora, CiteSeer).

Both datasets' raw files share an identical structure -- tab-separated
``<paper_id> <word_attrs...> <label>`` for the ``.content`` file, and
``<cited_id> <citing_id>`` for the ``.cites`` file, with the same
"direction is right to left" convention in both -- so a single loader
handles both.

Both are known to be commonly redistributed in a *symmetrized* form (for
example, the standard PyTorch-Geometric ``Planetoid`` distribution, in
which 100% of edges have a present reverse). This module intentionally
targets raw, genuinely directed mirrors instead; :func:`load_citation_raw`
does not symmetrize edges, and callers are encouraged to verify
directedness themselves (see :func:`posetrank.diagnostics.verify_directed`
-- or simply check the fraction of edges with a present reverse) before
treating the result as a genuine partial order source.
"""

from typing import Dict

import torch


def load_citation_raw(content_path: str, cites_path: str) -> Dict:
    """Load a LINQS-format citation network from raw files.

    Parameters
    ----------
    content_path : str
        Path to the ``.content`` file (paper id, bag-of-words features, label).
    cites_path : str
        Path to the ``.cites`` file (``cited_id  citing_id`` per line).

    Returns
    -------
    dict
        ``x``: ``[V, D]`` float tensor of binary bag-of-words features.
        ``y``: ``[V]`` long tensor of class indices.
        ``class_names``: list of class label strings, ``class_names[y[i]]``
            gives the label name for node ``i``.
        ``paper_ids``: list of original string IDs, ``paper_ids[i]`` for node ``i``.
        ``raw_edges``: ``[2, E]`` long tensor of ``(cited_idx, citing_idx)``
            pairs -- genuine citation direction, not yet cleaned (see
            :func:`posetrank.diagnostics.clean_dag`).
        ``num_nodes``: number of papers.
        ``skipped_edges``: number of ``.cites`` lines dropped (dangling
            references not present in ``.content``, or self-citations).
    """
    paper_ids, features, labels = [], [], []
    with open(content_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            paper_ids.append(parts[0])
            features.append([int(v) for v in parts[1:-1]])
            labels.append(parts[-1])

    class_names = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor([class_to_idx[label] for label in labels], dtype=torch.long)

    src, tgt = [], []
    skipped = 0
    with open(cites_path) as f:
        for line in f:
            cited, citing = line.strip().split()
            if cited not in id_to_idx or citing not in id_to_idx:
                skipped += 1  # a handful of .cites entries reference ids not in .content
                continue
            if cited == citing:
                skipped += 1  # self-citation, drop
                continue
            src.append(id_to_idx[cited])
            tgt.append(id_to_idx[citing])

    raw_edges = torch.tensor([src, tgt], dtype=torch.long)
    return {
        "x": x,
        "y": y,
        "class_names": class_names,
        "paper_ids": paper_ids,
        "raw_edges": raw_edges,
        "num_nodes": len(paper_ids),
        "skipped_edges": skipped,
    }
