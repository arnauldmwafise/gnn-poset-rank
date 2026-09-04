"""
gnn_poset_rank.data.letor
======================

Loader for LETOR 4.0 MQ2008 (Qin and Liu, 2013), a conventional
learning-to-rank benchmark built from the TREC Million Query track over
the Gov2 web collection.

Unlike the graph-structured datasets in this package, MQ2008 is not a
single global comparison graph: it is a collection of independent
query-document groups, each with its own small candidate set and
relevance grades. :func:`load_letor_file` returns this query-grouped
structure directly, in the same ``{'x', 'relevance', 'n_docs'}`` shape
used by the synthetic ranking-data generator, so that model code written
against one works unmodified against the other.

MQ2008 is not redistributed by this package: Microsoft Research's LETOR
4.0 project page does not offer a single stable, automation-friendly
direct-download URL. Download ``MQ2008.rar`` (or a maintained mirror)
manually, extract it, and point this loader at ``Fold1/train.txt`` (etc.).
"""

from collections import defaultdict
from typing import Dict, List

import torch


def load_letor_file(path: str, max_feat: int = 46) -> List[Dict]:
    """Load one LETOR-format fold file (e.g. ``Fold1/train.txt``).

    Parameters
    ----------
    path : str
        Path to a LETOR-format file: one line per document, each of the
        form ``relevance qid:N 1:val 2:val ... #comment``.
    max_feat : int, default 46
        Number of features per document (46 for MQ2008/MQ2007).

    Returns
    -------
    list[dict]
        One entry per query, each with:
        ``x``: ``[n_docs, max_feat]`` float tensor of document features.
        ``relevance``: ``[n_docs]`` long tensor of relevance grades (0-2 for MQ2008).
        ``n_docs``: number of candidate documents for this query.
        ``qid``: the original integer query ID.
    """
    grouped = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")[0].strip().split()
            relevance = int(parts[0])
            qid = int(parts[1].split(":")[1])
            feat_vec = torch.zeros(max_feat)
            for tok in parts[2:]:
                idx, val = tok.split(":")
                feat_vec[int(idx) - 1] = float(val)
            grouped[qid].append((relevance, feat_vec))

    queries = []
    for qid, docs in grouped.items():
        relevance = torch.tensor([r for r, _ in docs], dtype=torch.long)
        x = torch.stack([f for _, f in docs])
        queries.append({"x": x, "relevance": relevance, "n_docs": len(docs), "qid": qid})
    return queries
