"""
gnn_poset_rank.data.chameleon
==========================

Loader for the Chameleon Wikipedia page-page network (Rozemberczki et al.,
2021), in the Geom-GCN preprocessing (Pei et al., 2020).

Source: pinned to commit ``f1fc0d14b3b019c562737240d06ec83b07d16a8f`` of
``graphdml-uiuc-jlu/geom-gcn``. The current ``master`` branch 404s for the
Chameleon and Squirrel files specifically (though not for others, e.g.
Cornell) -- consistent with a documented data-quality critique of these
two files by Platonov et al., and matching how PyTorch-Geometric's own
source has since pinned to this same older commit rather than tracking
master. Verified empirically before use in this project: 73.9% of edges
are genuinely one-directional (only 26.1% are part of a mutual pair),
decisively different from fully-symmetrized citation-network mirrors.

Unlike Cora/CiteSeer's topic labels, Chameleon's five classes are
genuinely ordinal (discretized page-traffic bins, higher index = more
traffic), which matters if used for anything beyond the directed-graph
diagnostics this package focuses on.
"""

from typing import Dict

import torch


def load_chameleon_raw(edges_path: str, features_path: str) -> Dict:
    """Load the Chameleon network from raw Geom-GCN-format files.

    Parameters
    ----------
    edges_path : str
        Path to ``out1_graph_edges.txt`` (tab-separated ``src  tgt``, one header line).
    features_path : str
        Path to ``out1_node_feature_label.txt`` (tab-separated
        ``node_id  comma_separated_features  label``, one header line).

    Returns
    -------
    dict
        ``x``: ``[V, 2325]`` float tensor of binary bag-of-words features.
        ``y``: ``[V]`` long tensor of class indices (0-4, ordinal traffic bins).
        ``class_names``: list of five ``"traffic_bin_i"`` strings.
        ``raw_edges``: ``[2, E]`` long tensor of directed hyperlink edges,
            not yet cleaned (see :func:`gnn_poset_rank.diagnostics.clean_dag`).
        ``num_nodes``: number of pages.
        ``skipped_edges``: number of self-loop edges dropped.
    """
    with open(features_path) as f:
        next(f)  # header
        node_ids, features, labels = [], [], []
        for line in f:
            node_id, feat_str, label = line.strip().split("\t")
            node_ids.append(int(node_id))
            features.append([int(v) for v in feat_str.split(",")])
            labels.append(int(label))

    # node_id should already be a dense 0..V-1 range in this file, but
    # build an explicit mapping rather than assume it.
    id_to_idx = {nid: i for i, nid in enumerate(sorted(set(node_ids)))}
    order = [id_to_idx[nid] for nid in node_ids]
    x = torch.zeros(len(node_ids), len(features[0]), dtype=torch.float)
    y = torch.zeros(len(node_ids), dtype=torch.long)
    for i, feat, label in zip(order, features, labels):
        x[i] = torch.tensor(feat, dtype=torch.float)
        y[i] = label

    src, tgt = [], []
    skipped = 0
    with open(edges_path) as f:
        next(f)
        for line in f:
            u, v = line.strip().split("\t")
            u, v = int(u), int(v)
            if u == v:
                skipped += 1
                continue
            src.append(id_to_idx[u])
            tgt.append(id_to_idx[v])

    raw_edges = torch.tensor([src, tgt], dtype=torch.long)
    class_names = [f"traffic_bin_{i}" for i in range(5)]
    return {
        "x": x,
        "y": y,
        "class_names": class_names,
        "raw_edges": raw_edges,
        "num_nodes": len(node_ids),
        "skipped_edges": skipped,
    }
