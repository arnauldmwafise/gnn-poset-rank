"""
gnn_poset_rank.experiments.mq2008_eval
====================================

Reproduces the MQ2008 real-world cross-check (Section 5.5): a
within-query-respecting graph model against a naive baseline that pools
all queries' documents into a single graph.

For both models, the graph structure used for message passing is built
purely from document *feature* similarity (a k-nearest-neighbor graph),
never from the relevance labels being predicted, at both training and
evaluation time; relevance labels are used exclusively as supervision
targets for the loss function and as the basis for computing evaluation
accuracy, never as input structure. See :func:`build_within_query_knn_edges`
and the module-level warning on :func:`build_within_query_edges_LEAKY_DO_NOT_USE`
below for why this distinction is enforced explicitly rather than left
implicit: an earlier version of this experiment used the leaky
label-derived graph, producing an artifactually large apparent effect
(Appendix B of the paper).

This module also demonstrates the "naive global graph" anti-pattern that
much graph-based re-ranking work falls into: pooling all documents across
all queries into one feature space and connecting each document to its
k nearest neighbors regardless of query membership, producing mostly
spurious cross-query edges with no ranking meaning.
"""

import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_poset_rank.models.encoders import DirectedOrderConv


class ScoreModel(nn.Module):
    """A single-score ranking model, used only for the MQ2008 within-query-vs-naive comparison.

    Distinct from :class:`~gnn_poset_rank.models.TotalOrderScoreModel`: this
    experiment asks a different question (does query-respecting message-
    passing structure help, versus a naive pooled graph) using a plain
    pairwise ranking objective, not the dominates/dominated/incomparable
    three-way classification used elsewhere in this package.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.conv1 = DirectedOrderConv(hidden_dim, hidden_dim)
        self.conv2 = DirectedOrderConv(hidden_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return self.score_head(h).squeeze(-1)


def pairwise_ranking_loss(scores: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Hinge ranking loss: ``edge_index`` edges are (higher-relevance -> lower-relevance)."""
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, requires_grad=True)
    diff = scores[edge_index[0]] - scores[edge_index[1]]
    return F.relu(1.0 - diff).mean()


def per_query_pairwise_accuracy(scores_by_query: List[torch.Tensor], relevance_by_query: List[torch.Tensor]) -> float:
    """Fraction of within-query relevance-ordered pairs the model scores in the correct order."""
    accs = []
    for scores, relevance in zip(scores_by_query, relevance_by_query):
        n = relevance.size(0)
        correct, total = 0, 0
        for i in range(n):
            for j in range(n):
                if relevance[i] > relevance[j]:
                    total += 1
                    if scores[i] > scores[j]:
                        correct += 1
        if total > 0:
            accs.append(correct / total)
    return sum(accs) / len(accs) if accs else float("nan")


def build_within_query_edges_LEAKY_DO_NOT_USE(relevance: torch.Tensor) -> torch.Tensor:
    """Builds a graph directly from ground-truth labels. NEVER use this for message passing.

    .. warning::
        This function is kept **only** for documentation and transparency
        about a data-leakage bug identified and corrected during this
        project (see Appendix B of the paper): an edge ``i -> j`` is added
        whenever ``relevance[i] > relevance[j]``, so a model given this
        graph as *input structure* can trivially recover the ranking from
        node in/out-degree alone, without learning anything. It is safe
        to use ``relevance`` as a *loss target* (that is normal
        supervision); it must never be used to construct the
        message-passing graph itself. Use
        :func:`build_within_query_knn_edges` for that.
    """
    n = relevance.size(0)
    src, tgt = [], []
    for i in range(n):
        for j in range(n):
            if relevance[i] > relevance[j]:
                src.append(i)
                tgt.append(j)
    if not src:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)


def build_within_query_knn_edges(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Build a k-nearest-neighbor graph from document features only, never from labels.

    This is the leakage-safe structure used for message passing, at both
    training and evaluation time, by :func:`train_per_query_ensemble`.

    Parameters
    ----------
    x : torch.Tensor
        ``[n_docs, feature_dim]`` document features for one query.
    k : int, default 5
        Number of nearest neighbors per document.

    Returns
    -------
    torch.Tensor
        ``[2, n_docs * k]`` long tensor of directed k-NN edges.
    """
    n = x.size(0)
    k = min(k, n - 1)
    if k <= 0:
        return torch.zeros(2, 0, dtype=torch.long)
    sims = x @ x.t()
    sims.fill_diagonal_(-1e9)
    _, topk_idx = sims.topk(k, dim=1)
    src = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    tgt = topk_idx.reshape(-1)
    return torch.stack([src, tgt])


def build_naive_cross_query_graph(queries: List[Dict], k: int = 5):
    """Pool all documents across all queries and connect each to its k nearest neighbors.

    The anti-pattern this experiment is designed to test against: most of
    the resulting edges connect documents from unrelated queries, which
    is spurious structure with no ranking meaning.

    Parameters
    ----------
    queries : list[dict]
        Query-grouped data, as returned by :func:`gnn_poset_rank.data.load_letor_file`.
    k : int, default 5
        Number of nearest neighbors per document.

    Returns
    -------
    all_x, cross_query_edges, all_relevance, query_id, offsets, frac_cross_query
        Pooled features, the resulting k-NN edges, pooled relevance
        labels, a per-document query-id tensor, per-query start offsets
        into the pooled arrays, and the fraction of edges that cross a
        query boundary.
    """
    all_x = torch.cat([q["x"] for q in queries], dim=0)
    offsets = [0]
    for q in queries:
        offsets.append(offsets[-1] + q["n_docs"])
    n_total = all_x.size(0)

    sims = all_x @ all_x.t()
    sims.fill_diagonal_(-1e9)
    _, topk_idx = sims.topk(k, dim=1)
    src = torch.arange(n_total).unsqueeze(1).expand(-1, k).reshape(-1)
    tgt = topk_idx.reshape(-1)
    cross_query_edges = torch.stack([src, tgt])

    all_relevance = torch.cat([q["relevance"] for q in queries])
    query_id = torch.zeros(n_total, dtype=torch.long)
    for qi, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        query_id[start:end] = qi

    frac_cross_query = (query_id[cross_query_edges[0]] != query_id[cross_query_edges[1]]).float().mean().item()
    return all_x, cross_query_edges, all_relevance, query_id, offsets, frac_cross_query


def train_naive_global(train_queries: List[Dict], test_queries: List[Dict], epochs: int = 100) -> float:
    """Train and evaluate the naive pooled-graph baseline.

    Returns
    -------
    float
        Per-query pairwise ranking accuracy on ``test_queries``.
    """
    all_x, cross_edges, _all_rel, _query_id, offsets, _frac_cross = build_naive_cross_query_graph(train_queries, k=5)
    model = ScoreModel(feature_dim=all_x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Supervision is still within-query (the only labels that exist), but
    # message passing runs over the spurious cross-query graph.
    within_edges_list = [build_within_query_edges_LEAKY_DO_NOT_USE(q["relevance"]) for q in train_queries]
    combined_src, combined_tgt = [], []
    for start, e in zip(offsets[:-1], within_edges_list):
        if e.size(1) > 0:
            combined_src.append(e[0] + start)
            combined_tgt.append(e[1] + start)
    train_pairs = torch.stack([torch.cat(combined_src), torch.cat(combined_tgt)])

    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        scores = model(all_x, cross_edges)
        loss = pairwise_ranking_loss(scores, train_pairs)
        loss.backward()
        opt.step()

    model.eval()
    test_x, test_cross_edges, _test_rel, _test_qid, test_offsets, _ = build_naive_cross_query_graph(test_queries, k=5)
    with torch.no_grad():
        scores = model(test_x, test_cross_edges)
    scores_by_query = [scores[s:e] for s, e in zip(test_offsets[:-1], test_offsets[1:])]
    relevance_by_query = [q["relevance"] for q in test_queries]
    return per_query_pairwise_accuracy(scores_by_query, relevance_by_query)


def train_per_query_ensemble(train_queries: List[Dict], test_queries: List[Dict], epochs: int = 100) -> float:
    """Train and evaluate the within-query-respecting, leakage-safe model.

    Message-passing structure is built from document features only
    (:func:`build_within_query_knn_edges`), at both training and
    evaluation time; relevance labels are used exclusively as loss
    targets, never as input structure.

    Returns
    -------
    float
        Per-query pairwise ranking accuracy on ``test_queries``.
    """
    model = ScoreModel(feature_dim=train_queries[0]["x"].size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_struct_edges = [build_within_query_knn_edges(q["x"], k=5) for q in train_queries]
    train_loss_pairs = [build_within_query_edges_LEAKY_DO_NOT_USE(q["relevance"]) for q in train_queries]

    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for q, struct_edges, loss_pairs in zip(train_queries, train_struct_edges, train_loss_pairs):
            scores = model(q["x"], struct_edges)
            loss = pairwise_ranking_loss(scores, loss_pairs)
            total_loss = total_loss + loss
        (total_loss / len(train_queries)).backward()
        opt.step()

    model.eval()
    scores_by_query, relevance_by_query = [], []
    with torch.no_grad():
        for q in test_queries:
            struct_edges = build_within_query_knn_edges(q["x"], k=5)  # features only, same as training
            scores = model(q["x"], struct_edges)
            scores_by_query.append(scores)
            relevance_by_query.append(q["relevance"])
    return per_query_pairwise_accuracy(scores_by_query, relevance_by_query)


def run_mq2008_comparison(train_queries: List[Dict], test_queries: List[Dict], epochs: int = 100) -> Dict:
    """Run both the naive and within-query models and report the comparison.

    Parameters
    ----------
    train_queries, test_queries : list[dict]
        As returned by :func:`gnn_poset_rank.data.load_letor_file`.
    epochs : int, default 100
        Training epochs for both models.

    Returns
    -------
    dict
        ``naive_accuracy``, ``within_query_accuracy``: per-query pairwise
            accuracy for each model.
        ``advantage``: within-query accuracy minus naive accuracy.
    """
    t0 = time.time()
    acc_naive = train_naive_global(train_queries, test_queries, epochs=epochs)
    print(f"Naive global (spurious cross-query edges): {acc_naive:.4f}  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    acc_ensemble = train_per_query_ensemble(train_queries, test_queries, epochs=epochs)
    print(f"Per-query ensemble (within-query, leakage-safe): {acc_ensemble:.4f}  ({time.time() - t0:.1f}s)")

    return {
        "naive_accuracy": acc_naive,
        "within_query_accuracy": acc_ensemble,
        "advantage": acc_ensemble - acc_naive,
    }
