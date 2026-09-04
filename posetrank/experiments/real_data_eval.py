"""
posetrank.experiments.real_data_eval
=======================================

Reproduces Table 3: the real-data validation on four graph-structured
datasets (Cora, CiteSeer, Chameleon, Amazon0302), all in the high-SIR
regime (Section 5.4).

Ground truth: ``dominates[i, j]`` is True iff ``j`` is reachable from
``i`` in the transitive closure of the dataset's cleaned DAG -- the exact
same reachability notion used to compute the Structural Incomparability
Rate for these datasets (:mod:`posetrank.diagnostics`), keeping the
definition of "dominates" and "incomparable" fully consistent between the
Table 1 audit and this modeling comparison.

Leakage discipline
-------------------
Cover edges are split into train and test sets *before* message passing:
the model only ever sees training-set edges as its input graph, at both
training and evaluation time. Evaluation pairs are held-out test cover
edges (genuine dominance relations the model never saw) plus sampled
genuinely-incomparable pairs, checked against the *full* true DAG
(training and test edges combined), so that a pair connected only through
a held-out edge is correctly excluded from the incomparable class rather
than mislabeled.
"""

import random
import time
from typing import Dict, Tuple

import networkx as nx
import torch

from posetrank.models import PartialOrderModel, TotalOrderScoreModel
from posetrank.training import train_and_eval


def split_cover_edges(cover_edges: torch.Tensor, test_frac: float = 0.15, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly split cover edges into train and test sets.

    Returns
    -------
    train_edges, test_edges : torch.Tensor
        ``[2, E']`` tensors partitioning ``cover_edges``.
    """
    g = torch.Generator().manual_seed(seed)
    n_edges = cover_edges.size(1)
    perm = torch.randperm(n_edges, generator=g)
    n_test = int(test_frac * n_edges)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return cover_edges[:, train_idx], cover_edges[:, test_idx]


def sample_incomparable_pairs(DAG: nx.DiGraph, num_nodes: int, n_samples: int, seed: int = 0):
    """Sample genuinely incomparable pairs, checked against the full true DAG.

    Uses the same reachability check as
    :func:`posetrank.diagnostics.compute_order_diagnostics`'s SIR
    computation. Bounded rejection sampling (up to ``50 * n_samples``
    attempts), since sparse graphs can have very few comparable pairs and
    a naive unbounded search would be needlessly slow but is not at risk
    of hanging (unlike :func:`posetrank.training.sample_stratified_pairs`,
    which handles the symmetric near-total-order case).

    Returns
    -------
    list[tuple[int, int]]
        Up to ``n_samples`` genuinely incomparable ``(a, b)`` pairs.
    """
    rng = random.Random(seed)
    pairs = []
    attempts = 0
    max_attempts = n_samples * 50
    desc_cache = {}
    while len(pairs) < n_samples and attempts < max_attempts:
        attempts += 1
        a = rng.randrange(num_nodes)
        b = rng.randrange(num_nodes)
        if a == b:
            continue
        if a not in desc_cache:
            desc_cache[a] = nx.descendants(DAG, a)
        if b in desc_cache[a]:
            continue
        if b not in desc_cache:
            desc_cache[b] = nx.descendants(DAG, b)
        if a in desc_cache[b]:
            continue
        pairs.append((a, b))
    return pairs


def build_train_test_pairs(
    cover_edges: torch.Tensor,
    num_nodes: int,
    n_per_class_train: int = 400,
    n_per_class_test: int = 150,
    seed: int = 0,
):
    """Build leakage-safe train/test pairs for a real graph dataset.

    Returns
    -------
    train_edges : torch.Tensor
        ``[2, E']`` message-passing edges (train-only; never test edges).
    train_i, train_j, train_labels, test_i, test_j, test_labels : torch.Tensor
        Pair indices and three-way labels for training and evaluation.
    """
    train_edges, test_edges = split_cover_edges(cover_edges, test_frac=0.15, seed=seed)

    full_DAG = nx.DiGraph()
    full_DAG.add_nodes_from(range(num_nodes))
    full_DAG.add_edges_from(zip(cover_edges[0].tolist(), cover_edges[1].tolist()))

    train_i_list, train_j_list = train_edges[0].tolist(), train_edges[1].tolist()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_i_list), generator=g)[:n_per_class_train]
    tr_i = torch.tensor([train_i_list[k] for k in perm])
    tr_j = torch.tensor([train_j_list[k] for k in perm])
    tr_rev_i, tr_rev_j = tr_j.clone(), tr_i.clone()
    tr_inc_pairs = sample_incomparable_pairs(full_DAG, num_nodes, len(perm), seed=seed + 1)
    tr_inc_i = torch.tensor([a for a, b in tr_inc_pairs])
    tr_inc_j = torch.tensor([b for a, b in tr_inc_pairs])

    train_all_i = torch.cat([tr_i, tr_rev_i, tr_inc_i])
    train_all_j = torch.cat([tr_j, tr_rev_j, tr_inc_j])
    train_labels = torch.cat(
        [
            torch.zeros(len(tr_i), dtype=torch.long),
            torch.ones(len(tr_rev_i), dtype=torch.long),
            torch.full((len(tr_inc_i),), 2, dtype=torch.long),
        ]
    )

    test_i_list, test_j_list = test_edges[0].tolist(), test_edges[1].tolist()
    n_test = min(n_per_class_test, len(test_i_list))
    perm_test = torch.randperm(len(test_i_list), generator=g)[:n_test]
    te_i = torch.tensor([test_i_list[k] for k in perm_test])
    te_j = torch.tensor([test_j_list[k] for k in perm_test])
    te_rev_i, te_rev_j = te_j.clone(), te_i.clone()
    te_inc_pairs = sample_incomparable_pairs(full_DAG, num_nodes, n_test, seed=seed + 2)
    te_inc_i = torch.tensor([a for a, b in te_inc_pairs])
    te_inc_j = torch.tensor([b for a, b in te_inc_pairs])

    test_all_i = torch.cat([te_i, te_rev_i, te_inc_i])
    test_all_j = torch.cat([te_j, te_rev_j, te_inc_j])
    test_labels = torch.cat(
        [
            torch.zeros(len(te_i), dtype=torch.long),
            torch.ones(len(te_rev_i), dtype=torch.long),
            torch.full((len(te_inc_i),), 2, dtype=torch.long),
        ]
    )

    return train_edges, train_all_i, train_all_j, train_labels, test_all_i, test_all_j, test_labels


def run_real_data_comparison(name: str, x: torch.Tensor, cover_edges: torch.Tensor, num_nodes: int, seed: int = 0, verbose: bool = True) -> Dict:
    """Run the full TotalOrderScoreModel-vs-PartialOrderModel comparison on one real dataset.

    Parameters
    ----------
    name : str
        Dataset name, used only for logging.
    x : torch.Tensor
        ``[N, feature_dim]`` node features.
    cover_edges : torch.Tensor
        ``[2, E]`` long tensor, the dataset's full Hasse diagram (from
        :func:`posetrank.diagnostics.clean_dag`).
    num_nodes : int
        Number of nodes ``N``.
    seed : int, default 0
        Random seed, driving the train/test split, pair sampling, and
        model initialization together.
    verbose : bool, default True
        If True, print progress and results.

    Returns
    -------
    dict
        ``TotalOrderScoreModel``, ``PartialOrderModel``: each a dict with
        ``overall`` accuracy and ``per_class`` accuracy breakdown.
        ``advantage``: PartialOrderModel overall accuracy minus
            TotalOrderScoreModel overall accuracy.
        ``inc_advantage``: the same, restricted to the incomparable class.
    """
    if verbose:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    t0 = time.time()
    train_edges, tr_i, tr_j, tr_lab, te_i, te_j, te_lab = build_train_test_pairs(cover_edges, num_nodes, seed=seed)
    if verbose:
        print(f"train pairs: {tr_i.size(0)}, test pairs: {te_i.size(0)} (built in {time.time() - t0:.1f}s)")

    results = {}
    for model_name, model_cls in [
        ("TotalOrderScoreModel", TotalOrderScoreModel),
        ("PartialOrderModel", PartialOrderModel),
    ]:
        t0 = time.time()
        overall, per_class = train_and_eval(
            model_cls, x, train_edges, tr_i, tr_j, tr_lab, te_i, te_j, te_lab, init_seed=seed
        )
        dt = time.time() - t0
        if verbose:
            print(
                f"{model_name}: overall={overall:.4f}  "
                f"dom={per_class['i_dominates_j']:.4f} rev={per_class['j_dominates_i']:.4f} "
                f"inc={per_class['incomparable']:.4f}  ({dt:.1f}s)"
            )
        results[model_name] = {"overall": overall, "per_class": per_class}

    advantage = results["PartialOrderModel"]["overall"] - results["TotalOrderScoreModel"]["overall"]
    inc_advantage = (
        results["PartialOrderModel"]["per_class"]["incomparable"]
        - results["TotalOrderScoreModel"]["per_class"]["incomparable"]
    )
    if verbose:
        print(f"advantage: {advantage:+.4f}  (incomparable-class: {inc_advantage:+.4f})")
    results["advantage"] = advantage
    results["inc_advantage"] = inc_advantage
    return results
