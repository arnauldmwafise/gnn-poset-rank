"""
posetrank.experiments.skyline_eval
=====================================

Reproduces Tables 4, 4a, and 5: the real-world validation of the
moderate-SIR peak on the NBA skyline dataset (Section 5.6), and the
cross-domain replication check on HOUSE (Section 5.7).

Both datasets use :func:`posetrank.data.find_best_dimension_pair` to
locate a real dimension pair whose induced Pareto dominance relation
sits near the target Structural Incomparability Rate, then
:func:`posetrank.data.build_skyline_dataset` to construct the resulting
dataset, then the same five-seed resampling protocol used throughout
Section 5 (data split, pair sampling, and model initialization all
varying independently per seed).
"""

import statistics
from typing import Dict, Sequence, Tuple

import torch

from posetrank.data.skyline import build_skyline_dataset
from posetrank.models import PartialOrderModel, TotalOrderScoreModel
from posetrank.training import sample_stratified_pairs, train_and_eval


def run_skyline_multiseed(
    dataset: Dict, seeds: Sequence[int] = (0, 1, 2, 3, 4)
) -> Dict:
    """Run the TotalOrderScoreModel-vs-PartialOrderModel comparison on a skyline dataset.

    Parameters
    ----------
    dataset : dict
        As returned by :func:`posetrank.data.build_skyline_dataset`.
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Random seeds; each independently drives the observed-edge
        subsample, pair sampling, and model initialization.

    Returns
    -------
    dict
        ``advantage_mean``, ``advantage_std``: mean and standard
            deviation of the accuracy advantage across seeds.
        ``inc_advantage_mean``, ``inc_advantage_std``: the same,
            restricted to the incomparable class.
        ``sign_stable``: True if the advantage has the same sign in every seed.
        ``n_negative``: number of seeds with negative advantage.
        ``raw_advantages``: per-seed advantage values.
    """
    x, dominates, cover_edges, n = dataset["x"], dataset["dominates"], dataset["cover_edges"], dataset["num_items"]
    advantages, inc_advantages = [], []
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(cover_edges.size(1), generator=g)
        observed_edges = cover_edges[:, perm[: int(0.15 * cover_edges.size(1))]]

        train_i, train_j, train_labels = sample_stratified_pairs(dominates, n, n_per_class=500, seed=seed + 100)
        test_i, test_j, test_labels = sample_stratified_pairs(dominates, n, n_per_class=200, seed=seed + 200)

        overall_total, per_class_total = train_and_eval(
            TotalOrderScoreModel, x, observed_edges, train_i, train_j, train_labels,
            test_i, test_j, test_labels, init_seed=seed,
        )
        overall_partial, per_class_partial = train_and_eval(
            PartialOrderModel, x, observed_edges, train_i, train_j, train_labels,
            test_i, test_j, test_labels, init_seed=seed,
        )
        advantages.append(overall_partial - overall_total)
        inc_advantages.append(per_class_partial["incomparable"] - per_class_total["incomparable"])

    return {
        "advantage_mean": statistics.mean(advantages),
        "advantage_std": statistics.stdev(advantages) if len(advantages) > 1 else 0.0,
        "inc_advantage_mean": statistics.mean(inc_advantages),
        "inc_advantage_std": statistics.stdev(inc_advantages) if len(inc_advantages) > 1 else 0.0,
        "sign_stable": all(a >= 0 for a in advantages) or all(a <= 0 for a in advantages),
        "n_negative": sum(1 for a in advantages if a < 0),
        "raw_advantages": advantages,
    }


def run_skyline_experiment(
    data, label_dims: Tuple[int, ...], n: int = 3000, seeds: Sequence[int] = (0, 1, 2, 3, 4)
) -> Dict:
    """Build a skyline dataset for a given dimension pair and run the full comparison.

    Convenience wrapper combining :func:`~posetrank.data.build_skyline_dataset`
    and :func:`run_skyline_multiseed`.

    Parameters
    ----------
    data : numpy.ndarray
        As returned by :func:`posetrank.data.load_skyline_csv`.
    label_dims : tuple[int, ...]
        Column indices defining the dominance relation (typically found
        via :func:`posetrank.data.find_best_dimension_pair`).
    n : int, default 3000
        Working sample size.
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Random seeds for the multi-seed protocol.

    Returns
    -------
    dict
        As returned by :func:`run_skyline_multiseed`.
    """
    dataset = build_skyline_dataset(data, label_dims, n=n, seed=0)
    return run_skyline_multiseed(dataset, seeds=seeds)
