"""
gnn_poset_rank.experiments.k_scan
==============================

Reproduces Table 2: the synthetic K-scan predictive study (Section 5.3).

Sweeps the number of quality dimensions ``K`` in the synthetic
Pareto-dominance generator, which controls Structural Incomparability
Rate directly, and measures the resulting advantage of
:class:`~gnn_poset_rank.models.PartialOrderModel` over
:class:`~gnn_poset_rank.models.TotalOrderScoreModel` at each SIR level, with
multiple random seeds per level to characterize seed-to-seed variance
rather than report single-run point estimates.
"""

import statistics
from typing import Dict, Sequence

from gnn_poset_rank.data import generate_pareto_ranking_data
from gnn_poset_rank.diagnostics import compute_order_diagnostics
from gnn_poset_rank.models import PartialOrderModel, TotalOrderScoreModel
from gnn_poset_rank.training import sample_stratified_pairs, train_and_eval


def run_k_scan_point(
    k: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    num_items: int = 1500,
    feature_dim: int = 64,
    feature_noise: float = 0.4,
    observed_edge_fraction: float = 0.15,
) -> Dict:
    """Run one point of the K-scan (one value of K, multiple seeds).

    Each seed independently regenerates the synthetic dataset (not a
    resampled split of one fixed dataset), resamples training/test pairs,
    and reinitializes both models, so the resulting statistics capture
    genuine end-to-end variance, not only data-sampling variance.

    Parameters
    ----------
    k : int
        Number of quality dimensions (controls SIR).
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Random seeds; each drives data generation, pair sampling, and
        model initialization independently for that seed's run.
    num_items, feature_dim, feature_noise, observed_edge_fraction
        Passed through to :func:`~gnn_poset_rank.data.generate_pareto_ranking_data`.

    Returns
    -------
    dict
        ``k``: the input K.
        ``sir_mean``: mean measured SIR across seeds.
        ``advantage_mean``, ``advantage_std``: mean and standard
            deviation of (PartialOrderModel accuracy - TotalOrderScoreModel
            accuracy) across seeds.
        ``sign_stable``: True if the advantage has the same sign in every seed.
        ``n_negative``: number of seeds with a negative advantage.
        ``raw_advantages``: the per-seed advantage values.
    """
    sirs, advantages = [], []
    for seed in seeds:
        data = generate_pareto_ranking_data(
            num_items=num_items,
            num_quality_dims=k,
            feature_dim=feature_dim,
            feature_noise=feature_noise,
            observed_edge_fraction=observed_edge_fraction,
            seed=seed,
        )
        n = data["num_items"]
        diag = compute_order_diagnostics(
            data["cover_edges"], n, sample_pairs_for_sir=200_000, seed=seed + 100
        )
        sirs.append(diag["structural_incomparability_rate"])

        dominates = data["dominates"]
        train_i, train_j, train_labels = sample_stratified_pairs(dominates, n, n_per_class=500, seed=seed + 200)
        test_i, test_j, test_labels = sample_stratified_pairs(dominates, n, n_per_class=200, seed=seed + 300)

        overall_total, _ = train_and_eval(
            TotalOrderScoreModel, data["x"], data["observed_edges"],
            train_i, train_j, train_labels, test_i, test_j, test_labels, init_seed=seed,
        )
        overall_partial, _ = train_and_eval(
            PartialOrderModel, data["x"], data["observed_edges"],
            train_i, train_j, train_labels, test_i, test_j, test_labels, init_seed=seed,
        )
        advantages.append(overall_partial - overall_total)

    return {
        "k": k,
        "sir_mean": statistics.mean(sirs),
        "advantage_mean": statistics.mean(advantages),
        "advantage_std": statistics.stdev(advantages) if len(advantages) > 1 else 0.0,
        "sign_stable": all(a >= 0 for a in advantages) or all(a <= 0 for a in advantages),
        "n_negative": sum(1 for a in advantages if a < 0),
        "raw_advantages": advantages,
    }


def run_k_scan(k_values: Sequence[int] = (1, 2, 3, 4, 5, 6), seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> Dict[int, Dict]:
    """Run the full K-scan, reproducing every row of Table 2.

    Parameters
    ----------
    k_values : sequence of int, default (1, 2, 3, 4, 5, 6)
        The K values to sweep.
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Seeds used for every K value.

    Returns
    -------
    dict[int, dict]
        Maps each K to its :func:`run_k_scan_point` result.
    """
    return {k: run_k_scan_point(k, seeds=seeds) for k in k_values}
