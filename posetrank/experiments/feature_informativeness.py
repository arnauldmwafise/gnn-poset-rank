"""
posetrank.experiments.feature_informativeness
================================================

Reproduces Table 6: the controlled synthetic test of feature
informativeness (Section 5.7.1).

Fixes the number of quality dimensions at K = 2 (Structural
Incomparability Rate approx. 50%, the moderate-SIR peak identified by
the K-scan) and varies only the feature-noise level, which controls how
informative the observed node features are about the true quality
vector without affecting SIR at all (see
:mod:`posetrank.data.synthetic`). This isolates feature informativeness
as the sole manipulated variable, in the same systematic, multi-seed
spirit as the K-scan (:mod:`posetrank.experiments.k_scan`) isolates SIR.
"""

import statistics
from typing import Dict, Sequence

import torch

from posetrank.data import generate_pareto_ranking_data
from posetrank.diagnostics import compute_order_diagnostics
from posetrank.models import PartialOrderModel, TotalOrderScoreModel
from posetrank.training import sample_stratified_pairs, train_and_eval


def measure_feature_informativeness(data: Dict) -> float:
    """Maximum absolute correlation between any feature dimension and any true quality dimension.

    Identical metric to the one used for the real-world NBA-versus-HOUSE
    comparison (Section 5.7), for direct comparability between the
    controlled synthetic study and that real-data observation.

    Parameters
    ----------
    data : dict
        As returned by :func:`posetrank.data.generate_pareto_ranking_data`
        (must contain ``x`` and ``quality``).

    Returns
    -------
    float
        The maximum absolute Pearson correlation, in ``[0, 1]``.
    """
    x = data["x"]
    q = data["quality"]
    x_c = x - x.mean(0, keepdim=True)
    q_c = q - q.mean(0, keepdim=True)
    x_std = x_c.std(0, keepdim=True) + 1e-8
    q_std = q_c.std(0, keepdim=True) + 1e-8
    corr = (x_c.t() @ q_c) / (x.size(0) - 1) / (x_std.t() @ q_std)
    return corr.abs().max().item()


def run_informativeness_point(
    sigma: float,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    num_items: int = 1500,
    num_quality_dims: int = 2,
    feature_dim: int = 64,
    observed_edge_fraction: float = 0.15,
) -> Dict:
    """Run one point of the feature-informativeness scan (one noise level, multiple seeds).

    Parameters
    ----------
    sigma : float
        The feature-noise standard deviation
        (:func:`posetrank.data.generate_pareto_ranking_data`'s
        ``feature_noise`` parameter).
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Random seeds; each independently drives data generation, pair
        sampling, and model initialization.
    num_items, num_quality_dims, feature_dim, observed_edge_fraction
        Passed through to the synthetic generator. ``num_quality_dims``
        defaults to 2 (SIR approx. 50%), matching the paper.

    Returns
    -------
    dict
        ``sigma``: the input noise level.
        ``sir_mean``: mean measured SIR across seeds (should stay close
            to the K=2 baseline regardless of sigma; verifying this is
            what confirms the manipulation is clean).
        ``informativeness_mean``: mean feature informativeness across seeds.
        ``advantage_mean``, ``advantage_std``: mean and standard
            deviation of the accuracy advantage across seeds.
        ``sign_stable``: True if the advantage has the same sign in every seed.
        ``n_negative``: number of seeds with negative advantage.
        ``raw_advantages``: per-seed advantage values.
    """
    sirs, informativeness, advantages = [], [], []
    for seed in seeds:
        data = generate_pareto_ranking_data(
            num_items=num_items,
            num_quality_dims=num_quality_dims,
            feature_dim=feature_dim,
            feature_noise=sigma,
            observed_edge_fraction=observed_edge_fraction,
            seed=seed,
        )
        n = data["num_items"]
        diag = compute_order_diagnostics(
            data["cover_edges"], n, sample_pairs_for_sir=200_000, seed=seed + 100
        )
        sirs.append(diag["structural_incomparability_rate"])
        informativeness.append(measure_feature_informativeness(data))

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
        "sigma": sigma,
        "sir_mean": statistics.mean(sirs),
        "informativeness_mean": statistics.mean(informativeness),
        "advantage_mean": statistics.mean(advantages),
        "advantage_std": statistics.stdev(advantages) if len(advantages) > 1 else 0.0,
        "sign_stable": all(a >= 0 for a in advantages) or all(a <= 0 for a in advantages),
        "n_negative": sum(1 for a in advantages if a < 0),
        "raw_advantages": advantages,
    }


def run_informativeness_scan(
    sigma_values: Sequence[float] = (0.05, 0.4, 1.0, 2.0, 4.0, 8.0, 16.0),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> Dict[float, Dict]:
    """Run the full feature-informativeness scan, reproducing every row of Table 6.

    Parameters
    ----------
    sigma_values : sequence of float, default (0.05, 0.4, 1.0, 2.0, 4.0, 8.0, 16.0)
        The feature-noise levels to sweep (chosen to span from
        near-perfect to near-zero feature informativeness).
    seeds : sequence of int, default (0, 1, 2, 3, 4)
        Seeds used for every noise level.

    Returns
    -------
    dict[float, dict]
        Maps each sigma to its :func:`run_informativeness_point` result.
    """
    return {sigma: run_informativeness_point(sigma, seeds=seeds) for sigma in sigma_values}
