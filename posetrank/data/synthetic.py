"""
posetrank.data.synthetic
==========================

Synthetic ranking/comparison dataset generator with a *genuine* partial
order as ground truth, used throughout the paper for the controlled
K-scan (Section 5.3) and feature-informativeness study (Section 5.7.1).

Design
------
Each of ``num_items`` items has a ``num_quality_dims``-dimensional latent
quality vector drawn i.i.d. uniformly. The true preference relation is
Pareto dominance: item A dominates item B iff A is at least as good as B
on every quality dimension and strictly better on at least one. This is
a mathematical fact, not an empirical property: Pareto dominance is
always antisymmetric and transitive, and therefore always defines a
genuine strict partial order with no possibility of cycles (verified
computationally on generated data as a correctness check, in addition to
relying on the mathematical guarantee).

``num_quality_dims`` controls the resulting Structural Incomparability
Rate directly and predictably: at 1 dimension, dominance reduces to a
total order and SIR is 0 by construction; SIR increases rapidly as the
dimension count grows (the well-documented "curse of dimensionality" for
Pareto/skyline queries), giving direct experimental control over SIR via
a single integer parameter that real data does not offer.

Node features are a noisy linear projection of the true quality vector,
never the quality vector itself, so the feature-to-label relationship a
model must learn is realistically indirect. The observed comparison
graph made available for message passing is a sparse random subsample of
the true cover relation (Hasse diagram), mirroring how real comparison
data typically reveals only a sparse subset of the relations that hold
in the underlying ground truth. ``feature_noise`` independently controls
how informative those features are about the true quality vector,
without affecting the dominance relation (and therefore SIR) at all --
this independence is what makes the feature-informativeness study in
Section 5.7.1 a clean, single-variable manipulation.
"""

from typing import Dict

import networkx as nx
import torch


def generate_pareto_ranking_data(
    num_items: int = 2000,
    num_quality_dims: int = 4,
    feature_dim: int = 64,
    feature_noise: float = 0.4,
    observed_edge_fraction: float = 0.15,
    seed: int = 0,
) -> Dict:
    """Generate a synthetic Pareto-dominance ranking dataset.

    Parameters
    ----------
    num_items : int, default 2000
        Number of items ``N``.
    num_quality_dims : int, default 4
        Number ``K`` of latent quality dimensions. Controls SIR directly:
        ``K=1`` gives SIR approx. 0%; ``K=2`` gives SIR approx. 50% (the
        moderate-SIR peak used throughout the paper); larger ``K`` gives
        rapidly increasing SIR.
    feature_dim : int, default 64
        Dimensionality of the observed node features.
    feature_noise : float, default 0.4
        Standard deviation of the i.i.d. Gaussian noise added to the
        linear projection of the quality vector when generating features.
        Controls feature informativeness; does not affect SIR (see module
        docstring).
    observed_edge_fraction : float, default 0.15
        Fraction of the true cover relation's edges randomly retained as
        the "observed" message-passing graph given to a model.
    seed : int, default 0
        Random seed; every source of randomness (quality vector,
        projection matrix, feature noise, observed-edge subsampling) is
        derived from this single seed for full reproducibility.

    Returns
    -------
    dict
        ``x``: ``[N, feature_dim]`` float tensor, noisy observed features.
        ``quality``: ``[N, K]`` float tensor, the true (not observed) quality vector.
        ``num_items``: ``N``.
        ``dominates``: ``[N, N]`` bool tensor, ground-truth dominance relation.
        ``cover_edges``: ``[2, E]`` long tensor, the full true Hasse diagram.
        ``observed_edges``: ``[2, E']`` long tensor, the sparse graph given to a model.
        ``incomparable_rate``: fraction of ordered pairs with neither
            direction of dominance (a fast, exact companion to the
            sampled Structural Incomparability Rate in
            :mod:`posetrank.diagnostics`, computed directly from
            ``dominates`` without needing graph reachability).
        ``n_dominant_pairs``: number of ordered pairs with ``i`` dominating ``j``.

    Examples
    --------
    >>> data = generate_pareto_ranking_data(num_items=200, num_quality_dims=1, seed=0)
    >>> data['incomparable_rate']  # K=1 reduces to a total order
    0.0
    """
    g = torch.Generator().manual_seed(seed)

    quality = torch.rand(num_items, num_quality_dims, generator=g)

    # Noisy, higher-dimensional observation of quality -- what a model
    # actually receives as node features, never the ground-truth vector.
    proj = torch.randn(num_quality_dims, feature_dim, generator=g)
    x = quality @ proj + feature_noise * torch.randn(num_items, feature_dim, generator=g)

    dominates_or_equal = (quality.unsqueeze(0) >= quality.unsqueeze(1)).all(dim=-1)
    strictly_better_somewhere = (quality.unsqueeze(0) > quality.unsqueeze(1)).any(dim=-1)
    dominates = dominates_or_equal & strictly_better_somewhere
    dominates.fill_diagonal_(False)

    n_dominant_pairs = dominates.sum().item()
    total_pairs = num_items * (num_items - 1)
    incomparable_pairs = total_pairs - 2 * n_dominant_pairs
    incomparable_rate = incomparable_pairs / total_pairs if total_pairs > 0 else float("nan")

    src, tgt = dominates.nonzero(as_tuple=True)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_items))
    G.add_edges_from(zip(src.tolist(), tgt.tolist()))
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError(
            "Pareto dominance must be a strict partial order (acyclic by "
            "construction); a cycle here indicates a bug, not a property "
            "of the sampled quality vectors."
        )
    TR = nx.transitive_reduction(G)
    cover_edges = torch.tensor(list(TR.edges()), dtype=torch.long).t().contiguous()

    n_cover = cover_edges.size(1)
    perm = torch.randperm(n_cover, generator=g)
    n_observed = int(observed_edge_fraction * n_cover)
    observed_edges = cover_edges[:, perm[:n_observed]]

    return {
        "x": x,
        "quality": quality,
        "num_items": num_items,
        "dominates": dominates,
        "cover_edges": cover_edges,
        "observed_edges": observed_edges,
        "incomparable_rate": incomparable_rate,
        "n_dominant_pairs": n_dominant_pairs,
    }
