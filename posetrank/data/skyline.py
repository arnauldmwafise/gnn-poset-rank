"""
posetrank.data.skyline
========================

Loader for the NBA and HOUSE skyline-query benchmark datasets, drawn from
the SkyBench repository (Chester, https://github.com/sean-chester/SkyBench).

NBA (17,264 player-season records, 8 normalized statistical dimensions)
and HOUSE (127,931 real-estate records, 6 normalized dimensions) are used
in the paper to construct real Pareto dominance relations at a
deliberately targeted Structural Incomparability Rate near the
moderate-SIR peak identified by the synthetic K-scan (Section 5.3),
providing real-world validation of that peak (Section 5.6) and a
cross-domain replication check (Section 5.7).

Both files were independently verified to be genuine real-world data,
not synthetically generated surrogates, by inspecting their empirical
inter-dimension correlation structure: real data shows varied,
substantial correlations, whereas a synthetically-generated
"uncorrelated" dataset (as the source filenames' "U" suffix might
suggest) would show near-zero correlation between every dimension pair.
See :func:`verify_correlation_structure`.
"""

import itertools
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

from posetrank.diagnostics import compute_order_diagnostics


def load_skyline_csv(path: str, n_dims: int) -> np.ndarray:
    """Load a raw SkyBench-format CSV (comma-separated, no header, trailing comma).

    Parameters
    ----------
    path : str
        Path to the CSV file (e.g. ``nba-U-8-17264.csv``).
    n_dims : int
        Number of data columns (8 for NBA, 6 for HOUSE).

    Returns
    -------
    numpy.ndarray
        ``[n_rows, n_dims]`` array of normalized values in ``[0, 1]``.
    """
    return np.loadtxt(path, delimiter=",", usecols=range(n_dims))


def verify_correlation_structure(data: np.ndarray) -> Dict:
    """Report the inter-dimension correlation range, to distinguish real from synthetic data.

    Parameters
    ----------
    data : numpy.ndarray
        ``[n_rows, n_dims]`` array, as returned by :func:`load_skyline_csv`.

    Returns
    -------
    dict
        ``min_correlation``, ``max_correlation``: extremes of the
        pairwise Pearson correlation matrix (excluding the diagonal).
        A genuinely uncorrelated synthetic dataset would show both
        values close to zero; real data typically does not.
    """
    corr = np.corrcoef(data.T)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {"min_correlation": float(off_diag.min()), "max_correlation": float(off_diag.max())}


def find_best_dimension_pair(
    data: np.ndarray, target_sir: float = 0.5, sample_n: int = 1500, seed: int = 0
) -> List[Tuple[Tuple[int, int], float]]:
    """Search all dimension pairs for the one closest to a target SIR.

    For each pair of columns, constructs the induced Pareto dominance
    relation on a random subsample of rows, computes its Structural
    Incomparability Rate, and returns all pairs sorted by closeness to
    ``target_sir``.

    Parameters
    ----------
    data : numpy.ndarray
        ``[n_rows, n_dims]`` array, as returned by :func:`load_skyline_csv`.
    target_sir : float, default 0.5
        Target SIR to search for (0.5 matches the moderate-SIR peak).
    sample_n : int, default 1500
        Number of rows to subsample for the (comparatively expensive)
        diagnostic search; the final dataset used for model training can
        use a different, typically larger, sample (see
        :func:`build_skyline_dataset`).
    seed : int, default 0
        Random seed for subsampling.

    Returns
    -------
    list[tuple[tuple[int, int], float]]
        ``(dimension_pair, sir)`` tuples, sorted by ``|sir - target_sir|`` ascending.
    """
    n_dims = data.shape[1]
    np.random.seed(seed)  # legacy global-state API, matching the exact
    # seed behavior used to produce the paper's reported numbers -- NOT
    # np.random.default_rng, which is a different (also valid, but
    # numerically different) PRNG stream even at the "same" seed value
    idx = np.random.choice(len(data), min(sample_n, len(data)), replace=False)
    sub = torch.tensor(data[idx], dtype=torch.float)
    n = len(sub)

    results = []
    for combo in itertools.combinations(range(n_dims), 2):
        q = sub[:, list(combo)]
        dom_eq = (q.unsqueeze(0) >= q.unsqueeze(1)).all(dim=-1)
        strict = (q.unsqueeze(0) > q.unsqueeze(1)).any(dim=-1)
        dominates = dom_eq & strict
        dominates.fill_diagonal_(False)
        src, tgt = dominates.nonzero(as_tuple=True)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(src.tolist(), tgt.tolist()))
        TR = nx.transitive_reduction(G)
        cover_edges_list = list(TR.edges())
        if not cover_edges_list:
            continue
        cover_edges = torch.tensor(cover_edges_list, dtype=torch.long).t().contiguous()
        diag = compute_order_diagnostics(cover_edges, n, sample_pairs_for_sir=50_000, seed=seed)
        results.append((combo, diag["structural_incomparability_rate"]))

    results.sort(key=lambda item: abs(item[1] - target_sir))
    return results


def build_skyline_dataset(data: np.ndarray, label_dims: Tuple[int, ...], n: int = 3000, seed: int = 0) -> Dict:
    """Build a Pareto-dominance dataset from selected dimensions of skyline data.

    The dimensions in ``label_dims`` define the ground-truth dominance
    relation; every *other* dimension in the source data is used as a
    node feature, so that the features available to a model do not
    directly overlap with the dimensions defining the label, while still
    allowing the realistic partial correlation real features would have
    with the comparison criteria.

    Parameters
    ----------
    data : numpy.ndarray
        ``[n_rows, n_dims]`` array, as returned by :func:`load_skyline_csv`.
    label_dims : tuple[int, ...]
        Column indices (typically a pair) defining the dominance relation.
    n : int, default 3000
        Number of rows to subsample for the working dataset.
    seed : int, default 0
        Random seed for subsampling.

    Returns
    -------
    dict
        ``x``: ``[n, n_dims - len(label_dims)]`` float tensor of node features.
        ``quality``: ``[n, len(label_dims)]`` float tensor, the true label dimensions.
        ``dominates``: ``[n, n]`` bool tensor, ground-truth dominance relation.
        ``cover_edges``: ``[2, E]`` long tensor, the Hasse diagram.
        ``num_items``: ``n``.
    """
    n_dims = data.shape[1]
    np.random.seed(seed)  # see the matching note in find_best_dimension_pair
    idx = np.random.choice(len(data), n, replace=False)
    full = torch.tensor(data[idx], dtype=torch.float)

    feature_dims = [d for d in range(n_dims) if d not in label_dims]
    quality = full[:, list(label_dims)]
    x = full[:, feature_dims]

    dom_eq = (quality.unsqueeze(0) >= quality.unsqueeze(1)).all(dim=-1)
    strict = (quality.unsqueeze(0) > quality.unsqueeze(1)).any(dim=-1)
    dominates = dom_eq & strict
    dominates.fill_diagonal_(False)

    src, tgt = dominates.nonzero(as_tuple=True)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), tgt.tolist()))
    TR = nx.transitive_reduction(G)
    cover_edges = torch.tensor(list(TR.edges()), dtype=torch.long).t().contiguous()

    return {"x": x, "quality": quality, "dominates": dominates, "cover_edges": cover_edges, "num_items": n}
