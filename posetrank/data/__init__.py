"""Dataset loaders for every dataset used in the paper."""

from posetrank.data.citation import load_citation_raw
from posetrank.data.chameleon import load_chameleon_raw
from posetrank.data.amazon import load_amazon0302_subgraph, build_structural_features, compute_ranks
from posetrank.data.letor import load_letor_file
from posetrank.data.skyline import load_skyline_csv, find_best_dimension_pair, build_skyline_dataset
from posetrank.data.synthetic import generate_pareto_ranking_data

__all__ = [
    "load_citation_raw", "load_chameleon_raw",
    "load_amazon0302_subgraph", "build_structural_features", "compute_ranks",
    "load_letor_file",
    "load_skyline_csv", "find_best_dimension_pair", "build_skyline_dataset",
    "generate_pareto_ranking_data",
]
