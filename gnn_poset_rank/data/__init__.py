"""Dataset loaders for every dataset used in the paper."""

from gnn_poset_rank.data.citation import load_citation_raw
from gnn_poset_rank.data.chameleon import load_chameleon_raw
from gnn_poset_rank.data.amazon import load_amazon0302_subgraph, build_structural_features, compute_ranks
from gnn_poset_rank.data.letor import load_letor_file
from gnn_poset_rank.data.skyline import load_skyline_csv, find_best_dimension_pair, build_skyline_dataset
from gnn_poset_rank.data.synthetic import generate_pareto_ranking_data

__all__ = [
    "load_citation_raw", "load_chameleon_raw",
    "load_amazon0302_subgraph", "build_structural_features", "compute_ranks",
    "load_letor_file",
    "load_skyline_csv", "find_best_dimension_pair", "build_skyline_dataset",
    "generate_pareto_ranking_data",
]
