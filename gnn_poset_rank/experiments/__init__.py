"""Scripts reproducing each table in the paper."""

from gnn_poset_rank.experiments.k_scan import run_k_scan, run_k_scan_point
from gnn_poset_rank.experiments.real_data_eval import run_real_data_comparison
from gnn_poset_rank.experiments.skyline_eval import run_skyline_multiseed, run_skyline_experiment
from gnn_poset_rank.experiments.feature_informativeness import (
    measure_feature_informativeness,
    run_informativeness_point,
    run_informativeness_scan,
)
from gnn_poset_rank.experiments.mq2008_eval import run_mq2008_comparison

__all__ = [
    "run_k_scan", "run_k_scan_point",
    "run_real_data_comparison",
    "run_skyline_multiseed", "run_skyline_experiment",
    "measure_feature_informativeness", "run_informativeness_point", "run_informativeness_scan",
    "run_mq2008_comparison",
]
