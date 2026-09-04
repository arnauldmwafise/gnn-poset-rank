"""Shared encoder and the two prediction heads compared throughout the paper."""

from gnn_poset_rank.models.encoders import DirectedOrderConv, SharedEncoder
from gnn_poset_rank.models.heads import TotalOrderScoreModel, PartialOrderModel, DOMINATES, DOMINATED, INCOMPARABLE

__all__ = [
    "DirectedOrderConv", "SharedEncoder",
    "TotalOrderScoreModel", "PartialOrderModel",
    "DOMINATES", "DOMINATED", "INCOMPARABLE",
]
