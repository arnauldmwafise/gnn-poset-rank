"""Shared encoder and the two prediction heads compared throughout the paper."""

from posetrank.models.encoders import DirectedOrderConv, SharedEncoder
from posetrank.models.heads import TotalOrderScoreModel, PartialOrderModel, DOMINATES, DOMINATED, INCOMPARABLE

__all__ = [
    "DirectedOrderConv", "SharedEncoder",
    "TotalOrderScoreModel", "PartialOrderModel",
    "DOMINATES", "DOMINATED", "INCOMPARABLE",
]
