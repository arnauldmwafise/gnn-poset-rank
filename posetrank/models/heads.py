"""
posetrank.models.heads
========================

The two prediction heads compared throughout the paper. Both share an
identical :class:`~posetrank.models.encoders.SharedEncoder`; they differ
only in how -- or whether -- they represent incomparability, isolating
the effect of the prediction head's expressiveness from any difference
in encoder capacity.

TotalOrderScoreModel
    A total-order (single-score) baseline architecturally aligned with
    GNNRank (He et al., 2022): produces one scalar score per item, and
    classifies a pair by thresholding the score difference against a
    learned margin. This thresholding rule is the *only* mechanism by
    which a single-score architecture can express incomparability at
    all -- it does not learn a separate representation of incomparability
    so much as decline to commit to a direction when the score
    difference is small.

PartialOrderModel
    The partial-order-aware architecture: retains a full vector embedding
    per item, and feeds the concatenation of both endpoint embeddings and
    their difference into a small classification head that predicts one
    of three classes directly (dominates / dominated / incomparable).
    Incomparability is a direct classification outcome, never the
    residual case left over after thresholding a scalar difference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from posetrank.models.encoders import SharedEncoder

#: Class index convention used by both models' ``predict_pairs`` and loss
#: functions: 0 = "i dominates j", 1 = "j dominates i", 2 = "incomparable".
DOMINATES, DOMINATED, INCOMPARABLE = 0, 1, 2


class TotalOrderScoreModel(nn.Module):
    """Total-order (single-score) baseline, in the GNNRank lineage.

    Parameters
    ----------
    feature_dim : int
        Input node feature dimensionality.
    hidden_dim : int, default 64
        Encoder hidden layer width.
    embed_dim : int, default 32
        Encoder output embedding dimensionality (before score projection).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.encoder = SharedEncoder(feature_dim, hidden_dim, embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)
        self.margin = nn.Parameter(torch.tensor(0.5))  # learned deadband for "incomparable"

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute one scalar score per node.

        Returns
        -------
        torch.Tensor
            ``[N]`` scores, one per node.
        """
        e = self.encoder(x, edge_index)
        return self.score_head(e).squeeze(-1)

    def predict_pairs(self, scores: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """Classify pairs by thresholding the score difference against the learned margin.

        Parameters
        ----------
        scores : torch.Tensor
            ``[N]`` per-node scores, as returned by :meth:`forward`.
        i_idx, j_idx : torch.Tensor
            ``[P]`` index tensors identifying the query pairs.

        Returns
        -------
        torch.Tensor
            ``[P]`` long tensor of predicted labels (0/1/2, see module docstring).
        """
        diff = scores[i_idx] - scores[j_idx]
        m = self.margin.abs()
        pred = torch.full_like(diff, INCOMPARABLE, dtype=torch.long)
        pred[diff > m] = DOMINATES
        pred[diff < -m] = DOMINATED
        return pred

    def pairwise_loss(
        self, scores: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Margin-ranking loss for dominance pairs, hinge loss for incomparable pairs.

        Closely mirrors GNNRank's "upset" formulation: penalizes score
        order disagreeing with the observed comparison direction, plus a
        hinge term encouraging ``|diff| < margin`` for pairs labeled
        incomparable.
        """
        diff = scores[i_idx] - scores[j_idx]
        m = self.margin.abs()
        dom_mask = labels == DOMINATES
        rev_mask = labels == DOMINATED
        inc_mask = labels == INCOMPARABLE
        loss = dom_mask.float() * F.relu(m - diff + 0.1)
        loss = loss + rev_mask.float() * F.relu(m + diff + 0.1)
        loss = loss + inc_mask.float() * F.relu(diff.abs() - m + 0.1)
        return loss.mean()


class PartialOrderModel(nn.Module):
    """Partial-order-aware architecture: direct three-way pairwise classification.

    Parameters
    ----------
    feature_dim : int
        Input node feature dimensionality.
    hidden_dim : int, default 64
        Encoder hidden layer width, and the pairwise head's hidden width.
    embed_dim : int, default 32
        Encoder output embedding dimensionality.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.encoder = SharedEncoder(feature_dim, hidden_dim, embed_dim)
        self.pair_head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute one embedding vector per node (never a scalar score).

        Returns
        -------
        torch.Tensor
            ``[N, embed_dim]`` node embeddings.
        """
        return self.encoder(x, edge_index)

    def predict_pairs(self, embeddings: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """Classify pairs directly from a three-way softmax over both endpoint embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            ``[N, embed_dim]`` per-node embeddings, as returned by :meth:`forward`.
        i_idx, j_idx : torch.Tensor
            ``[P]`` index tensors identifying the query pairs.

        Returns
        -------
        torch.Tensor
            ``[P]`` long tensor of predicted labels (0/1/2, see module docstring).
        """
        e_i, e_j = embeddings[i_idx], embeddings[j_idx]
        feat = torch.cat([e_i, e_j, e_i - e_j], dim=-1)
        logits = self.pair_head(feat)
        return logits.argmax(dim=-1)

    def pairwise_loss(
        self, embeddings: torch.Tensor, i_idx: torch.Tensor, j_idx: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss over the direct three-way classification."""
        e_i, e_j = embeddings[i_idx], embeddings[j_idx]
        feat = torch.cat([e_i, e_j, e_i - e_j], dim=-1)
        logits = self.pair_head(feat)
        return F.cross_entropy(logits, labels)
