"""Tests for posetrank.models."""

import torch

from posetrank.models import DirectedOrderConv, PartialOrderModel, SharedEncoder, TotalOrderScoreModel
from posetrank.models.heads import DOMINATES, DOMINATED, INCOMPARABLE


def _toy_graph(n=20, e=30, feat_dim=16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(n, feat_dim)
    edge_index = torch.randint(0, n, (2, e))
    return x, edge_index


class TestDirectedOrderConv:
    def test_output_shape(self):
        x, edge_index = _toy_graph()
        conv = DirectedOrderConv(in_channels=16, out_channels=8)
        out = conv(x, edge_index)
        assert out.shape == (20, 8)

    def test_isolated_node_does_not_crash(self):
        # node 19 has no incident edges at all
        x, _ = _toy_graph(n=20)
        edge_index = torch.randint(0, 19, (2, 30))  # never touches node 19
        conv = DirectedOrderConv(in_channels=16, out_channels=8)
        out = conv(x, edge_index)
        assert torch.isfinite(out).all()

    def test_direction_matters(self):
        # swapping edge direction should, in general, change the output
        # (confirms the two streams are not accidentally symmetric)
        x, edge_index = _toy_graph()
        conv = DirectedOrderConv(in_channels=16, out_channels=8)
        out_forward = conv(x, edge_index)
        out_reversed = conv(x, edge_index.flip(0))
        assert not torch.allclose(out_forward, out_reversed)


class TestSharedEncoder:
    def test_output_shape(self):
        x, edge_index = _toy_graph()
        enc = SharedEncoder(feature_dim=16, hidden_dim=32, embed_dim=8)
        out = enc(x, edge_index)
        assert out.shape == (20, 8)


class TestTotalOrderScoreModel:
    def test_forward_gives_one_score_per_node(self):
        x, edge_index = _toy_graph()
        model = TotalOrderScoreModel(feature_dim=16)
        scores = model(x, edge_index)
        assert scores.shape == (20,)

    def test_predict_pairs_returns_valid_labels(self):
        x, edge_index = _toy_graph()
        model = TotalOrderScoreModel(feature_dim=16)
        scores = model(x, edge_index)
        i_idx = torch.tensor([0, 1, 2])
        j_idx = torch.tensor([3, 4, 5])
        preds = model.predict_pairs(scores, i_idx, j_idx)
        assert preds.shape == (3,)
        assert set(preds.tolist()).issubset({DOMINATES, DOMINATED, INCOMPARABLE})

    def test_large_score_gap_never_predicts_incomparable(self):
        # construct scores with a huge, unambiguous gap and confirm the
        # margin-threshold logic classifies it as a clear dominance, not
        # incomparable -- this is the core behavior the margin exists for
        model = TotalOrderScoreModel(feature_dim=4)
        scores = torch.tensor([100.0, -100.0])
        pred = model.predict_pairs(scores, torch.tensor([0]), torch.tensor([1]))
        assert pred.item() == DOMINATES

    def test_loss_is_finite_and_scalar(self):
        x, edge_index = _toy_graph()
        model = TotalOrderScoreModel(feature_dim=16)
        scores = model(x, edge_index)
        i_idx = torch.tensor([0, 1, 2])
        j_idx = torch.tensor([3, 4, 5])
        labels = torch.tensor([DOMINATES, DOMINATED, INCOMPARABLE])
        loss = model.pairwise_loss(scores, i_idx, j_idx, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)


class TestPartialOrderModel:
    def test_forward_gives_embedding_never_a_scalar(self):
        x, edge_index = _toy_graph()
        model = PartialOrderModel(feature_dim=16, embed_dim=8)
        out = model(x, edge_index)
        assert out.shape == (20, 8)
        assert out.shape[-1] > 1  # never collapses to a single scalar score

    def test_predict_pairs_returns_valid_labels(self):
        x, edge_index = _toy_graph()
        model = PartialOrderModel(feature_dim=16, embed_dim=8)
        emb = model(x, edge_index)
        i_idx = torch.tensor([0, 1, 2])
        j_idx = torch.tensor([3, 4, 5])
        preds = model.predict_pairs(emb, i_idx, j_idx)
        assert preds.shape == (3,)
        assert set(preds.tolist()).issubset({DOMINATES, DOMINATED, INCOMPARABLE})

    def test_loss_is_finite_and_scalar(self):
        x, edge_index = _toy_graph()
        model = PartialOrderModel(feature_dim=16, embed_dim=8)
        emb = model(x, edge_index)
        i_idx = torch.tensor([0, 1, 2])
        j_idx = torch.tensor([3, 4, 5])
        labels = torch.tensor([DOMINATES, DOMINATED, INCOMPARABLE])
        loss = model.pairwise_loss(emb, i_idx, j_idx, labels)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_one_gradient_step_reduces_loss_on_easy_data(self):
        # a coarse sanity check that gradients actually flow: on a trivial,
        # separable toy problem, loss after several optimizer steps should
        # be lower than the initial loss
        torch.manual_seed(0)
        x = torch.randn(10, 8)
        edge_index = torch.randint(0, 10, (2, 15))
        i_idx = torch.tensor([0, 1, 2, 3])
        j_idx = torch.tensor([4, 5, 6, 7])
        labels = torch.tensor([DOMINATES, DOMINATED, INCOMPARABLE, DOMINATES])

        model = PartialOrderModel(feature_dim=8, embed_dim=8)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        emb = model(x, edge_index)
        initial_loss = model.pairwise_loss(emb, i_idx, j_idx, labels).item()

        for _ in range(50):
            opt.zero_grad()
            emb = model(x, edge_index)
            loss = model.pairwise_loss(emb, i_idx, j_idx, labels)
            loss.backward()
            opt.step()

        final_loss = model.pairwise_loss(model(x, edge_index), i_idx, j_idx, labels).item()
        assert final_loss < initial_loss
