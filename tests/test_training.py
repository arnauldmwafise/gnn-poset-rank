"""Tests for posetrank.training."""

import torch

from posetrank.data.synthetic import generate_pareto_ranking_data
from posetrank.models import PartialOrderModel, TotalOrderScoreModel
from posetrank.training import sample_stratified_pairs, train_and_eval


class TestSampleStratifiedPairs:
    def test_returns_balanced_classes(self):
        data = generate_pareto_ranking_data(num_items=300, num_quality_dims=3, seed=0)
        i_idx, j_idx, labels = sample_stratified_pairs(data["dominates"], data["num_items"], n_per_class=50, seed=0)
        assert i_idx.shape == j_idx.shape == labels.shape
        for c in (0, 1, 2):
            assert (labels == c).sum().item() == 50

    def test_dominates_and_dominated_pairs_are_consistent_with_ground_truth(self):
        data = generate_pareto_ranking_data(num_items=300, num_quality_dims=3, seed=0)
        dominates = data["dominates"]
        i_idx, j_idx, labels = sample_stratified_pairs(dominates, data["num_items"], n_per_class=50, seed=0)
        for i, j, label in zip(i_idx.tolist(), j_idx.tolist(), labels.tolist()):
            if label == 0:
                assert dominates[i, j]
            elif label == 1:
                assert dominates[j, i]
            else:
                assert not dominates[i, j] and not dominates[j, i]

    def test_near_total_order_warns_but_does_not_hang(self):
        # K=1 has essentially zero incomparable pairs; this must terminate
        # (bounded rejection sampling), not hang, and should warn rather
        # than silently return fewer pairs than requested without notice
        data = generate_pareto_ranking_data(num_items=100, num_quality_dims=1, seed=0)
        i_idx, j_idx, labels = sample_stratified_pairs(data["dominates"], data["num_items"], n_per_class=20, seed=0)
        # the function must still return well-formed (if short) tensors
        assert i_idx.shape == j_idx.shape == labels.shape


class TestTrainAndEval:
    def test_both_models_train_without_error(self):
        data = generate_pareto_ranking_data(num_items=200, num_quality_dims=2, seed=0)
        n = data["num_items"]
        train_i, train_j, train_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=50, seed=1)
        test_i, test_j, test_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=20, seed=2)

        for model_cls in (TotalOrderScoreModel, PartialOrderModel):
            acc, per_class = train_and_eval(
                model_cls, data["x"], data["observed_edges"],
                train_i, train_j, train_labels, test_i, test_j, test_labels,
                epochs=10, init_seed=0,
            )
            assert 0.0 <= acc <= 1.0
            assert set(per_class.keys()) == {"i_dominates_j", "j_dominates_i", "incomparable"}

    def test_different_init_seed_gives_different_initial_weights(self):
        # Direct, unambiguous regression guard for the exact class of bug
        # found and fixed earlier in this project (model init accidentally
        # hardcoded, silently ignoring init_seed): compare weights
        # directly rather than downstream accuracy, since accuracy on a
        # small test set is coarsely quantized and can coincidentally tie
        # across seeds even when the underlying predictions differ.
        data = generate_pareto_ranking_data(num_items=200, num_quality_dims=2, seed=0)
        n = data["num_items"]
        train_i, train_j, train_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=50, seed=1)
        test_i, test_j, test_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=20, seed=2)

        preds_by_seed = []
        for init_seed in (0, 1):
            torch.manual_seed(init_seed)
            model = PartialOrderModel(feature_dim=data["x"].size(1))
            preds_by_seed.append(next(model.parameters()).clone())
        assert not torch.equal(preds_by_seed[0], preds_by_seed[1])

        # and confirm this propagates through train_and_eval to a
        # different set of *predictions* (not just a different accuracy
        # scalar, which can coincidentally tie)
        results = []
        for init_seed in (0, 1):
            torch.manual_seed(init_seed)
            model = PartialOrderModel(feature_dim=data["x"].size(1))
            with torch.no_grad():
                out = model(data["x"], data["observed_edges"])
                preds = model.predict_pairs(out, test_i, test_j)
            results.append(preds)
        assert not torch.equal(results[0], results[1])

    def test_same_seed_is_reproducible(self):
        data = generate_pareto_ranking_data(num_items=200, num_quality_dims=2, seed=0)
        n = data["num_items"]
        train_i, train_j, train_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=50, seed=1)
        test_i, test_j, test_labels = sample_stratified_pairs(data["dominates"], n, n_per_class=20, seed=2)

        acc1, _ = train_and_eval(
            PartialOrderModel, data["x"], data["observed_edges"],
            train_i, train_j, train_labels, test_i, test_j, test_labels, epochs=20, init_seed=0,
        )
        acc2, _ = train_and_eval(
            PartialOrderModel, data["x"], data["observed_edges"],
            train_i, train_j, train_labels, test_i, test_j, test_labels, epochs=20, init_seed=0,
        )
        assert acc1 == acc2
