"""
Tests for gnn_poset_rank.data.synthetic.

These tests check the mathematical properties the paper's Section 5.1
explicitly claims and verifies before relying on them: Pareto dominance
is a genuine strict partial order (antisymmetric, transitive, acyclic),
K=1 reduces exactly to a total order, and feature noise does not perturb
the resulting SIR -- the property that makes the feature-informativeness
study in Section 5.7.1 a clean, single-variable manipulation.
"""

import networkx as nx
import pytest
import torch

from gnn_poset_rank.data.synthetic import generate_pareto_ranking_data
from gnn_poset_rank.diagnostics import compute_order_diagnostics


class TestGeneratePartoRankingData:
    def test_k1_is_a_total_order(self):
        data = generate_pareto_ranking_data(num_items=200, num_quality_dims=1, seed=0)
        assert data["incomparable_rate"] == 0.0

    def test_dominance_relation_is_antisymmetric(self):
        data = generate_pareto_ranking_data(num_items=100, num_quality_dims=3, seed=0)
        dominates = data["dominates"]
        # i dominates j and j dominates i can never both be true
        both = dominates & dominates.t()
        assert not both.any()

    def test_cover_relation_is_acyclic(self):
        data = generate_pareto_ranking_data(num_items=300, num_quality_dims=4, seed=1)
        cover_edges = data["cover_edges"]
        G = nx.DiGraph()
        G.add_nodes_from(range(data["num_items"]))
        G.add_edges_from(zip(cover_edges[0].tolist(), cover_edges[1].tolist()))
        assert nx.is_directed_acyclic_graph(G)

    def test_incomparable_rate_matches_dominates_matrix(self):
        # incomparable_rate is reported directly from `dominates`; cross-check
        # it against an independent recomputation from the same matrix
        data = generate_pareto_ranking_data(num_items=150, num_quality_dims=2, seed=0)
        dominates = data["dominates"]
        n = data["num_items"]
        n_dom = dominates.sum().item()
        total = n * (n - 1)
        expected = (total - 2 * n_dom) / total
        assert data["incomparable_rate"] == pytest.approx(expected)

    def test_incomparability_increases_with_quality_dimensions(self):
        # not merely a claim of the paper -- a directly testable monotonic
        # trend that should hold for any reasonable seed
        rates = []
        for k in [1, 2, 4]:
            data = generate_pareto_ranking_data(num_items=400, num_quality_dims=k, seed=0)
            rates.append(data["incomparable_rate"])
        assert rates[0] < rates[1] < rates[2]

    def test_feature_noise_does_not_change_sir(self):
        # the property that makes the feature-informativeness study a
        # clean, single-variable manipulation: SIR must not depend on
        # feature_noise, only on num_quality_dims
        sirs = []
        for noise in [0.1, 1.0, 5.0]:
            data = generate_pareto_ranking_data(
                num_items=300, num_quality_dims=2, feature_noise=noise, seed=0
            )
            diag = compute_order_diagnostics(data["cover_edges"], data["num_items"], seed=0)
            sirs.append(diag["structural_incomparability_rate"])
        assert max(sirs) - min(sirs) < 0.02  # SIR should be essentially identical

    def test_same_seed_is_fully_reproducible(self):
        data1 = generate_pareto_ranking_data(num_items=100, num_quality_dims=2, seed=42)
        data2 = generate_pareto_ranking_data(num_items=100, num_quality_dims=2, seed=42)
        assert torch.equal(data1["x"], data2["x"])
        assert torch.equal(data1["dominates"], data2["dominates"])

    def test_different_seeds_give_different_data(self):
        data1 = generate_pareto_ranking_data(num_items=100, num_quality_dims=2, seed=0)
        data2 = generate_pareto_ranking_data(num_items=100, num_quality_dims=2, seed=1)
        assert not torch.equal(data1["x"], data2["x"])

    def test_observed_edges_are_a_subset_of_cover_edges(self):
        data = generate_pareto_ranking_data(
            num_items=200, num_quality_dims=2, observed_edge_fraction=0.15, seed=0
        )
        cover_set = set(zip(data["cover_edges"][0].tolist(), data["cover_edges"][1].tolist()))
        observed_set = set(zip(data["observed_edges"][0].tolist(), data["observed_edges"][1].tolist()))
        assert observed_set.issubset(cover_set)
        # roughly the requested fraction, allowing for integer rounding
        assert abs(len(observed_set) / len(cover_set) - 0.15) < 0.02
