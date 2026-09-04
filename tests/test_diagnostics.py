"""
Tests for posetrank.diagnostics.

The SIR test against brute-force ground truth mirrors the exact
verification performed before trusting this tool on any real dataset in
the paper: naive Monte Carlo sampling silently under-covers small graphs
unless explicitly checked against exhaustive enumeration, which is
precisely the bug this test is designed to catch if it were reintroduced.
"""

import networkx as nx
import pytest
import torch

from posetrank.diagnostics import clean_dag, compute_order_diagnostics, greedy_feedback_arc_set, verify_directed


def brute_force_sir(edges: torch.Tensor, num_nodes: int) -> float:
    """Exact SIR via networkx's transitive closure, independent of our own implementation."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edges[0].tolist(), edges[1].tolist()))
    TC = nx.transitive_closure(G)
    total_pairs = num_nodes * (num_nodes - 1)
    comparable_ordered_pairs = 2 * TC.number_of_edges()
    return 1.0 - comparable_ordered_pairs / total_pairs


class TestGreedyFeedbackArcSet:
    def test_acyclic_input_removes_nothing(self):
        edges = torch.tensor([[0, 1], [1, 2]])
        clean_edges, removed, _ordering = greedy_feedback_arc_set(edges, num_nodes=3)
        assert removed.size(1) == 0
        assert clean_edges.size(1) == 2

    def test_two_cycle_removes_one_edge(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        clean_edges, removed, _ordering = greedy_feedback_arc_set(edges, num_nodes=2)
        assert removed.size(1) == 1
        assert clean_edges.size(1) == 1

    def test_result_is_always_acyclic(self):
        torch.manual_seed(0)
        edges = torch.randint(0, 20, (2, 80))
        clean_edges, _removed, _ordering = greedy_feedback_arc_set(edges, num_nodes=20)
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        G.add_edges_from(zip(clean_edges[0].tolist(), clean_edges[1].tolist()))
        assert nx.is_directed_acyclic_graph(G)


class TestCleanDag:
    def test_diamond_has_zero_cir_and_trr(self):
        # 0 -> 1 -> 3, 0 -> 2 -> 3: already acyclic and already a valid
        # cover relation (no shortcut edge 0 -> 3 present), so both CIR
        # and TRR should be exactly zero.
        edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        result = clean_dag(edges, num_nodes=4)
        assert result["stats"]["fas_removal_fraction"] == 0.0
        assert result["stats"]["shortcut_edges_removed_by_reduction"] == 0
        assert result["cover_edges"].size(1) == 4

    def test_transitive_shortcut_is_removed(self):
        # 0 -> 1 -> 2 plus the redundant shortcut 0 -> 2: TRR should remove
        # exactly the shortcut, leaving the two essential cover edges.
        edges = torch.tensor([[0, 1, 0], [1, 2, 2]])
        result = clean_dag(edges, num_nodes=3)
        assert result["stats"]["shortcut_edges_removed_by_reduction"] == 1
        assert result["cover_edges"].size(1) == 2


class TestComputeOrderDiagnostics:
    def test_matches_brute_force_sir_on_small_graph(self):
        edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        result = compute_order_diagnostics(edges, num_nodes=5, sample_pairs_for_sir=1000, seed=0)
        expected = brute_force_sir(edges, num_nodes=5)
        assert result["structural_incomparability_rate"] == pytest.approx(expected, abs=1e-9)

    def test_exhaustive_and_monte_carlo_agree_at_moderate_scale(self):
        torch.manual_seed(0)
        edges = torch.randint(0, 60, (2, 150))
        edges = edges[:, edges[0] != edges[1]]
        exact = compute_order_diagnostics(edges, num_nodes=60, sample_pairs_for_sir=100_000, seed=0)
        sampled = compute_order_diagnostics(edges, num_nodes=60, sample_pairs_for_sir=500, seed=1)
        assert exact["n_pairs_sampled"] == 60 * 59  # confirms exhaustive path was actually taken
        assert sampled["structural_incomparability_rate"] == pytest.approx(
            exact["structural_incomparability_rate"], abs=0.05
        )

    def test_total_order_has_zero_sir(self):
        # a simple chain 0 -> 1 -> 2 -> 3 is a total order: every pair is comparable
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        result = compute_order_diagnostics(edges, num_nodes=4)
        assert result["structural_incomparability_rate"] == pytest.approx(0.0, abs=1e-9)

    def test_no_edges_gives_sir_near_one(self):
        edges = torch.zeros(2, 0, dtype=torch.long)
        result = compute_order_diagnostics(edges, num_nodes=10)
        assert result["structural_incomparability_rate"] == pytest.approx(1.0, abs=1e-9)


class TestVerifyDirected:
    def test_fully_symmetrized_graph_detected(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        result = verify_directed(edges, num_nodes=2)
        assert result["reverse_present_fraction"] == pytest.approx(1.0)

    def test_fully_one_directional_graph_detected(self):
        edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
        result = verify_directed(edges, num_nodes=3)
        assert result["reverse_present_fraction"] == pytest.approx(0.0)
