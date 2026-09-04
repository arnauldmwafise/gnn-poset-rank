"""
posetrank.diagnostics.cleaning
===============================

Generic pipeline for converting a raw, possibly-cyclic, possibly
transitively-redundant directed comparison graph into a validated Hasse
diagram (cover relation).

This module is deliberately domain-agnostic: it is used identically by
every dataset loader in :mod:`posetrank.data` (citation networks, the
Wikipedia hyperlink network, the co-purchase network) and by the
diagnostic framework in :mod:`posetrank.diagnostics.order_diagnostics`,
which reuses it to compute the Cycle Inconsistency Rate and to obtain the
cleaned graph needed for the Structural Incomparability Rate.

Pipeline
--------
1. :func:`greedy_feedback_arc_set` removes the smallest defensible set of
   edges needed to make the graph acyclic (a well-established O(V+E)
   heuristic approximation to the NP-hard minimum feedback arc set
   problem).
2. :func:`clean_dag` applies that removal, verifies the result is
   genuinely acyclic, and then applies transitive reduction to obtain the
   cover relation (Hasse diagram) -- the minimal edge set with the same
   transitive closure as the cleaned graph.
"""

from typing import Dict, List, Tuple

import networkx as nx
import torch


def greedy_feedback_arc_set(
    edges: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Remove a small feedback arc set to make a directed graph acyclic.

    Uses the classic Eades-Lin-Smyth greedy sequential heuristic:
    repeatedly strip sinks to the back of a node ordering and sources to
    the front; when neither exists, remove whichever remaining node
    maximizes (out-degree - in-degree) within the remaining subgraph and
    place it at the front. Edges consistent with the resulting ordering
    are kept; edges that run counter to it constitute the feedback arc
    set and are dropped.

    This is a heuristic, not an exact minimum feedback arc set (computing
    the true minimum is NP-hard), but is a well-established O(V+E)
    approximation that performs well when cycles are short and sparse, as
    is typical of real citation and hyperlink data.

    Parameters
    ----------
    edges : torch.Tensor
        ``[2, E]`` long tensor of directed edges ``(source, target)``.
    num_nodes : int
        Total number of nodes in the graph.

    Returns
    -------
    clean_edges : torch.Tensor
        ``[2, E']`` edges consistent with the discovered acyclic ordering.
    removed_edges : torch.Tensor
        ``[2, E - E']`` edges removed to break cycles (the feedback arc set).
    ordering : list[int]
        The node ordering used to classify edges as forward/backward.

    Notes
    -----
    All set iteration in this function is performed over ``sorted(...)``
    rather than raw Python sets, so that tie-breaking among multiple
    simultaneous sinks/sources (or among nodes tied on out-degree minus
    in-degree) is deterministic across machines and Python versions,
    rather than depending on incidental set-iteration order.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edges[0].tolist(), edges[1].tolist()))

    s1: List[int] = []
    s2: List[int] = []
    remaining = set(G.nodes())
    out_adj = {n: set(G.successors(n)) for n in G.nodes()}
    in_adj = {n: set(G.predecessors(n)) for n in G.nodes()}

    while remaining:
        progressed = True
        while progressed:
            progressed = False
            sinks = [n for n in sorted(remaining) if not (out_adj[n] & remaining)]
            for n in sinks:
                s2.insert(0, n)
                remaining.discard(n)
                progressed = True
            if not remaining:
                break
            sources = [n for n in sorted(remaining) if not (in_adj[n] & remaining)]
            for n in sources:
                s1.append(n)
                remaining.discard(n)
                progressed = True
        if remaining:
            best = max(
                sorted(remaining),
                key=lambda n: len(out_adj[n] & remaining) - len(in_adj[n] & remaining),
            )
            s1.append(best)
            remaining.discard(best)

    ordering = s1 + s2
    position = {n: i for i, n in enumerate(ordering)}

    src, tgt = edges[0].tolist(), edges[1].tolist()
    keep_src, keep_tgt, drop_src, drop_tgt = [], [], [], []
    for u, v in zip(src, tgt):
        if position[u] < position[v]:
            keep_src.append(u)
            keep_tgt.append(v)
        else:
            drop_src.append(u)
            drop_tgt.append(v)

    clean_edges = torch.tensor([keep_src, keep_tgt], dtype=torch.long)
    removed_edges = torch.tensor([drop_src, drop_tgt], dtype=torch.long)
    return clean_edges, removed_edges, ordering


def clean_dag(
    raw_edges: torch.Tensor, num_nodes: int, verbose: bool = False
) -> Dict:
    """Convert raw directed edges into a validated Hasse diagram.

    Applies :func:`greedy_feedback_arc_set` to obtain an acyclic edge set,
    verifies acyclicity explicitly (rather than assuming the heuristic
    succeeded), and then applies transitive reduction to obtain the cover
    relation -- the minimal edge set inducing the same transitive closure
    as the cleaned graph.

    Parameters
    ----------
    raw_edges : torch.Tensor
        ``[2, E]`` long tensor of raw directed edges, possibly cyclic and
        possibly transitively redundant.
    num_nodes : int
        Total number of nodes.
    verbose : bool, default False
        If True, print a short summary of edges removed at each stage.

    Returns
    -------
    dict
        ``cover_edges``: ``[2, E'']`` long tensor, the Hasse diagram.
        ``stats``: dict of edge counts at each pipeline stage, including
        ``fas_removal_fraction`` (the Cycle Inconsistency Rate) and
        ``shortcut_edges_removed_by_reduction`` (used to derive the
        Transitive Redundancy Rate).
    """
    clean_edges, removed_fas, _ordering = greedy_feedback_arc_set(raw_edges, num_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(clean_edges[0].tolist(), clean_edges[1].tolist()))
    if not nx.is_directed_acyclic_graph(G):
        raise RuntimeError(
            "greedy_feedback_arc_set failed to produce a DAG; this should "
            "never happen and indicates a bug in the heuristic, not a "
            "property of the input data."
        )

    reduced = nx.transitive_reduction(G)
    if reduced.number_of_edges() > 0:
        cover_edges = torch.tensor(list(reduced.edges()), dtype=torch.long).t().contiguous()
    else:
        cover_edges = torch.zeros(2, 0, dtype=torch.long)

    stats = {
        "raw_edges": raw_edges.size(1),
        "edges_after_fas_removal": clean_edges.size(1),
        "edges_removed_by_fas": removed_fas.size(1),
        "fas_removal_fraction": removed_fas.size(1) / raw_edges.size(1) if raw_edges.size(1) else 0.0,
        "cover_edges_after_transitive_reduction": cover_edges.size(1),
        "shortcut_edges_removed_by_reduction": clean_edges.size(1) - cover_edges.size(1),
    }

    if verbose:
        print(f"Raw edges:                              {stats['raw_edges']:,}")
        print(
            f"Removed by feedback-arc-set (cycles):   {stats['edges_removed_by_fas']:,} "
            f"({stats['fas_removal_fraction']:.2%})"
        )
        print(f"Edges after cycle removal (valid DAG):  {stats['edges_after_fas_removal']:,}")
        print(f"Shortcut edges removed by reduction:    {stats['shortcut_edges_removed_by_reduction']:,}")
        print(f"Final Hasse diagram (cover) edges:      {stats['cover_edges_after_transitive_reduction']:,}")

    return {"cover_edges": cover_edges, "stats": stats}
