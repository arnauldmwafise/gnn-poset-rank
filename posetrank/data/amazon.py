"""
posetrank.data.amazon
=======================

Loader for the Amazon0302 co-purchase network (Leskovec, Adamic, and
Huberman, 2007): a directed edge from product ``i`` to product ``j``
indicates that ``j`` was listed under ``i``'s "customers who bought this
item also bought" recommendations.

This is distinct from the unrelated ``com-Amazon`` community-detection
dataset from the same research group, which is fully symmetrized by
construction and unsuitable for genuine directed-partial-order analysis;
Amazon0302 was independently verified to be genuinely directed before
use in this project (45.7% of edges one-directional, no reverse present).

The full graph (262,111 nodes, 1,234,877 edges) is downsampled to a
connected subgraph via breadth-first search for computational
tractability, matching the 8,000-node subgraph used throughout the paper.
"""

import random
from typing import Dict, Tuple

import networkx as nx
import torch


def load_amazon0302_edges(path: str) -> Tuple[list, int]:
    """Parse the raw Amazon0302 edge list (SNAP format).

    Parameters
    ----------
    path : str
        Path to the raw ``Amazon0302.txt`` file (``#``-prefixed comment
        lines, then tab-separated ``FromNodeId  ToNodeId`` per line).

    Returns
    -------
    edges : list[tuple[int, int]]
        All directed edges as ``(source, target)`` node-id pairs.
    """
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = line.split("\t")
            edges.append((int(u), int(v)))
    return edges


def sample_bfs_subgraph(edges: list, target_size: int = 8000, seed: int = 0) -> Dict:
    """Sample a connected subgraph via breadth-first search from a random seed node.

    Traverses both successors and predecessors (i.e. treats the directed
    graph as undirected for the purpose of reachability during sampling
    only) so the resulting subgraph is weakly connected, while the
    returned edges retain their original direction.

    Parameters
    ----------
    edges : list[tuple[int, int]]
        Full edge list, as returned by :func:`load_amazon0302_edges`.
    target_size : int, default 8000
        Approximate number of nodes to include in the subgraph.
    seed : int, default 0
        Random seed for both the starting node and neighbor traversal order.

    Returns
    -------
    dict
        ``raw_edges``: ``[2, E]`` long tensor of directed edges, reindexed
            to a dense ``0..num_nodes-1`` range.
        ``num_nodes``: number of nodes in the sampled subgraph.
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)

    rng = random.Random(seed)
    seed_node = rng.choice(list(G.nodes()))
    visited = {seed_node}
    frontier = [seed_node]
    while frontier and len(visited) < target_size:
        next_frontier = []
        for node in frontier:
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
                    if len(visited) >= target_size:
                        break
            if len(visited) >= target_size:
                break
        frontier = next_frontier

    sub = G.subgraph(visited).copy()
    nodes = sorted(sub.nodes())
    id_map = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)
    src = [id_map[u] for u, v in sub.edges()]
    tgt = [id_map[v] for u, v in sub.edges()]
    return {"raw_edges": torch.tensor([src, tgt], dtype=torch.long), "num_nodes": n}


def load_amazon0302_subgraph(path: str, target_size: int = 8000, seed: int = 0) -> Dict:
    """Load Amazon0302 and return a BFS-sampled connected subgraph.

    Convenience wrapper combining :func:`load_amazon0302_edges` and
    :func:`sample_bfs_subgraph`.

    Parameters
    ----------
    path : str
        Path to the raw ``Amazon0302.txt`` file.
    target_size : int, default 8000
        Approximate number of nodes in the sampled subgraph.
    seed : int, default 0
        Random seed for sampling.

    Returns
    -------
    dict
        Same shape as :func:`sample_bfs_subgraph`.
    """
    edges = load_amazon0302_edges(path)
    return sample_bfs_subgraph(edges, target_size=target_size, seed=seed)


def compute_ranks(cover_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute each node's rank in a Hasse diagram (longest path from any minimal element).

    Parameters
    ----------
    cover_edges : torch.Tensor
        ``[2, E]`` long tensor of cover-relation edges (``source -> target``
        means source is covered by target, i.e. source < target with
        nothing in between).
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        ``[num_nodes]`` long tensor of ranks (0 for minimal elements).

    Raises
    ------
    ValueError
        If ``cover_edges`` contains a cycle (not a valid poset cover relation).
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(cover_edges[0].tolist(), cover_edges[1].tolist()))
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("compute_ranks: cover_edges contains a cycle -- not a valid poset cover relation.")

    topo = list(nx.topological_sort(G))
    ranks = [0] * num_nodes
    for v in topo:
        preds = list(G.predecessors(v))
        if preds:
            ranks[v] = 1 + max(ranks[u] for u in preds)
    return torch.tensor(ranks, dtype=torch.long)


def build_structural_features(
    raw_edges: torch.Tensor, ranks: torch.Tensor, num_nodes: int, embed_dim: int = 16, seed: int = 0
) -> torch.Tensor:
    """Build node features for Amazon0302 from graph structure alone.

    Amazon0302 has no product-content features available; this builds a
    feature vector purely from structural signals (log in/out-degree and
    Hasse-diagram rank), standardized, concatenated with a fixed random
    embedding to give the model additional per-node capacity to
    distinguish otherwise-similar nodes.

    Parameters
    ----------
    raw_edges : torch.Tensor
        ``[2, E]`` long tensor of raw (uncleaned) directed edges, used
        for computing in/out-degree.
    ranks : torch.Tensor
        ``[num_nodes]`` long or float tensor of Hasse-diagram rank per node.
    num_nodes : int
        Number of nodes.
    embed_dim : int, default 16
        Dimensionality of the appended random embedding.
    seed : int, default 0
        Random seed for the fixed embedding.

    Returns
    -------
    torch.Tensor
        ``[num_nodes, 3 + embed_dim]`` float tensor of node features.
    """
    torch.manual_seed(seed)
    in_deg = torch.zeros(num_nodes)
    out_deg = torch.zeros(num_nodes)
    out_deg.index_add_(0, raw_edges[0], torch.ones(raw_edges.size(1)))
    in_deg.index_add_(0, raw_edges[1], torch.ones(raw_edges.size(1)))

    struct_feat = torch.stack([torch.log1p(in_deg), torch.log1p(out_deg), ranks.float()], dim=1)
    struct_feat = (struct_feat - struct_feat.mean(0)) / (struct_feat.std(0) + 1e-6)

    random_embed = torch.randn(num_nodes, embed_dim) * 0.5
    return torch.cat([struct_feat, random_embed], dim=1)
