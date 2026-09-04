"""
gnn_poset_rank.models.encoders
============================

The directed dual-stream graph convolutional encoder shared by both
prediction heads compared throughout the paper.

The dual-stream design maintains two separate learned aggregations per
node -- one over the node's incoming edges (its ancestors in the
comparison graph, i.e. items it dominates) and one over its outgoing
edges (its descendants, i.e. items that dominate it) -- combined with a
self-transformation of the node's own features, rather than pooling
incoming and outgoing neighbors into a single undirected aggregator as a
standard graph convolutional network would.

This architectural choice is not novel to this project: directed,
direction-separated message passing of this kind was formalized and
shown to substantially improve performance on heterophilic and directed
graphs by Rossi et al. (2024, "Edge Directionality Improves Learning on
Heterophilic Graphs", LoG), who further proved that separating incoming
and outgoing aggregation matches the expressivity of the directed
Weisfeiler-Lehman test, exceeding that of standard (undirected-pooling)
message-passing networks. A related two-stream design was independently
introduced for signed directed link prediction by He et al. (2022,
"Two-stream signed directed graph convolutional network for link
prediction", Physica A). It is used here because a comparison relation
is inherently asymmetric -- "i dominates j" and "j dominates i" are
different statements with different truth values -- and pooling incoming
and outgoing edges together would discard exactly the directional
information a ranking task depends on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedOrderConv(nn.Module):
    """A single directed dual-stream graph convolution layer.

    Maintains three separate learned linear transforms: one applied to
    the mean-aggregated features of each node's ancestors (in-edges),
    one to the mean-aggregated features of its descendants (out-edges),
    and one to the node's own features (a self-loop / residual term).
    The three contributions are summed.

    Parameters
    ----------
    in_channels : int
        Input feature dimensionality.
    out_channels : int
        Output feature dimensionality.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_ancestor = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_descendant = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    @staticmethod
    def _mean_aggregate(x: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Mean-aggregate ``x[src]`` into each ``tgt`` index (isolated nodes get a zero vector)."""
        out = torch.zeros(num_nodes, x.size(1), device=x.device)
        out.index_add_(0, tgt, x[src])
        count = torch.zeros(num_nodes, device=x.device)
        count.index_add_(0, tgt, torch.ones(tgt.size(0), device=x.device))
        count = count.clamp(min=1.0)
        return out / count.unsqueeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one directed dual-stream convolution.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, in_channels]`` node features.
        edge_index : torch.Tensor
            ``[2, E]`` long tensor of directed edges ``(source, target)``,
            where an edge means "source dominates target".

        Returns
        -------
        torch.Tensor
            ``[N, out_channels]`` updated node features.
        """
        num_nodes = x.size(0)
        src, tgt = edge_index[0], edge_index[1]

        # Downstream: aggregate from ancestors (things this node dominates,
        # i.e. edges pointing INTO this node) into each target node.
        msg_down = self._mean_aggregate(x, src, tgt, num_nodes)
        out_downstream = self.lin_ancestor(msg_down)

        # Upstream: aggregate from descendants (things that dominate this
        # node, i.e. edges pointing OUT of this node) into each source node.
        msg_up = self._mean_aggregate(x, tgt, src, num_nodes)
        out_upstream = self.lin_descendant(msg_up)

        return out_downstream + out_upstream + self.lin_self(x)


class SharedEncoder(nn.Module):
    """Two-layer directed dual-stream encoder, shared by both prediction heads.

    Parameters
    ----------
    feature_dim : int
        Input node feature dimensionality.
    hidden_dim : int, default 64
        Hidden layer width.
    embed_dim : int, default 32
        Output embedding dimensionality.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.conv1 = DirectedOrderConv(hidden_dim, hidden_dim)
        self.conv2 = DirectedOrderConv(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features into embeddings.

        Parameters
        ----------
        x : torch.Tensor
            ``[N, feature_dim]`` node features.
        edge_index : torch.Tensor
            ``[2, E]`` long tensor of directed message-passing edges.

        Returns
        -------
        torch.Tensor
            ``[N, embed_dim]`` node embeddings.
        """
        h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return self.output_proj(h)
