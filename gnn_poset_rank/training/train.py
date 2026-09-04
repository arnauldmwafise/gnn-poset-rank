"""
gnn_poset_rank.training.train
==========================

Shared pair-sampling and train/eval routines used by every experiment in
the paper, ensuring the two models are always compared under an
identical training and evaluation protocol -- same epoch count, same
optimizer settings, same held-out test pairs -- so that any difference
in outcome is attributable to the prediction head, not to a difference
in training budget or evaluation set.
"""

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn


def sample_stratified_pairs(
    dominates: torch.Tensor, num_items: int, n_per_class: int, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a class-balanced set of dominates/dominated/incomparable pairs.

    Parameters
    ----------
    dominates : torch.Tensor
        ``[N, N]`` bool tensor, ground-truth dominance relation
        (``dominates[i, j]`` True iff ``i`` dominates ``j``).
    num_items : int
        Number of items ``N``.
    n_per_class : int
        Number of pairs to sample for each of the three classes.
    seed : int, default 0
        Random seed.

    Returns
    -------
    i_idx, j_idx : torch.Tensor
        ``[3 * n_per_class]`` long tensors of item indices for each pair.
    labels : torch.Tensor
        ``[3 * n_per_class]`` long tensor of labels (0 = i dominates j,
        1 = j dominates i, 2 = incomparable).

    Notes
    -----
    Incomparable pairs are found by bounded rejection sampling (200
    attempts, not an unbounded loop): a near-total-order dataset (for
    example, the synthetic generator at ``num_quality_dims=1``) can have
    vanishingly few or zero genuinely incomparable pairs, which would
    otherwise hang forever. A warning is printed if fewer than
    ``n_per_class`` incomparable pairs are found after all attempts.
    """
    g = torch.Generator().manual_seed(seed)
    i_dom, j_dom = dominates.nonzero(as_tuple=True)
    perm = torch.randperm(i_dom.size(0), generator=g)
    i_dom, j_dom = i_dom[perm][:n_per_class], j_dom[perm][:n_per_class]

    # Reverse of the label-0 pairs guarantees genuine dominated cases,
    # rather than independently resampling (which could, by chance, miss
    # rare dominance directions that only appear reversed).
    i_rev, j_rev = j_dom.clone(), i_dom.clone()

    inc_i, inc_j = [], []
    max_attempts = 200
    for _ in range(max_attempts):
        if len(inc_i) >= n_per_class:
            break
        cand_i = torch.randint(0, num_items, (n_per_class * 3,), generator=g)
        cand_j = torch.randint(0, num_items, (n_per_class * 3,), generator=g)
        mask = (~dominates[cand_i, cand_j]) & (~dominates[cand_j, cand_i]) & (cand_i != cand_j)
        inc_i.extend(cand_i[mask].tolist())
        inc_j.extend(cand_j[mask].tolist())
    if len(inc_i) < n_per_class:
        print(
            f"  [warning] only found {len(inc_i)}/{n_per_class} genuinely incomparable "
            f"pairs after {max_attempts} attempts -- this dataset has very few/no "
            f"incomparable pairs (expected for near-total-order data)"
        )
    inc_i = torch.tensor(inc_i[:n_per_class], dtype=torch.long)
    inc_j = torch.tensor(inc_j[:n_per_class], dtype=torch.long)

    all_i = torch.cat([i_dom, i_rev, inc_i])
    all_j = torch.cat([j_dom, j_rev, inc_j])
    all_labels = torch.cat(
        [
            torch.zeros(len(i_dom), dtype=torch.long),
            torch.ones(len(i_rev), dtype=torch.long),
            torch.full((len(inc_i),), 2, dtype=torch.long),
        ]
    )
    return all_i, all_j, all_labels


def train_and_eval(
    model_cls: Type[nn.Module],
    x: torch.Tensor,
    observed_edges: torch.Tensor,
    train_i: torch.Tensor,
    train_j: torch.Tensor,
    train_labels: torch.Tensor,
    test_i: torch.Tensor,
    test_j: torch.Tensor,
    test_labels: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.01,
    init_seed: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Train a model and evaluate its three-way pairwise classification accuracy.

    Both :class:`~gnn_poset_rank.models.TotalOrderScoreModel` and
    :class:`~gnn_poset_rank.models.PartialOrderModel` are trained and
    evaluated through this exact same function, with the same
    hyperparameters, so that any difference in outcome between them is
    attributable to the prediction head alone.

    Parameters
    ----------
    model_cls : type
        The model class to instantiate (``TotalOrderScoreModel`` or ``PartialOrderModel``).
    x : torch.Tensor
        ``[N, feature_dim]`` node features.
    observed_edges : torch.Tensor
        ``[2, E]`` long tensor of message-passing edges (only ever the
        training-visible edges; held-out edges must be excluded by the caller).
    train_i, train_j, train_labels : torch.Tensor
        Training pairs and labels, as returned by :func:`sample_stratified_pairs`.
    test_i, test_j, test_labels : torch.Tensor
        Held-out evaluation pairs and labels.
    epochs : int, default 150
        Number of full-batch training epochs.
    lr : float, default 0.01
        Adam learning rate.
    init_seed : int, default 0
        Random seed for model weight initialization. Distinct from the
        seed(s) used for data generation and pair sampling, so that a
        multi-seed protocol can vary model initialization independently
        of data sampling.

    Returns
    -------
    overall_acc : float
        Overall three-way classification accuracy on the test pairs.
    per_class_acc : dict[str, float]
        Accuracy restricted to each of the three classes
        (``'i_dominates_j'``, ``'j_dominates_i'``, ``'incomparable'``);
        ``nan`` for any class with zero test examples.
    """
    torch.manual_seed(init_seed)
    model = model_cls(feature_dim=x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for _epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, observed_edges)
        loss = model.pairwise_loss(out, train_i, train_j, train_labels)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out = model(x, observed_edges)
        preds = model.predict_pairs(out, test_i, test_j)

    overall_acc = (preds == test_labels).float().mean().item()
    per_class_acc = {}
    for c, name in [(0, "i_dominates_j"), (1, "j_dominates_i"), (2, "incomparable")]:
        mask = test_labels == c
        per_class_acc[name] = (
            (preds[mask] == test_labels[mask]).float().mean().item() if mask.sum() > 0 else float("nan")
        )
    return overall_acc, per_class_acc
