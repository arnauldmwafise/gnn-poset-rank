#!/usr/bin/env python3
"""
Reproduce the MQ2008 real-world cross-check (Section 5.5): a
within-query-respecting graph model against a naive baseline that pools
all queries into a single graph.

Requires MQ2008 (LETOR 4.0) already downloaded and extracted manually --
see the README for why this cannot be automated, and where to get it.

Usage
-----
    python scripts/run_mq2008.py --fold-dir data/MQ2008/Fold1
"""

import argparse
import os

from gnn_poset_rank.data import load_letor_file
from gnn_poset_rank.diagnostics import compute_order_diagnostics
from gnn_poset_rank.experiments.mq2008_eval import run_mq2008_comparison


def compute_mq2008_sir(queries):
    """Query-grouped SIR: fraction of same-query document pairs with tied relevance."""
    total_pairs, total_incomparable = 0, 0
    for q in queries:
        rel = q["relevance"]
        n = rel.size(0)
        if n < 2:
            continue
        n_pairs = n * (n - 1)
        same = (rel.unsqueeze(0) == rel.unsqueeze(1)).sum().item() - n
        total_pairs += n_pairs
        total_incomparable += same
    return total_incomparable / total_pairs if total_pairs > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True, help="Directory containing train.txt, vali.txt, test.txt")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train_queries = load_letor_file(os.path.join(args.fold_dir, "train.txt"))
    vali_queries = load_letor_file(os.path.join(args.fold_dir, "vali.txt"))
    test_queries = load_letor_file(os.path.join(args.fold_dir, "test.txt"))
    all_queries = train_queries + vali_queries + test_queries

    sir = compute_mq2008_sir(all_queries)
    print(f"MQ2008: {len(all_queries)} queries, query-grouped SIR = {sir:.1%}")

    result = run_mq2008_comparison(train_queries, test_queries, epochs=args.epochs)
    print(f"\nAdvantage (within-query minus naive): {result['advantage']:+.4f}")


if __name__ == "__main__":
    main()
