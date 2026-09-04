#!/usr/bin/env python3
"""
Reproduce Table 3: the real-data validation on Cora, CiteSeer, Chameleon,
and Amazon0302 (Section 5.4).

Usage
-----
    python scripts/run_table3_real_data.py --data-dir data --seeds 0 1 2 3 4
"""

import argparse
import os
import statistics

from posetrank.data import load_chameleon_raw, load_citation_raw
from posetrank.data.amazon import build_structural_features, compute_ranks, load_amazon0302_subgraph
from posetrank.diagnostics import clean_dag
from posetrank.experiments.real_data_eval import run_real_data_comparison


def run_dataset_multiseed(name, x, cover_edges, num_nodes, seeds):
    advantages = []
    for seed in seeds:
        result = run_real_data_comparison(name, x, cover_edges, num_nodes, seed=seed, verbose=False)
        advantages.append(result["advantage"])
    adv_mean = statistics.mean(advantages)
    adv_std = statistics.stdev(advantages) if len(advantages) > 1 else 0.0
    sign_stable = all(a >= 0 for a in advantages) or all(a <= 0 for a in advantages)
    n_neg = sum(1 for a in advantages if a < 0)
    print(f"{name:<12}{adv_mean:+.4f} +/- {adv_std:.4f}    {'Yes' if sign_stable else f'No ({n_neg}/{len(seeds)} neg)'}")
    return adv_mean, adv_std


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--amazon-subgraph-size", type=int, default=8000)
    args = parser.parse_args()

    print(f"{'Dataset':<12}{'Advantage (mean +/- std)':<28}{'Sign-stable?'}")

    cora = load_citation_raw(
        os.path.join(args.data_dir, "cora_raw", "cora.content"), os.path.join(args.data_dir, "cora_raw", "cora.cites")
    )
    cora_clean = clean_dag(cora["raw_edges"], cora["num_nodes"])
    run_dataset_multiseed("Cora", cora["x"], cora_clean["cover_edges"], cora["num_nodes"], args.seeds)

    citeseer = load_citation_raw(
        os.path.join(args.data_dir, "citeseer_raw", "citeseer.content"),
        os.path.join(args.data_dir, "citeseer_raw", "citeseer.cites"),
    )
    citeseer_clean = clean_dag(citeseer["raw_edges"], citeseer["num_nodes"])
    run_dataset_multiseed("CiteSeer", citeseer["x"], citeseer_clean["cover_edges"], citeseer["num_nodes"], args.seeds)

    chameleon = load_chameleon_raw(
        os.path.join(args.data_dir, "chameleon_raw", "out1_graph_edges.txt"),
        os.path.join(args.data_dir, "chameleon_raw", "out1_node_feature_label.txt"),
    )
    chameleon_clean = clean_dag(chameleon["raw_edges"], chameleon["num_nodes"])
    run_dataset_multiseed("Chameleon", chameleon["x"], chameleon_clean["cover_edges"], chameleon["num_nodes"], args.seeds)

    amazon = load_amazon0302_subgraph(
        os.path.join(args.data_dir, "amazon0302_raw", "Amazon0302.txt"), target_size=args.amazon_subgraph_size, seed=0
    )
    amazon_clean = clean_dag(amazon["raw_edges"], amazon["num_nodes"])
    amazon_ranks = compute_ranks(amazon_clean["cover_edges"], amazon["num_nodes"])
    amazon_x = build_structural_features(amazon["raw_edges"], amazon_ranks, amazon["num_nodes"])
    run_dataset_multiseed("Amazon0302", amazon_x, amazon_clean["cover_edges"], amazon["num_nodes"], args.seeds)


if __name__ == "__main__":
    main()
