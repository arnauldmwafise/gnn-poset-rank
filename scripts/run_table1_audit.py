#!/usr/bin/env python3
"""
Reproduce Table 1: the cross-domain audit (CIR, TRR, SIR) on Cora,
CiteSeer, Chameleon, and Amazon0302.

Usage
-----
    python scripts/run_table1_audit.py --data-dir data/

Expects raw dataset files already downloaded into ``--data-dir`` (see
``scripts/download_data.py``).
"""

import argparse
import os

from gnn_poset_rank.data import load_chameleon_raw, load_citation_raw
from gnn_poset_rank.data.amazon import load_amazon0302_subgraph
from gnn_poset_rank.diagnostics import compute_order_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root directory containing raw dataset files")
    parser.add_argument("--amazon-subgraph-size", type=int, default=8000)
    args = parser.parse_args()

    rows = []

    cora = load_citation_raw(
        os.path.join(args.data_dir, "cora_raw", "cora.content"),
        os.path.join(args.data_dir, "cora_raw", "cora.cites"),
    )
    diag = compute_order_diagnostics(cora["raw_edges"], cora["num_nodes"], sample_pairs_for_sir=200_000, seed=0)
    rows.append(("Cora", diag))

    citeseer = load_citation_raw(
        os.path.join(args.data_dir, "citeseer_raw", "citeseer.content"),
        os.path.join(args.data_dir, "citeseer_raw", "citeseer.cites"),
    )
    diag = compute_order_diagnostics(citeseer["raw_edges"], citeseer["num_nodes"], sample_pairs_for_sir=200_000, seed=0)
    rows.append(("CiteSeer", diag))

    chameleon = load_chameleon_raw(
        os.path.join(args.data_dir, "chameleon_raw", "out1_graph_edges.txt"),
        os.path.join(args.data_dir, "chameleon_raw", "out1_node_feature_label.txt"),
    )
    diag = compute_order_diagnostics(chameleon["raw_edges"], chameleon["num_nodes"], sample_pairs_for_sir=200_000, seed=0)
    rows.append(("Chameleon", diag))

    amazon = load_amazon0302_subgraph(
        os.path.join(args.data_dir, "amazon0302_raw", "Amazon0302.txt"),
        target_size=args.amazon_subgraph_size,
        seed=0,
    )
    diag = compute_order_diagnostics(amazon["raw_edges"], amazon["num_nodes"], sample_pairs_for_sir=200_000, seed=0)
    rows.append(("Amazon0302", diag))

    print(f"{'Dataset':<12}{'N':<8}{'Edges':<10}{'CIR':<8}{'TRR':<8}{'SIR'}")
    for name, d in rows:
        print(
            f"{name:<12}{d['n_nodes']:<8}{d['n_raw_edges']:<10}"
            f"{d['cycle_inconsistency_rate']:<8.1%}{d['transitive_redundancy_rate']:<8.1%}"
            f"{d['structural_incomparability_rate']:.1%}"
        )
    print(
        "\nNote: MQ2008's SIR (75.5% in the paper) uses a different, query-grouped "
        "definition (tied relevance within a query) -- see Section 4.5/4.7 of the paper "
        "and scripts/run_table3_mq2008.py."
    )


if __name__ == "__main__":
    main()
