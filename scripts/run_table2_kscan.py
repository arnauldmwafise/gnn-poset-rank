#!/usr/bin/env python3
"""
Reproduce Table 2: the synthetic K-scan predictive study (Section 5.3).

Usage
-----
    python scripts/run_table2_kscan.py
    python scripts/run_table2_kscan.py --k-values 1 2 3 --seeds 0 1 2  # quicker smoke test
"""

import argparse

from gnn_poset_rank.experiments.k_scan import run_k_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    results = run_k_scan(k_values=args.k_values, seeds=args.seeds)

    print(f"{'K':<4}{'SIR (mean)':<14}{'Advantage (mean +/- std)':<28}{'Sign-stable?'}")
    for k, r in results.items():
        stable = "Yes -- always positive" if r["sign_stable"] and r["advantage_mean"] >= 0 else (
            "Yes -- always negative" if r["sign_stable"] else f"No -- {r['n_negative']}/{len(args.seeds)} seeds negative"
        )
        print(f"{k:<4}{r['sir_mean']:<14.1%}{r['advantage_mean']:+.4f} +/- {r['advantage_std']:.4f}      {stable}")


if __name__ == "__main__":
    main()
