#!/usr/bin/env python3
"""
Reproduce Table 6: the controlled synthetic test of feature
informativeness at fixed SIR approx. 50% (Section 5.7.1).

Usage
-----
    python scripts/run_table6_informativeness.py
    python scripts/run_table6_informativeness.py --sigma-values 0.4 4.0 --seeds 0 1  # quicker smoke test
"""

import argparse

from posetrank.experiments.feature_informativeness import run_informativeness_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-values", type=float, nargs="+", default=[0.05, 0.4, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    results = run_informativeness_scan(sigma_values=args.sigma_values, seeds=args.seeds)

    print(f"{'Informativeness':<18}{'SIR':<10}{'Advantage (mean +/- std)':<28}{'Sign-stable?'}")
    for sigma, r in results.items():
        stable = "Yes" if r["sign_stable"] else f"No ({r['n_negative']}/{len(args.seeds)} negative)"
        print(f"{r['informativeness_mean']:<18.3f}{r['sir_mean']:<10.1%}{r['advantage_mean']:+.4f} +/- {r['advantage_std']:.4f}      {stable}")


if __name__ == "__main__":
    main()
