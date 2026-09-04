#!/usr/bin/env python3
"""
Reproduce Tables 4, 4a, and 5: the NBA moderate-SIR peak validation
(Section 5.6) and the HOUSE cross-domain replication check (Section 5.7).

Usage
-----
    python scripts/run_tables4_5_skyline.py --data-dir data --seeds 0 1 2 3 4
"""

import argparse
import os

from posetrank.data.skyline import build_skyline_dataset, find_best_dimension_pair, load_skyline_csv
from posetrank.experiments.skyline_eval import run_skyline_multiseed


def report(name, dims, sir, dataset, seeds):
    result = run_skyline_multiseed(dataset, seeds=seeds)
    stable = "Yes" if result["sign_stable"] else f"No ({result['n_negative']}/{len(seeds)} negative)"
    print(
        f"{name:<10}{str(dims):<10}{sir:<8.1%}"
        f"{result['advantage_mean']:+.4f} +/- {result['advantage_std']:.4f}    {stable}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-pairs-to-test", type=int, default=4, help="How many top dimension pairs to test per dataset")
    args = parser.parse_args()

    print(f"{'Dataset':<10}{'Dims':<10}{'SIR':<8}{'Advantage (mean +/- std)':<28}{'Sign-stable?'}")

    nba_data = load_skyline_csv(os.path.join(args.data_dir, "skyline_raw", "nba.csv"), n_dims=8)
    nba_pairs = find_best_dimension_pair(nba_data, target_sir=0.5, seed=0)
    for dims, sir in nba_pairs[: args.n_pairs_to_test]:
        dataset = build_skyline_dataset(nba_data, dims, n=3000, seed=0)
        report("NBA", dims, sir, dataset, args.seeds)

    house_data = load_skyline_csv(os.path.join(args.data_dir, "skyline_raw", "house.csv"), n_dims=6)
    house_pairs = find_best_dimension_pair(house_data, target_sir=0.5, seed=0)
    # closest-to-target pair, plus the closest pair on the SAME side of
    # 50% as NBA's successful pairs (all fractionally below), matching
    # the paper's peak-asymmetry check in Section 5.7
    dims, sir = house_pairs[0]
    dataset = build_skyline_dataset(house_data, dims, n=3000, seed=0)
    report("HOUSE", dims, sir, dataset, args.seeds)

    below_50 = [p for p in house_pairs if p[1] < 0.5]
    if below_50:
        dims, sir = below_50[0]
        dataset = build_skyline_dataset(house_data, dims, n=3000, seed=0)
        report("HOUSE", dims, sir, dataset, args.seeds)


if __name__ == "__main__":
    main()
