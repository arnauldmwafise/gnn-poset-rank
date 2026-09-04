#!/usr/bin/env python3
"""
Download every dataset used in the paper that can be fetched
automatically: Cora, CiteSeer, Chameleon, Amazon0302, and the NBA/HOUSE
skyline benchmarks.

MQ2008 (LETOR 4.0) is not included: Microsoft Research's project page
does not offer a single stable, automation-friendly direct-download URL.
See the README for manual download instructions.

Usage
-----
    python scripts/download_data.py --data-dir data
"""

import argparse
import gzip
import os
import shutil
import urllib.request


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  already exists, skipping: {dest}")
        return
    print(f"  downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    d = args.data_dir

    print("Cora, CiteSeer:")
    download(
        "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites",
        os.path.join(d, "cora_raw", "cora.cites"),
    )
    download(
        "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.content",
        os.path.join(d, "cora_raw", "cora.content"),
    )
    download(
        "https://raw.githubusercontent.com/ZPowerZ/citeseer-dataset/master/citeseer.cites",
        os.path.join(d, "citeseer_raw", "citeseer.cites"),
    )
    download(
        "https://raw.githubusercontent.com/ZPowerZ/citeseer-dataset/master/citeseer.content",
        os.path.join(d, "citeseer_raw", "citeseer.content"),
    )

    print("Chameleon:")
    commit = "f1fc0d14b3b019c562737240d06ec83b07d16a8f"  # master 404s for chameleon/squirrel specifically
    base = f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/{commit}/new_data/chameleon/"
    download(base + "out1_graph_edges.txt", os.path.join(d, "chameleon_raw", "out1_graph_edges.txt"))
    download(base + "out1_node_feature_label.txt", os.path.join(d, "chameleon_raw", "out1_node_feature_label.txt"))

    print("Amazon0302:")
    gz_path = os.path.join(d, "amazon0302_raw", "amazon0302.txt.gz")
    download("https://snap.stanford.edu/data/amazon0302.txt.gz", gz_path)
    txt_path = os.path.join(d, "amazon0302_raw", "Amazon0302.txt")
    if not os.path.exists(txt_path):
        print(f"  extracting {gz_path} -> {txt_path}")
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("NBA, HOUSE (skyline benchmarks):")
    download(
        "https://raw.githubusercontent.com/sean-chester/SkyBench/master/workloads/nba-U-8-17264.csv",
        os.path.join(d, "skyline_raw", "nba.csv"),
    )
    download(
        "https://raw.githubusercontent.com/sean-chester/SkyBench/master/workloads/house-U-6-127931.csv",
        os.path.join(d, "skyline_raw", "house.csv"),
    )

    print(
        "\nMQ2008 (LETOR 4.0) is not downloaded automatically. Download MQ2008.rar (or a "
        "maintained mirror) from Microsoft Research's LETOR 4.0 project page, extract it, "
        "and pass --fold-dir to scripts/run_mq2008.py pointing at the Fold1 directory."
    )


if __name__ == "__main__":
    main()
