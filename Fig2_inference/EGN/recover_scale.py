#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse


def load_spatial_folder(folder):
    folder = Path(folder)

    cnts_path = folder / "cnts.mtx"
    genes_path = folder / "genes.tsv"

    assert cnts_path.exists()
    assert genes_path.exists()

    X = spio.mmread(cnts_path)

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    genes = (
        pd.read_csv(genes_path, sep="\t", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )

    return X, genes


def normalize_log_cpm(X, scale_factor=1e4):
    X = X.astype(np.float32).tocsr()

    libsize = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    libsize[libsize <= 0] = 1.0

    scale = scale_factor / libsize

    X = X.multiply(scale[:, None]).tocsr()

    X.data = np.log1p(X.data)

    return X


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        required=True,
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help="EGN_OC_HD_to_Xenium_out directory",
    )

    args = parser.parse_args()

    selected_gene_path = os.path.join(
        args.model_dir,
        "selected_genes.tsv",
    )

    ymin_path = os.path.join(
        args.model_dir,
        "ymin.npy",
    )

    ymax_path = os.path.join(
        args.model_dir,
        "ymax.npy",
    )

    out_path = os.path.join(
        args.model_dir,
        "train_expr_scaled.npy",
    )

    print("=" * 80)
    print("[Load train data]")
    print("=" * 80)

    X, genes = load_spatial_folder(args.train_dir)

    print("X shape:", X.shape)
    print("n genes:", len(genes))

    print("=" * 80)
    print("[Normalize]")
    print("=" * 80)

    X_log = normalize_log_cpm(X)

    print("=" * 80)
    print("[Load selected genes]")
    print("=" * 80)

    selected_genes = (
        pd.read_csv(selected_gene_path, header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )

    print("selected genes:", len(selected_genes))

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    missing = [g for g in selected_genes if g not in gene_to_idx]

    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} genes missing from train data"
        )

    selected_idx = [gene_to_idx[g] for g in selected_genes]

    print("=" * 80)
    print("[Build expression matrix]")
    print("=" * 80)

    Y = X_log[:, selected_idx].toarray().astype(np.float32)

    print("Y shape:", Y.shape)

    print("=" * 80)
    print("[Load ymin/ymax]")
    print("=" * 80)

    ymin = np.load(ymin_path)
    ymax = np.load(ymax_path)

    print("ymin shape:", ymin.shape)
    print("ymax shape:", ymax.shape)

    print("=" * 80)
    print("[Scale]")
    print("=" * 80)

    denom = ymax - ymin
    denom[denom < 1e-6] = 1.0

    Y_scaled = (Y - ymin) / denom

    Y_scaled = Y_scaled.astype(np.float32)

    print("scaled shape:", Y_scaled.shape)

    print("=" * 80)
    print("[Save]")
    print("=" * 80)

    np.save(out_path, Y_scaled)

    print("saved:")
    print(out_path)

    print("Done.")


if __name__ == "__main__":
    main()
