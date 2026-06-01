#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None


def load_counts_h5(path):
    with h5py.File(path, "r") as f:
        if "counts" not in f:
            raise KeyError(f"{path} does not contain dataset 'counts'")
        X = f["counts"][:].astype(np.float32)
    return X


def load_sample(tile_root, sample):
    sample_dir = Path(tile_root) / "samples" / sample
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample dir: {sample_dir}")

    image_path = open(sample_dir / "image_path.txt").read().strip()
    coords = np.load(sample_dir / "tile_coords.npy").astype(np.float32)
    meta = pd.read_csv(sample_dir / "tile_meta.csv")
    X = load_counts_h5(sample_dir / "tile_counts.h5")

    if coords.shape[0] != X.shape[0]:
        raise ValueError(f"{sample}: coords rows {coords.shape[0]} != counts rows {X.shape[0]}")

    if len(meta) != X.shape[0]:
        raise ValueError(f"{sample}: meta rows {len(meta)} != counts rows {X.shape[0]}")

    return {
        "sample": sample,
        "sample_dir": str(sample_dir),
        "image_path": image_path,
        "coords": coords,
        "meta": meta,
        "X": X,
    }


def plot_one_sample(
    tile_root,
    sample,
    out_dir,
    point_size=6,
    alpha=0.85,
    max_points=0,
    log_total=True,
    invert_y=False,
    cmap="viridis",
):
    data = load_sample(tile_root, sample)

    image = Image.open(data["image_path"]).convert("RGB")
    coords = data["coords"]
    X = data["X"]
    meta = data["meta"]

    total = X.sum(axis=1).astype(np.float32)
    value = np.log1p(total) if log_total else total

    keep = np.arange(coords.shape[0])
    if max_points and max_points > 0 and max_points < len(keep):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(keep, size=max_points, replace=False))

    coords_plot = coords[keep]
    value_plot = value[keep]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. HE + tile total count overlay
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image)

    sc = ax.scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        c=value_plot,
        s=point_size,
        alpha=alpha,
        cmap=cmap,
        linewidths=0,
    )

    ax.set_title(
        f"{sample}\nHE overlay: {'log1p(total count)' if log_total else 'total count'}",
        fontsize=14,
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")

    if invert_y:
        ax.invert_yaxis()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log1p(total count)" if log_total else "total count")

    ax.set_aspect("equal")
    plt.tight_layout()

    out_png = out_dir / f"{sample}_HE_tile_totalcount_overlay.png"
    out_pdf = out_dir / f"{sample}_HE_tile_totalcount_overlay.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    # ----------------------------------------------------
    # 2. tile-only spatial plot
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        coords_plot[:, 0],
        coords_plot[:, 1],
        c=value_plot,
        s=point_size,
        alpha=alpha,
        cmap=cmap,
        linewidths=0,
    )

    ax.set_title(
        f"{sample}\nTile spatial map: {'log1p(total count)' if log_total else 'total count'}",
        fontsize=14,
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log1p(total count)" if log_total else "total count")

    plt.tight_layout()

    out_png2 = out_dir / f"{sample}_tile_totalcount_spatial_only.png"
    out_pdf2 = out_dir / f"{sample}_tile_totalcount_spatial_only.pdf"
    fig.savefig(out_png2, dpi=300)
    fig.savefig(out_pdf2)
    plt.close(fig)

    # ----------------------------------------------------
    # 3. QC summary
    # ----------------------------------------------------
    summary = {
        "sample": sample,
        "n_tiles": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "total_count_min": float(total.min()),
        "total_count_median": float(np.median(total)),
        "total_count_mean": float(total.mean()),
        "total_count_max": float(total.max()),
        "nonzero_fraction": float((X > 0).mean()),
        "x_min": float(coords[:, 0].min()),
        "x_max": float(coords[:, 0].max()),
        "y_min": float(coords[:, 1].min()),
        "y_max": float(coords[:, 1].max()),
        "image_path": data["image_path"],
        "overlay_png": str(out_png),
        "spatial_png": str(out_png2),
    }

    pd.DataFrame([summary]).to_csv(out_dir / f"{sample}_qc_summary.csv", index=False)

    print("[Saved]")
    print(f"  {out_png}")
    print(f"  {out_pdf}")
    print(f"  {out_png2}")
    print(f"  {out_pdf2}")
    print(f"  {out_dir / f'{sample}_qc_summary.csv'}")
    print("[Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_root", default="./EGNv1_tile_input_allgenes")
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--out_dir", default="./tile_input_qc_plots")
    parser.add_argument("--point_size", type=float, default=6)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--invert_y_on_HE", action="store_true")
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()

    all_summary = []
    for sample in args.samples:
        s = plot_one_sample(
            tile_root=args.tile_root,
            sample=sample,
            out_dir=args.out_dir,
            point_size=args.point_size,
            alpha=args.alpha,
            max_points=args.max_points,
            log_total=not args.no_log,
            invert_y=args.invert_y_on_HE,
            cmap=args.cmap,
        )
        all_summary.append(s)

    pd.DataFrame(all_summary).to_csv(Path(args.out_dir) / "all_qc_summary.csv", index=False)
    print(f"[Done] {Path(args.out_dir) / 'all_qc_summary.csv'}")


if __name__ == "__main__":
    main()
