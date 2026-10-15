#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare tile-level input for EGNv1-style all-gene training/prediction.

Input folder format follows your existing code:

sample/
  he-raw.jpg
  cnts.mtx
  genes.tsv
  barcodes.tsv
  locs-raw.tsv   # must contain x, y columns

Output:

EGNv1_tile_input_allgenes/
  manifest.tsv
  samples/
    OV_VisiumHD_all_new/
      image_path.txt
      genes.tsv
      tile_counts.h5
      tile_coords.npy
      tile_meta.csv
    HCC_VisiumHD_all_new/
      ...
    OV_Xenium_all_new/
      ...
"""

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import io as spio
from scipy import sparse
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


DEFAULT_SAMPLES = {
    "OV_VisiumHD_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_VisiumHD_all_new",
    "HCC_VisiumHD_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_VisiumHD_all_new",

    "OC_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/aligned_from_adata_for_path2space/OC_all_new_aligned",
    "CC_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/aligned_from_adata_for_path2space/CC_all_new_aligned",
    "HCC_Xenium_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new",
    "OV_Xenium_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium_all_new",
}


def read_matrix_auto(mtx_path, n_obs, n_genes):
    X = spio.mmread(str(mtx_path))
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    else:
        X = X.tocsr()

    # expected: rows = bins/spots, cols = genes
    if X.shape == (n_obs, n_genes):
        return X.tocsr()

    # sometimes mtx is genes x spots
    if X.shape == (n_genes, n_obs):
        print(f"[INFO] transpose matrix: {mtx_path}, {X.shape} -> {(n_obs, n_genes)}")
        return X.T.tocsr()

    raise ValueError(
        f"Matrix shape does not match in {mtx_path}\n"
        f"  X.shape = {X.shape}\n"
        f"  n_obs   = {n_obs}\n"
        f"  n_genes = {n_genes}"
    )


def load_spatial_folder(folder):
    folder = Path(folder)

    img_path = folder / "he-raw.jpg"
    cnts_path = folder / "cnts.mtx"
    genes_path = folder / "genes.tsv"
    barcodes_path = folder / "barcodes.tsv"
    locs_path = folder / "locs-raw.tsv"

    required = [img_path, cnts_path, genes_path, barcodes_path, locs_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    image = Image.open(img_path).convert("RGB")
    image_w, image_h = image.size

    genes = pd.read_csv(genes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()

    locs = pd.read_csv(locs_path, sep="\t")
    if "x" not in locs.columns or "y" not in locs.columns:
        raise ValueError(f"{locs_path} must contain columns: x, y")

    coords = locs[["x", "y"]].values.astype(np.float32)
    X = read_matrix_auto(cnts_path, n_obs=coords.shape[0], n_genes=len(genes))

    if len(barcodes) != X.shape[0]:
        print(f"[WARN] barcodes length {len(barcodes)} != matrix rows {X.shape[0]}; use row order anyway.")

    return {
        "folder": str(folder),
        "image_path": str(img_path),
        "image_w": image_w,
        "image_h": image_h,
        "X": X,
        "genes": genes,
        "barcodes": barcodes,
        "coords": coords,
    }


def aggregate_to_tiles(X, coords, tile_size=256, min_bins_per_tile=1, agg="sum"):
    """
    Aggregate bin/spot-level expression to non-overlapping tile-level expression.

    Tile id is floor(x / tile_size), floor(y / tile_size).
    Tile coordinate is the tile center in full-resolution image coordinate.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    tile_x = np.floor(x / tile_size).astype(np.int64)
    tile_y = np.floor(y / tile_size).astype(np.int64)

    tile_key = pd.Series([f"{a}_{b}" for a, b in zip(tile_x, tile_y)])
    codes, unique_keys = pd.factorize(tile_key, sort=True)

    n_tiles = len(unique_keys)
    n_obs = X.shape[0]

    G = sparse.csr_matrix(
        (
            np.ones(n_obs, dtype=np.float32),
            (codes, np.arange(n_obs)),
        ),
        shape=(n_tiles, n_obs),
    )

    tile_counts = (G @ X).tocsr()
    tile_n_bins = np.asarray(G.sum(axis=1)).ravel().astype(np.int64)

    if agg == "mean":
        denom = tile_n_bins.astype(np.float32).copy()
        denom[denom <= 0] = 1.0
        tile_counts = sparse.diags(1.0 / denom) @ tile_counts
        tile_counts = tile_counts.tocsr()
    elif agg != "sum":
        raise ValueError("--agg must be sum or mean")

    keep = tile_n_bins >= min_bins_per_tile
    tile_counts = tile_counts[keep].tocsr()
    tile_n_bins = tile_n_bins[keep]
    kept_keys = np.asarray(unique_keys)[keep]

    tile_ids = []
    tile_coords = []

    for key in kept_keys:
        a, b = key.split("_")
        a = int(a)
        b = int(b)
        tile_ids.append(key)
        tile_coords.append([(a + 0.5) * tile_size, (b + 0.5) * tile_size])

    tile_coords = np.asarray(tile_coords, dtype=np.float32)
    return tile_ids, tile_coords, tile_counts, tile_n_bins


def write_tile_h5(out_h5, tile_counts):
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    tile_counts = tile_counts.astype(np.float32).tocsr()

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "counts",
            data=tile_counts.toarray().astype(np.float32),
            compression="gzip",
            chunks=(min(512, tile_counts.shape[0]), min(512, tile_counts.shape[1])),
        )


def process_one_sample(sample_name, sample_path, out_root, tile_size, min_bins_per_tile, agg):
    print("\n" + "=" * 80)
    print(f"[Sample] {sample_name}")
    print(f"[Path]   {sample_path}")
    print("=" * 80)

    sample = load_spatial_folder(sample_path)

    tile_ids, tile_coords, tile_counts, tile_n_bins = aggregate_to_tiles(
        sample["X"],
        sample["coords"],
        tile_size=tile_size,
        min_bins_per_tile=min_bins_per_tile,
        agg=agg,
    )

    out_dir = Path(out_root) / "samples" / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "image_path.txt", "w") as f:
        f.write(sample["image_path"] + "\n")

    pd.Series(sample["genes"]).to_csv(out_dir / "genes.tsv", sep="\t", index=False, header=False)
    np.save(out_dir / "tile_coords.npy", tile_coords)
    write_tile_h5(out_dir / "tile_counts.h5", tile_counts)

    meta = pd.DataFrame({
        "tile_id": tile_ids,
        "x": tile_coords[:, 0],
        "y": tile_coords[:, 1],
        "n_bins": tile_n_bins,
    })
    meta.to_csv(out_dir / "tile_meta.csv", index=False)

    nonzero_fraction = float(tile_counts.nnz / (tile_counts.shape[0] * tile_counts.shape[1]))

    summary = {
        "sample": sample_name,
        "source_path": sample_path,
        "image_path": sample["image_path"],
        "image_w": sample["image_w"],
        "image_h": sample["image_h"],
        "n_raw_bins": sample["X"].shape[0],
        "n_tiles": tile_counts.shape[0],
        "n_genes": tile_counts.shape[1],
        "mean_bins_per_tile": float(np.mean(tile_n_bins)),
        "median_bins_per_tile": float(np.median(tile_n_bins)),
        "min_bins_per_tile": int(np.min(tile_n_bins)),
        "max_bins_per_tile": int(np.max(tile_n_bins)),
        "nonzero_fraction": nonzero_fraction,
        "prepared_dir": str(out_dir),
    }

    print("[Summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./EGNv1_tile_input_allgenes")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--min_bins_per_tile", type=int, default=1)
    parser.add_argument("--agg", choices=["sum", "mean"], default="sum")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help=(
            "Optional custom samples in format name=path. "
            "If not given, use the six hard-coded SQUALL paths."
        ),
    )
    args = parser.parse_args()

    if args.samples is None or len(args.samples) == 0:
        sample_dict = DEFAULT_SAMPLES
    else:
        sample_dict = {}
        for item in args.samples:
            if "=" not in item:
                raise ValueError(f"Invalid --samples item: {item}. Expected name=path")
            name, path = item.split("=", 1)
            sample_dict[name] = path

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[Config]")
    print(f"  out_dir           : {args.out_dir}")
    print(f"  tile_size         : {args.tile_size}")
    print(f"  min_bins_per_tile : {args.min_bins_per_tile}")
    print(f"  agg               : {args.agg}")
    print(f"  n_samples         : {len(sample_dict)}")

    rows = []
    for sample_name, sample_path in sample_dict.items():
        rows.append(
            process_one_sample(
                sample_name=sample_name,
                sample_path=sample_path,
                out_root=out_root,
                tile_size=args.tile_size,
                min_bins_per_tile=args.min_bins_per_tile,
                agg=args.agg,
            )
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_root / "manifest.tsv", sep="\t", index=False)

    print("\n" + "=" * 80)
    print("[Done]")
    print(f"Manifest saved to: {out_root / 'manifest.tsv'}")
    print("=" * 80)
    print(manifest)


if __name__ == "__main__":
    main()
