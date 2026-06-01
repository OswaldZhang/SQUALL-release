#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict-only tile-level Path2Space-style prediction.

This script does NOT train.

Main functions:
  1. Load already trained Path2Space-style MLP ensemble.
  2. Load external test spatial folders.
  3. Apply coordinate correction before tile grouping / patch extraction / h5 saving:
       --swap_xy
       --flip_x
       --flip_y
       --flip_basis image|coords
  4. Build tile-level groups.
  5. Extract CTransPath image features for each tile.
  6. Predict selected genes using trained ensemble.
  7. Save tile-level h5/csv.
  8. Optional input debug plots:
       --plot_input_debug

Input test folder format:
  sample/
    he-raw.jpg
    cnts.mtx
    genes.tsv
    barcodes.tsv
    locs-raw.tsv   # must contain x, y columns

Output:
  out_dir/
    predict_only_run_config.json
    coordinate_transform_used.json
    selected_genes.tsv
    ensemble_model_paths.txt
    SAMPLE_test_tile_metadata.csv
    input_debug/SAMPLE/
      SAMPLE_raw_vs_transformed_spots_on_HE.png/pdf
      SAMPLE_tile_centers_on_HE.png/pdf
      SAMPLE_actual_input_patch_montage.png/pdf
      SAMPLE_debug_tile_table.csv
    SAMPLE/
      path2space_tilelevel_predicted_expression_ensemble.h5
      optional csv
"""

import os
import sys
import json
import glob
import h5py
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import io as spio
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(x):
    x = str(x).rstrip("/").split("/")[-1]
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in x])


def fmt_float_for_tag(x):
    return str(float(x)).replace(".", "p").replace("-", "m")


# ============================================================
# Input loading
# ============================================================

def has_spatial_files(folder):
    folder = Path(folder)
    required = [
        "he-raw.jpg",
        "cnts.mtx",
        "genes.tsv",
        "barcodes.tsv",
        "locs-raw.tsv",
    ]
    return all((folder / x).exists() for x in required)


def discover_spatial_folders(path):
    """
    If path itself is a spatial folder, return [path].
    Otherwise, return immediate child folders that look like spatial folders.
    """
    path = Path(path)

    if has_spatial_files(path):
        return [path]

    folders = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and has_spatial_files(child):
            folders.append(child)

    if len(folders) == 0:
        raise FileNotFoundError(
            f"No valid spatial folder found under {path}. "
            "Expected he-raw.jpg, cnts.mtx, genes.tsv, barcodes.tsv, locs-raw.tsv."
        )

    return folders


def load_spatial_folder(folder):
    folder = Path(folder)

    img_path = folder / "he-raw.jpg"
    cnts_path = folder / "cnts.mtx"
    genes_path = folder / "genes.tsv"
    barcodes_path = folder / "barcodes.tsv"
    locs_path = folder / "locs-raw.tsv"

    assert img_path.exists(), f"Missing {img_path}"
    assert cnts_path.exists(), f"Missing {cnts_path}"
    assert genes_path.exists(), f"Missing {genes_path}"
    assert barcodes_path.exists(), f"Missing {barcodes_path}"
    assert locs_path.exists(), f"Missing {locs_path}"

    image = Image.open(img_path).convert("RGB")

    X = spio.mmread(cnts_path)
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = sparse.csr_matrix(X)

    genes = pd.read_csv(genes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:, 0].astype(str).tolist()

    locs = pd.read_csv(locs_path, sep="\t")
    if "x" not in locs.columns or "y" not in locs.columns:
        raise ValueError(f"{locs_path} must contain columns: x, y")

    coords = locs[["x", "y"]].values.astype(float)

    # Safe handling for gene x spot matrix.
    if X.shape[0] != coords.shape[0] and X.shape[1] == coords.shape[0]:
        print(f"[INFO] Transpose matrix in {folder}: original shape={X.shape}")
        X = X.T.tocsr()

    if X.shape[0] != coords.shape[0]:
        raise ValueError(f"X rows {X.shape[0]} != coords rows {coords.shape[0]} in {folder}")

    if X.shape[1] != len(genes):
        raise ValueError(f"X cols {X.shape[1]} != genes {len(genes)} in {folder}")

    if len(barcodes) != X.shape[0]:
        print(
            f"[WARN] barcodes {len(barcodes)} != X rows {X.shape[0]} in {folder}; "
            "use fallback barcodes."
        )
        barcodes = [f"{sanitize_name(folder)}_spot{i}" for i in range(X.shape[0])]

    return {
        "folder": str(folder),
        "sample": sanitize_name(folder),
        "image": image,
        "X": X,
        "genes": genes,
        "barcodes": barcodes,
        "coords": coords,
        "coords_raw": coords.copy(),
    }


# ============================================================
# Coordinate transform fix
# ============================================================

def build_coord_transform_tag(args):
    tag = (
        f"scale{fmt_float_for_tag(args.test_coord_scale)}"
        f"_swap{int(args.swap_xy)}"
        f"_flipx{int(args.flip_x)}"
        f"_flipy{int(args.flip_y)}"
        f"_basis{args.flip_basis}"
    )
    return sanitize_name(tag)


def print_coord_range(name, coords, image=None):
    coords = np.asarray(coords, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]

    print(
        f"[Coord range] {name}: "
        f"x=({np.nanmin(x):.2f}, {np.nanmax(x):.2f}), "
        f"y=({np.nanmin(y):.2f}, {np.nanmax(y):.2f})"
    )

    if image is not None:
        W, H = image.size
        out_x = int(((x < 0) | (x >= W)).sum())
        out_y = int(((y < 0) | (y >= H)).sum())
        print(
            f"[Coord vs image] image W,H=({W}, {H}); "
            f"out_of_x={out_x}/{len(x)}, out_of_y={out_y}/{len(y)}"
        )


def transform_coords_for_prediction(
    coords,
    image,
    coord_scale=1.0,
    swap_xy=False,
    flip_x=False,
    flip_y=False,
    flip_basis="image",
):
    """
    Transform raw locs-raw.tsv coordinates into image-space coordinates used for:
      1. tile grouping
      2. patch extraction
      3. h5/csv saved tile center coords

    Order:
      raw coords
        -> scale
        -> optional swap_xy
        -> optional flip_x / flip_y

    flip_basis:
      image:
        flip around image width/height:
          x' = (W - 1) - x
          y' = (H - 1) - y
        Recommended when x/y are image pixel coordinates.

      coords:
        flip around coordinate range:
          x' = x_max - (x - x_min)
          y' = y_max - (y - y_min)
        Useful if coordinates are cropped/local and not in full image pixel frame.
    """
    coords = np.asarray(coords, dtype=np.float32).copy()

    # Apply coordinate scale first so everything below is in image pixel units.
    coords = coords * float(coord_scale)

    if swap_xy:
        coords = coords[:, [1, 0]]

    if flip_basis not in ["image", "coords"]:
        raise ValueError("--flip_basis must be one of: image, coords")

    if flip_x or flip_y:
        if flip_basis == "image":
            W, H = image.size

            if flip_x:
                coords[:, 0] = (float(W) - 1.0) - coords[:, 0]

            if flip_y:
                coords[:, 1] = (float(H) - 1.0) - coords[:, 1]

        else:
            x_min, x_max = float(np.nanmin(coords[:, 0])), float(np.nanmax(coords[:, 0]))
            y_min, y_max = float(np.nanmin(coords[:, 1])), float(np.nanmax(coords[:, 1]))

            if flip_x:
                coords[:, 0] = x_max - (coords[:, 0] - x_min)

            if flip_y:
                coords[:, 1] = y_max - (coords[:, 1] - y_min)

    return coords.astype(np.float32)


def apply_coordinate_transform_to_sample(sample, args):
    """
    Return a shallow-copied sample with transformed sample["coords"].
    """
    sample = dict(sample)

    raw_coords = np.asarray(sample["coords_raw"], dtype=np.float32)
    image = sample["image"]

    print_coord_range(f"{sample['sample']} raw", raw_coords, image=image)

    coords_transformed = transform_coords_for_prediction(
        coords=raw_coords,
        image=image,
        coord_scale=args.test_coord_scale,
        swap_xy=args.swap_xy,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_basis=args.flip_basis,
    )

    sample["coords"] = coords_transformed
    sample["coord_transform_tag"] = build_coord_transform_tag(args)
    sample["coord_transform_info"] = {
        "coord_scale": float(args.test_coord_scale),
        "swap_xy": bool(args.swap_xy),
        "flip_x": bool(args.flip_x),
        "flip_y": bool(args.flip_y),
        "flip_basis": str(args.flip_basis),
        "image_width": int(image.size[0]),
        "image_height": int(image.size[1]),
        "tag": sample["coord_transform_tag"],
    }

    print(
        f"[Coordinate transform] sample={sample['sample']} "
        f"tag={sample['coord_transform_tag']}"
    )
    print_coord_range(f"{sample['sample']} transformed", coords_transformed, image=image)

    return sample


# ============================================================
# Input debug plotting
# ============================================================

def downsample_image_for_plot(image, max_side=3000):
    """
    Downsample huge HE image for fast plotting.
    Return image array and scale factor from original pixel to displayed pixel.
    """
    W, H = image.size
    max_dim = max(W, H)

    if max_dim <= max_side:
        scale = 1.0
        img_small = image.convert("RGB")
    else:
        scale = max_side / float(max_dim)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        img_small = image.convert("RGB").resize((new_w, new_h), Image.LANCZOS)

    return np.asarray(img_small), scale


def sample_indices_evenly(n, max_n, seed=42):
    if n <= max_n:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    return np.sort(idx)


def plot_raw_vs_transformed_spots_on_he(
    sample_raw,
    sample_transformed,
    out_dir,
    max_spots=50000,
    dpi=200,
):
    """
    Plot raw locs coords and transformed coords on HE.
    This helps diagnose swap_xy / flip_x / flip_y.
    """
    os.makedirs(out_dir, exist_ok=True)

    image = sample_raw["image"]
    img_arr, scale = downsample_image_for_plot(image, max_side=3000)

    raw = np.asarray(sample_raw["coords_raw"], dtype=np.float32)
    transformed = np.asarray(sample_transformed["coords"], dtype=np.float32)

    n = raw.shape[0]
    idx = sample_indices_evenly(n, max_spots, seed=42)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].imshow(img_arr)
    axs[0].scatter(
        raw[idx, 0] * scale,
        raw[idx, 1] * scale,
        s=1,
        c="red",
        alpha=0.35,
        linewidths=0,
    )
    axs[0].set_title(
        f"{sample_raw['sample']} raw locs on HE\n"
        f"n_spots={n}, shown={len(idx)}",
        fontsize=11,
    )
    axs[0].axis("off")

    axs[1].imshow(img_arr)
    axs[1].scatter(
        transformed[idx, 0] * scale,
        transformed[idx, 1] * scale,
        s=1,
        c="cyan",
        alpha=0.35,
        linewidths=0,
    )
    axs[1].set_title(
        f"{sample_raw['sample']} transformed locs on HE\n"
        f"{sample_transformed.get('coord_transform_tag', 'NA')}",
        fontsize=11,
    )
    axs[1].axis("off")

    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{sample_raw['sample']}_raw_vs_transformed_spots_on_HE.png")
    out_pdf = os.path.join(out_dir, f"{sample_raw['sample']}_raw_vs_transformed_spots_on_HE.pdf")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Input debug saved] {out_png}")
    print(f"[Input debug saved] {out_pdf}")


def plot_tile_centers_on_he(
    sample,
    tile_df,
    out_dir,
    max_tiles=20000,
    dpi=200,
):
    """
    Plot tile centers on HE.
    These are the coordinates used for patch extraction and saved to h5 coords.
    """
    os.makedirs(out_dir, exist_ok=True)

    image = sample["image"]
    img_arr, scale = downsample_image_for_plot(image, max_side=3000)

    centers = tile_df[["center_x", "center_y"]].values.astype(np.float32)
    n_spots = tile_df["n_spots"].values.astype(float)

    idx = sample_indices_evenly(centers.shape[0], max_tiles, seed=43)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img_arr)

    sc = ax.scatter(
        centers[idx, 0] * scale,
        centers[idx, 1] * scale,
        c=n_spots[idx],
        s=5,
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )

    ax.set_title(
        f"{sample['sample']} tile centers on HE\n"
        f"n_tiles={centers.shape[0]}, shown={len(idx)} | {sample.get('coord_transform_tag', 'NA')}",
        fontsize=11,
    )
    ax.axis("off")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("n_spots per tile")

    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{sample['sample']}_tile_centers_on_HE.png")
    out_pdf = os.path.join(out_dir, f"{sample['sample']}_tile_centers_on_HE.pdf")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Input debug saved] {out_png}")
    print(f"[Input debug saved] {out_pdf}")


def plot_tile_patch_montage(
    sample,
    tile_df,
    out_dir,
    tile_size=224,
    n_patches=36,
    dpi=200,
    seed=44,
):
    """
    Plot actual image patches used as CTransPath input.
    This directly shows what the model sees.
    """
    os.makedirs(out_dir, exist_ok=True)

    image = sample["image"]
    n_tiles = tile_df.shape[0]

    if n_tiles == 0:
        print(f"[WARN] No tiles for patch montage: {sample['sample']}")
        return

    rng = np.random.default_rng(seed)

    if n_tiles <= n_patches:
        chosen = np.arange(n_tiles)
    else:
        chosen = rng.choice(n_tiles, size=n_patches, replace=False)
        chosen = np.sort(chosen)

    n_show = len(chosen)
    ncol = int(np.ceil(np.sqrt(n_show)))
    nrow = int(np.ceil(n_show / ncol))

    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.4))

    if nrow == 1 and ncol == 1:
        axs = np.array([[axs]])
    elif nrow == 1:
        axs = axs.reshape(1, -1)
    elif ncol == 1:
        axs = axs.reshape(-1, 1)

    for ax in axs.ravel():
        ax.axis("off")

    for panel_i, tile_i in enumerate(chosen):
        ax = axs.ravel()[panel_i]

        row = tile_df.iloc[int(tile_i)]
        x = float(row["center_x"])
        y = float(row["center_y"])

        patch = extract_patch(image, x, y, tile_size)

        if tile_size != 224:
            patch_show = patch.resize((224, 224))
        else:
            patch_show = patch

        ax.imshow(patch_show)
        ax.set_title(
            f"tile={row['tile_id']}\n"
            f"x={x:.0f}, y={y:.0f}, n={int(row['n_spots'])}",
            fontsize=7,
        )
        ax.axis("off")

    fig.suptitle(
        f"{sample['sample']} actual CTransPath input patches\n"
        f"tile_size={tile_size} | {sample.get('coord_transform_tag', 'NA')}",
        fontsize=12,
    )

    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{sample['sample']}_actual_input_patch_montage.png")
    out_pdf = os.path.join(out_dir, f"{sample['sample']}_actual_input_patch_montage.pdf")

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Input debug saved] {out_png}")
    print(f"[Input debug saved] {out_pdf}")


def plot_input_debug_all(
    sample_raw,
    sample_transformed,
    tile_df,
    out_dir,
    tile_size=224,
    max_spots=50000,
    max_tiles=20000,
    n_patches=36,
    dpi=200,
):
    """
    One-stop input debug plotting.
    """
    debug_dir = os.path.join(out_dir, "input_debug", sample_transformed["sample"])
    os.makedirs(debug_dir, exist_ok=True)

    plot_raw_vs_transformed_spots_on_he(
        sample_raw=sample_raw,
        sample_transformed=sample_transformed,
        out_dir=debug_dir,
        max_spots=max_spots,
        dpi=dpi,
    )

    plot_tile_centers_on_he(
        sample=sample_transformed,
        tile_df=tile_df,
        out_dir=debug_dir,
        max_tiles=max_tiles,
        dpi=dpi,
    )

    plot_tile_patch_montage(
        sample=sample_transformed,
        tile_df=tile_df,
        out_dir=debug_dir,
        tile_size=tile_size,
        n_patches=n_patches,
        dpi=dpi,
    )

    out_csv = os.path.join(debug_dir, f"{sample_transformed['sample']}_debug_tile_table.csv")
    tile_df.drop(columns=["spot_indices"], errors="ignore").to_csv(out_csv, index=False)
    print(f"[Input debug saved] {out_csv}")


# ============================================================
# CTransPath and Macenko
# ============================================================

def import_ctranspath_and_macenko(path2space_feature_dir, use_macenko=True):
    """
    Import:
      - CTransPath architecture from func.ctrans_model
      - Macenko normalizer from func.utils_color_norm only when use_macenko=True
    """
    path2space_feature_dir = str(path2space_feature_dir)

    if path2space_feature_dir not in sys.path:
        sys.path.insert(0, path2space_feature_dir)

    try:
        from func.ctrans_model import CTransPath
    except Exception as e:
        raise ImportError(
            "Cannot import CTransPath from func.ctrans_model.\n"
            "Please check --path2space_feature_dir contains func/ctrans_model.py.\n"
            f"Original error: {repr(e)}"
        )

    color_normalizer = None

    if use_macenko:
        try:
            import func.utils_color_norm as utils_color_norm
            color_normalizer = utils_color_norm.macenko_normalizer()
        except Exception as e:
            raise ImportError(
                "Cannot import Macenko normalizer from func.utils_color_norm.\n"
                "If you want to skip Macenko, run with --no_macenko.\n"
                f"Original error: {repr(e)}"
            )

    return CTransPath, color_normalizer


def load_ctranspath_model(CTransPath, weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"CTransPath weight not found: {weight_path}")

    print(f"[Load CTransPath weight] {weight_path}")

    model = CTransPath(num_classes=0).to(device)

    ckpt = torch.load(weight_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    msg = model.load_state_dict(new_state, strict=False)
    print("[CTransPath load msg]", msg)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def get_ctranspath_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Load trained artifacts
# ============================================================

def load_trained_artifacts(trained_out_dir):
    """
    Load selected genes, trained model paths, and model metadata.
    """
    if trained_out_dir is None:
        raise ValueError("--trained_out_dir is required.")

    if not os.path.isdir(trained_out_dir):
        raise FileNotFoundError(f"trained_out_dir not found: {trained_out_dir}")

    selected_genes_path = os.path.join(trained_out_dir, "selected_genes.tsv")
    if not os.path.exists(selected_genes_path):
        raise FileNotFoundError(f"selected_genes.tsv not found: {selected_genes_path}")

    selected_genes = (
        pd.read_csv(selected_genes_path, sep="\t", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )

    ensemble_paths_file = os.path.join(trained_out_dir, "ensemble_model_paths.txt")

    if os.path.exists(ensemble_paths_file):
        model_paths = (
            pd.read_csv(ensemble_paths_file, header=None)
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )
    else:
        model_paths = sorted(
            glob.glob(os.path.join(trained_out_dir, "result_*_*", "model_trained.pth"))
        )

    resolved_model_paths = []
    for p in model_paths:
        if os.path.exists(p):
            resolved_model_paths.append(p)
        else:
            p2 = os.path.join(trained_out_dir, p)
            if os.path.exists(p2):
                resolved_model_paths.append(p2)

    model_paths = resolved_model_paths

    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No trained model found under {trained_out_dir}. "
            "Expected ensemble_model_paths.txt or result_*_*/model_trained.pth"
        )

    first_result_dir = os.path.dirname(model_paths[0])
    meta_path = os.path.join(first_result_dir, "model_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        print(f"[WARN] model_meta.json not found: {meta_path}. Use fallback defaults.")
        meta = {}

    n_inputs = int(meta.get("n_inputs", 768))
    hidden = int(meta.get("hidden", 768))
    n_outputs = int(meta.get("n_outputs", len(selected_genes)))
    dropout = float(meta.get("dropout", 0.2))
    output_relu = bool(meta.get("output_relu", True))

    if n_outputs != len(selected_genes):
        print(
            f"[WARN] n_outputs from meta={n_outputs}, "
            f"but len(selected_genes)={len(selected_genes)}. "
            "Use len(selected_genes)."
        )
        n_outputs = len(selected_genes)

    print("[Loaded trained artifacts]")
    print("  trained_out_dir:", trained_out_dir)
    print("  selected_genes:", len(selected_genes))
    print("  n_models:", len(model_paths))
    print("  n_inputs:", n_inputs)
    print("  hidden:", hidden)
    print("  n_outputs:", n_outputs)
    print("  dropout:", dropout)
    print("  output_relu:", output_relu)

    for p in model_paths[:5]:
        print("  model:", p)
    if len(model_paths) > 5:
        print(f"  ... {len(model_paths) - 5} more models")

    return selected_genes, model_paths, n_inputs, hidden, n_outputs, dropout, output_relu


# ============================================================
# Tile grouping
# ============================================================

def build_tile_table_from_coords(
    coords,
    tile_size=224,
    tile_stride=224,
    coord_scale=1.0,
    min_spots_per_tile=1,
):
    """
    Convert spot coordinates into tile-level groups.
    In this fixed script, coords have usually already been scaled/transformed.
    Therefore main() calls this with coord_scale=1.0.
    """
    coords_scaled = coords.astype(float) * float(coord_scale)
    x = coords_scaled[:, 0]
    y = coords_scaled[:, 1]

    tile_x = np.floor(x / float(tile_stride)).astype(np.int64)
    tile_y = np.floor(y / float(tile_stride)).astype(np.int64)

    df = pd.DataFrame({
        "spot_index": np.arange(coords.shape[0], dtype=np.int64),
        "tile_x": tile_x,
        "tile_y": tile_y,
    })

    rows = []
    for (tx, ty), sub in df.groupby(["tile_x", "tile_y"], sort=True):
        spot_indices = sub["spot_index"].values.astype(np.int64)

        if len(spot_indices) < min_spots_per_tile:
            continue

        center_x = tx * tile_stride + tile_size / 2.0
        center_y = ty * tile_stride + tile_size / 2.0

        tile_id = f"{tx}x{ty}"

        rows.append({
            "tile_id": tile_id,
            "tile_x": int(tx),
            "tile_y": int(ty),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "n_spots": int(len(spot_indices)),
            "spot_indices": spot_indices,
        })

    tile_df = pd.DataFrame(rows)

    if tile_df.shape[0] == 0:
        raise ValueError(
            "No tile left after grouping. "
            "Try smaller --min_spots_per_tile or check coordinates/tile size."
        )

    return tile_df


# ============================================================
# Patch extraction and direct CTransPath feature extraction
# ============================================================

def extract_patch(image, x, y, window):
    """
    x/y are PIL coordinates:
      x = horizontal coordinate
      y = vertical coordinate
    """
    x = int(round(x))
    y = int(round(y))
    half = window // 2

    left = x - half
    upper = y - half
    right = left + window
    lower = upper + window

    W, H = image.size

    crop_left = max(left, 0)
    crop_upper = max(upper, 0)
    crop_right = min(right, W)
    crop_lower = min(lower, H)

    patch = Image.new("RGB", (window, window), (255, 255, 255))

    if crop_right > crop_left and crop_lower > crop_upper:
        crop = image.crop((crop_left, crop_upper, crop_right, crop_lower))
        paste_x = crop_left - left
        paste_y = crop_upper - upper
        patch.paste(crop, (paste_x, paste_y))

    return patch


def normalize_tile_macenko(patch, color_normalizer):
    if color_normalizer is None:
        return patch.convert("RGB")

    arr = np.asarray(patch.convert("RGB"))

    try:
        arr_norm = color_normalizer.transform(arr)
        arr_norm = np.asarray(arr_norm)
        arr_norm = np.clip(arr_norm, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_norm).convert("RGB")
    except Exception as e:
        print(f"[WARN] Macenko failed for one tile, use raw tile. Error: {repr(e)}")
        return patch.convert("RGB")


@torch.no_grad()
def ctranspath_forward_tiles(
    tiles,
    model,
    transform,
    device,
):
    batch = torch.stack(
        [transform(tile.convert("RGB")) for tile in tiles],
        dim=0,
    ).to(device, non_blocking=True)

    feat = model(batch)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    if hasattr(feat, "last_hidden_state"):
        feat = feat.last_hidden_state[:, 0]

    if feat.dim() > 2:
        feat = torch.flatten(feat, start_dim=1)

    return feat.detach().cpu().float()


def extract_ctrans_features_for_tile_table(
    sample,
    tile_df,
    ctrans_model,
    ctrans_transform,
    color_normalizer,
    cache_dir,
    tile_size=224,
    batch_size=128,
    force_recompute=False,
    use_macenko=True,
    device="cuda",
    split_name="test",
):
    """
    Extract one CTransPath feature per tile.
    """
    os.makedirs(cache_dir, exist_ok=True)

    sample_name = sample["sample"]
    macenko_tag = "macenko" if use_macenko else "raw"

    cache_path = os.path.join(
        cache_dir,
        f"{sample_name}.{split_name}.tilelevel.direct_ctranspath."
        f"{macenko_tag}.tile{tile_size}.features.pt"
    )

    if os.path.exists(cache_path) and not force_recompute:
        print(f"[Load tile feature cache] {cache_path}")
        obj = torch.load(cache_path, map_location="cpu")
        return obj["features"].float()

    image = sample["image"]
    centers = tile_df[["center_x", "center_y"]].values.astype(float)

    print(
        f"[Extract tile-level CTransPath features] "
        f"sample={sample_name}, split={split_name}, n_tiles={len(centers)}, "
        f"tile_size={tile_size}, macenko={use_macenko}, device={device}"
    )

    all_features = []

    for start in tqdm(
        range(0, len(centers), batch_size),
        desc=f"Tile -> CTransPath {sample_name} {split_name}",
    ):
        end = min(start + batch_size, len(centers))

        tiles = []
        for i in range(start, end):
            x, y = centers[i]
            patch = extract_patch(image, x, y, tile_size)

            if tile_size != 224:
                patch = patch.resize((224, 224))

            if use_macenko:
                patch = normalize_tile_macenko(patch, color_normalizer)

            tiles.append(patch)

        feats = ctranspath_forward_tiles(
            tiles=tiles,
            model=ctrans_model,
            transform=ctrans_transform,
            device=device,
        )

        all_features.append(feats)

    features = torch.cat(all_features, dim=0).contiguous()

    if features.shape[0] != len(centers):
        raise RuntimeError(
            f"Feature rows {features.shape[0]} != n_tiles {len(centers)} for {sample_name}"
        )

    torch.save(
        {
            "features": features,
            "tile_meta": tile_df.drop(columns=["spot_indices"], errors="ignore").copy(),
            "sample": sample_name,
            "tile_size": tile_size,
            "use_macenko": use_macenko,
            "feature_extractor": "direct_CTransPath_tilelevel",
            "coord_transform_info": sample.get("coord_transform_info", {}),
        },
        cache_path,
    )

    print(f"[Saved tile feature cache] {cache_path}")
    print(f"[Tile feature shape] {sample_name}: {tuple(features.shape)}")

    return features


# ============================================================
# MLP model
# ============================================================

class TileFeatureDataset(Dataset):
    def __init__(self, features, y=None):
        self.features = features.float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        x = self.features[idx].unsqueeze(0)
        if self.y is None:
            return x
        return x, self.y[idx]


class MLPRegressionReluTwo(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_hiddens,
        n_outputs,
        dropout=0.2,
        bias_init=None,
        output_relu=True,
    ):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer1_linear = nn.Linear(n_hiddens, n_outputs)
        self.layer1_relu = nn.ReLU() if output_relu else nn.Identity()

        if bias_init is not None:
            with torch.no_grad():
                self.layer1_linear.bias.copy_(torch.as_tensor(bias_init).float())

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.layer0(x)
        x = self.layer1_linear(x)
        x = self.layer1_relu(x)
        x = torch.mean(x, dim=1)
        return x


@torch.no_grad()
def predict_with_model(
    model,
    features,
    batch_size=128,
    num_workers=4,
    device="cuda",
):
    ds = TileFeatureDataset(features, y=None)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    preds = []
    model.eval()

    for xb in tqdm(loader, desc="Predict one model"):
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(preds, axis=0)


def load_fold_model(
    model_path,
    n_inputs,
    hidden,
    n_outputs,
    dropout,
    output_relu,
    device,
):
    model = MLPRegressionReluTwo(
        n_inputs=n_inputs,
        n_hiddens=hidden,
        n_outputs=n_outputs,
        dropout=dropout,
        bias_init=None,
        output_relu=output_relu,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def ensemble_predict_from_folds(
    model_paths,
    test_features,
    n_inputs,
    hidden,
    n_outputs,
    dropout,
    output_relu,
    batch_size=128,
    num_workers=4,
    device="cuda",
    save_each_model=False,
):
    sum_pred = None
    each_pred = []

    for m, model_path in enumerate(model_paths):
        print(f"[Ensemble predict] model {m + 1}/{len(model_paths)}: {model_path}")

        model = load_fold_model(
            model_path=model_path,
            n_inputs=n_inputs,
            hidden=hidden,
            n_outputs=n_outputs,
            dropout=dropout,
            output_relu=output_relu,
            device=device,
        )

        pred = predict_with_model(
            model=model,
            features=test_features,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        if sum_pred is None:
            sum_pred = pred.astype(np.float64)
        else:
            sum_pred += pred.astype(np.float64)

        if save_each_model:
            each_pred.append(pred.astype(np.float32))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_mean = (sum_pred / len(model_paths)).astype(np.float32)

    if save_each_model:
        pred_stack = np.stack(each_pred, axis=0).astype(np.float32)
    else:
        pred_stack = None

    return pred_mean, pred_stack


# ============================================================
# Save outputs
# ============================================================

def save_prediction_h5(
    out_h5,
    pred_mean,
    pred_stack,
    genes,
    tile_meta,
    sample_name,
    model_paths,
    coord_transform_info=None,
):
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    tile_meta = tile_meta.copy()

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "pred_lognorm",
            data=pred_mean.astype(np.float32),
            compression="gzip",
            chunks=(min(1024, pred_mean.shape[0]), min(256, pred_mean.shape[1])),
        )

        if pred_stack is not None:
            f.create_dataset(
                "pred_lognorm_each_model",
                data=pred_stack.astype(np.float32),
                compression="gzip",
                chunks=(1, min(1024, pred_stack.shape[1]), min(256, pred_stack.shape[2])),
            )

        f.create_dataset("genes", data=np.asarray(genes, dtype="S"))
        f.create_dataset("tile_id", data=np.asarray(tile_meta["tile_id"].astype(str).values, dtype="S"))
        f.create_dataset("tile_x", data=tile_meta["tile_x"].values.astype(np.int64))
        f.create_dataset("tile_y", data=tile_meta["tile_y"].values.astype(np.int64))
        f.create_dataset("coords", data=tile_meta[["center_x", "center_y"]].values.astype(np.float32))
        f.create_dataset("n_spots", data=tile_meta["n_spots"].values.astype(np.int64))
        f.create_dataset("model_paths", data=np.asarray(model_paths, dtype="S"))

        f.attrs["sample"] = sample_name
        f.attrs["prediction_level"] = "tile"
        f.attrs["prediction_space"] = "log1p(CPM_1e4)"
        f.attrs["feature"] = "direct_CTransPath_tilelevel"
        f.attrs["ensemble"] = "mean over all result_{ik}_{il} models"

        if coord_transform_info is not None:
            f.attrs["coord_transform_tag"] = str(coord_transform_info.get("tag", "NA"))
            f.attrs["coord_scale"] = float(coord_transform_info.get("coord_scale", 1.0))
            f.attrs["coord_swap_xy"] = int(bool(coord_transform_info.get("swap_xy", False)))
            f.attrs["coord_flip_x"] = int(bool(coord_transform_info.get("flip_x", False)))
            f.attrs["coord_flip_y"] = int(bool(coord_transform_info.get("flip_y", False)))
            f.attrs["coord_flip_basis"] = str(coord_transform_info.get("flip_basis", "NA"))
            f.attrs["coord_image_width"] = int(coord_transform_info.get("image_width", -1))
            f.attrs["coord_image_height"] = int(coord_transform_info.get("image_height", -1))

    print(f"[Saved h5] {out_h5}")


def save_prediction_csv(out_csv, pred_mean, genes, tile_meta, sample_name, coord_transform_info=None):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = pd.DataFrame(pred_mean, columns=genes)
    meta = tile_meta.drop(columns=["spot_indices"], errors="ignore").copy()
    meta.insert(0, "sample", sample_name)

    if coord_transform_info is not None:
        meta.insert(1, "coord_transform_tag", str(coord_transform_info.get("tag", "NA")))

    out = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    out.to_csv(out_csv, index=False)

    print(f"[Saved csv] {out_csv}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--trained_out_dir", required=True)

    parser.add_argument(
        "--path2space_feature_dir",
        required=True,
        help="Used to import func/ctrans_model.py and func/utils_color_norm.py",
    )

    parser.add_argument(
        "--ctranspath_weight",
        default="/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/ctranspath.pth",
        help="Path to ctranspath.pth",
    )

    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--tile_stride", type=int, default=224)
    parser.add_argument("--min_spots_per_tile", type=int, default=1)

    parser.add_argument("--test_coord_scale", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--no_macenko", action="store_true")
    parser.add_argument("--max_test_tiles", type=int, default=0)

    parser.add_argument("--force_recompute_features", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--save_each_model_pred", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--swap_xy",
        action="store_true",
        help="Swap locs-raw.tsv x/y before tile grouping and patch extraction.",
    )

    parser.add_argument(
        "--flip_x",
        action="store_true",
        help="Mirror x coordinate before tile grouping and patch extraction. Use this for left-right reversed tissue mask.",
    )

    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="Mirror y coordinate before tile grouping and patch extraction.",
    )

    parser.add_argument(
        "--flip_basis",
        type=str,
        default="image",
        choices=["image", "coords"],
        help=(
            "Basis for flip_x/flip_y. "
            "'image': flip by image width/height; recommended for image pixel coordinates. "
            "'coords': flip by coordinate min/max."
        ),
    )

    parser.add_argument(
        "--plot_input_debug",
        action="store_true",
        help="Plot raw/transformed spots, tile centers, and actual input patches before prediction.",
    )

    parser.add_argument(
        "--input_debug_max_spots",
        type=int,
        default=50000,
        help="Max spots to display in input debug coordinate plots.",
    )

    parser.add_argument(
        "--input_debug_max_tiles",
        type=int,
        default=20000,
        help="Max tile centers to display in input debug plot.",
    )

    parser.add_argument(
        "--input_debug_n_patches",
        type=int,
        default=36,
        help="Number of actual input patches to show in montage.",
    )

    parser.add_argument(
        "--input_debug_dpi",
        type=int,
        default=200,
        help="DPI for input debug plots.",
    )

    parser.add_argument(
        "--debug_only",
        action="store_true",
        help="Only generate input debug plots and metadata, then skip feature extraction/prediction.",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    coord_transform_tag = build_coord_transform_tag(args)

    with open(os.path.join(args.out_dir, "predict_only_run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    with open(os.path.join(args.out_dir, "coordinate_transform_used.json"), "w") as f:
        json.dump(
            {
                "coord_transform_tag": coord_transform_tag,
                "test_coord_scale": args.test_coord_scale,
                "swap_xy": args.swap_xy,
                "flip_x": args.flip_x,
                "flip_y": args.flip_y,
                "flip_basis": args.flip_basis,
                "note": "Coordinates are transformed before tile grouping, patch extraction, and h5/csv saving.",
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Load trained genes/models/meta
    # ------------------------------------------------------------
    selected_genes, model_paths, n_inputs, hidden, n_outputs, dropout, output_relu = load_trained_artifacts(
        args.trained_out_dir
    )

    pd.Series(selected_genes).to_csv(
        os.path.join(args.out_dir, "selected_genes.tsv"),
        sep="\t",
        index=False,
        header=False,
    )

    pd.Series(model_paths).to_csv(
        os.path.join(args.out_dir, "ensemble_model_paths.txt"),
        index=False,
        header=False,
    )

    # ------------------------------------------------------------
    # Load CTransPath unless debug_only without feature extraction
    # ------------------------------------------------------------
    use_macenko = not args.no_macenko

    ctrans_model = None
    ctrans_transform = None
    color_normalizer = None

    if not args.debug_only:
        CTransPath, color_normalizer = import_ctranspath_and_macenko(
            args.path2space_feature_dir,
            use_macenko=use_macenko,
        )

        ctrans_model = load_ctranspath_model(
            CTransPath=CTransPath,
            weight_path=args.ctranspath_weight,
            device=device,
        )

        ctrans_transform = get_ctranspath_transform()

    # ------------------------------------------------------------
    # Load test samples
    # ------------------------------------------------------------
    test_folders = discover_spatial_folders(args.test_dir)

    print("[Test folders]")
    for x in test_folders:
        print(" ", x)

    test_samples = [load_spatial_folder(x) for x in test_folders]

    # ------------------------------------------------------------
    # Predict test tiles
    # ------------------------------------------------------------
    print("[Predict-only external test samples at tile level]")

    for sample_raw in test_samples:
        print("=" * 80)
        print(f"[Test sample] {sample_raw['sample']}")
        print("=" * 80)

        # Important:
        # Apply scale/swap/flip before tile grouping, patch extraction, and h5 saving.
        sample = apply_coordinate_transform_to_sample(sample_raw, args)

        # coords already scaled/transformed, so coord_scale=1.0 here.
        test_tile_df = build_tile_table_from_coords(
            coords=sample["coords"],
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            coord_scale=1.0,
            min_spots_per_tile=args.min_spots_per_tile,
        )

        if args.max_test_tiles and args.max_test_tiles > 0:
            keep = np.arange(min(args.max_test_tiles, test_tile_df.shape[0]))
            test_tile_df = test_tile_df.iloc[keep].reset_index(drop=True)
            print(f"[Subsample test tiles] {test_tile_df.shape[0]}")

        test_tile_meta_save = test_tile_df.drop(columns=["spot_indices"], errors="ignore").copy()
        test_tile_meta_save["coord_transform_tag"] = sample["coord_transform_tag"]

        test_tile_meta_save.to_csv(
            os.path.join(args.out_dir, f"{sample['sample']}_test_tile_metadata.csv"),
            index=False,
        )

        print(f"n_test_tiles: {test_tile_df.shape[0]}")

        # ------------------------------------------------------------
        # Input debug plots: verify actual prediction input
        # ------------------------------------------------------------
        if args.plot_input_debug or args.debug_only:
            plot_input_debug_all(
                sample_raw=sample_raw,
                sample_transformed=sample,
                tile_df=test_tile_df,
                out_dir=args.out_dir,
                tile_size=args.tile_size,
                max_spots=args.input_debug_max_spots,
                max_tiles=args.input_debug_max_tiles,
                n_patches=args.input_debug_n_patches,
                dpi=args.input_debug_dpi,
            )

        if args.debug_only:
            print("[debug_only] Skip feature extraction and prediction.")
            continue

        # Include transform tag in cache split name so old non-flipped feature cache is not reused.
        split_name = f"test_{sample['coord_transform_tag']}"

        test_features = extract_ctrans_features_for_tile_table(
            sample=sample,
            tile_df=test_tile_df,
            ctrans_model=ctrans_model,
            ctrans_transform=ctrans_transform,
            color_normalizer=color_normalizer,
            cache_dir=cache_dir,
            tile_size=args.tile_size,
            batch_size=args.feature_batch_size,
            force_recompute=args.force_recompute_features,
            use_macenko=use_macenko,
            device=device,
            split_name=split_name,
        )

        pred_mean, pred_stack = ensemble_predict_from_folds(
            model_paths=model_paths,
            test_features=test_features,
            n_inputs=n_inputs,
            hidden=hidden,
            n_outputs=n_outputs,
            dropout=dropout,
            output_relu=output_relu,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            save_each_model=args.save_each_model_pred,
        )

        sample_out_dir = os.path.join(args.out_dir, sample["sample"])
        os.makedirs(sample_out_dir, exist_ok=True)

        out_h5 = os.path.join(
            sample_out_dir,
            "path2space_tilelevel_predicted_expression_ensemble.h5",
        )

        save_prediction_h5(
            out_h5=out_h5,
            pred_mean=pred_mean,
            pred_stack=pred_stack,
            genes=selected_genes,
            tile_meta=test_tile_meta_save,
            sample_name=sample["sample"],
            model_paths=model_paths,
            coord_transform_info=sample.get("coord_transform_info", None),
        )

        if args.save_csv:
            out_csv = os.path.join(
                sample_out_dir,
                "path2space_tilelevel_predicted_expression_ensemble.csv",
            )
            save_prediction_csv(
                out_csv=out_csv,
                pred_mean=pred_mean,
                genes=selected_genes,
                tile_meta=test_tile_meta_save,
                sample_name=sample["sample"],
                coord_transform_info=sample.get("coord_transform_info", None),
            )

    print("Done predict_only.")


if __name__ == "__main__":
    main()
