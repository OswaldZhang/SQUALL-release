#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.io import mmwrite
from PIL import Image
import tifffile
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


# ======================================================
# Config
# ======================================================

SAMPLES = {
    "OC_all_new_aligned": {
        "adata": "/lustre1/zxzeng/bwqin/STORM/Xenium/OC_Xenium_public/adata.h5ad",
        "public_dir": "/lustre1/zxzeng/bwqin/STORM/Xenium/OC_Xenium_public",
    },
    "CC_all_new_aligned": {
        "adata": "/lustre1/zxzeng/bwqin/STORM/Xenium/CC_Xenium_public/adata.h5ad",
        "public_dir": "/lustre1/zxzeng/bwqin/STORM/Xenium/CC_Xenium_public",
    },
}

OUT_ROOT = "/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/aligned_from_adata_for_path2space"

TARGET_MPP = 0.5
TILE_SIZE = 224
TILE_STRIDE = 224

# 画 QC 图用
DOWNSAMPLE_FOR_PLOT = 6
N_SHOW_POINTS = 200000
RANDOM_SEED = 0


# ======================================================
# Helpers
# ======================================================

def find_he_ome_tif(public_dir):
    patterns = [
        os.path.join(public_dir, "*_he_image.ome.tif"),
        os.path.join(public_dir, "*.ome.tif"),
        os.path.join(public_dir, "*.tif"),
        os.path.join(public_dir, "*.tiff"),
    ]

    hits = []
    for pat in patterns:
        hits.extend(glob.glob(pat))

    hits = sorted(set(hits))

    if len(hits) == 0:
        raise FileNotFoundError(f"No HE OME-TIFF found under {public_dir}")

    # 优先 he_image
    he_hits = [x for x in hits if "he_image" in os.path.basename(x)]
    if len(he_hits) > 0:
        return he_hits[0]

    return hits[0]


def read_he_as_rgb(he_path):
    arr = tifffile.imread(he_path)

    # 常见 OME-TIFF 有时候是 C,H,W
    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [3, 4]:
        arr = np.moveaxis(arr, 0, -1)

    # 如果多层，只取第一层
    if arr.ndim > 3:
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [3, 4]:
            arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [0.5, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return arr


def get_original_mpp(adata):
    if "H&E resolution" not in adata.uns:
        raise KeyError("adata.uns['H&E resolution'] not found.")

    res = np.asarray(adata.uns["H&E resolution"]).astype(float)

    if res.size == 1:
        return float(res[0])

    # 一般 x/y 一样；这里取均值
    return float(np.mean(res[:2]))


def resize_he_to_target_mpp(he_np, original_mpp, target_mpp):
    """
    如果原图 original_mpp 较小，比如 0.2125，
    target_mpp=0.5 时，需要下采样：
        new_size = old_size / (target_mpp / original_mpp)
    """
    H, W = he_np.shape[:2]

    scale_factor = target_mpp / original_mpp

    new_W = int(round(W / scale_factor))
    new_H = int(round(H / scale_factor))

    print(f"[Resize HE]")
    print(f"  original_mpp : {original_mpp}")
    print(f"  target_mpp   : {target_mpp}")
    print(f"  scale_factor : {scale_factor:.6f}")
    print(f"  old W,H      : {W}, {H}")
    print(f"  new W,H      : {new_W}, {new_H}")

    he_resized = cv2.resize(
        he_np,
        (new_W, new_H),
        interpolation=cv2.INTER_AREA,
    )

    return he_resized, new_W, new_H, scale_factor


def get_aligned_pixel_coords(adata, target_mpp):
    """
    Your previous construction:
        adata.obsm["aligned_spatial"] = aligned_pixel_coords * resolution

    So aligned_spatial is in µm.
    Convert to pixel coordinate on target_mpp image:
        pixel = aligned_spatial_um / target_mpp
    """
    if "aligned_spatial" not in adata.obsm:
        raise KeyError("adata.obsm['aligned_spatial'] not found.")

    aligned_um = np.asarray(adata.obsm["aligned_spatial"]).astype(float)

    coords_px = aligned_um[:, :2] / float(target_mpp)

    return coords_px


def make_counts_matrix(adata):
    X = adata.X

    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    else:
        X = X.tocsr()

    return X


def save_path2space_inputs(adata, he_resized, coords_px, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------
    # HE image
    # ------------------------------
    he_path = os.path.join(out_dir, "he-raw.jpg")
    Image.fromarray(he_resized).save(he_path, quality=95)
    print(f"[Saved] {he_path}")

    # ------------------------------
    # locs-raw.tsv
    # ------------------------------
    locs = pd.DataFrame({
        "barcode": adata.obs_names.astype(str),
        "x": coords_px[:, 0],
        "y": coords_px[:, 1],
    })

    locs_path = os.path.join(out_dir, "locs-raw.tsv")
    locs.to_csv(locs_path, sep="\t", index=False)
    print(f"[Saved] {locs_path}")

    # ------------------------------
    # cnts.mtx
    # ------------------------------
    X = make_counts_matrix(adata)

    cnts_path = os.path.join(out_dir, "cnts.mtx")
    mmwrite(cnts_path, X)
    print(f"[Saved] {cnts_path} shape={X.shape}")

    # ------------------------------
    # genes.tsv
    # ------------------------------
    genes = pd.Series(adata.var_names.astype(str))
    genes_path = os.path.join(out_dir, "genes.tsv")
    genes.to_csv(genes_path, sep="\t", index=False, header=False)
    print(f"[Saved] {genes_path} n={len(genes)}")

    # ------------------------------
    # barcodes.tsv
    # ------------------------------
    barcodes = pd.Series(adata.obs_names.astype(str))
    barcodes_path = os.path.join(out_dir, "barcodes.tsv")
    barcodes.to_csv(barcodes_path, sep="\t", index=False, header=False)
    print(f"[Saved] {barcodes_path} n={len(barcodes)}")


def in_image_mask(coords_px, W, H):
    x = coords_px[:, 0]
    y = coords_px[:, 1]

    return (
        np.isfinite(x) &
        np.isfinite(y) &
        (x >= 0) &
        (x < W) &
        (y >= 0) &
        (y < H)
    )


def sample_indices(n, max_n=N_SHOW_POINTS, seed=RANDOM_SEED):
    if n <= max_n:
        return np.arange(n)

    rng = np.random.RandomState(seed)
    return rng.choice(n, size=max_n, replace=False)


def save_align_check_plot(he_resized, coords_px, out_dir, sample_name):
    H, W = he_resized.shape[:2]

    small_W = max(1, W // DOWNSAMPLE_FOR_PLOT)
    small_H = max(1, H // DOWNSAMPLE_FOR_PLOT)

    img_small = cv2.resize(
        he_resized,
        (small_W, small_H),
        interpolation=cv2.INTER_AREA,
    )

    ok = in_image_mask(coords_px, W, H)

    idx_all = np.where(ok)[0]
    idx_show = idx_all[sample_indices(len(idx_all))]

    x = coords_px[idx_show, 0] / DOWNSAMPLE_FOR_PLOT
    y = coords_px[idx_show, 1] / DOWNSAMPLE_FOR_PLOT

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_small)
    ax.scatter(x, y, s=0.6, c="red", alpha=0.55, linewidths=0)
    ax.set_xlim(0, small_W)
    ax.set_ylim(small_H, 0)
    ax.axis("off")
    ax.set_title(
        f"{sample_name}: aligned_spatial / {TARGET_MPP} on resized HE\n"
        f"in-image={ok.mean():.4f} ({ok.sum()}/{len(ok)}), shown={len(idx_show)}",
        fontsize=12,
    )

    out_png = os.path.join(out_dir, "align_check_points.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_png}")


def build_tile_coverage(coords_px, W, H, tile_size=TILE_SIZE, tile_stride=TILE_STRIDE):
    ok = in_image_mask(coords_px, W, H)
    coords_ok = coords_px[ok]

    tile_x = np.floor(coords_ok[:, 0] / tile_stride).astype(np.int64)
    tile_y = np.floor(coords_ok[:, 1] / tile_stride).astype(np.int64)

    df = pd.DataFrame({
        "tile_x": tile_x,
        "tile_y": tile_y,
    })

    cov = (
        df.groupby(["tile_x", "tile_y"], sort=True)
        .size()
        .reset_index(name="n_spots")
    )

    cov["x0"] = cov["tile_x"] * tile_stride
    cov["y0"] = cov["tile_y"] * tile_stride
    cov["center_x"] = cov["x0"] + tile_size / 2
    cov["center_y"] = cov["y0"] + tile_size / 2

    return cov


def save_tile_coverage_plot(he_resized, coords_px, out_dir, sample_name):
    H, W = he_resized.shape[:2]

    small_W = max(1, W // DOWNSAMPLE_FOR_PLOT)
    small_H = max(1, H // DOWNSAMPLE_FOR_PLOT)

    img_small = cv2.resize(
        he_resized,
        (small_W, small_H),
        interpolation=cv2.INTER_AREA,
    )

    cov = build_tile_coverage(coords_px, W, H)

    ok = in_image_mask(coords_px, W, H)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_small)

    for _, row in cov.iterrows():
        rect = Rectangle(
            (row["x0"] / DOWNSAMPLE_FOR_PLOT, row["y0"] / DOWNSAMPLE_FOR_PLOT),
            TILE_SIZE / DOWNSAMPLE_FOR_PLOT,
            TILE_SIZE / DOWNSAMPLE_FOR_PLOT,
            linewidth=0.25,
            edgecolor=(0, 0, 0, 0.60),
            facecolor=(1, 1, 0, 0.30),
        )
        ax.add_patch(rect)

    ax.set_xlim(0, small_W)
    ax.set_ylim(small_H, 0)
    ax.axis("off")
    ax.set_title(
        f"{sample_name}: tile coverage from aligned_spatial\n"
        f"tile={TILE_SIZE}, n_tiles={len(cov)}, in-image={ok.mean():.4f}",
        fontsize=12,
    )

    out_png = os.path.join(out_dir, "align_check_tile_coverage.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out_png}")

    cov.to_csv(os.path.join(out_dir, "tile_coverage.csv"), index=False)


def process_one(sample_name, adata_path, public_dir, out_root):
    print("\n" + "=" * 100)
    print(f"[Sample] {sample_name}")
    print("=" * 100)

    out_dir = os.path.join(out_root, sample_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Load adata] {adata_path}")
    adata = sc.read_h5ad(adata_path)

    print(f"  adata shape: {adata.shape}")
    print(f"  obsm keys  : {list(adata.obsm.keys())}")
    print(f"  uns keys   : {list(adata.uns.keys())}")

    he_ome_path = find_he_ome_tif(public_dir)
    print(f"[HE OME] {he_ome_path}")

    original_mpp = get_original_mpp(adata)

    he_np = read_he_as_rgb(he_ome_path)
    he_resized, new_W, new_H, scale_factor = resize_he_to_target_mpp(
        he_np,
        original_mpp=original_mpp,
        target_mpp=TARGET_MPP,
    )

    coords_px = get_aligned_pixel_coords(adata, target_mpp=TARGET_MPP)

    ok = in_image_mask(coords_px, new_W, new_H)

    print("[Aligned coords]")
    print(f"  coords_px shape : {coords_px.shape}")
    print(f"  x min/max       : {np.nanmin(coords_px[:, 0]):.3f}, {np.nanmax(coords_px[:, 0]):.3f}")
    print(f"  y min/max       : {np.nanmin(coords_px[:, 1]):.3f}, {np.nanmax(coords_px[:, 1]):.3f}")
    print(f"  image W,H       : {new_W}, {new_H}")
    print(f"  in-image        : {ok.sum()}/{len(ok)} ({ok.mean():.4f})")

    # Save Path2Space input files
    save_path2space_inputs(
        adata=adata,
        he_resized=he_resized,
        coords_px=coords_px,
        out_dir=out_dir,
    )

    # Save QC figures
    save_align_check_plot(
        he_resized=he_resized,
        coords_px=coords_px,
        out_dir=out_dir,
        sample_name=sample_name,
    )

    save_tile_coverage_plot(
        he_resized=he_resized,
        coords_px=coords_px,
        out_dir=out_dir,
        sample_name=sample_name,
    )

    # Save metadata
    meta = {
        "sample": sample_name,
        "adata_path": adata_path,
        "public_dir": public_dir,
        "he_ome_path": he_ome_path,
        "target_mpp": TARGET_MPP,
        "original_mpp": original_mpp,
        "resize_scale_factor": scale_factor,
        "image_W": new_W,
        "image_H": new_H,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_in_image": int(ok.sum()),
        "in_image_fraction": float(ok.mean()),
    }

    with open(os.path.join(out_dir, "prepare_aligned_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    metas = []

    for sample_name, cfg in SAMPLES.items():
        meta = process_one(
            sample_name=sample_name,
            adata_path=cfg["adata"],
            public_dir=cfg["public_dir"],
            out_root=OUT_ROOT,
        )
        metas.append(meta)

    summary = pd.DataFrame(metas)
    summary_path = os.path.join(OUT_ROOT, "prepare_aligned_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"OUT_ROOT: {OUT_ROOT}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
