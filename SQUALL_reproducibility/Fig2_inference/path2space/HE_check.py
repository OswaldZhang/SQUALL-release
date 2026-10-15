#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ======================================================
# Input folders
# ======================================================

SAMPLE_DIRS = {
    "OV_Xenium_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium_all_new",
    "HCC_Xenium_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new",
    "OC_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OC_all_new",
    "CC_all_new": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/CC_all_new",
}

OUT_DIR = "check_xenium_locs_on_HE_4samples"
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================
# Config
# ======================================================

N_SHOW = 50000
POINT_SIZE = 0.25
ALPHA = 0.35
RANDOM_SEED = 0

# HE 
DOWNSAMPLE = 8

#  scale=1
#  [0.8, 0.9, 1.0, 1.1, 1.2]
SCALE_LIST = [1.0]


# ======================================================
# File discovery
# ======================================================

def find_first_existing(folder, names):
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def discover_files(sample_dir):
    """
    Expected Path2Space / iSTAR style:
      he-raw.jpg
      locs-raw.tsv

    Also tries a few alternative image names.
    """
    he_path = find_first_existing(
        sample_dir,
        [
            "he-raw.jpg",
            "he-raw.png",
            "HE.jpg",
            "HE.png",
            "tissue_hires_image.png",
            "image.jpg",
            "image.png",
        ],
    )

    locs_path = find_first_existing(
        sample_dir,
        [
            "locs-raw.tsv",
            "locs.tsv",
            "locations.tsv",
            "spatial_locs.tsv",
        ],
    )

    if he_path is None:
        raise FileNotFoundError(f"No HE image found in {sample_dir}")

    if locs_path is None:
        raise FileNotFoundError(f"No locs file found in {sample_dir}")

    return he_path, locs_path


# ======================================================
# Load locs
# ======================================================

def load_locs(locs_path):
    if locs_path.endswith(".csv"):
        df = pd.read_csv(locs_path)
    else:
        df = pd.read_csv(locs_path, sep="\t")

    print(f"[locs columns] {locs_path}")
    print(df.columns.tolist())

    #  x/y
    if "x" in df.columns and "y" in df.columns:
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        return x, y, df, "x", "y"

    # 
    candidates = [
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
        ("col", "row"),
        ("X", "Y"),
    ]

    for xc, yc in candidates:
        if xc in df.columns and yc in df.columns:
            x = df[xc].values.astype(float)
            y = df[yc].values.astype(float)
            return x, y, df, xc, yc

    raise ValueError(
        f"Cannot find x/y columns in {locs_path}. "
        f"Existing columns: {df.columns.tolist()}"
    )


# ======================================================
# Transform
# ======================================================

def transform_xy(
    x,
    y,
    W,
    H,
    scale=1.0,
    swap=False,
    flipx=False,
    flipy=False,
    x_offset=0.0,
    y_offset=0.0,
):
    x2 = np.asarray(x, dtype=float).copy()
    y2 = np.asarray(y, dtype=float).copy()

    if swap:
        x2, y2 = y2.copy(), x2.copy()

    x2 = x2 * float(scale)
    y2 = y2 * float(scale)

    if flipx:
        x2 = W - x2

    if flipy:
        y2 = H - y2

    x2 = x2 + float(x_offset)
    y2 = y2 + float(y_offset)

    return x2, y2


def summarize_coords(name, x, y, W, H):
    print("\n" + "=" * 80)
    print(f"[{name}] coordinate summary")
    print("=" * 80)
    print(f"image W,H     : {W}, {H}")
    print(f"n spots       : {len(x)}")
    print(f"x min/max     : {np.nanmin(x):.3f}, {np.nanmax(x):.3f}")
    print(f"y min/max     : {np.nanmin(y):.3f}, {np.nanmax(y):.3f}")
    print(f"x/W           : {np.nanmax(x) / W:.4f}")
    print(f"y/H           : {np.nanmax(y) / H:.4f}")

    #  x/y 
    score_normal = abs(np.nanmax(x) / W - 1) + abs(np.nanmax(y) / H - 1)
    score_swap = abs(np.nanmax(x) / H - 1) + abs(np.nanmax(y) / W - 1)

    print(f"score_normal x->W,y->H : {score_normal:.4f}")
    print(f"score_swap   x->H,y->W : {score_swap:.4f}")

    if score_normal < score_swap:
        print(">>> likely normal x/y")
    else:
        print(">>> likely swapped x/y")


# ======================================================
# Plotting
# ======================================================

def sample_points(x, y, n_show=N_SHOW, seed=RANDOM_SEED):
    n = len(x)
    if n <= n_show:
        return x, y

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n_show, replace=False)
    return x[idx], y[idx]


def plot_overlay(
    ax,
    img_small,
    x,
    y,
    W,
    H,
    title,
    color="cyan",
    downsample=DOWNSAMPLE,
):
    xd = x / downsample
    yd = y / downsample

    xs, ys = sample_points(xd, yd)

    ax.imshow(img_small)
    ax.scatter(
        xs,
        ys,
        s=POINT_SIZE,
        c=color,
        alpha=ALPHA,
        linewidths=0,
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, W / downsample)
    ax.set_ylim(H / downsample, 0)
    ax.axis("off")


def save_raw_overlay(sample_name, img_small, x, y, W, H, sample_out_dir):
    fig, ax = plt.subplots(figsize=(10, 10))

    plot_overlay(
        ax,
        img_small,
        x,
        y,
        W,
        H,
        title=f"{sample_name} raw locs on HE\nn_spots={len(x)}, shown={min(len(x), N_SHOW)}",
        color="red",
    )

    out_png = os.path.join(sample_out_dir, f"{sample_name}_raw_locs_on_HE.png")
    out_pdf = os.path.join(sample_out_dir, f"{sample_name}_raw_locs_on_HE.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


def save_transform_grid(sample_name, img_small, x, y, W, H, sample_out_dir, scale=1.0):
    fig, axs = plt.subplots(2, 4, figsize=(22, 11))
    axs = axs.ravel()

    k = 0
    for swap in [False, True]:
        for flipx in [False, True]:
            for flipy in [False, True]:
                xt, yt = transform_xy(
                    x,
                    y,
                    W=W,
                    H=H,
                    scale=scale,
                    swap=swap,
                    flipx=flipx,
                    flipy=flipy,
                )

                title = (
                    f"scale={scale}, "
                    f"swap={int(swap)}, "
                    f"flipx={int(flipx)}, "
                    f"flipy={int(flipy)}"
                )

                plot_overlay(
                    axs[k],
                    img_small,
                    xt,
                    yt,
                    W,
                    H,
                    title=title,
                    color="cyan",
                )
                k += 1

    plt.suptitle(
        f"{sample_name}: 8 transform candidates\n"
        f"Choose the one where spots follow tissue morphology best",
        fontsize=14,
    )

    plt.tight_layout()

    scale_tag = str(scale).replace(".", "p")
    out_png = os.path.join(sample_out_dir, f"{sample_name}_transform_grid_scale{scale_tag}.png")
    out_pdf = os.path.join(sample_out_dir, f"{sample_name}_transform_grid_scale{scale_tag}.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_png}")
    print(f"[Saved] {out_pdf}")


# ======================================================
# Optional offset grid after choosing transform
# ======================================================

def save_offset_grid(
    sample_name,
    img_small,
    x,
    y,
    W,
    H,
    sample_out_dir,
    scale=1.0,
    swap=False,
    flipx=False,
    flipy=False,
    x_offsets=(-1000, -500, 0, 500, 1000),
    y_offsets=(-1000, -500, 0, 500, 1000),
):
    fig, axs = plt.subplots(len(y_offsets), len(x_offsets), figsize=(20, 20))

    for iy, yoff in enumerate(y_offsets):
        for ix, xoff in enumerate(x_offsets):
            ax = axs[iy, ix]

            xt, yt = transform_xy(
                x,
                y,
                W=W,
                H=H,
                scale=scale,
                swap=swap,
                flipx=flipx,
                flipy=flipy,
                x_offset=xoff,
                y_offset=yoff,
            )

            title = f"xoff={xoff}, yoff={yoff}"

            plot_overlay(
                ax,
                img_small,
                xt,
                yt,
                W,
                H,
                title=title,
                color="yellow",
            )

    plt.suptitle(
        f"{sample_name}: offset grid\n"
        f"scale={scale}, swap={int(swap)}, flipx={int(flipx)}, flipy={int(flipy)}",
        fontsize=14,
    )

    plt.tight_layout()

    out_png = os.path.join(
        sample_out_dir,
        f"{sample_name}_offset_grid_scale{scale}_swap{int(swap)}_flipx{int(flipx)}_flipy{int(flipy)}.png",
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_png}")


# ======================================================
# Main
# ======================================================

def process_one_sample(sample_name, sample_dir):
    print("\n" + "#" * 100)
    print(f"[Sample] {sample_name}")
    print(f"[Dir] {sample_dir}")
    print("#" * 100)

    sample_out_dir = os.path.join(OUT_DIR, sample_name)
    os.makedirs(sample_out_dir, exist_ok=True)

    he_path, locs_path = discover_files(sample_dir)

    print(f"HE   : {he_path}")
    print(f"locs : {locs_path}")

    img = Image.open(he_path).convert("RGB")
    W, H = img.size

    img_small = img.resize((max(1, W // DOWNSAMPLE), max(1, H // DOWNSAMPLE)))
    img_small = np.asarray(img_small)

    x, y, df, x_col, y_col = load_locs(locs_path)

    print(f"Using columns: x={x_col}, y={y_col}")

    summarize_coords(sample_name, x, y, W, H)

    save_raw_overlay(
        sample_name=sample_name,
        img_small=img_small,
        x=x,
        y=y,
        W=W,
        H=H,
        sample_out_dir=sample_out_dir,
    )

    for scale in SCALE_LIST:
        save_transform_grid(
            sample_name=sample_name,
            img_small=img_small,
            x=x,
            y=y,
            W=W,
            H=H,
            sample_out_dir=sample_out_dir,
            scale=scale,
        )

    #  summary
    summary = {
        "sample": sample_name,
        "sample_dir": sample_dir,
        "he_path": he_path,
        "locs_path": locs_path,
        "x_col": x_col,
        "y_col": y_col,
        "image_W": W,
        "image_H": H,
        "n_spots": len(x),
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "y_min": float(np.nanmin(y)),
        "y_max": float(np.nanmax(y)),
        "x_max_over_W": float(np.nanmax(x) / W),
        "y_max_over_H": float(np.nanmax(y) / H),
        "x_max_over_H": float(np.nanmax(x) / H),
        "y_max_over_W": float(np.nanmax(y) / W),
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(sample_out_dir, f"{sample_name}_coord_summary.csv"),
        index=False,
    )


def main():
    for sample_name, sample_dir in SAMPLE_DIRS.items():
        try:
            process_one_sample(sample_name, sample_dir)
        except Exception as e:
            print(f"[ERROR] {sample_name}: {repr(e)}")


if __name__ == "__main__":
    main()
