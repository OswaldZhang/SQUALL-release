#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

TASKS = {
    "HCC": {
        "out_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_HCC_to_Xenium_tilelevel_original_setting",
        "genes": ["CDK3", "VEGFA"],
    },
    "OV": {
        "out_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_Xenium_tilelevel_original_setting",
        "genes": ["EPCAM", "AKT1", "CDK1"],
    },
    "OC": {
        "out_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_OC_tilelevel_predict_only_swapxy",
        "genes": ["IFNGR1", "STAT1"],
    },
    "CC": {
        "out_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_CC_tilelevel_predict_only_swapxy",
        "genes": ["CD8A", "MTOR"],
    },
}

OUT_PLOT_DIR = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_tilelevel_gene_plots_orientation_check"

POINT_SIZE = 8
ALPHA = 0.95
CMAP = "viridis"

VMIN_PERCENTILE = 1
VMAX_PERCENTILE = 99

SAVE_PDF = True
SAVE_PNG = True
DPI = 300

#  4-panel 
MAKE_ORIENTATION_CHECK = True

# 
MAKE_FINAL_PLOT = True

# 
# : "normal", "flip_x", "flip_y", "flip_xy"
#  task  task  "flip_x"
FINAL_ORIENTATION_BY_TASK = {
    "HCC": "normal",
    "OV": "normal",
    "OC": "normal",
    #"CC": "normal",
    #  CC 
    "OC": "flip_x",
    "CC": "flip_x",
}

#  y 
#  ax.invert_yaxis()
INVERT_Y_AXIS = True


# ============================================================
# IO helpers
# ============================================================

def decode_array(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def safe_gene_name(name):
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in str(name)])


def find_prediction_h5(out_dir):
    patterns = [
        os.path.join(out_dir, "path2space_tilelevel_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "*", "path2space_tilelevel_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "**", "path2space_tilelevel_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "path2space_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "*", "path2space_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "**", "path2space_predicted_expression_ensemble.h5"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    files = sorted(list(set(files)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"No prediction h5 found under {out_dir}. "
            "Expected path2space_tilelevel_predicted_expression_ensemble.h5 "
            "or path2space_predicted_expression_ensemble.h5"
        )

    return files


def inspect_h5(h5_path):
    print(f"\n[Inspect h5] {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(f"  dataset: {k}, shape={obj.shape}, dtype={obj.dtype}")
        for k, v in f.attrs.items():
            print(f"  attr: {k} = {v}")


def read_prediction_h5(h5_path):
    inspect_h5(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "pred_lognorm" in f:
            pred_key = "pred_lognorm"
        elif "pred" in f:
            pred_key = "pred"
        elif "prediction" in f:
            pred_key = "prediction"
        elif "predictions" in f:
            pred_key = "predictions"
        else:
            raise KeyError(
                f"No prediction dataset found in {h5_path}. "
                "Expected pred_lognorm / pred / prediction / predictions."
            )

        pred = f[pred_key][:].astype(np.float32)

        if "genes" not in f:
            raise KeyError(f"{h5_path} does not contain dataset 'genes'")
        genes = decode_array(f["genes"][:])

        if "coords" not in f:
            raise KeyError(f"{h5_path} does not contain dataset 'coords'")
        coords = f["coords"][:].astype(np.float32)

        if "tile_id" in f:
            tile_id = decode_array(f["tile_id"][:])
        else:
            tile_id = [f"tile_{i}" for i in range(pred.shape[0])]

        if "n_spots" in f:
            n_spots = f["n_spots"][:]
        else:
            n_spots = np.ones(pred.shape[0], dtype=int)

        sample = f.attrs.get("sample", os.path.basename(os.path.dirname(h5_path)))
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8")
        else:
            sample = str(sample)

    if pred.shape[0] != coords.shape[0]:
        raise ValueError(
            f"pred rows != coords rows: pred={pred.shape}, coords={coords.shape}"
        )

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    print(
        f"[Read h5] sample={sample}, pred_key={pred_key}, "
        f"pred={pred.shape}, coords={coords.shape}, n_genes={len(genes)}"
    )

    return {
        "h5_path": h5_path,
        "pred_key": pred_key,
        "sample": sample,
        "pred": pred,
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "coords": coords,
        "tile_id": tile_id,
        "n_spots": n_spots,
    }


# ============================================================
# Coordinate transform
# ============================================================

def get_transformed_coords(coords, orientation="normal"):
    """
    orientation:
        normal  : x, y
        flip_x  : mirror x within current coordinate range
        flip_y  : mirror y within current coordinate range
        flip_xy : mirror both x and y
    """
    coords = np.asarray(coords, dtype=np.float32)

    x = coords[:, 0].copy()
    y = coords[:, 1].copy()

    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)

    if orientation in ["flip_x", "flip_xy"]:
        x = x_max - (x - x_min)

    if orientation in ["flip_y", "flip_xy"]:
        y = y_max - (y - y_min)

    return x, y


def get_color_limits(expr, vmin_pct=1, vmax_pct=99):
    finite = np.isfinite(expr)

    if finite.sum() == 0:
        return None, None

    vmin = np.percentile(expr[finite], vmin_pct)
    vmax = np.percentile(expr[finite], vmax_pct)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(expr[finite]))
        vmax = float(np.nanmax(expr[finite]))

    return vmin, vmax


# ============================================================
# Plotting
# ============================================================

def plot_gene_orientation_check(
    sample_data,
    gene,
    task_name,
    out_dir,
    point_size=8,
    alpha=0.95,
    cmap="viridis",
    vmin_pct=1,
    vmax_pct=99,
):
    gene_to_idx = sample_data["gene_to_idx"]

    if gene not in gene_to_idx:
        print(f"[WARN] {gene} not found in {sample_data['sample']} genes. Skip orientation check.")
        return None

    idx = gene_to_idx[gene]
    expr = sample_data["pred"][:, idx]
    coords = sample_data["coords"]

    vmin, vmax = get_color_limits(expr, vmin_pct=vmin_pct, vmax_pct=vmax_pct)
    if vmin is None:
        print(f"[WARN] {gene} all non-finite in {sample_data['sample']}. Skip orientation check.")
        return None

    orientations = ["normal", "flip_x", "flip_y", "flip_xy"]

    os.makedirs(out_dir, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    sc = None

    for ax, orientation in zip(axs, orientations):
        x, y = get_transformed_coords(coords, orientation=orientation)

        sc = ax.scatter(
            x,
            y,
            c=expr,
            s=point_size,
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )

        ax.set_title(
            f"{task_name} | {sample_data['sample']} | {gene}\n{orientation}",
            fontsize=11,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

        if INVERT_Y_AXIS:
            ax.invert_yaxis()

    cbar = plt.colorbar(sc, ax=axs.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(f"{gene} predicted expression")

    safe_sample = safe_gene_name(sample_data["sample"])
    safe_gene = safe_gene_name(gene)

    prefix = os.path.join(
        out_dir,
        f"{task_name}_{safe_sample}_{safe_gene}_orientation_check"
    )

    if SAVE_PNG:
        fig.savefig(prefix + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(prefix + ".pdf", dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[Saved orientation check] {prefix}.png/.pdf")
    return prefix


def plot_gene_final(
    sample_data,
    gene,
    task_name,
    out_dir,
    orientation="normal",
    point_size=8,
    alpha=0.95,
    cmap="viridis",
    vmin_pct=1,
    vmax_pct=99,
):
    gene_to_idx = sample_data["gene_to_idx"]

    if gene not in gene_to_idx:
        print(f"[WARN] {gene} not found in {sample_data['sample']} genes. Skip final plot.")
        return None

    idx = gene_to_idx[gene]
    expr = sample_data["pred"][:, idx]
    coords = sample_data["coords"]

    vmin, vmax = get_color_limits(expr, vmin_pct=vmin_pct, vmax_pct=vmax_pct)
    if vmin is None:
        print(f"[WARN] {gene} all non-finite in {sample_data['sample']}. Skip final plot.")
        return None

    x, y = get_transformed_coords(coords, orientation=orientation)

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(
        x,
        y,
        c=expr,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_title(
        f"{task_name} | {sample_data['sample']} | {gene}\n"
        f"Path2Space tile-level predicted lognorm expression | {orientation}",
        fontsize=12,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    if INVERT_Y_AXIS:
        ax.invert_yaxis()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{gene} predicted expression")

    plt.tight_layout()

    safe_sample = safe_gene_name(sample_data["sample"])
    safe_gene = safe_gene_name(gene)

    prefix = os.path.join(
        out_dir,
        f"{task_name}_{safe_sample}_{safe_gene}_tilelevel_pred_{orientation}"
    )

    if SAVE_PNG:
        fig.savefig(prefix + ".png", dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(prefix + ".pdf", dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[Saved final] {prefix}.png/.pdf")
    return prefix


def export_gene_values(sample_data, genes, task_name, out_dir):
    coords = sample_data["coords"]

    meta = pd.DataFrame({
        "sample": sample_data["sample"],
        "tile_id": sample_data["tile_id"],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "n_spots": sample_data["n_spots"],
    })

    for gene in genes:
        if gene not in sample_data["gene_to_idx"]:
            print(f"[WARN] {gene} not found in {sample_data['sample']} genes. Skip csv value.")
            continue

        idx = sample_data["gene_to_idx"][gene]
        meta[gene] = sample_data["pred"][:, idx]

    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(
        out_dir,
        f"{task_name}_{safe_gene_name(sample_data['sample'])}_selected_gene_values.csv"
    )

    meta.to_csv(out_csv, index=False)

    print(f"[Saved values] {out_csv}")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUT_PLOT_DIR, exist_ok=True)

    for task_name, cfg in TASKS.items():
        out_dir = cfg["out_dir"]
        genes = cfg["genes"]

        print("=" * 80)
        print(f"[Task] {task_name}")
        print(f"out_dir: {out_dir}")
        print(f"genes: {genes}")
        print("=" * 80)

        h5_files = find_prediction_h5(out_dir)

        print(f"[Found h5] n={len(h5_files)}")
        for h5_path in h5_files:
            print(" ", h5_path)

        task_plot_dir = os.path.join(OUT_PLOT_DIR, task_name)
        os.makedirs(task_plot_dir, exist_ok=True)

        for h5_path in h5_files:
            sample_data = read_prediction_h5(h5_path)

            print(
                f"[Sample] task={task_name}, sample={sample_data['sample']}, "
                f"pred={sample_data['pred'].shape}, n_genes={len(sample_data['genes'])}"
            )

            export_gene_values(
                sample_data=sample_data,
                genes=genes,
                task_name=task_name,
                out_dir=task_plot_dir,
            )

            final_orientation = FINAL_ORIENTATION_BY_TASK.get(task_name, "normal")

            for gene in genes:
                if MAKE_ORIENTATION_CHECK:
                    plot_gene_orientation_check(
                        sample_data=sample_data,
                        gene=gene,
                        task_name=task_name,
                        out_dir=task_plot_dir,
                        point_size=POINT_SIZE,
                        alpha=ALPHA,
                        cmap=CMAP,
                        vmin_pct=VMIN_PERCENTILE,
                        vmax_pct=VMAX_PERCENTILE,
                    )

                if MAKE_FINAL_PLOT:
                    plot_gene_final(
                        sample_data=sample_data,
                        gene=gene,
                        task_name=task_name,
                        out_dir=task_plot_dir,
                        orientation=final_orientation,
                        point_size=POINT_SIZE,
                        alpha=ALPHA,
                        cmap=CMAP,
                        vmin_pct=VMIN_PERCENTILE,
                        vmax_pct=VMAX_PERCENTILE,
                    )

    print("Done.")


if __name__ == "__main__":
    main()
