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
    "OC":{
        "out_dir":"/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_OC_tilelevel_predict_only_aligned/OC_all_new_aligned",
        "genes": ["IFNGR1","STAT1"],
    },
    "CC":{
        "out_dir":"/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_CC_tilelevel_predict_only_aligned/CC_all_new_aligned",
        "genes": ["CD8A", "MTOR"],
    }
    }

OUT_PLOT_DIR = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_gene_sample_final"

POINT_SIZE = 8
ALPHA = 0.95
CMAP = "viridis"

# percentile color scale; avoids one extreme tile dominating the colorbar
VMIN_PERCENTILE = 1
VMAX_PERCENTILE = 99

SAVE_PDF = True
SAVE_PNG = True
DPI = 300


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


def find_prediction_h5(out_dir):
    patterns = [
        os.path.join(out_dir, "*", "path2space_tilelevel_predicted_expression_ensemble.h5"),
        os.path.join(out_dir, "**", "path2space_tilelevel_predicted_expression_ensemble.h5"),
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
            "Expected path2space_tilelevel_predicted_expression_ensemble.h5"
        )

    return files


def read_prediction_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        pred = f["pred_lognorm"][:].astype(np.float32)
        genes = decode_array(f["genes"][:])
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

    gene_to_idx = {g: i for i, g in enumerate(genes)}

    return {
        "h5_path": h5_path,
        "sample": sample,
        "pred": pred,
        "genes": genes,
        "gene_to_idx": gene_to_idx,
        "coords": coords,
        "tile_id": tile_id,
        "n_spots": n_spots,
    }


# ============================================================
# Plotting
# ============================================================

def safe_gene_name(gene):
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in gene])


def plot_gene_spatial(
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
        print(f"[WARN] {gene} not found in {sample_data['sample']} genes. Skip.")
        return None

    idx = gene_to_idx[gene]
    expr = sample_data["pred"][:, idx]
    coords = sample_data["coords"]

    x = coords[:, 0]
    y = coords[:, 1]

    finite = np.isfinite(expr)
    if finite.sum() == 0:
        print(f"[WARN] {gene} all non-finite in {sample_data['sample']}. Skip.")
        return None

    vmin = np.percentile(expr[finite], vmin_pct)
    vmax = np.percentile(expr[finite], vmax_pct)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(expr))
        vmax = float(np.nanmax(expr))

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
        f"Path2Space tile-level predicted lognorm expression",
        fontsize=12,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    # image coordinate convention: y increases downward
    ax.invert_yaxis()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{gene} predicted expression")

    plt.tight_layout()

    safe_sample = safe_gene_name(sample_data["sample"])
    safe_gene = safe_gene_name(gene)

    prefix = os.path.join(out_dir, f"{task_name}_{safe_sample}_{safe_gene}_tilelevel_pred")

    if SAVE_PDF:
        fig.savefig(prefix + ".pdf", dpi=DPI, bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(prefix + ".png", dpi=DPI, bbox_inches="tight")

    plt.close(fig)

    print(f"[Saved] {prefix}.pdf/.png")

    return prefix


def export_gene_values(sample_data, genes, task_name, out_dir):
    rows = []

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
                f"[Read] task={task_name}, sample={sample_data['sample']}, "
                f"pred={sample_data['pred'].shape}, n_genes={len(sample_data['genes'])}"
            )

            export_gene_values(
                sample_data=sample_data,
                genes=genes,
                task_name=task_name,
                out_dir=task_plot_dir,
            )

            for gene in genes:
                plot_gene_spatial(
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

    print("Done.")


if __name__ == "__main__":
    main()
