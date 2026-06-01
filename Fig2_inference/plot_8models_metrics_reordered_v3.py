#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot heatmap panels for OC / CC / HCC with:
    GT | STNET | iSTAR | EGNv1 | Hist2ST | DeepPT | Path2Space | SQUALL

Fixes in this version:
  1. Hist2ST and DeepPT are actually added to panels/metrics loop.
  2. H5 mapping supports:
       - tile_x/tile_y
       - coords / spatial / tile_coords
       - parsable tile_id / obs_names / barcodes strings
       - row-order fallback for DeepPT h5 files without coords.

Important:
    SQUALL is always the right-most model panel.
"""

import os
import re
import json
import h5py
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from sklearn.preprocessing import quantile_transform
from matplotlib.colors import LinearSegmentedColormap

Image.MAX_IMAGE_PIXELS = None


# ======================================================
# Sample configs
# ======================================================

SAMPLES = {
    "HCC": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/HCC/Xenium/HCC",
        "genes": ["MET", "CD47", "RHOA", "RAC1", "NRAS", "APEX1"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_HCC_to_HCCXenium/xenium_predicted_expression.h5",
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_HCC_to_HCCXenium_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_HCCmodel_predict_HCCXenium_ensemble5/HCC_Xenium_all_new_predicted_expression_ensemble5_fold0.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_HCC_to_Xenium_tilelevel_original_setting/HCC_Xenium_all_new/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    },
    "OV": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/OV_Xenium",
        "genes": ["EPCAM", "AKT1", "CDK1"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OV_to_OVXenium/xenium_predicted_expression.h5",
        # Keep your original paths here; change them manually if OV-specific files are elsewhere.
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OV_to_OVXenium_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_OVXenium_ensemble5/OV_Xenium_all_new_predicted_expression_ensemble5_fold0.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_Xenium_tilelevel_original_setting/OV_Xenium_all_new/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    },
    "OC": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/OC_Xenium_public",
        "genes": ["IFNGR1", "STAT1"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OC_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OVmodel_to_OC/xenium_predicted_expression.h5",
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OVmodel_to_OC_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_OC_ensemble5/OC_all_new_predicted_expression_ensemble5_fold0.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_OC_tilelevel_predict_only_aligned/OC_all_new_aligned/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    },
    "CC": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/CC_Xenium_public",
        "genes": ["CD8A", "MTOR"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/CC_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OVmodel_to_CC/xenium_predicted_expression.h5",
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OVmodel_to_CC_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_CC_fold0_0/CC_all_new_predicted_expression.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_CC_tilelevel_predict_only_aligned/CC_all_new_aligned/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    }
}
SAMPLES = {
 "CC": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/CC_Xenium_public",
        "genes": ["PTPN2","STAT1","PTEN","PTPN6","XBP1"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/CC_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OVmodel_to_CC/xenium_predicted_expression.h5",
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OVmodel_to_CC_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_CC_fold0_0/CC_all_new_predicted_expression.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_CC_tilelevel_predict_only_aligned/CC_all_new_aligned/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    }
}

'''
SAMPLES = {
    "OV": {
        "root": "/lustre1/zxzeng/bwqin/SQUALL/Xenium/OV_Xenium",
        "genes": ["EPCAM", "AKT1", "CDK1"],
        "stnet_dir": "STNET",
        "storm_dir": "geneplot_v3_tumor_all",
        "istar_dir": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium_all_new/cnts-super",
        "egnv1_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OV_to_OVXenium/xenium_predicted_expression.h5",
        # Keep your original paths here; change them manually if OV-specific files are elsewhere.
        "hist2st_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OV_to_OVXenium_coordzero/xenium_predicted_expression.h5",
        "deeppt_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_OVXenium_ensemble5/OV_Xenium_all_new_predicted_expression_ensemble5_fold0.h5",
        "path2space_h5": "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_Xenium_tilelevel_original_setting/OV_Xenium_all_new/path2space_tilelevel_predicted_expression_ensemble.h5",
        "gt_dir": "expression_all",
        "valid_genes": "valid_genes_all.json",
    }
    }
'''
# ======================================================
# Global config
# ======================================================

OUT_DIR = "heatmap_panels_8models_final"
OUT_DIR = "heatmap_panels_8models_final_CC"
os.makedirs(OUT_DIR, exist_ok=True)

HE_FILE = "HE_rescaled_0.5mpp.tiff"

TILE_SIZE = 224.0
COORD_SCALE = 1.0
X_OFFSET = 0.0
Y_OFFSET = 0.0

HE_DOWNSAMPLE = 20
HE_ALPHA = 1.0
HEATMAP_ALPHA = 0.70
VMIN = 0.01
VMAX = 1.0
ALLOW_MISSING = True

PANEL_ORDER = [
    "GT",
    "STNET",
    "ISTAR",
    "EGNv1",
    "Hist2ST",
    "DeepPT",
    "Path2Space",
    "SQUALL",
]

PANEL_LABEL = {
    "GT": "Ground truth",
    "STNET": "ST-Net",
    "ISTAR": "iSTAR",
    "EGNv1": "EGNv1",
    "Hist2ST": "Hist2ST",
    "DeepPT": "DeepPT",
    "Path2Space": "Path2Space",
    "SQUALL": "SQUALL",
}

PREFERRED_H5_KEYS = [
    "pred_lognorm",
    "pred",
    "prediction",
    "predictions",
    "pred_expr",
    "expr_pred",
    "X",
    "pred_scaled",
]

H5_STRING_ID_KEYS = [
    "tile_id", "tile_ids", "tile_name", "tile_names", "obs_names", "barcodes", "barcode",
    "spot_id", "spot_ids", "ids", "names",
]
H5_X_KEYS = ["tile_x", "x", "grid_x", "pos_x", "posX"]
H5_Y_KEYS = ["tile_y", "y", "grid_y", "pos_y", "posY"]
H5_COORD_KEYS = ["coords", "coord", "coordinates", "spatial", "tile_coords"]
H5_GENE_KEYS = ["genes", "gene_names", "var_names", "features"]

vortex_cmap = LinearSegmentedColormap.from_list(
    "vortex_cmap",
    ["#3b0f70", "#1c6db7", "#4a8d6e", "#fdb863", "#a91e2c"],
)
vortex_cmap.set_under("black")
CMAP = vortex_cmap


# ======================================================
# Basic helpers
# ======================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_join(root, path):
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def parse_pos_from_string(s):
    s = str(s)
    patterns = [
        r"posX_(\d+)_posY_(\d+)",
        r"posx_(\d+)_posy_(\d+)",
        r"posX(\d+)_posY(\d+)",
        r"posx(\d+)_posy(\d+)",
        r"x_(\d+)_y_(\d+)",
        r"X_(\d+)_Y_(\d+)",
        r"x(\d+)_y(\d+)",
        r"X(\d+)_Y(\d+)",
        r"tile_(\d+)_(\d+)",
        r"(\d+)x(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def decode_array(arr):
    out = []
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def load_pickle_any(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def quantile_normalize(arr, mask=None, n_quantiles=1000):
    arr = np.asarray(arr, dtype=np.float32)
    arr_flat = arr.flatten()

    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        valid = arr_flat[mask_flat]
    else:
        mask_flat = np.isfinite(arr_flat)
        valid = arr_flat[mask_flat]

    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.float32)

    normalized = quantile_transform(
        valid.reshape(-1, 1),
        n_quantiles=min(n_quantiles, len(valid)),
        output_distribution="uniform",
        copy=True,
    ).flatten().astype(np.float32)

    arr_out = np.zeros_like(arr_flat, dtype=np.float32)
    if mask is not None:
        idx = np.where(mask_flat)[0]
        # If mask contains non-finite values, only fill finite positions in order.
        finite_idx = idx[np.isfinite(arr_flat[idx])]
        arr_out[finite_idx] = normalized
    else:
        idx = np.where(np.isfinite(arr_flat))[0]
        arr_out[idx] = normalized

    return arr_out.reshape(arr.shape).astype(np.float32)


def compute_metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    mask = np.isfinite(pred_flat) & np.isfinite(gt_flat)

    if mask.sum() > 1:
        try:
            pearson_corr = pearsonr(pred_flat[mask], gt_flat[mask])[0]
        except Exception:
            pearson_corr = np.nan
    else:
        pearson_corr = np.nan

    try:
        data_range = float(np.nanmax(gt) - np.nanmin(gt))
        ssim_val = np.nan if data_range <= 0 else ssim(pred, gt, data_range=data_range)
    except Exception:
        ssim_val = np.nan

    try:
        rmsd_val = np.sqrt(np.nanmean((pred - gt) ** 2))
    except Exception:
        rmsd_val = np.nan

    return pearson_corr, ssim_val, rmsd_val


# ======================================================
# GT loading
# ======================================================

def infer_gt_files_and_shape(gt_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith("_Xenium_expr.pt")])
    if len(gt_files) == 0:
        raise RuntimeError(f"No *_Xenium_expr.pt found in {gt_dir}")

    file_to_xy = {}
    tile_positions = []
    for fname in gt_files:
        xy = parse_pos_from_string(fname)
        if xy is None:
            continue
        x, y = xy
        file_to_xy[fname] = (x, y)
        tile_positions.append((x, y))

    if len(tile_positions) == 0:
        raise RuntimeError(f"No posX/posY pattern found in GT files under {gt_dir}")

    max_x = max(x for x, y in tile_positions)
    max_y = max(y for x, y in tile_positions)
    gt_shape = (max_y + 1, max_x + 1)
    n_tiles = gt_shape[0] * gt_shape[1]

    gt_mask_flat = np.zeros(n_tiles, dtype=bool)
    for x, y in tile_positions:
        flat = y * gt_shape[1] + x
        gt_mask_flat[flat] = True

    return gt_files, file_to_xy, gt_shape, gt_mask_flat


def load_gt_gene(gene, gt_dir, gt_files, file_to_xy, gt_shape, valid_gene_to_idx):
    if gene not in valid_gene_to_idx:
        return None, f"{gene} not in valid_genes"

    gene_idx = valid_gene_to_idx[gene]
    n_tiles = gt_shape[0] * gt_shape[1]
    raw = np.zeros(n_tiles, dtype=np.float32)
    gt_mask_flat = np.zeros(n_tiles, dtype=bool)

    for fname in gt_files:
        if fname not in file_to_xy:
            continue
        x, y = file_to_xy[fname]
        flat = y * gt_shape[1] + x
        fpath = os.path.join(gt_dir, fname)
        expr = torch.load(fpath, map_location="cpu")

        if isinstance(expr, torch.Tensor):
            arr = expr.detach().cpu().numpy()
        else:
            arr = np.asarray(expr)

        if arr.ndim == 2:
            arr = arr[0]
        elif arr.ndim != 1:
            raise ValueError(f"Unexpected GT expr shape {arr.shape} in {fpath}")

        if gene_idx < len(arr):
            raw[flat] = float(arr[gene_idx])
            gt_mask_flat[flat] = True

    gt_mask = gt_mask_flat.reshape(gt_shape)
    gt_arr = raw.reshape(gt_shape)
    gt_q = quantile_normalize(gt_arr, mask=gt_mask)
    gt_q[~gt_mask] = 0
    return gt_q.astype(np.float32), "ok"


# ======================================================
# File model loading: STNET / iSTAR / SQUALL
# ======================================================

def load_model_output_original(path, shape, mask=None):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = load_pickle_any(path)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = resize(arr, shape, preserve_range=True, anti_aliasing=True).astype(np.float32)

    if mask is not None:
        arr[~mask] = 0
    arr = quantile_normalize(arr, mask)
    if mask is not None:
        arr[~mask] = 0
    return arr.astype(np.float32)


def load_stnet_gene(gene, stnet_dir, gt_shape, gt_mask):
    path = os.path.join(stnet_dir, f"{gene}.pkl")
    if not os.path.exists(path):
        return None, f"missing:{path}"

    obj = load_pickle_any(path)
    if not (isinstance(obj, dict) and "pred" in obj):
        return None, f"bad_format:{path}"

    arr = np.asarray(obj["pred"], dtype=np.float32)
    arr = resize(arr, gt_shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    stnet_mask = np.isfinite(arr) & gt_mask
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = quantile_normalize(arr, mask=stnet_mask)
    arr[~stnet_mask] = 0
    return arr.astype(np.float32), path


def load_istar_gene(gene, istar_dir, gt_shape, gt_mask):
    path = os.path.join(istar_dir, f"{gene}.pickle")
    if not os.path.exists(path):
        return None, f"missing:{path}"
    arr = load_model_output_original(path, gt_shape, mask=gt_mask)
    return arr, path


def load_storm_gene(gene, storm_dir, gt_shape, gt_mask):
    path = os.path.join(storm_dir, f"{gene}_expression.npy")
    if not os.path.exists(path):
        return None, f"missing:{path}"
    arr = load_model_output_original(path, gt_shape, mask=gt_mask)
    return arr, path


# ======================================================
# H5 loading: EGNv1 / Hist2ST / DeepPT / Path2Space
# ======================================================

def h5_all_datasets(f):
    out = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    f.visititems(visitor)
    return out


def get_dataset_if_exists(f, candidate_keys):
    all_dsets = h5_all_datasets(f)
    for key in candidate_keys:
        if key in all_dsets:
            return key, all_dsets[key]
    for name, ds in all_dsets.items():
        base = name.split("/")[-1]
        if base in candidate_keys:
            return name, ds
    return None, None


def choose_prediction_key(h5_path, preferred_key=None):
    with h5py.File(h5_path, "r") as f:
        all_dsets = h5_all_datasets(f)
        all_keys = list(all_dsets.keys())

        if preferred_key is not None:
            if preferred_key not in all_dsets:
                raise KeyError(f"{preferred_key} not found in {h5_path}. Available: {all_keys}")
            return preferred_key

        for k in PREFERRED_H5_KEYS:
            if k in all_dsets:
                return k
            # basename match for nested datasets
            for name in all_keys:
                if name.split("/")[-1] == k:
                    return name

        for k, ds in all_dsets.items():
            if len(ds.shape) == 2 and np.issubdtype(ds.dtype, np.number):
                return k

    raise KeyError(f"No numeric 2D prediction dataset found in {h5_path}")


def parse_xy_arrays_from_string_ids(ids):
    tile_x = []
    tile_y = []
    ok = []
    for s in ids:
        xy = parse_pos_from_string(s)
        if xy is None:
            ok.append(False)
            tile_x.append(-1)
            tile_y.append(-1)
        else:
            x, y = xy
            ok.append(True)
            tile_x.append(x)
            tile_y.append(y)
    return (
        np.asarray(tile_x, dtype=np.int64),
        np.asarray(tile_y, dtype=np.int64),
        np.asarray(ok, dtype=bool),
    )


def load_h5_meta(h5_path, preferred_key=None):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    key = choose_prediction_key(h5_path, preferred_key=preferred_key)

    with h5py.File(h5_path, "r") as f:
        pred_shape = f[key].shape

        gene_key, gene_ds = get_dataset_if_exists(f, H5_GENE_KEYS)
        if gene_ds is None:
            raise KeyError(f"No genes/gene_names/var_names/features dataset found in {h5_path}")
        genes = decode_array(gene_ds[:])

        x_key, x_ds = get_dataset_if_exists(f, H5_X_KEYS)
        y_key, y_ds = get_dataset_if_exists(f, H5_Y_KEYS)
        tile_x = x_ds[:].astype(np.int64) if x_ds is not None else None
        tile_y = y_ds[:].astype(np.int64) if y_ds is not None else None

        coord_key, coord_ds = get_dataset_if_exists(f, H5_COORD_KEYS)
        coords = coord_ds[:].astype(np.float32) if coord_ds is not None else None

        id_key, id_ds = get_dataset_if_exists(f, H5_STRING_ID_KEYS)
        tile_id = decode_array(id_ds[:]) if id_ds is not None else None

    if len(pred_shape) != 2:
        raise ValueError(f"{h5_path}: prediction dataset {key} is not 2D, shape={pred_shape}")

    if pred_shape[1] != len(genes):
        raise ValueError(f"{h5_path}: pred shape {pred_shape}, n_genes={len(genes)}. Expected rows x genes.")

    return {
        "h5_path": h5_path,
        "pred_key": key,
        "pred_shape": pred_shape,
        "genes": genes,
        "gene_to_col": {g: i for i, g in enumerate(genes)},
        "tile_x": tile_x,
        "tile_y": tile_y,
        "coords": coords,
        "tile_id": tile_id,
    }


def build_h5_to_gt_mapping(h5_obj, gt_shape, gt_mask_flat):
    n_y, n_x = gt_shape
    n_tiles = n_y * n_x
    n_rows = h5_obj["pred_shape"][0]
    gt_valid = int(gt_mask_flat.sum())

    tile_x = None
    tile_y = None
    source = None

    if h5_obj.get("tile_x") is not None and h5_obj.get("tile_y") is not None:
        tile_x = h5_obj["tile_x"].astype(np.int64)
        tile_y = h5_obj["tile_y"].astype(np.int64)
        if len(tile_x) != n_rows or len(tile_y) != n_rows:
            raise ValueError("tile_x/tile_y length does not match pred rows")
        source = "tile_x/tile_y"

    elif h5_obj.get("coords") is not None:
        coords = np.asarray(h5_obj["coords"], dtype=np.float32)
        if coords.shape[0] != n_rows:
            raise ValueError("coords rows do not match pred rows")
        x = coords[:, 0] * COORD_SCALE
        y = coords[:, 1] * COORD_SCALE
        tile_x = np.floor((x - X_OFFSET) / TILE_SIZE).astype(np.int64)
        tile_y = np.floor((y - Y_OFFSET) / TILE_SIZE).astype(np.int64)
        source = "coords"

    elif h5_obj.get("tile_id") is not None:
        ids = h5_obj["tile_id"]
        if len(ids) != n_rows:
            raise ValueError("tile_id length does not match pred rows")
        tile_x0, tile_y0, ok = parse_xy_arrays_from_string_ids(ids)
        if ok.sum() == 0:
            raise RuntimeError(f"{h5_obj['h5_path']} has tile_id but no parsable x/y pattern")
        tile_x = tile_x0
        tile_y = tile_y0
        source = "tile_id"

    else:
        valid_tile_flat = np.where(gt_mask_flat)[0].astype(np.int64)

        if n_rows == gt_valid:
            row_idx = np.arange(n_rows, dtype=np.int64)
            tile_flat = valid_tile_flat
            tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)
            source = "row_order_gt_valid_tiles"
            print(
                f"[H5 mapping] {os.path.basename(h5_obj['h5_path'])} source={source} "
                f"rows={n_rows}, kept={len(row_idx)}, tiles={(tile_counts > 0).sum()}/{gt_valid}"
            )
            return {"row_idx": row_idx, "tile_flat": tile_flat, "tile_counts": tile_counts, "source": source}

        if n_rows == n_tiles:
            row_idx = valid_tile_flat.copy()
            tile_flat = valid_tile_flat.copy()
            tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)
            source = "row_order_full_grid"
            print(
                f"[H5 mapping] {os.path.basename(h5_obj['h5_path'])} source={source} "
                f"rows={n_rows}, kept={len(row_idx)}, tiles={(tile_counts > 0).sum()}/{gt_valid}"
            )
            return {"row_idx": row_idx, "tile_flat": tile_flat, "tile_counts": tile_counts, "source": source}

        # Tolerant fallback for exporters with 1-few extra rows.
        # This is useful when the h5 has a leading/header/out-of-bound tile but no coords.
        if n_rows > gt_valid and (n_rows - gt_valid) <= 5:
            row_idx = np.arange(gt_valid, dtype=np.int64)
            tile_flat = valid_tile_flat
            tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)
            source = f"row_order_gt_valid_tiles_truncate_first_{gt_valid}_of_{n_rows}"
            print(
                f"[WARN] {os.path.basename(h5_obj['h5_path'])} has no coords and rows={n_rows}, "
                f"gt_valid={gt_valid}. Using first {gt_valid} rows by row-order fallback."
            )
            print(
                f"[H5 mapping] {os.path.basename(h5_obj['h5_path'])} source={source} "
                f"rows={n_rows}, kept={len(row_idx)}, tiles={(tile_counts > 0).sum()}/{gt_valid}"
            )
            return {"row_idx": row_idx, "tile_flat": tile_flat, "tile_counts": tile_counts, "source": source}

        raise RuntimeError(
            f"{h5_obj['h5_path']} has neither tile_x/tile_y, coords, parsable tile_id, "
            f"nor a row count matching GT valid/full grid. "
            f"n_rows={n_rows}, gt_valid={gt_valid}, full_grid={n_tiles}"
        )

    in_bound = (
        (tile_x >= 0) & (tile_x < n_x) &
        (tile_y >= 0) & (tile_y < n_y)
    )
    tile_flat_all = tile_y * n_x + tile_x

    keep = np.zeros(n_rows, dtype=bool)
    valid_rows = np.where(in_bound)[0]
    keep[valid_rows] = gt_mask_flat[tile_flat_all[valid_rows]]

    row_idx = np.where(keep)[0].astype(np.int64)
    tile_flat = tile_flat_all[row_idx].astype(np.int64)
    tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)

    print(
        f"[H5 mapping] {os.path.basename(h5_obj['h5_path'])} source={source} "
        f"rows={n_rows}, kept={len(row_idx)}, tiles={(tile_counts > 0).sum()}/{gt_valid}"
    )
    return {"row_idx": row_idx, "tile_flat": tile_flat, "tile_counts": tile_counts, "source": source}


def load_h5_gene_to_gt_grid(h5_obj, mapping, gene, gt_shape, gt_mask_flat):
    if gene not in h5_obj["gene_to_col"]:
        return None, f"{gene} missing in h5"

    col = int(h5_obj["gene_to_col"][gene])
    n_tiles = gt_shape[0] * gt_shape[1]
    row_idx = mapping["row_idx"]
    tile_flat = mapping["tile_flat"]
    tile_counts = mapping["tile_counts"]

    if len(row_idx) == 0:
        return None, "no rows mapped to GT"

    try:
        with h5py.File(h5_obj["h5_path"], "r") as f:
            ds = f[h5_obj["pred_key"]]
            values = ds[row_idx, col].astype(np.float32)
    except Exception as e:
        return None, f"h5_read_error:{repr(e)}"

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tile_sum = np.zeros(n_tiles, dtype=np.float32)
    np.add.at(tile_sum, tile_flat, values)

    pred_tile = np.zeros(n_tiles, dtype=np.float32)
    nonzero = tile_counts > 0
    pred_tile[nonzero] = tile_sum[nonzero] / tile_counts[nonzero]

    arr = pred_tile.reshape(gt_shape)
    gt_mask = gt_mask_flat.reshape(gt_shape)
    arr[~gt_mask] = 0
    arr = quantile_normalize(arr, mask=gt_mask)
    arr[~gt_mask] = 0
    return arr.astype(np.float32), "ok"


# ======================================================
# HE and plotting
# ======================================================

def load_he_down(root):
    he_path = os.path.join(root, HE_FILE)
    if not os.path.exists(he_path):
        print(f"[WARN] HE not found: {he_path}")
        return None
    img = Image.open(he_path).convert("RGB")
    w = max(1, img.width // HE_DOWNSAMPLE)
    h = max(1, img.height // HE_DOWNSAMPLE)
    img_down = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img_down)


def array_to_overlay(arr, gt_mask, he_shape):
    out_h, out_w = he_shape[:2]
    arr_up = resize(arr, (out_h, out_w), preserve_range=True, order=1, anti_aliasing=True).astype(np.float32)
    mask_up = resize(gt_mask.astype(np.float32), (out_h, out_w), preserve_range=True, order=0, anti_aliasing=False) > 0.5
    arr_up = np.nan_to_num(arr_up, nan=0.0, posinf=0.0, neginf=0.0)
    arr_up[~mask_up] = 0
    return arr_up, mask_up


def plot_gene_panel(sample_name, gene, panels, gt_arr, gt_mask, he_down, out_dir):
    n = len(PANEL_ORDER)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.5), gridspec_kw={"wspace": 0.01})
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, PANEL_ORDER):
        arr = panels.get(model, None)

        if arr is None:
            blank = np.zeros_like(gt_arr) if gt_arr is not None else np.zeros_like(gt_mask, dtype=np.float32)
            ax.imshow(blank, cmap=CMAP, vmin=VMIN, vmax=VMAX)
            title = f"{PANEL_LABEL.get(model, model)}\nmissing"
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            continue

        ax.imshow(arr, cmap=CMAP, vmin=VMIN, vmax=VMAX)

        if model == "GT":
            title = f"{gene}\nGround Truth"
        else:
            p, s, r = compute_metrics(arr, gt_arr)
            title = f"{PANEL_LABEL.get(model, model)}\nP={p:.2f}, S={s:.2f}"

        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.01)

    png = os.path.join(out_dir, f"{sample_name}_{gene}_8models_black_background.png")
    pdf = os.path.join(out_dir, f"{sample_name}_{gene}_8models_black_background.pdf")
    svg = os.path.join(out_dir, f"{sample_name}_{gene}_8models_black_background.svg")

    plt.savefig(png, dpi=600, bbox_inches="tight", transparent=True)
    plt.savefig(pdf, dpi=600, bbox_inches="tight", transparent=True)
    plt.savefig(svg, dpi=600, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"[Saved] {png}")


# ======================================================
# Process one sample
# ======================================================

def process_one_sample(sample_name, cfg):
    root = cfg["root"]
    out_dir = os.path.join(OUT_DIR, sample_name)
    ensure_dir(out_dir)

    print("\n" + "=" * 120)
    print(f"[Sample] {sample_name}")
    print(f"root: {root}")
    print("=" * 120)

    gt_dir = safe_join(root, cfg["gt_dir"])
    valid_genes_path = safe_join(root, cfg["valid_genes"])
    stnet_dir = safe_join(root, cfg["stnet_dir"])
    storm_dir = safe_join(root, cfg["storm_dir"])
    istar_dir = cfg["istar_dir"]

    with open(valid_genes_path) as f:
        valid_genes = json.load(f)
    valid_gene_to_idx = {g: i for i, g in enumerate(valid_genes)}

    gt_files, file_to_xy, gt_shape, gt_mask_flat = infer_gt_files_and_shape(gt_dir)
    gt_mask = gt_mask_flat.reshape(gt_shape)
    print(f"[GT shape] {gt_shape}, valid tiles={int(gt_mask.sum())}")

    he_down = load_he_down(root)
    if he_down is not None:
        print(f"[HE down] {he_down.shape}")

    h5_objects = {}
    h5_mappings = {}
    h5_configs = {
        "EGNv1": cfg["egnv1_h5"],
        "Hist2ST": cfg["hist2st_h5"],
        "DeepPT": cfg["deeppt_h5"],
        "Path2Space": cfg["path2space_h5"],
    }

    for model, h5_path in h5_configs.items():
        try:
            print(f"[Load h5 meta] {model}: {h5_path}")
            obj = load_h5_meta(h5_path)
            mapping = build_h5_to_gt_mapping(obj, gt_shape, gt_mask_flat)
            h5_objects[model] = obj
            h5_mappings[model] = mapping
            print(
                f"  pred_key={obj['pred_key']}, pred_shape={obj['pred_shape']}, "
                f"n_genes={len(obj['genes'])}, mapping={mapping['source']}"
            )
        except Exception as e:
            print(f"[WARN] failed to load h5 {model}: {repr(e)}")
            h5_objects[model] = None
            h5_mappings[model] = None

    metrics_rows = []

    for gene in cfg["genes"]:
        print("\n" + "-" * 100)
        print(f"[Gene] {sample_name} {gene}")
        print("-" * 100)

        panels = {m: None for m in PANEL_ORDER}
        statuses = {}

        gt_arr, status = load_gt_gene(
            gene=gene,
            gt_dir=gt_dir,
            gt_files=gt_files,
            file_to_xy=file_to_xy,
            gt_shape=gt_shape,
            valid_gene_to_idx=valid_gene_to_idx,
        )
        statuses["GT"] = status
        panels["GT"] = gt_arr

        if gt_arr is None:
            print(f"[SKIP] GT missing for {gene}: {status}")
            if not ALLOW_MISSING:
                continue

        try:
            arr, status = load_stnet_gene(gene, stnet_dir, gt_shape, gt_mask)
            panels["STNET"] = arr
            statuses["STNET"] = status
        except Exception as e:
            panels["STNET"] = None
            statuses["STNET"] = f"error:{repr(e)}"

        try:
            arr, status = load_istar_gene(gene, istar_dir, gt_shape, gt_mask)
            panels["ISTAR"] = arr
            statuses["ISTAR"] = status
        except Exception as e:
            panels["ISTAR"] = None
            statuses["ISTAR"] = f"error:{repr(e)}"

        # H5-based panels. This is the main fix: include Hist2ST and DeepPT here.
        for model in ["EGNv1", "Hist2ST", "DeepPT", "Path2Space"]:
            if h5_objects.get(model) is None:
                panels[model] = None
                statuses[model] = "h5_meta_missing"
                continue
            try:
                arr, status = load_h5_gene_to_gt_grid(
                    h5_obj=h5_objects[model],
                    mapping=h5_mappings[model],
                    gene=gene,
                    gt_shape=gt_shape,
                    gt_mask_flat=gt_mask_flat,
                )
                panels[model] = arr
                statuses[model] = status
            except Exception as e:
                panels[model] = None
                statuses[model] = f"error:{repr(e)}"

        try:
            arr, status = load_storm_gene(gene, storm_dir, gt_shape, gt_mask)
            panels["SQUALL"] = arr
            statuses["SQUALL"] = status
        except Exception as e:
            panels["SQUALL"] = None
            statuses["SQUALL"] = f"error:{repr(e)}"

        row = {"Sample": sample_name, "Gene": gene}
        for model in PANEL_ORDER:
            row[f"{model}_status"] = statuses.get(model, "")
            if model == "GT":
                continue
            arr = panels.get(model, None)
            if arr is not None and gt_arr is not None:
                p, s, r = compute_metrics(arr, gt_arr)
            else:
                p, s, r = np.nan, np.nan, np.nan
            row[f"{model}_Pearson"] = p
            row[f"{model}_SSIM"] = s
            row[f"{model}_RMSD"] = r
        metrics_rows.append(row)

        plot_gene_panel(
            sample_name=sample_name,
            gene=gene,
            panels=panels,
            gt_arr=gt_arr,
            gt_mask=gt_mask,
            he_down=he_down,
            out_dir=out_dir,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(out_dir, f"{sample_name}_selected_gene_panel_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[Saved metrics] {metrics_csv}")
    return metrics_df


# ======================================================
# Main
# ======================================================

def main():
    all_metrics = []
    for sample_name, cfg in SAMPLES.items():
        df = process_one_sample(sample_name, cfg)
        all_metrics.append(df)

    all_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "all_selected_gene_panel_metrics.csv")
    all_df.to_csv(out_csv, index=False)

    print("\nDone.")
    print(f"Output folder: {OUT_DIR}")
    print(f"All metrics: {out_csv}")


if __name__ == "__main__":
    main()

