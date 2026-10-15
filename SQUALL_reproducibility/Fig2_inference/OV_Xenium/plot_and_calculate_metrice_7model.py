#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast metric-only version with 7 models:
    STNET | iSTAR | EGNv1 | Path2Space | Hist2ST | DeepPT | SQUALL

Preserved original metric logic:
    - quantile_normalize()
    - STNET original processing
    - iSTAR/SQUALL original load_model_output processing
    - compute_metrics(): Pearson + full-image SSIM + RMSD

Added:
    - H5 tile-level models: EGNv1 / Path2Space / Hist2ST / DeepPT
      Supported mapping priority:
        1) tile_x + tile_y
        2) tile_id strings containing posX/posY or x/y
        3) coords -> tile grid by tile_size/coord_scale/offset

Example:
python plot_Xenium_SQUALL_ISTAR_STNET_wsi_all_v11_add_path2space_metrice_only_fast.py \
  --out_csv gene_metrics_summary_7models_no_EGNv2_add_Hist2ST_DeepPT.csv \
  --overall_csv gene_metrics_summary_7models_no_EGNv2_add_Hist2ST_DeepPT_overall.csv \
  --num_workers 6 \
  --chunk_size 128 \
  --use_cache 1
"""

import os
import re
import json
import h5py
import torch
import pickle
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
from sklearn.preprocessing import quantile_transform
from concurrent.futures import ProcessPoolExecutor, as_completed


# ======================================================
# Default paths
# ======================================================

DEFAULT_STNET_DIR = "STNET"
DEFAULT_SQUALL_DIR = "geneplot_v3_tumor_all"
DEFAULT_ISTAR_DIR = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium_all_new/cnts-super"

DEFAULT_EGNV1_H5 = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/EGNv1/outputs/EGNv1_HD_original_OV_to_OVXenium/xenium_predicted_expression.h5"

DEFAULT_PATH2SPACE_H5 = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/path2space/outputs/Path2Space_OV_to_Xenium_tilelevel_original_setting/OV_Xenium_all_new/path2space_tilelevel_predicted_expression_ensemble.h5"
DEFAULT_HIST2ST_H5 = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/Hist2ST/outputs/Hist2ST_HD_native_OV_to_OVXenium_coordzero/xenium_predicted_expression.h5"
DEFAULT_DEEPPT_H5 = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/DeepPT/outputs/DeepPT_fold0_OVmodel_predict_OVXenium_ensemble5/OV_Xenium_all_new_predicted_expression_ensemble5_fold0.h5"

DEFAULT_GT_DIR = "expression_all"
DEFAULT_VALID_GENES_PATH = "valid_genes_all.json"

PREFERRED_H5_KEYS = [
    "pred_lognorm",
    "pred",
    "prediction",
    "predictions",
    "expression",
    "X",
    "pred_scaled",
]


# ======================================================
# Original quantile normalize
# ======================================================

def quantile_normalize(arr, mask=None, n_quantiles=1000):
    arr = np.asarray(arr, dtype=np.float32)
    arr_flat = arr.flatten()

    if mask is not None:
        mask_flat = mask.flatten()
        valid = arr_flat[mask_flat]
    else:
        valid = arr_flat[np.isfinite(arr_flat)]

    if len(valid) == 0:
        return np.zeros_like(arr, dtype=np.float32)

    valid_reshaped = valid.reshape(-1, 1)

    normalized = quantile_transform(
        valid_reshaped,
        n_quantiles=min(n_quantiles, len(valid)),
        output_distribution="uniform",
        copy=True,
    ).flatten().astype(np.float32)

    arr_out = arr.copy().flatten()

    if mask is not None:
        arr_out[mask_flat] = normalized
    else:
        arr_out[np.isfinite(arr_flat)] = normalized

    return arr_out.reshape(arr.shape).astype(np.float32)


# ======================================================
# Original compute_metrics
# ======================================================

def compute_metrics(pred, gt):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    mask = np.isfinite(pred_flat) & np.isfinite(gt_flat)

    pearson_corr = pearsonr(pred_flat[mask], gt_flat[mask])[0] if mask.sum() > 0 else np.nan

    try:
        ssim_val = ssim(pred, gt, data_range=gt.max() - gt.min())
    except Exception:
        ssim_val = np.nan

    rmsd_val = np.sqrt(np.nanmean((pred - gt) ** 2))

    return pearson_corr, ssim_val, rmsd_val


# ======================================================
# Helpers
# ======================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_pos_from_string(s):
    s = str(s)

    patterns = [
        r"posX_(\d+)_posY_(\d+)",
        r"posx_(\d+)_posy_(\d+)",
        r"x_(\d+)_y_(\d+)",
        r"X_(\d+)_Y_(\d+)",
        r"x(\d+)_y(\d+)",
        r"X(\d+)_Y(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return int(m.group(1)), int(m.group(2))

    return None


def decode_array(arr):
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_pickle_any(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def cache_path_for(cache_dir, model_name, gene):
    safe_gene = str(gene).replace("/", "_")
    return os.path.join(cache_dir, model_name, f"{safe_gene}.npy")


def maybe_load_cache(cache_dir, model_name, gene, gt_shape, use_cache):
    if not use_cache:
        return None

    path = cache_path_for(cache_dir, model_name, gene)

    if not os.path.exists(path):
        return None

    try:
        arr = np.load(path)
        if arr.shape == gt_shape:
            return arr.astype(np.float32)
    except Exception:
        return None

    return None


def maybe_save_cache(cache_dir, model_name, gene, arr, use_cache):
    if not use_cache:
        return

    path = cache_path_for(cache_dir, model_name, gene)
    ensure_dir(os.path.dirname(path))
    np.save(path, np.asarray(arr, dtype=np.float32))


# ======================================================
# GT loading
# ======================================================

def infer_gt_files_and_shape(gt_dir):
    gt_files = [
        f for f in os.listdir(gt_dir)
        if f.endswith("_Xenium_expr.pt")
    ]

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

    print("[GT]")
    print(f"  GT files       : {len(gt_files)}")
    print(f"  mapped files   : {len(file_to_xy)}")
    print(f"  gt_shape       : {gt_shape}")
    print(f"  valid GT tiles : {int(gt_mask_flat.sum())}")

    return gt_files, file_to_xy, gt_shape, gt_mask_flat


def build_gt_matrix_once(gt_dir, gt_files, file_to_xy, gt_shape, genes, valid_gene_to_idx):
    n_tiles = gt_shape[0] * gt_shape[1]
    n_genes = len(genes)

    gt_raw_matrix = np.zeros((n_tiles, n_genes), dtype=np.float32)
    gt_mask_flat = np.zeros(n_tiles, dtype=bool)

    local_cols = []
    source_cols = []

    for local_i, g in enumerate(genes):
        if g not in valid_gene_to_idx:
            continue
        local_cols.append(local_i)
        source_cols.append(valid_gene_to_idx[g])

    local_cols = np.asarray(local_cols, dtype=np.int64)
    source_cols = np.asarray(source_cols, dtype=np.int64)

    print("[Build GT matrix once]")
    print(f"  genes requested : {n_genes}")
    print(f"  genes available : {len(local_cols)}")
    print(f"  GT files        : {len(gt_files)}")

    skipped = 0

    for fname in tqdm(gt_files, desc="Load GT .pt once"):
        if fname not in file_to_xy:
            skipped += 1
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

        gt_raw_matrix[flat, local_cols] = arr[source_cols].astype(np.float32)
        gt_mask_flat[flat] = True

    print(f"  skipped files   : {skipped}")
    print(f"  valid GT tiles  : {int(gt_mask_flat.sum())}")

    return gt_raw_matrix, gt_mask_flat


def build_gt_quantile_matrix(gt_raw_matrix, gt_mask_flat, gt_shape, genes):
    n_tiles, n_genes = gt_raw_matrix.shape
    gt_mask = gt_mask_flat.reshape(gt_shape)

    gt_q_matrix = np.zeros((n_tiles, n_genes), dtype=np.float32)

    for j, gene in enumerate(tqdm(genes, desc="Quantile normalize GT")):
        gt_mat = gt_raw_matrix[:, j].reshape(gt_shape)
        gt_arr = quantile_normalize(gt_mat, mask=gt_mask)
        gt_q_matrix[:, j] = gt_arr.reshape(-1).astype(np.float32)

    return gt_q_matrix


# ======================================================
# Original file model processing
# ======================================================

def load_model_output_original(path, shape, mask=None):
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = pickle.load(open(path, "rb"))

    arr = np.nan_to_num(arr, nan=0.0)

    arr = resize(
        arr,
        shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    if mask is not None:
        arr[~mask] = 0

    return quantile_normalize(arr, mask).astype(np.float32)


def load_stnet_processed_original(gene, stnet_dir, gt_shape, gt_mask, cache_dir=None, use_cache=False):
    cached = maybe_load_cache(cache_dir, "STNET", gene, gt_shape, use_cache)
    if cached is not None:
        return cached, "cache"

    stnet_path = os.path.join(stnet_dir, f"{gene}.pkl")

    if not os.path.exists(stnet_path):
        return None, f"missing:{stnet_path}"

    obj = load_pickle_any(stnet_path)

    if not (isinstance(obj, dict) and "pred" in obj):
        return None, f"bad_format:{stnet_path}"

    stnet_pred = obj["pred"]

    stnet_arr = resize(
        stnet_pred,
        gt_shape,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    stnet_mask = np.isfinite(stnet_arr) & gt_mask

    stnet_arr = quantile_normalize(stnet_arr, mask=stnet_mask)
    stnet_arr[~stnet_mask] = 0

    stnet_arr = stnet_arr.astype(np.float32)

    maybe_save_cache(cache_dir, "STNET", gene, stnet_arr, use_cache)

    return stnet_arr, stnet_path


def load_istar_processed_original(gene, istar_dir, gt_shape, gt_mask, cache_dir=None, use_cache=False):
    cached = maybe_load_cache(cache_dir, "ISTAR", gene, gt_shape, use_cache)
    if cached is not None:
        return cached, "cache"

    istar_path = os.path.join(istar_dir, f"{gene}.pickle")

    if not os.path.exists(istar_path):
        return None, f"missing:{istar_path}"

    arr = load_model_output_original(istar_path, gt_shape, mask=gt_mask)

    maybe_save_cache(cache_dir, "ISTAR", gene, arr, use_cache)

    return arr, istar_path


def load_storm_processed_original(gene, storm_dir, gt_shape, gt_mask, cache_dir=None, use_cache=False):
    cached = maybe_load_cache(cache_dir, "SQUALL", gene, gt_shape, use_cache)
    if cached is not None:
        return cached, "cache"

    storm_path = os.path.join(storm_dir, f"{gene}_expression.npy")

    if not os.path.exists(storm_path):
        return None, f"missing:{storm_path}"

    arr = load_model_output_original(storm_path, gt_shape, mask=gt_mask)

    maybe_save_cache(cache_dir, "SQUALL", gene, arr, use_cache)

    return arr, storm_path


# ======================================================
# Parallel worker for STNET/iSTAR/SQUALL
# ======================================================

def compute_one_gene_file_models_worker(args_tuple):
    (
        gene,
        gene_idx,
        gt_arr_flat,
        gt_shape,
        gt_mask_flat,
        stnet_dir,
        istar_dir,
        storm_dir,
        cache_dir,
        use_cache,
    ) = args_tuple

    gt_arr = gt_arr_flat.reshape(gt_shape).astype(np.float32)
    gt_mask = gt_mask_flat.reshape(gt_shape)

    metrics = {
        "Gene": gene,
        "_gene_idx": gene_idx,
    }

    for model_name in ["STNET", "ISTAR", "SQUALL"]:
        metrics[f"{model_name}_Pearson"] = np.nan
        metrics[f"{model_name}_SSIM"] = np.nan
        metrics[f"{model_name}_RMSD"] = np.nan

    try:
        stnet_arr, status = load_stnet_processed_original(
            gene, stnet_dir, gt_shape, gt_mask, cache_dir, use_cache
        )
        if stnet_arr is not None:
            p, s, r = compute_metrics(stnet_arr, gt_arr)
            metrics["STNET_Pearson"] = p
            metrics["STNET_SSIM"] = s
            metrics["STNET_RMSD"] = r
        else:
            metrics["STNET_status"] = status
    except Exception as e:
        metrics["STNET_status"] = f"error:{repr(e)}"

    try:
        istar_arr, status = load_istar_processed_original(
            gene, istar_dir, gt_shape, gt_mask, cache_dir, use_cache
        )
        if istar_arr is not None:
            p, s, r = compute_metrics(istar_arr, gt_arr)
            metrics["ISTAR_Pearson"] = p
            metrics["ISTAR_SSIM"] = s
            metrics["ISTAR_RMSD"] = r
        else:
            metrics["ISTAR_status"] = status
    except Exception as e:
        metrics["ISTAR_status"] = f"error:{repr(e)}"

    try:
        storm_arr, status = load_storm_processed_original(
            gene, storm_dir, gt_shape, gt_mask, cache_dir, use_cache
        )
        if storm_arr is not None:
            p, s, r = compute_metrics(storm_arr, gt_arr)
            metrics["SQUALL_Pearson"] = p
            metrics["SQUALL_SSIM"] = s
            metrics["SQUALL_RMSD"] = r
        else:
            metrics["SQUALL_status"] = status
    except Exception as e:
        metrics["SQUALL_status"] = f"error:{repr(e)}"

    return metrics


def compute_file_models_metrics_parallel(
    genes,
    gt_q_matrix,
    gt_shape,
    gt_mask_flat,
    stnet_dir,
    istar_dir,
    storm_dir,
    cache_dir,
    use_cache,
    num_workers,
):
    tasks = []

    for j, gene in enumerate(genes):
        tasks.append(
            (
                gene,
                j,
                gt_q_matrix[:, j].astype(np.float32),
                gt_shape,
                gt_mask_flat.astype(bool),
                stnet_dir,
                istar_dir,
                storm_dir,
                cache_dir,
                use_cache,
            )
        )

    rows = []

    if num_workers <= 1:
        for t in tqdm(tasks, desc="File models STNET/iSTAR/SQUALL"):
            rows.append(compute_one_gene_file_models_worker(t))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(compute_one_gene_file_models_worker, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="File models STNET/iSTAR/SQUALL parallel"):
                rows.append(fut.result())

    df = pd.DataFrame(rows)
    df = df.sort_values("_gene_idx").drop(columns=["_gene_idx"])

    return df


# ======================================================
# Generic H5 helpers for EGN / Path2Space / Hist2ST / DeepPT
# ======================================================

def inspect_h5(h5_path):
    print(f"\n[Inspect h5] {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                print(f"  dataset: {k}, shape={obj.shape}, dtype={obj.dtype}")
        for k, v in f.attrs.items():
            print(f"  attr: {k}={v}")


def choose_prediction_key(h5_path, preferred_key=None):
    with h5py.File(h5_path, "r") as f:
        all_keys = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_keys.append(name)

        f.visititems(visitor)

        if preferred_key is not None:
            if preferred_key not in all_keys:
                raise KeyError(
                    f"{preferred_key} not found in {h5_path}. "
                    f"Available datasets: {all_keys}"
                )
            return preferred_key

        for k in PREFERRED_H5_KEYS:
            if k in all_keys:
                return k

        for k in all_keys:
            ds = f[k]
            if len(ds.shape) == 2 and np.issubdtype(ds.dtype, np.number):
                return k

    raise KeyError(f"No 2D numeric prediction dataset found in {h5_path}")


def load_generic_h5_meta(h5_path, pred_key=None, name="H5Model"):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Missing h5: {h5_path}")

    inspect_h5(h5_path)

    key = choose_prediction_key(h5_path, preferred_key=pred_key)

    with h5py.File(h5_path, "r") as f:
        pred_shape = f[key].shape

        if "genes" not in f:
            raise KeyError(f"{h5_path} does not contain dataset: genes")
        genes = decode_array(f["genes"][:])

        coords = None
        coords_key = None
        # Different models use different names for spatial/tile coordinates.
        # EGN/Hist2ST usually use "coords"; DeepPT uses "tile_coords".
        for cand in ["coords", "tile_coords", "coord", "coordinates"]:
            if cand in f:
                coords = f[cand][:].astype(np.float32)
                coords_key = cand
                break

        tile_x = None
        tile_y = None
        if "tile_x" in f and "tile_y" in f:
            tile_x = f["tile_x"][:].astype(np.int64)
            tile_y = f["tile_y"][:].astype(np.int64)

        tile_id = None
        if "tile_id" in f:
            tile_id = decode_array(f["tile_id"][:])

    if pred_shape[1] != len(genes):
        raise ValueError(
            f"{h5_path}: pred cols {pred_shape[1]} != genes {len(genes)}"
        )

    if coords is not None and pred_shape[0] != coords.shape[0]:
        raise ValueError(
            f"{h5_path}: pred rows {pred_shape[0]} != coords rows {coords.shape[0]}"
        )

    if tile_x is not None and pred_shape[0] != len(tile_x):
        raise ValueError(
            f"{h5_path}: pred rows {pred_shape[0]} != tile_x rows {len(tile_x)}"
        )

    if tile_id is not None and pred_shape[0] != len(tile_id):
        raise ValueError(
            f"{h5_path}: pred rows {pred_shape[0]} != tile_id rows {len(tile_id)}"
        )

    print(f"[Load {name} meta]")
    print(f"  h5       : {h5_path}")
    print(f"  pred_key : {key}")
    print(f"  pred     : {pred_shape}")
    print(f"  genes    : {len(genes)}")
    if coords is not None:
        if coords_key is None:
            print(f"  coords   : {coords.shape}")
        else:
            print(f"  coords   : {coords.shape}  from dataset '{coords_key}'")
    if tile_x is not None:
        print(f"  tile_x/y : {tile_x.shape}, {tile_y.shape}")
    if tile_id is not None:
        print(f"  tile_id  : {len(tile_id)}")

    return {
        "name": name,
        "h5_path": h5_path,
        "pred_key": key,
        "pred_shape": pred_shape,
        "coords": coords,
        "tile_x": tile_x,
        "tile_y": tile_y,
        "tile_id": tile_id,
        "genes": genes,
        "gene_to_col": {g: i for i, g in enumerate(genes)},
    }


def build_coord_to_tile_mapping(
    coords,
    gt_shape,
    gt_mask_flat,
    tile_size=224.0,
    coord_scale=1.0,
    x_offset=0.0,
    y_offset=0.0,
    swap_xy=False,
    flip_x=False,
    flip_y=False,
):
    coords = np.asarray(coords, dtype=np.float32).copy()

    if swap_xy:
        coords = coords[:, [1, 0]]

    x = coords[:, 0] * float(coord_scale)
    y = coords[:, 1] * float(coord_scale)

    n_y, n_x = gt_shape
    n_tiles = n_y * n_x

    tile_x_raw = np.floor((x - float(x_offset)) / float(tile_size)).astype(np.int64)
    tile_y_raw = np.floor((y - float(y_offset)) / float(tile_size)).astype(np.int64)

    if flip_x:
        tile_x = (n_x - 1) - tile_x_raw
    else:
        tile_x = tile_x_raw

    if flip_y:
        tile_y = (n_y - 1) - tile_y_raw
    else:
        tile_y = tile_y_raw

    in_bound = (
        (tile_x >= 0) & (tile_x < n_x) &
        (tile_y >= 0) & (tile_y < n_y)
    )

    tile_flat_all = tile_y * n_x + tile_x

    keep = np.zeros(coords.shape[0], dtype=bool)
    valid_rows = np.where(in_bound)[0]
    keep[valid_rows] = gt_mask_flat[tile_flat_all[valid_rows]]

    row_idx = np.where(keep)[0].astype(np.int64)
    tile_flat = tile_flat_all[row_idx].astype(np.int64)

    tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)

    print("[Coord -> tile mapping]")
    print(f"  total rows           : {coords.shape[0]}")
    print(f"  in-bound rows        : {int(in_bound.sum())}")
    print(f"  on GT valid tile rows: {int(keep.sum())}")
    print(f"  GT tiles with >=1 row: {int((tile_counts > 0).sum())} / {int(gt_mask_flat.sum())}")

    if row_idx.size == 0:
        raise RuntimeError("No rows mapped to GT tiles. Check mapping params.")

    return {
        "row_idx": row_idx,
        "tile_flat": tile_flat,
        "tile_counts": tile_counts,
    }


def build_tilexy_to_tile_mapping(tile_x, tile_y, gt_shape, gt_mask_flat):
    """
    Path2Space preferred mapping:
        h5 has tile_x/tile_y directly.
    """
    tile_x = np.asarray(tile_x, dtype=np.int64)
    tile_y = np.asarray(tile_y, dtype=np.int64)

    n_y, n_x = gt_shape
    n_tiles = n_y * n_x

    in_bound = (
        (tile_x >= 0) & (tile_x < n_x) &
        (tile_y >= 0) & (tile_y < n_y)
    )

    tile_flat_all = tile_y * n_x + tile_x

    keep = np.zeros(tile_x.shape[0], dtype=bool)
    valid_rows = np.where(in_bound)[0]
    keep[valid_rows] = gt_mask_flat[tile_flat_all[valid_rows]]

    row_idx = np.where(keep)[0].astype(np.int64)
    tile_flat = tile_flat_all[row_idx].astype(np.int64)

    tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)

    print("[tile_x/tile_y -> GT tile mapping]")
    print(f"  total rows           : {tile_x.shape[0]}")
    print(f"  in-bound rows        : {int(in_bound.sum())}")
    print(f"  on GT valid tile rows: {int(keep.sum())}")
    print(f"  GT tiles with >=1 row: {int((tile_counts > 0).sum())} / {int(gt_mask_flat.sum())}")

    if row_idx.size == 0:
        raise RuntimeError("No tile_x/tile_y rows mapped to GT tiles. Check gt_shape/tile index.")

    return {
        "row_idx": row_idx,
        "tile_flat": tile_flat,
        "tile_counts": tile_counts,
    }


def build_tileid_to_tile_mapping(tile_id, gt_shape, gt_mask_flat):
    """
    Generic fallback for h5 files that store tile IDs as strings.
    It parses patterns such as:
        posX_12_posY_34, x_12_y_34, x12_y34, X_12_Y_34
    """
    n_y, n_x = gt_shape
    n_tiles = n_y * n_x

    tile_x = np.full(len(tile_id), -1, dtype=np.int64)
    tile_y = np.full(len(tile_id), -1, dtype=np.int64)

    parsed = 0
    for i, tid in enumerate(tile_id):
        xy = parse_pos_from_string(tid)
        if xy is not None:
            tile_x[i], tile_y[i] = xy
            parsed += 1

    in_bound = (
        (tile_x >= 0) & (tile_x < n_x) &
        (tile_y >= 0) & (tile_y < n_y)
    )

    tile_flat_all = tile_y * n_x + tile_x

    keep = np.zeros(len(tile_id), dtype=bool)
    valid_rows = np.where(in_bound)[0]
    keep[valid_rows] = gt_mask_flat[tile_flat_all[valid_rows]]

    row_idx = np.where(keep)[0].astype(np.int64)
    tile_flat = tile_flat_all[row_idx].astype(np.int64)
    tile_counts = np.bincount(tile_flat, minlength=n_tiles).astype(np.int64)

    print("[tile_id -> GT tile mapping]")
    print(f"  total rows           : {len(tile_id)}")
    print(f"  parsed rows          : {parsed}")
    print(f"  in-bound rows        : {int(in_bound.sum())}")
    print(f"  on GT valid tile rows: {int(keep.sum())}")
    print(f"  GT tiles with >=1 row: {int((tile_counts > 0).sum())} / {int(gt_mask_flat.sum())}")

    if row_idx.size == 0:
        raise RuntimeError("No tile_id rows mapped to GT tiles. Check tile_id format.")

    return {
        "row_idx": row_idx,
        "tile_flat": tile_flat,
        "tile_counts": tile_counts,
    }


def infer_h5_mapping(
    h5_obj,
    gt_shape,
    gt_mask_flat,
    tile_size=224.0,
    coord_scale=1.0,
    x_offset=0.0,
    y_offset=0.0,
    swap_xy=False,
    flip_x=False,
    flip_y=False,
    force_coords=False,
):
    """
    Mapping priority for all h5 models:
      1. tile_x/tile_y, unless force_coords=True
      2. tile_id string parsing
      3. coords -> grid mapping
    """
    name = h5_obj.get("name", "H5Model")

    if (not force_coords) and h5_obj.get("tile_x") is not None and h5_obj.get("tile_y") is not None:
        print(f"[{name}] use tile_x/tile_y mapping")
        return build_tilexy_to_tile_mapping(
            tile_x=h5_obj["tile_x"],
            tile_y=h5_obj["tile_y"],
            gt_shape=gt_shape,
            gt_mask_flat=gt_mask_flat,
        )

    if (not force_coords) and h5_obj.get("tile_id") is not None:
        try:
            print(f"[{name}] use tile_id mapping")
            return build_tileid_to_tile_mapping(
                tile_id=h5_obj["tile_id"],
                gt_shape=gt_shape,
                gt_mask_flat=gt_mask_flat,
            )
        except Exception as e:
            print(f"[WARNING] {name} tile_id mapping failed: {repr(e)}; fallback to coords mapping.")

    if h5_obj.get("coords") is None:
        raise RuntimeError(
            f"{name} h5 has no usable mapping fields. Need one of: tile_x/tile_y, tile_id, coords/tile_coords."
        )

    print(f"[{name}] use coords mapping")
    return build_coord_to_tile_mapping(
        coords=h5_obj["coords"],
        gt_shape=gt_shape,
        gt_mask_flat=gt_mask_flat,
        tile_size=tile_size,
        coord_scale=coord_scale,
        x_offset=x_offset,
        y_offset=y_offset,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
    )


def aggregate_h5_columns_to_tiles_fast(h5_path, pred_key, h5_cols, mapping, gt_shape):
    h5_cols = np.asarray(h5_cols, dtype=np.int64)

    n_tiles = gt_shape[0] * gt_shape[1]
    B = len(h5_cols)

    if B == 0:
        return np.zeros((n_tiles, 0), dtype=np.float32)

    row_idx = mapping["row_idx"]
    tile_flat = mapping["tile_flat"]
    tile_counts = mapping["tile_counts"]

    order = np.argsort(h5_cols)
    sorted_cols = h5_cols[order]

    c0 = int(sorted_cols[0])
    c1 = int(sorted_cols[-1]) + 1

    with h5py.File(h5_path, "r") as f:
        ds = f[pred_key]
        block_all = ds[:, c0:c1].astype(np.float32)
        block = block_all[row_idx, :]

    local_cols = sorted_cols - c0
    block = block[:, local_cols]

    inv_order = np.argsort(order)
    block = block[:, inv_order]

    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    tile_sum = np.zeros((n_tiles, B), dtype=np.float32)
    np.add.at(tile_sum, tile_flat, block)

    pred_tile = np.zeros((n_tiles, B), dtype=np.float32)

    nonzero = tile_counts > 0
    pred_tile[nonzero, :] = tile_sum[nonzero, :] / tile_counts[nonzero, None]

    return pred_tile.astype(np.float32)


def process_h5_matrix_like_original(pred_tile_matrix, gt_mask_flat, gt_shape):
    n_tiles, B = pred_tile_matrix.shape
    gt_mask = gt_mask_flat.reshape(gt_shape)

    out = np.zeros((n_tiles, B), dtype=np.float32)

    for j in range(B):
        arr = pred_tile_matrix[:, j].reshape(gt_shape).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[~gt_mask] = 0
        arr = quantile_normalize(arr, gt_mask)
        out[:, j] = arr.reshape(-1)

    return out.astype(np.float32)


def compute_h5_tile_model_metrics_fast(
    prefix,
    h5_obj,
    mapping,
    genes,
    gt_q_matrix,
    gt_shape,
    gt_mask_flat,
    chunk_size=128,
):
    n_genes = len(genes)

    out = {
        f"{prefix}_Pearson": np.full(n_genes, np.nan, dtype=np.float32),
        f"{prefix}_SSIM": np.full(n_genes, np.nan, dtype=np.float32),
        f"{prefix}_RMSD": np.full(n_genes, np.nan, dtype=np.float32),
    }

    present = []
    missing = []

    gene_to_col = h5_obj["gene_to_col"]

    for local_i, g in enumerate(genes):
        if g in gene_to_col:
            present.append((local_i, g, gene_to_col[g]))
        else:
            missing.append(g)

    print(f"[{prefix}] genes present in h5: {len(present)} / {len(genes)}")
    print(f"[{prefix}] genes missing in h5: {len(missing)}")

    present = sorted(present, key=lambda x: x[2])

    for start in tqdm(range(0, len(present), chunk_size), desc=f"{prefix} chunks"):
        end = min(start + chunk_size, len(present))
        chunk = present[start:end]

        local_indices = [x[0] for x in chunk]
        h5_cols = [x[2] for x in chunk]

        pred_tile_chunk = aggregate_h5_columns_to_tiles_fast(
            h5_path=h5_obj["h5_path"],
            pred_key=h5_obj["pred_key"],
            h5_cols=h5_cols,
            mapping=mapping,
            gt_shape=gt_shape,
        )

        pred_q_chunk = process_h5_matrix_like_original(
            pred_tile_matrix=pred_tile_chunk,
            gt_mask_flat=gt_mask_flat,
            gt_shape=gt_shape,
        )

        for jj, local_i in enumerate(local_indices):
            pred_arr = pred_q_chunk[:, jj].reshape(gt_shape)
            gt_arr = gt_q_matrix[:, local_i].reshape(gt_shape)

            p, s, r = compute_metrics(pred_arr, gt_arr)

            out[f"{prefix}_Pearson"][local_i] = p
            out[f"{prefix}_SSIM"][local_i] = s
            out[f"{prefix}_RMSD"][local_i] = r

    df = pd.DataFrame({"Gene": genes})

    for k, v in out.items():
        df[k] = v

    return df


# ======================================================
# Summary
# ======================================================

def build_overall_summary(metrics_df):
    rows = []

    models = ["STNET", "ISTAR", "EGNv1", "Path2Space", "Hist2ST", "DeepPT", "SQUALL"]
    metrics = ["Pearson", "SSIM", "RMSD"]

    for model in models:
        row = {"Model": model}

        for metric in metrics:
            col = f"{model}_{metric}"

            if col not in metrics_df.columns:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_std"] = np.nan
                row[f"{metric}_valid_n"] = 0
                continue

            values = pd.to_numeric(metrics_df[col], errors="coerce").values
            valid = np.isfinite(values)

            row[f"{metric}_mean"] = np.nanmean(values) if valid.sum() > 0 else np.nan
            row[f"{metric}_median"] = np.nanmedian(values) if valid.sum() > 0 else np.nan
            row[f"{metric}_std"] = np.nanstd(values) if valid.sum() > 0 else np.nan
            row[f"{metric}_valid_n"] = int(valid.sum())

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================
# Main
# ======================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stnet_dir", type=str, default=DEFAULT_STNET_DIR)
    parser.add_argument("--storm_dir", type=str, default=DEFAULT_SQUALL_DIR)
    parser.add_argument("--istar_dir", type=str, default=DEFAULT_ISTAR_DIR)

    parser.add_argument("--egnv1_h5", type=str, default=DEFAULT_EGNV1_H5)
    parser.add_argument("--path2space_h5", type=str, default=DEFAULT_PATH2SPACE_H5)
    parser.add_argument("--hist2st_h5", type=str, default=DEFAULT_HIST2ST_H5)
    parser.add_argument("--deeppt_h5", type=str, default=DEFAULT_DEEPPT_H5)

    parser.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR)
    parser.add_argument("--valid_genes", type=str, default=DEFAULT_VALID_GENES_PATH)

    parser.add_argument("--out_csv", type=str, default="gene_metrics_summary_7models_no_EGNv2_add_Hist2ST_DeepPT.csv")
    parser.add_argument("--overall_csv", type=str, default="gene_metrics_summary_7models_no_EGNv2_add_Hist2ST_DeepPT_overall.csv")

    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--pred_key", type=str, default=None)
    parser.add_argument("--egnv1_pred_key", type=str, default=None)
    parser.add_argument("--path2space_pred_key", type=str, default=None)
    parser.add_argument("--hist2st_pred_key", type=str, default=None)
    parser.add_argument("--deeppt_pred_key", type=str, default=None)

    parser.add_argument("--tile_size", type=float, default=224.0)
    parser.add_argument("--coord_scale", type=float, default=1.0)
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--y_offset", type=float, default=0.0)
    parser.add_argument("--swap_xy", action="store_true")
    parser.add_argument("--flip_x", action="store_true")
    parser.add_argument("--flip_y", action="store_true")

    parser.add_argument(
        "--force_coords_mapping",
        action="store_true",
        help="Force all h5 models to use coords mapping instead of tile_x/tile_y or tile_id.",
    )

    parser.add_argument(
        "--only_genes",
        type=str,
        default=None,
        help="Optional comma-separated gene list, e.g. AKT1,CDK1,EPCAM",
    )

    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        help="1: cache processed STNET/ISTAR/SQUALL arrays; 0: no cache",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="metric_fast_cache_processed_arrays_same_logic",
    )

    args = parser.parse_args()

    use_cache = bool(args.use_cache)

    if use_cache:
        ensure_dir(args.cache_dir)

    # --------------------------------------------------
    # Genes
    # --------------------------------------------------
    with open(args.valid_genes) as f:
        valid_genes = json.load(f)

    valid_gene_to_idx = {g: i for i, g in enumerate(valid_genes)}

    if args.only_genes is not None:
        selected = [x.strip() for x in args.only_genes.split(",") if x.strip()]
        valid_set = set(valid_genes)
        genes = [g for g in selected if g in valid_set]
        missing = [g for g in selected if g not in valid_set]
        if len(missing) > 0:
            print(f"[WARNING] selected genes not in valid_genes: {missing}")
    else:
        genes = list(valid_genes)

    if len(genes) == 0:
        raise RuntimeError("No genes to process.")

    print(f"[Genes] {len(genes)}")

    # --------------------------------------------------
    # GT once
    # --------------------------------------------------
    gt_files, file_to_xy, gt_shape, gt_mask_flat_from_files = infer_gt_files_and_shape(args.gt_dir)

    gt_raw_matrix, gt_mask_flat_loaded = build_gt_matrix_once(
        gt_dir=args.gt_dir,
        gt_files=gt_files,
        file_to_xy=file_to_xy,
        gt_shape=gt_shape,
        genes=genes,
        valid_gene_to_idx=valid_gene_to_idx,
    )

    gt_mask_flat = gt_mask_flat_from_files & gt_mask_flat_loaded

    print("[Common GT]")
    print(f"  gt_shape       : {gt_shape}")
    print(f"  valid gt tiles : {int(gt_mask_flat.sum())}")

    gt_q_matrix = build_gt_quantile_matrix(
        gt_raw_matrix=gt_raw_matrix,
        gt_mask_flat=gt_mask_flat,
        gt_shape=gt_shape,
        genes=genes,
    )

    # --------------------------------------------------
    # File models: STNET / ISTAR / SQUALL in parallel
    # --------------------------------------------------
    file_df = compute_file_models_metrics_parallel(
        genes=genes,
        gt_q_matrix=gt_q_matrix,
        gt_shape=gt_shape,
        gt_mask_flat=gt_mask_flat,
        stnet_dir=args.stnet_dir,
        istar_dir=args.istar_dir,
        storm_dir=args.storm_dir,
        cache_dir=args.cache_dir,
        use_cache=use_cache,
        num_workers=args.num_workers,
    )

    partial_file_csv = args.out_csv.replace(".csv", ".partial_file_models.csv")
    file_df.to_csv(partial_file_csv, index=False)
    print(f"[Saved partial file models] {partial_file_csv}")

    # --------------------------------------------------
    # H5 tile models: EGNv1 / Path2Space / Hist2ST / DeepPT
    # EGNv2 is intentionally removed.
    # --------------------------------------------------
    h5_model_specs = [
        ("EGNv1", args.egnv1_h5, args.egnv1_pred_key if args.egnv1_pred_key is not None else args.pred_key),
        ("Path2Space", args.path2space_h5, args.path2space_pred_key),
        ("Hist2ST", args.hist2st_h5, args.hist2st_pred_key),
        ("DeepPT", args.deeppt_h5, args.deeppt_pred_key),
    ]

    h5_dfs = {}
    running_df = file_df.copy()

    for model_name, h5_path, pred_key in h5_model_specs:
        print("\n====================================")
        print(f"[H5 model] {model_name}")
        print("====================================")

        h5_obj = load_generic_h5_meta(
            h5_path,
            pred_key=pred_key,
            name=model_name,
        )

        mapping = infer_h5_mapping(
            h5_obj=h5_obj,
            gt_shape=gt_shape,
            gt_mask_flat=gt_mask_flat,
            tile_size=args.tile_size,
            coord_scale=args.coord_scale,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            swap_xy=args.swap_xy,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            force_coords=args.force_coords_mapping,
        )

        model_df = compute_h5_tile_model_metrics_fast(
            prefix=model_name,
            h5_obj=h5_obj,
            mapping=mapping,
            genes=genes,
            gt_q_matrix=gt_q_matrix,
            gt_shape=gt_shape,
            gt_mask_flat=gt_mask_flat,
            chunk_size=args.chunk_size,
        )

        h5_dfs[model_name] = model_df
        running_df = running_df.merge(model_df, on="Gene", how="left")

        partial_csv = args.out_csv.replace(".csv", f".partial_with_{model_name}.csv")
        running_df.to_csv(partial_csv, index=False)
        print(f"[Saved partial {model_name}] {partial_csv}")

    # --------------------------------------------------
    # Merge and save
    # --------------------------------------------------
    metrics_df = running_df.copy()

    preferred_cols = ["Gene"]

    for model in ["STNET", "ISTAR", "EGNv1", "Path2Space", "Hist2ST", "DeepPT", "SQUALL"]:
        for metric in ["Pearson", "SSIM", "RMSD"]:
            col = f"{model}_{metric}"
            if col in metrics_df.columns:
                preferred_cols.append(col)

    other_cols = [c for c in metrics_df.columns if c not in preferred_cols]
    metrics_df = metrics_df[preferred_cols + other_cols]

    metrics_df.to_csv(args.out_csv, index=False)
    print(f"[Saved metrics] {args.out_csv}")

    overall_df = build_overall_summary(metrics_df)
    overall_df.to_csv(args.overall_csv, index=False)

    print("\n====================================")
    print("[Overall summary]")
    print("====================================")
    with pd.option_context("display.max_rows", 100, "display.max_columns", 100, "display.width", 260):
        print(overall_df)

    print("\n====================================")
    print("[Done]")
    print(f"Metrics CSV : {args.out_csv}")
    print(f"Overall CSV : {args.overall_csv}")
    if use_cache:
        print(f"Cache dir   : {args.cache_dir}")
    print("Models      : STNET | ISTAR | EGNv1 | Path2Space | Hist2ST | DeepPT | SQUALL")
    print("====================================")


if __name__ == "__main__":
    main()
