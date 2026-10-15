#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tile-level Path2Space-style training + external test prediction.

Main difference from spot-level version:
  - Instead of extracting one CTransPath feature for every spot,
    this script bins spots into image tiles.
  - Each tile gets one CTransPath feature.
  - Train label for each tile = mean log1p(CPM_1e4) expression of spots inside that tile.
  - Test output is tile-level prediction.

Input folder format:
  sample/
    he-raw.jpg
    cnts.mtx
    genes.tsv
    barcodes.tsv
    locs-raw.tsv   # must contain x, y columns

Default:
  --tile_size 224
  --tile_stride 224
means non-overlapping 224x224 image tiles.
"""

import os
import sys
import json
import h5py
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import io as spio
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from sklearn.model_selection import KFold
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
    }


def normalize_log_cpm(X, scale_factor=1e4):
    X = X.astype(np.float32).tocsr()
    libsize = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    libsize[libsize <= 0] = 1.0
    scale = scale_factor / libsize
    X = X.multiply(scale[:, None]).tocsr()
    X.data = np.log1p(X.data)
    return X


# ============================================================
# CTransPath and Macenko
# ============================================================

def import_ctranspath_and_macenko(path2space_feature_dir, use_macenko=True):
    """
    Import:
      - CTransPath architecture from func.ctrans_model
      - Macenko normalizer from func.utils_color_norm only when use_macenko=True

    This script does not call Path2Space tiles2features().
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

    coords: original x/y coordinates.
    coord_scale: scale applied to coordinates before grouping and patch extraction.

    Return:
      tile_df with one row per tile:
        tile_id
        tile_x
        tile_y
        center_x
        center_y
        n_spots
        spot_indices
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


def build_tile_expression_matrix(
    sample,
    X_log,
    selected_genes,
    tile_df,
):
    """
    Aggregate spot-level expression to tile-level expression.

    For each tile:
      Y_tile = mean of log-normalized expression of spots inside tile.

    Return:
      Y_tile: [n_tiles, n_genes]
    """
    gene_to_idx = {g: i for i, g in enumerate(sample["genes"])}
    selected_idx = [gene_to_idx[g] for g in selected_genes]

    X_sel = X_log[:, selected_idx].tocsr()

    Ys = []
    for spot_indices in tqdm(tile_df["spot_indices"], desc=f"Aggregate expr {sample['sample']}"):
        Y = X_sel[spot_indices].mean(axis=0)
        Y = np.asarray(Y).ravel().astype(np.float32)
        Ys.append(Y)

    Y_tile = np.vstack(Ys).astype(np.float32)
    return Y_tile


def build_tile_level_train_data(
    train_samples,
    train_X_logs,
    selected_genes,
    tile_size=224,
    tile_stride=224,
    coord_scale=1.0,
    min_spots_per_tile=1,
):
    """
    Build tile-level training expression matrix and metadata.
    """
    Y_list = []
    meta_list = []

    for sample, X_log in zip(train_samples, train_X_logs):
        print(f"[Build train tiles] {sample['sample']}")

        tile_df = build_tile_table_from_coords(
            coords=sample["coords"],
            tile_size=tile_size,
            tile_stride=tile_stride,
            coord_scale=coord_scale,
            min_spots_per_tile=min_spots_per_tile,
        )

        Y_tile = build_tile_expression_matrix(
            sample=sample,
            X_log=X_log,
            selected_genes=selected_genes,
            tile_df=tile_df,
        )

        meta = tile_df.drop(columns=["spot_indices"]).copy()
        meta.insert(0, "sample", sample["sample"])

        Y_list.append(Y_tile)
        meta_list.append(meta)

        print(f"  n_tiles={tile_df.shape[0]}, Y_tile={Y_tile.shape}")

    Y_train_tile = np.concatenate(Y_list, axis=0).astype(np.float32)
    train_tile_meta = pd.concat(meta_list, axis=0, ignore_index=True)

    return Y_train_tile, train_tile_meta


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
    split_name="train",
):
    """
    Extract one CTransPath feature per tile.

    tile_df must contain:
      center_x, center_y, tile_id
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
        },
        cache_path,
    )

    print(f"[Saved tile feature cache] {cache_path}")
    print(f"[Tile feature shape] {sample_name}: {tuple(features.shape)}")

    return features


# ============================================================
# Gene selection
# ============================================================

def select_genes_across_train_and_test(
    train_samples,
    test_samples,
    train_X_logs,
    n_genes=1000,
    gene_list=None,
    ignore_test_gene_filter=False,
):
    train_gene_sets = [set(s["genes"]) for s in train_samples]
    common = set.intersection(*train_gene_sets)

    if not ignore_test_gene_filter:
        test_gene_sets = [set(s["genes"]) for s in test_samples]
        common = common.intersection(set.intersection(*test_gene_sets))

    common = sorted(list(common))

    if len(common) == 0:
        raise ValueError("No common genes found.")

    if gene_list is not None:
        wanted = pd.read_csv(gene_list, header=None).iloc[:, 0].astype(str).tolist()
        selected = [g for g in wanted if g in common]
        if len(selected) == 0:
            raise ValueError("No genes from gene_list exist in common gene set.")
        return selected

    print(f"[Gene selection] common genes = {len(common)}")

    total_n = 0
    total_sum = np.zeros(len(common), dtype=np.float64)
    total_sumsq = np.zeros(len(common), dtype=np.float64)

    for sample, Xlog in zip(train_samples, train_X_logs):
        gene_to_idx = {g: i for i, g in enumerate(sample["genes"])}
        idx = [gene_to_idx[g] for g in common]

        Xc = Xlog[:, idx].tocsr()
        n = Xc.shape[0]

        total_n += n
        total_sum += np.asarray(Xc.sum(axis=0)).ravel()
        total_sumsq += np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel()

    mean = total_sum / max(total_n, 1)
    mean_sq = total_sumsq / max(total_n, 1)
    var = mean_sq - mean ** 2

    if n_genes is None or n_genes <= 0 or n_genes > len(common):
        n_genes = len(common)

    top = np.argsort(-var)[:n_genes]
    selected = [common[i] for i in top]

    return selected


# ============================================================
# MLP
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


def pearson_by_gene_np(y_true, y_pred, eps=1e-8):
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)

    num = np.sum(yt * yp, axis=0)
    den = np.sqrt(np.sum(yt ** 2, axis=0) * np.sum(yp ** 2, axis=0)) + eps

    r = num / den
    r = r[np.isfinite(r)]

    if len(r) == 0:
        return np.nan

    return float(np.mean(r))


def make_nested_cv_splits(n_samples, n_outer=5, n_inner=5, seed=42):
    indices = np.arange(n_samples)

    if n_outer <= 1:
        outer_splits = [(indices, np.array([], dtype=int))]
    else:
        outer_kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        outer_splits = []
        for train_valid_idx, outer_test_idx in outer_kf.split(indices):
            outer_splits.append((train_valid_idx, outer_test_idx))

    all_splits = []

    for ik, (train_valid_idx, outer_test_idx) in enumerate(outer_splits):
        train_valid_idx = np.asarray(train_valid_idx)

        if n_inner <= 1:
            rng = np.random.RandomState(seed + ik)
            shuffled = train_valid_idx.copy()
            rng.shuffle(shuffled)

            n_valid = max(1, int(0.1 * len(shuffled)))
            valid_idx = shuffled[:n_valid]
            train_idx = shuffled[n_valid:]

            all_splits.append({
                "ik": ik,
                "il": 0,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "outer_test_idx": outer_test_idx,
            })
        else:
            inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=seed + ik)
            for il, (inner_train_pos, valid_pos) in enumerate(inner_kf.split(train_valid_idx)):
                train_idx = train_valid_idx[inner_train_pos]
                valid_idx = train_valid_idx[valid_pos]

                all_splits.append({
                    "ik": ik,
                    "il": il,
                    "train_idx": train_idx,
                    "valid_idx": valid_idx,
                    "outer_test_idx": outer_test_idx,
                })

    return all_splits


def train_one_fold(
    train_features,
    Y_train,
    train_idx,
    valid_idx,
    result_dir,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    hidden=768,
    dropout=0.2,
    patience=None,
    num_workers=4,
    device="cuda",
    output_relu=True,
):
    os.makedirs(result_dir, exist_ok=True)

    if patience is None:
        patience = max(5, epochs // 10)

    n_inputs = train_features.size(1)
    n_outputs = Y_train.shape[1]

    full_ds = TileFeatureDataset(train_features, Y_train)

    train_ds = Subset(full_ds, train_idx.tolist())
    valid_ds = Subset(full_ds, valid_idx.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    bias_init = Y_train[train_idx].mean(axis=0).astype(np.float32)

    model = MLPRegressionReluTwo(
        n_inputs=n_inputs,
        n_hiddens=hidden,
        n_outputs=n_outputs,
        dropout=dropout,
        bias_init=bias_init,
        output_relu=output_relu,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    best_path = os.path.join(result_dir, "model_trained.pth")
    fold_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(loss.item())

        model.eval()
        valid_losses = []
        valid_true = []
        valid_pred = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                valid_losses.append(loss.item())
                valid_true.append(yb.detach().cpu().numpy())
                valid_pred.append(pred.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))

        valid_true = np.concatenate(valid_true, axis=0)
        valid_pred = np.concatenate(valid_pred, axis=0)
        valid_gene_pearson = pearson_by_gene_np(valid_true, valid_pred)

        print(
            f"[{os.path.basename(result_dir)}][Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}, "
            f"valid_loss={valid_loss:.6f}, "
            f"valid_gene_pearson={valid_gene_pearson:.4f}"
        )

        fold_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_gene_pearson": valid_gene_pearson,
        })

        if valid_loss < best_val:
            best_val = valid_loss
            best_epoch = epoch
            bad_epochs = 0

            torch.save(model.state_dict(), best_path)

            meta = {
                "n_inputs": n_inputs,
                "hidden": hidden,
                "n_outputs": n_outputs,
                "dropout": dropout,
                "output_relu": output_relu,
                "best_epoch": best_epoch,
                "best_valid_loss": best_val,
            }
            with open(os.path.join(result_dir, "model_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  [Saved best] {best_path}")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(
                f"[Early stop] {os.path.basename(result_dir)} "
                f"best_epoch={best_epoch}, best_valid_loss={best_val:.6f}"
            )
            break

    fold_log = pd.DataFrame(fold_log)
    fold_log.to_csv(os.path.join(result_dir, "train_log.csv"), index=False)

    return {
        "best_path": best_path,
        "best_epoch": best_epoch,
        "best_valid_loss": best_val,
        "log": fold_log,
    }


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

    print(f"[Saved h5] {out_h5}")


def save_prediction_csv(out_csv, pred_mean, genes, tile_meta, sample_name):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = pd.DataFrame(pred_mean, columns=genes)
    meta = tile_meta.drop(columns=["spot_indices"], errors="ignore").copy()
    meta.insert(0, "sample", sample_name)

    out = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    out.to_csv(out_csv, index=False)

    print(f"[Saved csv] {out_csv}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--out_dir", required=True)

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

    parser.add_argument("--train_coord_scale", type=float, default=1.0)
    parser.add_argument("--test_coord_scale", type=float, default=1.0)

    parser.add_argument("--n_genes", type=int, default=1000)
    parser.add_argument("--gene_list", type=str, default=None)
    parser.add_argument("--ignore_test_gene_filter", action="store_true")

    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--no_output_relu", action="store_true")
    parser.add_argument("--no_macenko", action="store_true")

    parser.add_argument("--max_train_tiles", type=int, default=0)
    parser.add_argument("--max_test_tiles", type=int, default=0)

    parser.add_argument("--force_recompute_features", action="store_true")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--save_each_model_pred", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    use_macenko = not args.no_macenko

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
    # Load data
    # ------------------------------------------------------------
    train_folders = discover_spatial_folders(args.train_dir)
    test_folders = discover_spatial_folders(args.test_dir)

    print("[Train folders]")
    for x in train_folders:
        print(" ", x)

    print("[Test folders]")
    for x in test_folders:
        print(" ", x)

    train_samples = [load_spatial_folder(x) for x in train_folders]
    test_samples = [load_spatial_folder(x) for x in test_folders]

    # ------------------------------------------------------------
    # Expression processing
    # ------------------------------------------------------------
    print("[Normalize train expression]")
    train_X_logs = [normalize_log_cpm(s["X"]) for s in train_samples]

    print("[Select genes]")
    selected_genes = select_genes_across_train_and_test(
        train_samples=train_samples,
        test_samples=test_samples,
        train_X_logs=train_X_logs,
        n_genes=args.n_genes,
        gene_list=args.gene_list,
        ignore_test_gene_filter=args.ignore_test_gene_filter,
    )

    print("n_selected_genes:", len(selected_genes))

    pd.Series(selected_genes).to_csv(
        os.path.join(args.out_dir, "selected_genes.tsv"),
        sep="\t",
        index=False,
        header=False,
    )

    # ------------------------------------------------------------
    # Build train tile labels
    # ------------------------------------------------------------
    print("[Build tile-level train expression matrix]")
    Y_train, train_tile_meta = build_tile_level_train_data(
        train_samples=train_samples,
        train_X_logs=train_X_logs,
        selected_genes=selected_genes,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        coord_scale=args.train_coord_scale,
        min_spots_per_tile=args.min_spots_per_tile,
    )

    if args.max_train_tiles and args.max_train_tiles > 0:
        keep = np.random.RandomState(args.seed).choice(
            Y_train.shape[0],
            size=min(args.max_train_tiles, Y_train.shape[0]),
            replace=False,
        )
        keep = np.sort(keep)
        Y_train = Y_train[keep]
        train_tile_meta = train_tile_meta.iloc[keep].reset_index(drop=True)
        print(f"[Subsample train tiles] {Y_train.shape[0]}")

    train_tile_meta.to_csv(
        os.path.join(args.out_dir, "train_tile_metadata.csv"),
        index=False,
    )

    print("Y_train_tile:", Y_train.shape)
    print("n_train_tiles:", train_tile_meta.shape[0])

    # ------------------------------------------------------------
    # Extract train tile features
    # ------------------------------------------------------------
    print("[Extract train tile-level direct CTransPath features]")

    train_features_list = []
    offset = 0

    for sample in train_samples:
        n_this = int((train_tile_meta["sample"] == sample["sample"]).sum())
        meta_this = train_tile_meta[train_tile_meta["sample"] == sample["sample"]].copy()

        feat = extract_ctrans_features_for_tile_table(
            sample=sample,
            tile_df=meta_this,
            ctrans_model=ctrans_model,
            ctrans_transform=ctrans_transform,
            color_normalizer=color_normalizer,
            cache_dir=cache_dir,
            tile_size=args.tile_size,
            batch_size=args.feature_batch_size,
            force_recompute=args.force_recompute_features,
            use_macenko=use_macenko,
            device=device,
            split_name="train",
        )

        train_features_list.append(feat)
        offset += n_this

    train_features = torch.cat(train_features_list, dim=0).contiguous()

    print("train_features:", tuple(train_features.shape))

    if train_features.shape[0] != Y_train.shape[0]:
        raise RuntimeError(
            f"train_features rows {train_features.shape[0]} != Y_train rows {Y_train.shape[0]}"
        )

    n_inputs = train_features.shape[1]
    n_outputs = len(selected_genes)
    output_relu = not args.no_output_relu

    # Free CTransPath before MLP training.
    del ctrans_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Train fold models
    # ------------------------------------------------------------
    print("[Make nested CV splits]")
    splits = make_nested_cv_splits(
        n_samples=train_features.shape[0],
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        seed=args.seed,
    )

    split_summary = []
    for sp in splits:
        split_summary.append({
            "ik": sp["ik"],
            "il": sp["il"],
            "n_train": len(sp["train_idx"]),
            "n_valid": len(sp["valid_idx"]),
            "n_outer_test": len(sp["outer_test_idx"]),
        })

    split_summary_df = pd.DataFrame(split_summary)
    split_summary_df.to_csv(os.path.join(args.out_dir, "nested_cv_split_summary.csv"), index=False)
    print(split_summary_df)

    print("[Train fold models]")
    model_paths = []
    all_logs = []

    patience = None if args.patience <= 0 else args.patience

    for sp in splits:
        ik = sp["ik"]
        il = sp["il"]

        result_dir = os.path.join(args.out_dir, f"result_{ik}_{il}")

        print("=" * 80)
        print(f"[Fold] ik={ik}, il={il}")
        print(
            f"  train={len(sp['train_idx'])}, "
            f"valid={len(sp['valid_idx'])}, "
            f"outer_test={len(sp['outer_test_idx'])}"
        )
        print("=" * 80)

        fold_result = train_one_fold(
            train_features=train_features,
            Y_train=Y_train,
            train_idx=sp["train_idx"],
            valid_idx=sp["valid_idx"],
            result_dir=result_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            patience=patience,
            num_workers=args.num_workers,
            device=device,
            output_relu=output_relu,
        )

        model_paths.append(fold_result["best_path"])

        log_df = fold_result["log"].copy()
        log_df.insert(0, "ik", ik)
        log_df.insert(1, "il", il)
        log_df["best_epoch"] = fold_result["best_epoch"]
        log_df["best_valid_loss"] = fold_result["best_valid_loss"]
        all_logs.append(log_df)

    all_logs_df = pd.concat(all_logs, axis=0, ignore_index=True)
    all_logs_df.to_csv(os.path.join(args.out_dir, "train_log_all_folds.csv"), index=False)

    pd.Series(model_paths).to_csv(
        os.path.join(args.out_dir, "ensemble_model_paths.txt"),
        index=False,
        header=False,
    )

    print("[Ensemble models]")
    for p in model_paths:
        print(" ", p)

    # ------------------------------------------------------------
    # Re-load CTransPath for test feature extraction
    # ------------------------------------------------------------
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
    # Predict test tiles
    # ------------------------------------------------------------
    print("[Predict external test samples at tile level]")

    for sample in test_samples:
        print("=" * 80)
        print(f"[Test sample] {sample['sample']}")
        print("=" * 80)

        test_tile_df = build_tile_table_from_coords(
            coords=sample["coords"],
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            coord_scale=args.test_coord_scale,
            min_spots_per_tile=args.min_spots_per_tile,
        )

        if args.max_test_tiles and args.max_test_tiles > 0:
            keep = np.arange(min(args.max_test_tiles, test_tile_df.shape[0]))
            test_tile_df = test_tile_df.iloc[keep].reset_index(drop=True)
            print(f"[Subsample test tiles] {test_tile_df.shape[0]}")

        test_tile_meta_save = test_tile_df.drop(columns=["spot_indices"], errors="ignore").copy()
        test_tile_meta_save.to_csv(
            os.path.join(args.out_dir, f"{sample['sample']}_test_tile_metadata.csv"),
            index=False,
        )

        print(f"n_test_tiles: {test_tile_df.shape[0]}")

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
            split_name="test",
        )

        pred_mean, pred_stack = ensemble_predict_from_folds(
            model_paths=model_paths,
            test_features=test_features,
            n_inputs=n_inputs,
            hidden=args.hidden,
            n_outputs=n_outputs,
            dropout=args.dropout,
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
            )

    print("Done.")


if __name__ == "__main__":
    main()