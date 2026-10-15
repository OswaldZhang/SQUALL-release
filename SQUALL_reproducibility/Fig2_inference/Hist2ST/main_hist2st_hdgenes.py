#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hist2ST-native tile-level HD/Xenium pipeline.

Goal
----
Use the original Hist2ST model body as much as possible:
  - import Hist2ST from HIST2ST.py
  - keep ConvMixer + Transformer + GraphSAGE/GNN + optional ZINB/NB + bake/self-distillation
  - keep one graph/section forward style

Only the dataset/input-output wrapper is adapted to your tile input format:

  TILE_ROOT/
    samples/
      SAMPLE/
        image_path.txt
        genes.tsv
        tile_counts.h5       # dataset: counts, shape [n_tiles, n_genes]
        tile_coords.npy      # pixel centers, shape [n_tiles, 2], x/y in full-res image
        tile_meta.csv        # optional: tile_id, n_bins, grid/tile x/y columns

Main outputs
------------
  out_dir/
    best_model.pt
    last_model.pt
    selected_genes.tsv
    train_stats.npz
    model_meta.json
    xenium_predicted_expression.h5

Prediction h5 datasets
----------------------
  pred_lognorm : model output in Hist2ST-normalized log expression space
  genes        : selected genes
  coords       : pixel centers from tile_coords.npy
  tile_id      : copied from tile_meta.csv if present
  n_bins       : copied from tile_meta.csv if present
  gt_lognorm   : optional, if test counts contain selected genes

Important
---------
Original Hist2ST does graph-level attention over all nodes in a section. For very large
HD tile graphs, use --max_train_nodes / --max_pred_nodes to sample/subgraph nodes.
Default 0 means no sampling/cropping of graph nodes.
"""

import os
import json
import math
import argparse
import random
from pathlib import Path

import h5py
import numpy as np

# Compatibility patch:
# pytorch-lightning 1.x internally uses np.Inf,
# but NumPy 2.x removed np.Inf.
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import TensorBoardLogger
except Exception as e:
    raise ImportError("This script needs pytorch_lightning, same as original Hist2ST_train.py") from e

try:
    import scprep as scp
except Exception:
    scp = None

from graph_construction import calcADJ
from HIST2ST import Hist2ST

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# -------------------------
# Reproducibility / helpers
# -------------------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {x}")


def save_string_dataset(h5, name, values):
    arr = np.asarray([str(x).encode("utf-8") for x in values])
    h5.create_dataset(name, data=arr)


def read_counts_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "counts" not in f:
            raise KeyError(f"{path} does not contain dataset 'counts'")
        X = f["counts"][:].astype(np.float32)
    return X


def load_tile_sample(tile_root: str, sample_name: str):
    sample_dir = Path(tile_root) / "samples" / sample_name
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample dir: {sample_dir}")

    image_path = Path(open(sample_dir / "image_path.txt").read().strip())
    genes = pd.read_csv(sample_dir / "genes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    coords = np.load(sample_dir / "tile_coords.npy").astype(np.float32)
    counts = read_counts_h5(sample_dir / "tile_counts.h5")
    meta_path = sample_dir / "tile_meta.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame(index=np.arange(coords.shape[0]))

    if counts.shape[0] != coords.shape[0]:
        raise ValueError(f"{sample_name}: counts rows {counts.shape[0]} != coords rows {coords.shape[0]}")
    if counts.shape[1] != len(genes):
        raise ValueError(f"{sample_name}: counts cols {counts.shape[1]} != genes {len(genes)}")
    if len(meta) not in {0, coords.shape[0]}:
        raise ValueError(f"{sample_name}: tile_meta rows {len(meta)} != coords rows {coords.shape[0]}")

    return {
        "sample_name": sample_name,
        "sample_dir": str(sample_dir),
        "image_path": str(image_path),
        "genes": genes,
        "coords": coords,  # full-res pixel x/y centers
        "counts": counts,
        "meta": meta,
    }


# -------------------------
# Hist2ST-style expression normalization
# -------------------------

def hist2st_log_normalize_counts(X: np.ndarray, library_size: float = 1.0) -> np.ndarray:
    """
    Original uploaded Hist2ST dataset uses:
      scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))

    scprep's default library size is used if library_size <= 0. If scprep is unavailable,
    a simple counts/sum*median_sum then log1p fallback is used.
    """
    X = X.astype(np.float32)
    if scp is not None:
        if library_size and library_size > 0:
            Y = scp.normalize.library_size_normalize(X, library_size=library_size)
        else:
            Y = scp.normalize.library_size_normalize(X)
        Y = scp.transform.log(Y)
        return np.asarray(Y, dtype=np.float32)

    sums = X.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    scale = float(np.median(sums)) if (library_size is None or library_size <= 0) else float(library_size)
    return np.log1p(X / sums * scale).astype(np.float32)


def size_factors_for_zinb(X: np.ndarray) -> np.ndarray:
    n_counts = X.sum(axis=1).astype(np.float32)
    med = np.median(n_counts[n_counts > 0]) if np.any(n_counts > 0) else 1.0
    if med <= 0:
        med = 1.0
    return (n_counts / med).astype(np.float32)


# -------------------------
# Grid positions for Hist2ST positional embedding and graph
# -------------------------

def _rank_to_int(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values)
    uniq = np.unique(vals)
    order = {v: i for i, v in enumerate(sorted(uniq.tolist()))}
    return np.asarray([order[v] for v in vals], dtype=np.int64)



def normalize_grid_positions(grid: np.ndarray, n_pos: int = 64) -> np.ndarray:
    """
    Normalize each sample's x/y grid independently to [0, n_pos - 1].
    This avoids unseen coordinate embedding indices during external prediction.
    """
    grid = np.asarray(grid, dtype=np.float32)
    out = np.zeros_like(grid, dtype=np.int64)

    for j in range(2):
        v = grid[:, j]
        finite = np.isfinite(v)
        if finite.sum() == 0:
            out[:, j] = 0
            continue

        vmin = float(np.nanmin(v[finite]))
        vmax = float(np.nanmax(v[finite]))

        if vmax <= vmin:
            out[:, j] = 0
        else:
            out[:, j] = np.rint((v - vmin) / (vmax - vmin) * (n_pos - 1)).astype(np.int64)

    out = np.clip(out, 0, n_pos - 1)
    return out.astype(np.int64)


def infer_grid_positions(sample, coord_mode="auto", tile_size=224, n_pos=64):
    """
    Return integer grid positions [N, 2] for Hist2ST x/y embedding and calcADJ.

    Modes:
      auto:
        use tile_meta grid/tile/x/y columns if available, then compact-rank them.
      pixel_rank:
        compact-rank tile_coords.npy x/y.
      normalized:
        use tile_meta grid/tile/x/y columns if available, otherwise tile_coords.npy,
        and min-max normalize each sample independently to [0, n_pos - 1].
        This is recommended for cross-section / cross-platform prediction.
    """
    meta = sample["meta"]
    coords = sample["coords"]

    candidates = [
        ("grid_x", "grid_y"),
        ("tile_x", "tile_y"),
        ("x", "y"),
        ("array_col", "array_row"),
        ("col", "row"),
    ]

    if coord_mode == "zero":
        grid_i = np.zeros((coords.shape[0], 2), dtype=np.int64)
        print(f"[Grid positions] zero coordinates; max={grid_i.max(axis=0)}")
        return grid_i

    raw_grid = None
    raw_source = None

    for cx, cy in candidates:
        if cx in meta.columns and cy in meta.columns:
            gx = pd.to_numeric(meta[cx], errors="coerce").values
            gy = pd.to_numeric(meta[cy], errors="coerce").values
            if np.isfinite(gx).all() and np.isfinite(gy).all():
                raw_grid = np.stack([gx, gy], axis=1).astype(np.float32)
                raw_source = f"tile_meta columns: {cx}, {cy}"
                break

    if raw_grid is None:
        raw_grid = coords.astype(np.float32)
        raw_source = "tile_coords.npy x/y"

    if coord_mode == "normalized":
        grid_i = normalize_grid_positions(raw_grid, n_pos=int(n_pos))
        print(
            f"[Grid positions] normalized from {raw_source}; "
            f"raw_max={np.nanmax(raw_grid, axis=0)}, norm_max={grid_i.max(axis=0)}, n_pos={n_pos}"
        )
        return grid_i.astype(np.int64)

    if coord_mode == "pixel_rank":
        gx = _rank_to_int(coords[:, 0])
        gy = _rank_to_int(coords[:, 1])
        grid_i = np.stack([gx, gy], axis=1).astype(np.int64)
        print(f"[Grid positions] use rank(tile_coords.npy x/y); max={grid_i.max(axis=0)}")
        return grid_i

    # auto mode: use raw_grid source, then compact-rank
    grid_i = np.stack([_rank_to_int(raw_grid[:, 0]), _rank_to_int(raw_grid[:, 1])], axis=1)
    print(f"[Grid positions] use {raw_source}; max={grid_i.max(axis=0)}")
    return grid_i.astype(np.int64)

def subset_indices(n, max_nodes, seed, mode="random"):
    if max_nodes is None or max_nodes <= 0 or max_nodes >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if mode == "random":
        idx = rng.choice(n, size=max_nodes, replace=False)
        return np.sort(idx.astype(np.int64))
    if mode == "first":
        return np.arange(max_nodes, dtype=np.int64)
    raise ValueError(f"Unsupported subset mode: {mode}")


def split_train_val_indices(n, val_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    val = np.sort(order[:n_val].astype(np.int64)) if n_val > 0 else np.asarray([], dtype=np.int64)
    train = np.sort(order[n_val:].astype(np.int64))
    if len(train) == 0:
        train = val.copy()
    return train, val


# -------------------------
# Image patch extraction: raw RGB tensor like original dataset.py
# -------------------------

def crop_patch_pil(image: Image.Image, x, y, crop_size: int, fill=(255, 255, 255)) -> Image.Image:
    x = int(round(float(x)))
    y = int(round(float(y)))
    half = crop_size // 2
    left, upper = x - half, y - half
    right, lower = left + crop_size, upper + crop_size
    W, H = image.size

    crop_left = max(left, 0)
    crop_upper = max(upper, 0)
    crop_right = min(right, W)
    crop_lower = min(lower, H)

    patch = Image.new("RGB", (crop_size, crop_size), fill)
    if crop_right > crop_left and crop_lower > crop_upper:
        crop = image.crop((crop_left, crop_upper, crop_right, crop_lower))
        patch.paste(crop, (crop_left - left, crop_upper - upper))
    return patch


def extract_patches_raw(image_path, coords, indices, crop_size=224, fig_size=112):
    image = Image.open(image_path).convert("RGB")
    patches = torch.empty((len(indices), 3, fig_size, fig_size), dtype=torch.float32)
    for out_i, idx in enumerate(tqdm(indices, desc="Crop image patches")):
        x, y = coords[idx]
        patch = crop_patch_pil(image, x, y, crop_size=crop_size)
        if fig_size != crop_size:
            patch = patch.resize((fig_size, fig_size), Image.BILINEAR)
        arr = np.asarray(patch).astype(np.float32)  # original Hist2ST uses raw image tensor values
        patches[out_i] = torch.from_numpy(arr).permute(2, 0, 1)
    return patches



def make_adj_safe(adj):
    """
    Avoid zero-degree nodes causing NaN in Hist2ST GNN mean aggregation.
    Add self-loop for all nodes.
    """
    if not torch.is_tensor(adj):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj = adj.float().clone()
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square, got {tuple(adj.shape)}")
    adj.fill_diagonal_(1.0)
    row_sum = adj.sum(dim=1)
    bad = row_sum <= 0
    if bad.any():
        idx = torch.where(bad)[0]
        adj[idx, idx] = 1.0
    return adj

class Hist2STGraphDataset(Dataset):
    """
    A dataset item is one graph/section, matching original Hist2ST batch style:
      patch, center, exp, adj, ori_counts, size_factors, pixel_centers
    DataLoader(batch_size=1) gives tensors compatible with Hist2ST.training_step.
    """
    def __init__(self, sample, gene_indices, grid_positions, node_indices, args, mode="train"):
        self.sample = sample
        self.gene_indices = np.asarray(gene_indices, dtype=np.int64)
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.args = args
        self.mode = mode

        raw = sample["counts"][self.node_indices[:, None], self.gene_indices].astype(np.float32)
        self.exp = torch.from_numpy(hist2st_log_normalize_counts(raw, library_size=args.library_size)).float()
        self.ori = torch.from_numpy(raw).float()
        self.sfs = torch.from_numpy(size_factors_for_zinb(raw)).float()

        grid = grid_positions[self.node_indices].astype(np.int64)

        # IMPORTANT:
        # Do not re-rank coordinates inside each subgraph.
        # For external prediction, train/test must share the same coordinate embedding range.
        self.positions = torch.from_numpy(grid).long()
        self.pixel_centers = torch.from_numpy(sample["coords"][self.node_indices].astype(np.float32)).float()

        adj = calcADJ(grid.astype(np.float32), k=args.neighbor, pruneTag=args.prune)
        self.adj = make_adj_safe(adj)

        self.patches = extract_patches_raw(
            sample["image_path"],
            sample["coords"],
            self.node_indices,
            crop_size=args.crop_size,
            fig_size=args.fig_size,
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return [
            self.patches,
            self.positions,
            self.exp,
            self.adj,
            self.ori,
            self.sfs,
            self.pixel_centers,
        ]


# -------------------------
# Lightning compatibility and prediction
# -------------------------

def make_trainer(args, logger=None, callbacks=None):
    callbacks = callbacks or []
    kwargs = dict(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=False,
    )
    # PL 1.x accepts gpus=[id]; PL 2.x prefers accelerator/devices.
    try:
        return pl.Trainer(gpus=[args.gpu], **kwargs)
    except TypeError:
        if torch.cuda.is_available():
            return pl.Trainer(accelerator="gpu", devices=[args.gpu], **kwargs)
        return pl.Trainer(accelerator="cpu", devices=1, **kwargs)


class SaveBestByValLoss(pl.Callback):
    def __init__(self, out_dir):
        super().__init__()
        self.best = float("inf")
        self.out_dir = Path(out_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val = metrics.get("valid_loss")
        if val is None:
            return
        val_f = float(val.detach().cpu()) if torch.is_tensor(val) else float(val)
        if val_f < self.best:
            self.best = val_f
            path = self.out_dir / "best_model.pt"
            torch.save(pl_module.state_dict(), path)
            print(f"[Saved best] {path} valid_loss={val_f:.6f}")


@torch.no_grad()
def predict_graph(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = model.to(device)
    model.eval()
    preds = []
    centers = []
    gt = []
    for batch in tqdm(loader, desc="Predict graph"):
        patch, position, exp, adj, oris, sfs, center = batch
        patch = patch.to(device)
        position = position.to(device)
        adj = adj.to(device).squeeze(0)
        pred = model(patch, position, adj)[0].squeeze(0).detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
        centers.append(center.squeeze(0).cpu().numpy().astype(np.float32))
        gt.append(exp.squeeze(0).cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(centers, axis=0), np.concatenate(gt, axis=0)


def compute_gene_corr_np(y_true, y_pred):
    out = []
    for j in range(y_true.shape[1]):
        a, b = y_true[:, j], y_pred[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(out, dtype=np.float32)


def save_prediction_h5(out_h5, pred, gt, genes, coords, meta, node_indices):
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    meta_sub = meta.iloc[node_indices].reset_index(drop=True) if len(meta) == len(node_indices) or len(meta) > max(node_indices, default=-1) else meta

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("pred_lognorm", data=pred.astype(np.float32), compression="gzip")
        if gt is not None:
            f.create_dataset("gt_lognorm", data=gt.astype(np.float32), compression="gzip")
        save_string_dataset(f, "genes", genes)
        f.create_dataset("coords", data=coords.astype(np.float32))
        f.create_dataset("node_indices", data=node_indices.astype(np.int64))
        if "tile_id" in meta_sub.columns:
            save_string_dataset(f, "tile_id", meta_sub["tile_id"].astype(str).tolist())
        if "n_bins" in meta_sub.columns:
            f.create_dataset("n_bins", data=meta_sub["n_bins"].values.astype(np.int64))
    print(f"[Saved prediction] {out_h5}")


# -------------------------
# Main pipeline
# -------------------------

def build_model(args, n_genes, n_pos):
    kernel, patch, depth1, depth2, depth3, heads, channel = map(int, args.tag.split("-"))
    model = Hist2ST(
        depth1=depth1,
        depth2=depth2,
        depth3=depth3,
        n_genes=n_genes,
        learning_rate=args.lr,
        label=None,
        kernel_size=kernel,
        patch_size=patch,
        heads=heads,
        channel=channel,
        dropout=args.dropout,
        zinb=args.zinb,
        nb=(args.nb == "T"),
        bake=args.bake,
        lamb=args.lamb,
        policy=args.policy,
        fig_size=args.fig_size,
        n_pos=n_pos,
    )
    return model


def choose_genes(args, train_sample, test_sample=None, train_out_dir=None):
    if args.mode == "predict_only":
        genes = pd.read_csv(Path(train_out_dir) / "selected_genes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).tolist()
        return genes

    train_genes = list(map(str, train_sample["genes"]))
    if args.gene_source == "train" or test_sample is None:
        return train_genes

    test_genes = set(map(str, test_sample["genes"]))
    genes = [g for g in train_genes if g in test_genes]
    if len(genes) == 0:
        raise ValueError("No common genes between train and test sample")
    return genes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train_predict", "predict_only"], default="train_predict")
    parser.add_argument("--tile_root", required=True)
    parser.add_argument("--train_sample", required=True)
    parser.add_argument("--test_sample", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_out_dir", default=None)
    parser.add_argument("--ckpt", default=None)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--logger", type=str, default="logs/hist2st_hd")
    parser.add_argument("--name", type=str, default="Hist2ST_HD_native")

    # Original Hist2ST defaults: 5-7-2-8-4-16-32, bake=5, lamb=0.5, zinb=0.25
    parser.add_argument("--tag", type=str, default="5-7-2-8-4-16-32")
    parser.add_argument("--bake", type=int, default=5)
    parser.add_argument("--lamb", type=float, default=0.5)
    parser.add_argument("--nb", type=str, default="F", choices=["T", "F"])
    parser.add_argument("--zinb", type=float, default=0.25)
    parser.add_argument("--prune", type=str, default="Grid", choices=["Grid", "NA", "STD"])
    parser.add_argument("--policy", type=str, default="mean")
    parser.add_argument("--neighbor", type=int, default=4)

    # For HD tile-level adaptation
    parser.add_argument("--crop_size", type=int, default=224, help="full-res crop around tile center before optional resize")
    parser.add_argument("--fig_size", type=int, default=112, help="model input image size; original Hist2ST default is 112")
    parser.add_argument("--coord_mode", type=str, default="auto", choices=["auto", "pixel_rank", "normalized", "zero"])
    parser.add_argument("--n_pos", type=int, default=0, help="0=auto from train/test grid max + 1")
    parser.add_argument("--library_size", type=float, default=1.0, help="Hist2ST/scprep library_size_normalize target; 1.0 matches simple proportions")
    parser.add_argument("--gene_source", type=str, default="common", choices=["common", "train"])
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_nodes", type=int, default=0, help="0=use all train nodes; otherwise random subgraph for memory")
    parser.add_argument("--max_pred_nodes", type=int, default=0, help="0=use all test nodes; otherwise random subgraph for quick/memory-safe prediction")
    parser.add_argument("--subset_mode", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2)

    args = parser.parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}; gpu={args.gpu}")

    train_sample = load_tile_sample(args.tile_root, args.train_sample)
    test_sample = load_tile_sample(args.tile_root, args.test_sample)

    if args.mode == "predict_only":
        if args.train_out_dir is None:
            raise ValueError("--train_out_dir is required for predict_only")
        if args.ckpt is None:
            raise ValueError("--ckpt is required for predict_only")
        meta_json = Path(args.train_out_dir) / "model_meta.json"
        if not meta_json.exists():
            raise FileNotFoundError(f"Missing model_meta.json in train_out_dir: {meta_json}")
        train_meta = json.load(open(meta_json))
        genes = choose_genes(args, train_sample, test_sample, train_out_dir=args.train_out_dir)
        args.tag = train_meta["tag"]
        args.fig_size = int(train_meta["fig_size"])
        args.crop_size = int(train_meta.get("crop_size", args.crop_size))
        args.n_pos = int(train_meta["n_pos"])
        args.coord_mode = train_meta.get("coord_mode", args.coord_mode)
        args.prune = train_meta.get("prune", args.prune)
        args.neighbor = int(train_meta.get("neighbor", args.neighbor))
        args.policy = train_meta.get("policy", args.policy)
        args.zinb = float(train_meta.get("zinb", args.zinb))
        args.nb = train_meta.get("nb", args.nb)
        args.bake = int(train_meta.get("bake", args.bake))
        args.lamb = float(train_meta.get("lamb", args.lamb))
        args.library_size = float(train_meta.get("library_size", args.library_size))
        print(f"[Predict-only] loaded train metadata from {meta_json}")
    else:
        genes = choose_genes(args, train_sample, test_sample)

    gene_to_idx_train = {g: i for i, g in enumerate(train_sample["genes"])}
    gene_to_idx_test = {g: i for i, g in enumerate(test_sample["genes"])}
    missing_train = [g for g in genes if g not in gene_to_idx_train]
    if missing_train:
        raise ValueError(f"Selected genes missing from train sample, first few: {missing_train[:10]}")
    missing_test = [g for g in genes if g not in gene_to_idx_test]
    if missing_test:
        print(f"[WARN] {len(missing_test)} selected genes missing in test; gt_lognorm will not be complete")
    train_gene_idx = [gene_to_idx_train[g] for g in genes]
    test_gene_idx = [gene_to_idx_test[g] for g in genes if g in gene_to_idx_test]

    pd.Series(genes).to_csv(out_dir / "selected_genes.tsv", sep="\t", index=False, header=False)
    print(f"[Genes] n_selected={len(genes)} gene_source={args.gene_source}")

    coord_n_pos = int(args.n_pos) if args.n_pos and args.n_pos > 0 else 64

    train_grid = infer_grid_positions(
        train_sample,
        coord_mode=args.coord_mode,
        n_pos=coord_n_pos,
    )
    test_grid = infer_grid_positions(
        test_sample,
        coord_mode=args.coord_mode,
        n_pos=coord_n_pos,
    )

    if args.coord_mode == "zero":
        n_pos = 1
        n_pos_auto = 1
    elif args.coord_mode == "normalized":
        n_pos = coord_n_pos
        n_pos_auto = coord_n_pos
    else:
        n_pos_auto = int(max(train_grid.max(), test_grid.max()) + 2)
        n_pos = int(args.n_pos) if args.n_pos and args.n_pos > 0 else n_pos_auto

    print(f"[n_pos] {n_pos} (auto suggestion {n_pos_auto}, coord_mode={args.coord_mode})")

    if args.mode == "train_predict":
        all_train_nodes = subset_indices(
            train_sample["counts"].shape[0], args.max_train_nodes, args.seed, mode=args.subset_mode
        )
        local_train, local_val = split_train_val_indices(len(all_train_nodes), args.val_fraction, args.seed)
        train_nodes = all_train_nodes[local_train]
        val_nodes = all_train_nodes[local_val] if len(local_val) > 0 else all_train_nodes[local_train]
        print(f"[Train nodes] {len(train_nodes)}; [Val nodes] {len(val_nodes)}")

        train_ds = Hist2STGraphDataset(train_sample, train_gene_idx, train_grid, train_nodes, args, mode="train")
        val_ds = Hist2STGraphDataset(train_sample, train_gene_idx, train_grid, val_nodes, args, mode="val")
        train_loader = DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)

        model = build_model(args, n_genes=len(genes), n_pos=n_pos)
        if args.ckpt is not None:
            print(f"[Load ckpt before train] {args.ckpt}")
            state = torch.load(args.ckpt, map_location="cpu")
            model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state, strict=True)

        log_name = f"{args.name}_{args.train_sample}_to_{args.test_sample}_{args.tag}_zinb{args.zinb}_bake{args.bake}"
        logger = TensorBoardLogger(args.logger, name=log_name)
        best_cb = SaveBestByValLoss(out_dir)
        trainer = make_trainer(args, logger=logger, callbacks=[best_cb])
        print("[Train Hist2ST]")
        trainer.fit(model, train_loader, val_loader)

        torch.save(model.state_dict(), out_dir / "last_model.pt")
        best_path = out_dir / "best_model.pt"
        if best_path.exists():
            print(f"[Load best model] {best_path}")
            model.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
        else:
            print("[WARN] best_model.pt not found; using last model")
            torch.save(model.state_dict(), best_path)

        np.savez(
            out_dir / "train_stats.npz",
            genes=np.asarray(genes, dtype=object),
            train_sample=args.train_sample,
            library_size=args.library_size,
        )
        with open(out_dir / "model_meta.json", "w") as f:
            json.dump(
                {
                    "tag": args.tag,
                    "fig_size": args.fig_size,
                    "crop_size": args.crop_size,
                    "n_pos": n_pos,
                    "coord_mode": args.coord_mode,
                    "n_genes": len(genes),
                    "library_size": args.library_size,
                    "prune": args.prune,
                    "neighbor": args.neighbor,
                    "policy": args.policy,
                    "zinb": args.zinb,
                    "nb": args.nb,
                    "bake": args.bake,
                    "lamb": args.lamb,
                },
                f,
                indent=2,
            )

    else:
        model = build_model(args, n_genes=len(genes), n_pos=n_pos)
        print(f"[Load ckpt] {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state, strict=True)

    # Predict test sample
    pred_nodes = subset_indices(test_sample["counts"].shape[0], args.max_pred_nodes, args.seed, mode=args.subset_mode)
    print(f"[Predict nodes] {len(pred_nodes)} / {test_sample['counts'].shape[0]}")
    # test_gene_idx must correspond to all selected genes. If missing genes exist, raise for output shape consistency.
    if len(test_gene_idx) != len(genes):
        raise ValueError("Some selected genes are missing in test sample; use --gene_source common during training")
    test_ds = Hist2STGraphDataset(test_sample, test_gene_idx, test_grid, pred_nodes, args, mode="test")
    pred, pred_coords, gt = predict_graph(model, test_ds, device=device)
    save_prediction_h5(
        out_h5=out_dir / "xenium_predicted_expression.h5",
        pred=pred,
        gt=gt,
        genes=genes,
        coords=pred_coords,
        meta=test_sample["meta"],
        node_indices=pred_nodes,
    )

    corr = compute_gene_corr_np(gt, pred)
    pd.DataFrame({"Gene": genes, "Pearson": corr}).to_csv(out_dir / "test_gene_pearson.csv", index=False)
    print(f"[Test mean gene Pearson] {np.nanmean(corr):.6f}")
    print("[Done]")


if __name__ == "__main__":
    main()
