#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepPT-style tile-level main.py with:
  1. original DeepPT model components:
       ResNet50 feature extractor
       AutoEncoder 2048 -> 512 -> 2048
       MLP 512 -> 512 -> n_genes
  2. Macenko stain normalization before ResNet feature extraction
  3. nested CV-style train/valid/internal-test split on training tiles
  4. external prediction on Xenium / OC / CC samples
  5. h5 output for spatial evaluation

Input folder:

EGNv1_tile_input_allgenes/
  manifest.tsv
  samples/<sample_name>/
    image_path.txt
    genes.tsv
    tile_counts.h5          # dataset: counts, shape [n_tiles, n_genes]
    tile_coords.npy         # shape [n_tiles, 2], columns x,y
    tile_meta.csv

Output h5:

  predicted_expression      [n_tiles, n_genes]
  true_expression           [n_tiles, n_genes]
  genes                     [n_genes]
  tile_coords               [n_tiles, 2]

Important:
  This script requires the original DeepPT utils_color_norm.py to be available
  in the same directory or PYTHONPATH if --no_macenko is not used.
"""

import os
import json
import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# Random seed
# ============================================================

def init_random_seed(random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Follow original DeepPT style.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Original DeepPT model components
# ============================================================

class Feature_Extraction(nn.Module):
    """
    Original DeepPT-style ResNet50 feature extractor.
    Output: 2048-d avgpool feature before fc.
    """
    def __init__(self, model_type="load_from_saved_file"):
        super().__init__()

        if model_type == "load_from_saved_file":
            self.resnet = resnet50(weights=None)
        elif model_type == "load_from_internet":
            import torchvision
            self.resnet = resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )
        else:
            raise ValueError(
                "model_type must be load_from_saved_file or load_from_internet"
            )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class AutoEncoder(nn.Module):
    """
    Original DeepPT AutoEncoder:
      encoder: 2048 -> 512 + ReLU
      decoder: 512 -> 2048 + ReLU
    """
    def __init__(self, n_inputs=2048, n_hiddens=512, n_outputs=2048):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLP_regression(nn.Module):
    """
    Original DeepPT MLP architecture:
      Linear(512 -> 512)
      Dropout
      Linear(512 -> n_genes)

    Original DeepPT forward:
      tile predictions -> mean over tiles -> slide-level expression

    For this spatial benchmark:
      forward_tile() keeps tile-level predictions.
    """
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super(MLP_regression, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout)
        )

        self.layer1 = nn.Linear(n_hiddens, n_outputs)

        if bias_init is not None:
            with torch.no_grad():
                self.layer1.bias.copy_(bias_init)

    def forward_tile(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        return x

    def forward(self, x):
        x = self.forward_tile(x)
        x = torch.mean(x, dim=0)
        return x


# ============================================================
# IO utilities
# ============================================================

def read_first_line(path):
    return Path(path).read_text().strip().splitlines()[0]


def get_sample_dir(input_dir, sample_name):
    sample_dir = Path(input_dir) / "samples" / sample_name

    if not sample_dir.exists():
        raise FileNotFoundError(f"Cannot find sample directory: {sample_dir}")

    required = [
        "image_path.txt",
        "genes.tsv",
        "tile_counts.h5",
        "tile_coords.npy",
        "tile_meta.csv",
    ]

    for fn in required:
        p = sample_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    return sample_dir


def load_genes(sample_dir):
    genes = (
        pd.read_csv(Path(sample_dir) / "genes.tsv", sep="\t", header=None)
        .iloc[:, 0]
        .astype(str)
        .tolist()
    )
    return genes


def load_counts(sample_dir, gene_indices=None):
    h5_path = Path(sample_dir) / "tile_counts.h5"

    with h5py.File(h5_path, "r") as f:
        if "counts" not in f:
            raise KeyError(f"{h5_path} does not contain dataset 'counts'")

        ds = f["counts"]

        if gene_indices is None:
            X = ds[:]
        else:
            gene_indices = np.asarray(gene_indices, dtype=np.int64)

            # h5py requires sorted fancy indices.
            order = np.argsort(gene_indices)
            X_sorted = ds[:, gene_indices[order]]
            rev = np.argsort(order)
            X = X_sorted[:, rev]

    return X.astype(np.float32, copy=False)


def normalize_counts(X, mode="log1p_cpm", scale_factor=1e4):
    X = X.astype(np.float32, copy=False)

    if mode == "raw":
        return X

    if mode == "log1p":
        return np.log1p(X).astype(np.float32)

    if mode == "log1p_cpm":
        lib = X.sum(axis=1, keepdims=True)
        lib[lib <= 0] = 1.0
        X = X / lib * float(scale_factor)
        return np.log1p(X).astype(np.float32)

    raise ValueError("--norm must be raw, log1p, or log1p_cpm")


def common_gene_indices(train_genes, test_genes, requested_genes=None):
    train_map = {g: i for i, g in enumerate(train_genes)}
    test_map = {g: i for i, g in enumerate(test_genes)}

    if requested_genes is None:
        genes = [g for g in train_genes if g in test_map]
    else:
        genes = [g for g in requested_genes if g in train_map and g in test_map]

    if len(genes) == 0:
        raise ValueError("No common genes found.")

    train_idx = [train_map[g] for g in genes]
    test_idx = [test_map[g] for g in genes]

    return genes, train_idx, test_idx


# ============================================================
# Nested CV split
# ============================================================

def make_nested_cv_indices(
    n_items,
    n_outer_folds=5,
    n_inner_folds=5,
    ik_fold=0,
    il_fold=0,
    seed=42,
):
    """
    Tile-level nested CV split.

    Outer:
      one fold becomes internal_test_idx

    Inner:
      within outer-train pool, one fold becomes valid_idx,
      remaining folds become train_idx

    This mimics original DeepPT's ik_fold / il_fold logic, but applied to
    tile-level training data because current input is one sample's tile set.
    """
    if n_outer_folds < 2:
        raise ValueError("--n_outer_folds must be >= 2")
    if n_inner_folds < 2:
        raise ValueError("--n_inner_folds must be >= 2")
    if ik_fold < 0 or ik_fold >= n_outer_folds:
        raise ValueError(f"--ik_fold must be in [0, {n_outer_folds - 1}]")
    if il_fold < 0 or il_fold >= n_inner_folds:
        raise ValueError(f"--il_fold must be in [0, {n_inner_folds - 1}]")
    if n_items < n_outer_folds:
        raise ValueError("Number of training tiles is smaller than n_outer_folds")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_items)
    rng.shuffle(all_idx)

    outer_folds = np.array_split(all_idx, n_outer_folds)
    internal_test_idx = np.asarray(outer_folds[ik_fold], dtype=np.int64)

    outer_train_pool = np.concatenate(
        [outer_folds[i] for i in range(n_outer_folds) if i != ik_fold]
    )
    rng.shuffle(outer_train_pool)

    if len(outer_train_pool) < n_inner_folds:
        raise ValueError("Outer train pool is smaller than n_inner_folds")

    inner_folds = np.array_split(outer_train_pool, n_inner_folds)
    valid_idx = np.asarray(inner_folds[il_fold], dtype=np.int64)

    train_idx = np.concatenate(
        [inner_folds[i] for i in range(n_inner_folds) if i != il_fold]
    ).astype(np.int64)

    return train_idx, valid_idx, internal_test_idx


# ============================================================
# Macenko normalization
# ============================================================

def build_macenko_normalizer(use_macenko=True):
    if not use_macenko:
        print("[Macenko] disabled by --no_macenko")
        return None

    try:
        import utils_color_norm
        normalizer = utils_color_norm.macenko_normalizer()
        print("[Macenko] enabled: using utils_color_norm.macenko_normalizer()")
        return normalizer
    except Exception as e:
        raise RuntimeError(
            "Macenko normalization was requested, but utils_color_norm.py "
            "or its dependency spams could not be imported. "
            "Put the original DeepPT utils_color_norm.py in this directory "
            "and make sure spams is installed, or run with --no_macenko.\n"
            f"Original error: {repr(e)}"
        )

'''
def apply_macenko(tile_pil, normalizer):
    if normalizer is None:
        return tile_pil

    arr = np.asarray(tile_pil).astype(np.uint8)
    arr_norm = normalizer.transform(arr)
    return Image.fromarray(arr_norm.astype(np.uint8))
'''

def is_bad_tile_for_macenko(arr, white_thr=245, black_thr=10, min_tissue_frac=0.02, min_std=3.0):
    """
    Detect tiles that are likely to break Macenko:
      - almost all white background
      - almost all black / invalid
      - too little color variation
      - NaN / inf values
    """
    if arr is None:
        return True

    if arr.ndim != 3 or arr.shape[2] != 3:
        return True

    if not np.isfinite(arr).all():
        return True

    arr = arr.astype(np.uint8, copy=False)

    # grayscale-like intensity
    gray = arr.mean(axis=2)

    # tissue-like pixels: not too white and not too black
    tissue_mask = (gray < white_thr) & (gray > black_thr)
    tissue_frac = float(tissue_mask.mean())

    if tissue_frac < min_tissue_frac:
        return True

    if float(arr.std()) < min_std:
        return True

    return False


def apply_macenko(tile_pil, normalizer):
    """
    Robust Macenko wrapper.

    Original DeepPT applies Macenko after tile filtering.
    In our spatial tile setting, some tile centers may produce nearly blank tiles.
    For those tiles, Macenko can fail because OD is empty or stain matrix is unstable.
    We keep Macenko for valid tissue tiles and fallback to the original tile when it fails.
    """
    if normalizer is None:
        return tile_pil

    arr = np.asarray(tile_pil).astype(np.uint8)

    # Skip obvious bad/background tiles.
    if is_bad_tile_for_macenko(arr):
        return tile_pil

    try:
        arr_norm = normalizer.transform(arr)

        # Guard against NaN/Inf and collapsed output.
        if arr_norm is None:
            return tile_pil

        arr_norm = np.asarray(arr_norm)

        if arr_norm.shape != arr.shape:
            return tile_pil

        if not np.isfinite(arr_norm).all():
            return tile_pil

        arr_norm = np.clip(arr_norm, 0, 255).astype(np.uint8)

        # If normalization collapses to almost constant image, fallback.
        if float(arr_norm.std()) < 1.0:
            return tile_pil

        return Image.fromarray(arr_norm)

    except Exception:
        # Includes:
        #   LinAlgError: Eigenvalues did not converge
        #   divide by zero / invalid values propagated to linalg
        #   spams failures
        return tile_pil

# ============================================================
# Tile crop + ResNet features
# ============================================================

def crop_tile_from_center(image, cx, cy, tile_size):
    half = tile_size / 2.0

    left = int(round(cx - half))
    upper = int(round(cy - half))
    right = left + tile_size
    lower = upper + tile_size

    if left >= 0 and upper >= 0 and right <= image.width and lower <= image.height:
        return image.crop((left, upper, right, lower)).convert("RGB")

    canvas = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))

    crop_left = max(left, 0)
    crop_upper = max(upper, 0)
    crop_right = min(right, image.width)
    crop_lower = min(lower, image.height)

    if crop_right > crop_left and crop_lower > crop_upper:
        crop = image.crop((crop_left, crop_upper, crop_right, crop_lower)).convert("RGB")
        paste_x = crop_left - left
        paste_y = crop_upper - upper
        canvas.paste(crop, (paste_x, paste_y))

    return canvas


def load_resnet_weight(model, weight_path, device):
    if weight_path is None or str(weight_path).lower() == "none":
        print("[WARN] No ResNet weight provided. ResNet is randomly initialized.")
        return model

    state = torch.load(weight_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    state = {k.replace("module.", ""): v for k, v in state.items()}

    # First try loading into Feature_Extraction directly.
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ResNet] loaded weight into Feature_Extraction: {weight_path}")
        print(f"[ResNet] missing keys: {len(missing)}")
        print(f"[ResNet] unexpected keys: {len(unexpected)}")
        return model
    except Exception:
        pass

    # Then try loading into model.resnet.
    clean_state = {}
    for k, v in state.items():
        k2 = k
        if k2.startswith("resnet."):
            k2 = k2.replace("resnet.", "", 1)
        clean_state[k2] = v

    missing, unexpected = model.resnet.load_state_dict(clean_state, strict=False)

    print(f"[ResNet] loaded weight into model.resnet: {weight_path}")
    print(f"[ResNet] missing keys: {len(missing)}")
    print(f"[ResNet] unexpected keys: {len(unexpected)}")

    return model


@torch.no_grad()
def extract_or_load_resnet_features(
    sample_dir,
    tile_size,
    batch_size,
    resnet_weight,
    device,
    use_macenko=True,
    force_recompute=False,
):
    sample_dir = Path(sample_dir)

    suffix = "macenko" if use_macenko else "nomacenko"
    cache_path = sample_dir / f"deeppt_resnet50_features_tile{tile_size}_{suffix}.npy"

    coords = np.load(sample_dir / "tile_coords.npy").astype(np.float32)

    if cache_path.exists() and not force_recompute:
        features = np.load(cache_path).astype(np.float32)
        if features.shape[0] == coords.shape[0] and features.shape[1] == 2048:
            print(f"[Feature cache] loaded: {cache_path}, shape={features.shape}")
            return features
        else:
            print("[WARN] cached feature shape mismatch. Recomputing.")

    normalizer = build_macenko_normalizer(use_macenko=use_macenko)

    image_path = read_first_line(sample_dir / "image_path.txt")
    image = Image.open(image_path).convert("RGB")

    model = Feature_Extraction(model_type="load_from_saved_file")
    model = load_resnet_weight(model, resnet_weight, device)
    model.to(device)
    model.eval()

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    features = []
    batch = []

    print(f"[Extract ResNet features] sample={sample_dir.name}, n_tiles={coords.shape[0]}")
    print(f"[Image] {image_path}")
    print(f"[Macenko] use_macenko={use_macenko}")

    for i in tqdm(range(coords.shape[0])):
        cx, cy = float(coords[i, 0]), float(coords[i, 1])

        tile = crop_tile_from_center(image, cx, cy, tile_size)
        tile = apply_macenko(tile, normalizer)

        batch.append(data_transform(tile))

        if len(batch) == batch_size or i == coords.shape[0] - 1:
            xb = torch.stack(batch, dim=0).to(device)
            fb = model(xb).detach().cpu().numpy().astype(np.float32)
            features.append(fb)
            batch = []

    features = np.concatenate(features, axis=0).astype(np.float32)

    np.save(cache_path, features)
    print(f"[Feature cache] saved: {cache_path}, shape={features.shape}")

    return features


# ============================================================
# Dataset
# ============================================================

class FeatureDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X.astype(np.float32, copy=False)
        self.Y = None if Y is None else Y.astype(np.float32, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()

        if self.Y is None:
            return x

        y = torch.from_numpy(self.Y[idx]).float()
        return x, y


# ============================================================
# Metrics
# ============================================================

def mean_gene_pearson(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)

    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0)) + eps
    r = (yt * yp).sum(axis=0) / denom
    r = r[np.isfinite(r)]

    if len(r) == 0:
        return float("nan")

    return float(np.mean(r))


# ============================================================
# AE training / compression
# ============================================================

def train_autoencoder(X_train, out_dir, args, device):
    """
    Train original DeepPT AE on ResNet 2048-d tile features.

    Original DeepPT AE is unsupervised. Here it is trained only on training
    sample's ResNet features, not on external test sample.
    """
    out_dir = Path(out_dir)
    ae_ckpt = out_dir / "deeppt_model_AE.pth"

    rng = np.random.default_rng(args.seed)
    idx = np.arange(X_train.shape[0])
    rng.shuffle(idx)

    n_valid = max(1, int(X_train.shape[0] * args.ae_valid_fraction))
    ae_valid_idx = idx[:n_valid]
    ae_train_idx = idx[n_valid:]

    train_loader = DataLoader(
        FeatureDataset(X_train[ae_train_idx]),
        batch_size=args.ae_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        FeatureDataset(X_train[ae_valid_idx]),
        batch_size=args.ae_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = AutoEncoder(
        n_inputs=2048,
        n_hiddens=512,
        n_outputs=2048,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.ae_lr)
    loss_fn = nn.MSELoss()

    logs = []

    print("[Train AE] original DeepPT AE: 2048 -> 512 -> 2048")
    print(f"[AE split] train={len(ae_train_idx)}, valid={len(ae_valid_idx)}")

    for epoch in range(1, args.ae_epochs + 1):
        model.train()
        train_losses = []

        for xb in train_loader:
            xb = xb.to(device).float()
            pred = model(xb)
            loss = loss_fn(pred, xb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []

        with torch.no_grad():
            for xb in valid_loader:
                xb = xb.to(device).float()
                pred = model(xb)
                loss = loss_fn(pred, xb)
                valid_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))

        print(
            f"[AE Epoch {epoch:03d}/{args.ae_epochs}] "
            f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        })

    torch.save(
        {
            "model_state": model.state_dict(),
            "n_inputs": 2048,
            "n_hiddens": 512,
            "n_outputs": 2048,
            "args": vars(args),
        },
        ae_ckpt,
    )

    pd.DataFrame(logs).to_csv(out_dir / "ae_train_log.csv", index=False)

    print(f"[AE saved] {ae_ckpt}")

    return model, ae_ckpt


def load_autoencoder(ae_ckpt, device):
    ckpt = torch.load(ae_ckpt, map_location=device)

    model = AutoEncoder(
        n_inputs=int(ckpt.get("n_inputs", 2048)),
        n_hiddens=int(ckpt.get("n_hiddens", 512)),
        n_outputs=int(ckpt.get("n_outputs", 2048)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model


@torch.no_grad()
def compress_features_with_ae(model, X, batch_size, device):
    loader = DataLoader(
        FeatureDataset(X),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    features = []
    model.eval()

    for xb in loader:
        xb = xb.to(device).float()
        z = model.encoder(xb)
        features.append(z.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(features, axis=0).astype(np.float32)


# ============================================================
# MLP training / prediction
# ============================================================

def train_mlp_nested_cv(X_train_ae, Y_train, train_idx, valid_idx, internal_test_idx, out_dir, args, device):
    out_dir = Path(out_dir)
    mlp_ckpt = out_dir / "deeppt_model_MLP.pth"

    train_loader = DataLoader(
        FeatureDataset(X_train_ae[train_idx], Y_train[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        FeatureDataset(X_train_ae[valid_idx], Y_train[valid_idx]),
        batch_size=args.pred_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    internal_test_loader = DataLoader(
        FeatureDataset(X_train_ae[internal_test_idx], Y_train[internal_test_idx]),
        batch_size=args.pred_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    bias_init = torch.Tensor(Y_train[train_idx].mean(axis=0)).to(device)

    model = MLP_regression(
        n_inputs=512,
        n_hiddens=512,
        n_outputs=Y_train.shape[1],
        dropout=args.dropout,
        bias_init=bias_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_score = -1e9
    best_epoch = -1
    epoch_since_best = 0

    logs = []

    print("[Train MLP] original DeepPT MLP layers: 512 -> 512 -> n_genes")
    print(
        f"[Nested CV] ik_fold={args.ik_fold}/{args.n_outer_folds}, "
        f"il_fold={args.il_fold}/{args.n_inner_folds}"
    )
    print(
        f"[Nested CV sizes] train={len(train_idx)}, "
        f"valid={len(valid_idx)}, internal_test={len(internal_test_idx)}"
    )
    print("[Note] forward_tile() is used for tile-level supervision and h5 output.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            pred = model.forward_tile(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        valid_true = []
        valid_pred = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()

                pred = model.forward_tile(xb)
                loss = loss_fn(pred, yb)

                valid_losses.append(float(loss.detach().cpu()))
                valid_true.append(yb.detach().cpu().numpy())
                valid_pred.append(pred.detach().cpu().numpy())

        valid_true = np.concatenate(valid_true, axis=0)
        valid_pred = np.concatenate(valid_pred, axis=0)

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        valid_r = mean_gene_pearson(valid_true, valid_pred)

        print(
            f"[MLP Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.6f} "
            f"valid_loss={valid_loss:.6f} "
            f"valid_mean_gene_r={valid_r:.6f}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_mean_gene_r": valid_r,
        })

        if np.isfinite(valid_r) and valid_r > best_score:
            best_score = valid_r
            best_epoch = epoch
            epoch_since_best = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_inputs": 512,
                    "n_hiddens": 512,
                    "n_outputs": Y_train.shape[1],
                    "dropout": args.dropout,
                    "best_epoch": best_epoch,
                    "best_valid_mean_gene_r": best_score,
                    "ik_fold": args.ik_fold,
                    "il_fold": args.il_fold,
                    "n_outer_folds": args.n_outer_folds,
                    "n_inner_folds": args.n_inner_folds,
                    "args": vars(args),
                },
                mlp_ckpt,
            )
        else:
            epoch_since_best += 1

        if epoch_since_best >= args.patience:
            print(
                f"[Early stopping] epoch={epoch}, "
                f"best_epoch={best_epoch}, "
                f"best_valid_mean_gene_r={best_score:.6f}"
            )
            break

    pd.DataFrame(logs).to_csv(out_dir / "mlp_train_log.csv", index=False)

    ckpt = torch.load(mlp_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Internal outer-test evaluation.
    internal_true = []
    internal_pred = []
    internal_losses = []

    with torch.no_grad():
        for xb, yb in internal_test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            pred = model.forward_tile(xb)
            loss = loss_fn(pred, yb)

            internal_losses.append(float(loss.detach().cpu()))
            internal_true.append(yb.detach().cpu().numpy())
            internal_pred.append(pred.detach().cpu().numpy())

    internal_true = np.concatenate(internal_true, axis=0)
    internal_pred = np.concatenate(internal_pred, axis=0)

    internal_loss = float(np.mean(internal_losses))
    internal_r = mean_gene_pearson(internal_true, internal_pred)

    print(
        f"[Internal outer-test] loss={internal_loss:.6f}, "
        f"mean_gene_pearson={internal_r:.6f}"
    )

    with open(out_dir / "internal_test_metrics.json", "w") as f:
        json.dump(
            {
                "internal_test_loss": internal_loss,
                "internal_test_mean_gene_pearson": internal_r,
                "n_internal_test_tiles": int(len(internal_test_idx)),
            },
            f,
            indent=2,
        )

    print(f"[MLP saved] {mlp_ckpt}")

    return model, mlp_ckpt, internal_loss, internal_r


def load_mlp(mlp_ckpt, device):
    ckpt = torch.load(mlp_ckpt, map_location=device)

    model = MLP_regression(
        n_inputs=int(ckpt.get("n_inputs", 512)),
        n_hiddens=int(ckpt.get("n_hiddens", 512)),
        n_outputs=int(ckpt["n_outputs"]),
        dropout=float(ckpt.get("dropout", 0.2)),
        bias_init=None,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, ckpt


@torch.no_grad()
def predict_mlp_tile(model, X_ae, batch_size, device):
    loader = DataLoader(
        FeatureDataset(X_ae),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    preds = []
    model.eval()

    for xb in loader:
        xb = xb.to(device).float()
        pred = model.forward_tile(xb)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(preds, axis=0).astype(np.float32)


# ============================================================
# Output
# ============================================================

def save_prediction_h5(
    out_h5,
    pred,
    genes,
    coords,
    truth=None,
    norm="log1p_cpm",
    tile_meta_csv=None,
):
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "predicted_expression",
            data=pred.astype(np.float32),
            compression="gzip",
            chunks=(min(512, pred.shape[0]), min(512, pred.shape[1])),
        )

        if truth is not None:
            f.create_dataset(
                "true_expression",
                data=truth.astype(np.float32),
                compression="gzip",
                chunks=(min(512, truth.shape[0]), min(512, truth.shape[1])),
            )

        f.create_dataset(
            "genes",
            data=np.asarray(genes, dtype=object),
            dtype=str_dtype,
        )

        f.create_dataset(
            "tile_coords",
            data=coords.astype(np.float32),
            compression="gzip",
        )

        f.attrs["norm"] = norm

        if tile_meta_csv is not None:
            f.attrs["tile_meta_csv"] = str(tile_meta_csv)

    print(f"[Saved h5] {out_h5}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--train_sample", default=None)
    parser.add_argument("--test_sample", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--predict_only", action="store_true")

    parser.add_argument("--ae_ckpt", default=None)
    parser.add_argument("--mlp_ckpt", default=None)
    parser.add_argument("--gene_list", default=None)

    parser.add_argument("--resnet_weight", required=True)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--force_recompute_features", action="store_true")

    parser.add_argument(
        "--no_macenko",
        action="store_true",
        help="Disable Macenko normalization. Default is to use Macenko.",
    )

    parser.add_argument("--norm", choices=["raw", "log1p", "log1p_cpm"], default="log1p_cpm")
    parser.add_argument("--scale_factor", type=float, default=1e4)

    # Nested CV.
    parser.add_argument("--n_outer_folds", type=int, default=5)
    parser.add_argument("--n_inner_folds", type=int, default=5)
    parser.add_argument("--ik_fold", type=int, default=0)
    parser.add_argument("--il_fold", type=int, default=0)

    # Original DeepPT AE default: 500 epochs, lr=1e-4, batch_size=32.
    parser.add_argument("--ae_epochs", type=int, default=500)
    parser.add_argument("--ae_batch_size", type=int, default=32)
    parser.add_argument("--ae_lr", type=float, default=1e-4)
    parser.add_argument("--ae_valid_fraction", type=float, default=0.1)

    # Original DeepPT MLP default: 500 epochs, patience=50, dropout=0.2, lr=1e-4.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pred_batch_size", type=int, default=4096)
    parser.add_argument("--feature_batch_size", type=int, default=64)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")

    args = parser.parse_args()

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_random_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu"
    use_macenko = not args.no_macenko

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[Config]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print(f"  use_macenko: {use_macenko}")
    print("=" * 100)

    # -----------------------------
    # Load test sample
    # -----------------------------
    test_dir = get_sample_dir(args.input_dir, args.test_sample)
    test_genes_all = load_genes(test_dir)
    test_coords = np.load(test_dir / "tile_coords.npy").astype(np.float32)

    # -----------------------------
    # Predict-only mode
    # -----------------------------
    if args.predict_only:
        if args.ae_ckpt is None:
            raise ValueError("--predict_only requires --ae_ckpt")
        if args.mlp_ckpt is None:
            raise ValueError("--predict_only requires --mlp_ckpt")
        if args.gene_list is None:
            raise ValueError("--predict_only requires --gene_list")

        genes = (
            pd.read_csv(args.gene_list, sep="\t", header=None)
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )

        test_map = {g: i for i, g in enumerate(test_genes_all)}
        missing = [g for g in genes if g not in test_map]

        if len(missing) > 0:
            raise ValueError(
                f"Test sample missing {len(missing)} genes from training gene list. "
                f"First missing genes: {missing[:20]}"
            )

        test_idx = [test_map[g] for g in genes]

        X_test_resnet = extract_or_load_resnet_features(
            sample_dir=test_dir,
            tile_size=args.tile_size,
            batch_size=args.feature_batch_size,
            resnet_weight=args.resnet_weight,
            device=device,
            use_macenko=use_macenko,
            force_recompute=args.force_recompute_features,
        )

        ae_model = load_autoencoder(args.ae_ckpt, device=device)

        X_test_ae = compress_features_with_ae(
            ae_model,
            X_test_resnet,
            batch_size=args.pred_batch_size,
            device=device,
        )

        mlp_model, mlp_ckpt = load_mlp(args.mlp_ckpt, device=device)

        if int(mlp_ckpt["n_outputs"]) != len(genes):
            raise ValueError(
                f"MLP output dim = {mlp_ckpt['n_outputs']}, "
                f"but gene_list has {len(genes)} genes."
            )

        pred = predict_mlp_tile(
            mlp_model,
            X_test_ae,
            batch_size=args.pred_batch_size,
            device=device,
        )

        Y_test = normalize_counts(
            load_counts(test_dir, test_idx),
            mode=args.norm,
            scale_factor=args.scale_factor,
        )

        r = mean_gene_pearson(Y_test, pred)

        print(f"[External eval] {args.test_sample}: mean_gene_pearson={r:.6f}")

        out_h5 = out_dir / f"{args.test_sample}_predicted_expression.h5"

        save_prediction_h5(
            out_h5=out_h5,
            pred=pred,
            genes=genes,
            coords=test_coords,
            truth=Y_test,
            norm=args.norm,
            tile_meta_csv=test_dir / "tile_meta.csv",
        )

        pd.Series(genes).to_csv(out_dir / "gene_list.tsv", sep="\t", index=False, header=False)

        meta = {
            "mode": "predict_only",
            "test_sample": args.test_sample,
            "ae_ckpt": args.ae_ckpt,
            "mlp_ckpt": args.mlp_ckpt,
            "gene_list": args.gene_list,
            "prediction_h5": str(out_h5),
            "external_mean_gene_pearson": r,
            "use_macenko": use_macenko,
            "args": vars(args),
        }

        with open(out_dir / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print("[Done]")
        return

    # -----------------------------
    # Train + external test mode
    # -----------------------------
    if args.train_sample is None:
        raise ValueError("Training mode requires --train_sample")

    train_dir = get_sample_dir(args.input_dir, args.train_sample)
    train_genes_all = load_genes(train_dir)

    requested_genes = None
    if args.gene_list is not None:
        requested_genes = (
            pd.read_csv(args.gene_list, sep="\t", header=None)
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )

    genes, train_gene_idx, test_gene_idx = common_gene_indices(
        train_genes_all,
        test_genes_all,
        requested_genes=requested_genes,
    )

    print(
        f"[Genes] train={len(train_genes_all)}, "
        f"test={len(test_genes_all)}, "
        f"used={len(genes)}"
    )

    pd.Series(genes).to_csv(out_dir / "gene_list.tsv", sep="\t", index=False, header=False)

    # ResNet features with Macenko.
    X_train_resnet = extract_or_load_resnet_features(
        sample_dir=train_dir,
        tile_size=args.tile_size,
        batch_size=args.feature_batch_size,
        resnet_weight=args.resnet_weight,
        device=device,
        use_macenko=use_macenko,
        force_recompute=args.force_recompute_features,
    )

    X_test_resnet = extract_or_load_resnet_features(
        sample_dir=test_dir,
        tile_size=args.tile_size,
        batch_size=args.feature_batch_size,
        resnet_weight=args.resnet_weight,
        device=device,
        use_macenko=use_macenko,
        force_recompute=args.force_recompute_features,
    )

    Y_train = normalize_counts(
        load_counts(train_dir, train_gene_idx),
        mode=args.norm,
        scale_factor=args.scale_factor,
    )

    Y_test = normalize_counts(
        load_counts(test_dir, test_gene_idx),
        mode=args.norm,
        scale_factor=args.scale_factor,
    )

    print(f"[Train ResNet feature] {X_train_resnet.shape}")
    print(f"[Test ResNet feature]  {X_test_resnet.shape}")
    print(f"[Train expression]     {Y_train.shape}")
    print(f"[Test expression]      {Y_test.shape}")

    # Nested CV split on training sample tiles.
    train_idx, valid_idx, internal_test_idx = make_nested_cv_indices(
        n_items=X_train_resnet.shape[0],
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        ik_fold=args.ik_fold,
        il_fold=args.il_fold,
        seed=args.seed,
    )

    np.savez(
        out_dir / "nested_cv_indices.npz",
        train_idx=train_idx,
        valid_idx=valid_idx,
        internal_test_idx=internal_test_idx,
        ik_fold=args.ik_fold,
        il_fold=args.il_fold,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
    )

    print(
        f"[Nested CV indices saved] {out_dir / 'nested_cv_indices.npz'}"
    )

    # Train original DeepPT AE on training sample ResNet features.
    ae_model, ae_ckpt = train_autoencoder(
        X_train_resnet,
        out_dir=out_dir,
        args=args,
        device=device,
    )

    # Compress features using original DeepPT AE encoder.
    X_train_ae = compress_features_with_ae(
        ae_model,
        X_train_resnet,
        batch_size=args.pred_batch_size,
        device=device,
    )

    X_test_ae = compress_features_with_ae(
        ae_model,
        X_test_resnet,
        batch_size=args.pred_batch_size,
        device=device,
    )

    print(f"[Train AE feature] {X_train_ae.shape}")
    print(f"[Test AE feature]  {X_test_ae.shape}")

    # Train original DeepPT MLP with nested CV train/valid/internal-test.
    mlp_model, mlp_ckpt, internal_loss, internal_r = train_mlp_nested_cv(
        X_train_ae=X_train_ae,
        Y_train=Y_train,
        train_idx=train_idx,
        valid_idx=valid_idx,
        internal_test_idx=internal_test_idx,
        out_dir=out_dir,
        args=args,
        device=device,
    )

    # External test prediction.
    pred = predict_mlp_tile(
        mlp_model,
        X_test_ae,
        batch_size=args.pred_batch_size,
        device=device,
    )

    external_r = mean_gene_pearson(Y_test, pred)

    print(
        f"[External eval] {args.train_sample} -> {args.test_sample}: "
        f"mean_gene_pearson={external_r:.6f}"
    )

    out_h5 = out_dir / f"{args.test_sample}_predicted_expression.h5"

    save_prediction_h5(
        out_h5=out_h5,
        pred=pred,
        genes=genes,
        coords=test_coords,
        truth=Y_test,
        norm=args.norm,
        tile_meta_csv=test_dir / "tile_meta.csv",
    )

    meta = {
        "mode": "train_nested_cv_and_external_test",
        "train_sample": args.train_sample,
        "test_sample": args.test_sample,
        "n_genes": len(genes),
        "n_train_tiles": int(X_train_resnet.shape[0]),
        "n_external_test_tiles": int(X_test_resnet.shape[0]),
        "n_nested_train_tiles": int(len(train_idx)),
        "n_nested_valid_tiles": int(len(valid_idx)),
        "n_nested_internal_test_tiles": int(len(internal_test_idx)),
        "ik_fold": args.ik_fold,
        "il_fold": args.il_fold,
        "n_outer_folds": args.n_outer_folds,
        "n_inner_folds": args.n_inner_folds,
        "ae_ckpt": str(ae_ckpt),
        "mlp_ckpt": str(mlp_ckpt),
        "prediction_h5": str(out_h5),
        "internal_test_loss": internal_loss,
        "internal_test_mean_gene_pearson": internal_r,
        "external_mean_gene_pearson": external_r,
        "use_macenko": use_macenko,
        "args": vars(args),
    }

    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[Done]")


if __name__ == "__main__":
    main()
