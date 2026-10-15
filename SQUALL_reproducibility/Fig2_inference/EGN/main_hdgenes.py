#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tile-level HD-gene EGN using original EGN model body.

Key design:
  1. Import original EGN from egn.py:
       from egn import EGN

  2. Keep original EGN body:
       patch embedding + Transformer + EB + CSRA + MLP head

  3. Keep original exemplar/KNN idea:
       ResNet50 embedding -> cdist/topk -> op_feature/op_count
     But do NOT average K exemplars into one prior vector.

  4. Only patch 250-gene-specific layers at runtime:
       model.transformer.encoder.proe: Linear(mdim + 250, dim) -> Linear(mdim + gene_dim, dim)
       model.transformer.encoder.to_v: Linear(250, dim) -> Linear(gene_dim, dim)
       model.mlp_head[-1]            : Linear(dim*2, 250) -> Linear(dim*2, gene_dim)

Input root:
  EGNv1_tile_input_allgenes/
    samples/
      SAMPLE/
        image_path.txt
        genes.tsv
        tile_counts.h5   with dataset "counts"
        tile_coords.npy
        tile_meta.csv

Output:
  out_dir/
    best_model.pt
    selected_genes.tsv
    train_stats.npz
    xenium_predicted_expression.h5
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
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as T

from egn import EGN


Image.MAX_IMAGE_PIXELS = None


# =========================
# Utils
# =========================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in ["true", "1", "yes", "y"]:
        return True
    if x in ["false", "0", "no", "n"]:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool: {x}")


def read_counts_h5(path):
    with h5py.File(path, "r") as f:
        if "counts" not in f:
            raise KeyError(f"{path} does not contain dataset 'counts'")
        X = f["counts"][:].astype(np.float32)
    return X


def load_tile_sample(tile_root, sample_name):
    sample_dir = Path(tile_root) / "samples" / sample_name
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample dir: {sample_dir}")

    image_path = open(sample_dir / "image_path.txt").read().strip()
    genes = pd.read_csv(sample_dir / "genes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).tolist()
    coords = np.load(sample_dir / "tile_coords.npy").astype(np.float32)
    counts = read_counts_h5(sample_dir / "tile_counts.h5")
    meta = pd.read_csv(sample_dir / "tile_meta.csv")

    if counts.shape[0] != coords.shape[0]:
        raise ValueError(f"{sample_name}: counts rows {counts.shape[0]} != coords rows {coords.shape[0]}")
    if counts.shape[1] != len(genes):
        raise ValueError(f"{sample_name}: counts cols {counts.shape[1]} != genes {len(genes)}")

    return {
        "sample_name": sample_name,
        "sample_dir": str(sample_dir),
        "image_path": image_path,
        "genes": genes,
        "coords": coords,
        "counts": counts,
        "meta": meta,
    }


def log_transform_counts(X, mode="log10"):
    X = X.astype(np.float32)
    if mode == "log10":
        return np.log10(X + 1.0).astype(np.float32)
    if mode == "log1p":
        return np.log1p(X).astype(np.float32)
    raise ValueError("--log_mode must be log10 or log1p")


def fit_minmax(Y):
    ymin = Y.min(axis=0, keepdims=True).astype(np.float32)
    ymax = Y.max(axis=0, keepdims=True).astype(np.float32)
    denom = ymax - ymin
    denom[denom < 1e-8] = 1.0
    return ymin, ymax


def apply_minmax(Y, ymin, ymax):
    denom = ymax - ymin
    denom[denom < 1e-8] = 1.0
    return ((Y - ymin) / denom).astype(np.float32)


def inverse_minmax(Ys, ymin, ymax):
    return Ys * (ymax - ymin) + ymin


def save_string_dataset(h5, name, values):
    arr = np.asarray([str(x).encode("utf-8") for x in values])
    h5.create_dataset(name, data=arr)


def compute_gene_corr(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    corr = []
    for i in range(y_true.shape[1]):
        a = y_true[:, i]
        b = y_pred[:, i]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            corr.append(np.nan)
        else:
            corr.append(np.corrcoef(a, b)[0, 1])
    return np.asarray(corr, dtype=np.float32)


# =========================
# Image patch utils
# =========================

def crop_patch_pil(image, x, y, size=256):
    x = int(round(float(x)))
    y = int(round(float(y)))
    half = size // 2

    left = x - half
    upper = y - half
    right = left + size
    lower = upper + size

    W, H = image.size

    crop_left = max(left, 0)
    crop_upper = max(upper, 0)
    crop_right = min(right, W)
    crop_lower = min(lower, H)

    patch = Image.new("RGB", (size, size), (255, 255, 255))

    if crop_right > crop_left and crop_lower > crop_upper:
        crop = image.crop((crop_left, crop_upper, crop_right, crop_lower))
        patch.paste(crop, (crop_left - left, crop_upper - upper))

    return patch


class TilePatchDataset(Dataset):
    """
    Return original EGN image input patch.

    Original main.py used:
      ToTensor()
      Normalize(mean=[0.5476, 0.5218, 0.6881],
                std =[0.2461, 0.2101, 0.1649])
    """
    def __init__(self, image_path, coords, size=256):
        self.image = Image.open(image_path).convert("RGB")
        self.coords = coords.astype(np.float32)
        self.size = int(size)

        self.transform_egn = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.5476, 0.5218, 0.6881],
                std=[0.2461, 0.2101, 0.1649],
            ),
        ])

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = crop_patch_pil(self.image, x, y, size=self.size)
        img = self.transform_egn(patch)
        return img, idx


class ResNetPatchDataset(Dataset):
    """
    Return patches for original-style ResNet exemplar embedding.

    Original build_exemplar.py used:
      img = patch / 255
      return (img - 0.5) / 0.5
    """
    def __init__(self, image_path, coords, size=256):
        self.image = Image.open(image_path).convert("RGB")
        self.coords = coords.astype(np.float32)
        self.size = int(size)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = crop_patch_pil(self.image, x, y, size=self.size)
        arr = np.asarray(patch).astype(np.float32) / 255.0
        img = torch.from_numpy(arr).permute(2, 0, 1)
        img = (img - 0.5) / 0.5
        return img, idx


# =========================
# ResNet50 embedding for exemplar retrieval
# =========================

def build_resnet50_encoder(device, resnet_weight=None):
    model = torchvision.models.resnet50(weights=None)

    if resnet_weight is not None and str(resnet_weight).lower() != "none":
        print(f"[Load ResNet50 weights] {resnet_weight}")
        state = torch.load(resnet_weight, map_location="cpu")

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("resnet."):
                k = k[len("resnet."):]
            new_state[k] = v

        msg = model.load_state_dict(new_state, strict=False)
        print("[ResNet load msg]", msg)

    features = model.fc.in_features
    encoder = nn.Sequential(*list(model.children())[:-1])
    encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder, features


@torch.no_grad()
def compute_resnet_embeddings(
    sample,
    cache_path,
    size,
    batch_size,
    workers,
    device,
    resnet_weight,
):
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[Load embedding cache] {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    ds = ResNetPatchDataset(
        image_path=sample["image_path"],
        coords=sample["coords"],
        size=size,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    encoder, features = build_resnet50_encoder(device, resnet_weight=resnet_weight)

    embs = []
    for imgs, idx in tqdm(loader, desc=f"Extract ResNet50 embeddings: {sample['sample_name']}"):
        imgs = imgs.to(device, non_blocking=True)
        emb = encoder(imgs).view(imgs.size(0), features)
        embs.append(emb.detach().cpu())

    embs = torch.cat(embs, dim=0).contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, cache_path)

    print(f"[Saved embedding cache] {cache_path}, shape={tuple(embs.shape)}")
    return embs


# =========================
# Original-style KNN exemplar
# =========================

@torch.no_grad()
def build_knn_indices(
    query_emb,
    ref_emb,
    k,
    batch_size,
    device,
    exclude_self=False,
):
    """
    This follows original build_exemplar idea:
      distance = torch.cdist(query_emb, ref_emb)
      topk smallest distance
    It returns indices [N_query, k].
    """
    ref_gpu = ref_emb.float().to(device)
    all_inds = []

    n = query_emb.size(0)

    for start in tqdm(range(0, n, batch_size), desc="Build KNN exemplar index"):
        end = min(start + batch_size, n)
        q = query_emb[start:end].float().to(device)

        dist = torch.cdist(q, ref_gpu, p=2)

        if exclude_self:
            rows = torch.arange(end - start, device=device)
            cols = torch.arange(start, end, device=device)
            valid = cols < ref_gpu.size(0)
            dist[rows[valid], cols[valid]] = float("inf")

        kk = min(k, ref_gpu.size(0))
        vals, inds = torch.topk(dist, k=kk, dim=1, largest=False)

        if kk < k:
            pad = inds[:, -1:].repeat(1, k - kk)
            inds = torch.cat([inds, pad], dim=1)

        all_inds.append(inds.cpu())

    return torch.cat(all_inds, dim=0).long().contiguous()


# =========================
# Dataset feeding original EGN.forward(data)
# =========================

class EGNTrainDataset(Dataset):
    def __init__(
        self,
        image_path,
        coords,
        y_scaled,
        p_feature,
        ref_feature,
        ref_expr_scaled,
        knn_indices,
        size,
    ):
        self.patch_ds = TilePatchDataset(image_path=image_path, coords=coords, size=size)
        self.y_scaled = torch.from_numpy(y_scaled).float()
        self.p_feature = p_feature.float()
        self.ref_feature = ref_feature.float()
        self.ref_expr_scaled = ref_expr_scaled.float()
        self.knn_indices = knn_indices.long()

    def __len__(self):
        return self.y_scaled.size(0)

    def __getitem__(self, idx):
        img, _ = self.patch_ds[idx]
        inds = self.knn_indices[idx]

        return {
            "img": img,
            "count": self.y_scaled[idx],
            "p_feature": self.p_feature[idx].unsqueeze(0),
            "op_feature": self.ref_feature[inds],
            "op_count": self.ref_expr_scaled[inds],
        }


class EGNTestDataset(Dataset):
    def __init__(
        self,
        image_path,
        coords,
        p_feature,
        ref_feature,
        ref_expr_scaled,
        knn_indices,
        size,
    ):
        self.patch_ds = TilePatchDataset(image_path=image_path, coords=coords, size=size)
        self.p_feature = p_feature.float()
        self.ref_feature = ref_feature.float()
        self.ref_expr_scaled = ref_expr_scaled.float()
        self.knn_indices = knn_indices.long()

    def __len__(self):
        return self.p_feature.size(0)

    def __getitem__(self, idx):
        img, _ = self.patch_ds[idx]
        inds = self.knn_indices[idx]

        return {
            "img": img,
            "p_feature": self.p_feature[idx].unsqueeze(0),
            "op_feature": self.ref_feature[inds],
            "op_count": self.ref_expr_scaled[inds],
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =========================
# Patch original EGN 250-gene layers only
# =========================

def patch_original_egn_gene_dim(model, gene_dim, mdim, dim, linear_projection):
    """
    Only patch gene-dimension-specific layers.
    Other EGN body remains original.
    """
    device = next(model.parameters()).device

    model.transformer.encoder.proe = nn.Linear(mdim + gene_dim, dim).to(device)

    if linear_projection:
        model.transformer.encoder.to_v = nn.Linear(gene_dim, dim).to(device)
    else:
        model.transformer.encoder.to_v = nn.Sequential(
            nn.Linear(gene_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        ).to(device)

    model.mlp_head[-1] = nn.Linear(dim * 2, gene_dim).to(device)
    model.gene_dim = gene_dim

    return model


def build_egn_model(args, gene_dim, device):
    model = EGN(
        image_size=args.size,
        patch_size=args.patch_size,
        num_classes=250,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        bhead=args.bhead,
        bdim=args.bdim,
        bfre=args.bfre,
        mdim=args.mdim,
        player=args.player,
        linear_projection=args.linear_projection,
    )

    model.to(device)

    model = patch_original_egn_gene_dim(
        model=model,
        gene_dim=gene_dim,
        mdim=args.mdim,
        dim=args.dim,
        linear_projection=args.linear_projection,
    )

    return model


# =========================
# Training / prediction
# =========================

def correlation_loss(pred, target):
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=0) + 1e-8) * torch.sqrt((target_centered ** 2).sum(dim=0) + 1e-8)

    corr = numerator / (denominator + 1e-8)
    corr = torch.nan_to_num(corr, nan=0.0)
    return 1.0 - corr.mean()


def train_model(model, train_ds, args, out_dir, device):
    n_val = max(1, int(len(train_ds) * args.val_fraction))
    n_train = len(train_ds) - n_val

    gen = torch.Generator().manual_seed(args.seed)
    ds_train, ds_val = random_split(train_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    mse = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    best_path = Path(out_dir) / "best_model.pt"

    for epoch in range(1, args.epoch + 1):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epoch}"):
            batch = move_batch_to_device(batch, device)

            pred = model(batch)
            loss_mse = mse(pred, batch["count"])
            loss_corr = correlation_loss(pred, batch["count"])
            loss = loss_mse + args.corr_weight * loss_corr

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        pred_all = []
        true_all = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validate epoch {epoch}/{args.epoch}"):
                batch = move_batch_to_device(batch, device)

                pred = model(batch)
                loss = mse(pred, batch["count"])

                val_losses.append(float(loss.detach().cpu()))
                pred_all.append(pred.detach().cpu())
                true_all.append(batch["count"].detach().cpu())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        pred_cat = torch.cat(pred_all, dim=0)
        true_cat = torch.cat(true_all, dim=0)
        gene_corr = compute_gene_corr(true_cat, pred_cat)
        mean_corr = float(np.nanmean(gene_corr))

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_mse={val_loss:.6f} "
            f"val_mean_gene_corr={mean_corr:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"[Saved best] {best_path}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    return model


@torch.no_grad()
def predict_to_h5(
    model,
    test_ds,
    out_h5,
    genes,
    coords,
    meta,
    ymin,
    ymax,
    args,
    device,
):
    loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    model.to(device)

    n = len(test_ds)
    g = len(genes)

    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    ymin_t = torch.from_numpy(ymin).float().to(device)
    ymax_t = torch.from_numpy(ymax).float().to(device)

    with h5py.File(out_h5, "w") as f:
        d_scaled = f.create_dataset(
            "pred_scaled",
            shape=(n, g),
            dtype="float32",
            chunks=(min(256, n), min(512, g)),
            compression="gzip",
        )

        d_lognorm = f.create_dataset(
            "pred_lognorm",
            shape=(n, g),
            dtype="float32",
            chunks=(min(256, n), min(512, g)),
            compression="gzip",
        )

        save_string_dataset(f, "genes", genes)
        f.create_dataset("coords", data=coords.astype(np.float32))

        if "tile_id" in meta.columns:
            save_string_dataset(f, "tile_id", meta["tile_id"].astype(str).tolist())

        if "n_bins" in meta.columns:
            f.create_dataset("n_bins", data=meta["n_bins"].values.astype(np.int64))

        start = 0

        for batch in tqdm(loader, desc="Predict"):
            batch = move_batch_to_device(batch, device)

            pred_scaled = model(batch)
            pred_lognorm = inverse_minmax(pred_scaled, ymin_t, ymax_t)

            end = start + pred_scaled.size(0)
            d_scaled[start:end, :] = pred_scaled.detach().cpu().numpy().astype(np.float32)
            d_lognorm[start:end, :] = pred_lognorm.detach().cpu().numpy().astype(np.float32)
            start = end

    print(f"[Saved prediction] {out_h5}")


def prepare_train(args, out_dir, device):
    train_sample = load_tile_sample(args.tile_root, args.train_sample)

    genes = list(train_sample["genes"])
    gene_dim = len(genes)

    print(f"[Train sample] {args.train_sample}")
    print(f"[Train tiles]  {train_sample['counts'].shape[0]}")
    print(f"[Gene dim]     {gene_dim}")

    pd.Series(genes).to_csv(Path(out_dir) / "selected_genes.tsv", sep="\t", index=False, header=False)

    Y_log = log_transform_counts(train_sample["counts"], mode=args.log_mode)
    ymin, ymax = fit_minmax(Y_log)
    Y_scaled = apply_minmax(Y_log, ymin, ymax)

    np.savez(
        Path(out_dir) / "train_stats.npz",
        ymin=ymin,
        ymax=ymax,
        genes=np.asarray(genes, dtype=object),
        train_sample=args.train_sample,
        log_mode=args.log_mode,
    )

    train_emb = compute_resnet_embeddings(
        sample=train_sample,
        cache_path=Path(out_dir) / "cache" / f"{args.train_sample}_resnet50_emb.pt",
        size=args.size,
        batch_size=args.embed_batch_size,
        workers=args.workers,
        device=device,
        resnet_weight=args.resnet_weight,
    )

    train_expr_scaled = torch.from_numpy(Y_scaled).float()

    train_knn = build_knn_indices(
        query_emb=train_emb,
        ref_emb=train_emb,
        k=args.numk,
        batch_size=args.knn_batch,
        device=device,
        exclude_self=True,
    )

    train_ds = EGNTrainDataset(
        image_path=train_sample["image_path"],
        coords=train_sample["coords"],
        y_scaled=Y_scaled,
        p_feature=train_emb,
        ref_feature=train_emb,
        ref_expr_scaled=train_expr_scaled,
        knn_indices=train_knn,
        size=args.size,
    )

    return train_sample, genes, gene_dim, ymin, ymax, train_emb, train_expr_scaled, train_ds


def prepare_test(args, test_sample_name, train_emb, train_expr_scaled, out_dir, device):
    test_sample = load_tile_sample(args.tile_root, test_sample_name)

    print(f"[Test sample] {test_sample_name}")
    print(f"[Test tiles]  {test_sample['counts'].shape[0]}")

    test_emb = compute_resnet_embeddings(
        sample=test_sample,
        cache_path=Path(out_dir) / "cache" / f"{test_sample_name}_resnet50_emb.pt",
        size=args.size,
        batch_size=args.embed_batch_size,
        workers=args.workers,
        device=device,
        resnet_weight=args.resnet_weight,
    )

    test_knn = build_knn_indices(
        query_emb=test_emb,
        ref_emb=train_emb,
        k=args.numk,
        batch_size=args.knn_batch,
        device=device,
        exclude_self=False,
    )

    test_ds = EGNTestDataset(
        image_path=test_sample["image_path"],
        coords=test_sample["coords"],
        p_feature=test_emb,
        ref_feature=train_emb,
        ref_expr_scaled=train_expr_scaled,
        knn_indices=test_knn,
        size=args.size,
    )

    return test_sample, test_ds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train_predict", "predict_only"], default="train_predict")
    parser.add_argument("--tile_root", required=True)
    parser.add_argument("--train_sample", required=True)
    parser.add_argument("--test_sample", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--train_out_dir", default=None)

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--corr_weight", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=4096)
    parser.add_argument("--bhead", type=int, default=8)
    parser.add_argument("--bdim", type=int, default=64)
    parser.add_argument("--bfre", type=int, default=2)
    parser.add_argument("--mdim", type=int, default=2048)
    parser.add_argument("--player", type=int, default=1)
    parser.add_argument("--linear_projection", type=str2bool, default=True)

    parser.add_argument("--numk", type=int, default=16)
    parser.add_argument("--knn_batch", type=int, default=512)

    parser.add_argument("--resnet_weight", default=None)
    parser.add_argument("--log_mode", choices=["log10", "log1p"], default="log10")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.mode == "train_predict":
        train_sample, genes, gene_dim, ymin, ymax, train_emb, train_expr_scaled, train_ds = prepare_train(
            args=args,
            out_dir=out_dir,
            device=device,
        )

        print("[Build original EGN and patch only gene-dim layers]")
        model = build_egn_model(args, gene_dim=gene_dim, device=device)

        if args.ckpt is not None:
            print(f"[Load ckpt before train] {args.ckpt}")
            ckpt = torch.load(args.ckpt, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
            else:
                model.load_state_dict(ckpt, strict=True)

        print("[Train]")
        model = train_model(
            model=model,
            train_ds=train_ds,
            args=args,
            out_dir=out_dir,
            device=device,
        )

        test_sample, test_ds = prepare_test(
            args=args,
            test_sample_name=args.test_sample,
            train_emb=train_emb,
            train_expr_scaled=train_expr_scaled,
            out_dir=out_dir,
            device=device,
        )

        print("[Predict]")
        predict_to_h5(
            model=model,
            test_ds=test_ds,
            out_h5=out_dir / "xenium_predicted_expression.h5",
            genes=genes,
            coords=test_sample["coords"],
            meta=test_sample["meta"],
            ymin=ymin,
            ymax=ymax,
            args=args,
            device=device,
        )

    elif args.mode == "predict_only":
        if args.ckpt is None:
            raise ValueError("--ckpt is required in predict_only mode")
        if args.train_out_dir is None:
            raise ValueError("--train_out_dir is required in predict_only mode")

        train_out_dir = Path(args.train_out_dir)

        genes = pd.read_csv(train_out_dir / "selected_genes.tsv", sep="\t", header=None).iloc[:, 0].astype(str).tolist()
        gene_dim = len(genes)

        stats = np.load(train_out_dir / "train_stats.npz", allow_pickle=True)
        ymin = stats["ymin"].astype(np.float32)
        ymax = stats["ymax"].astype(np.float32)
        log_mode = str(stats["log_mode"]) if "log_mode" in stats.files else args.log_mode
        args.log_mode = log_mode

        train_sample = load_tile_sample(args.tile_root, args.train_sample)
        train_gene_to_idx = {g: i for i, g in enumerate(train_sample["genes"])}
        train_idx = [train_gene_to_idx[g] for g in genes]

        train_counts = train_sample["counts"][:, train_idx].astype(np.float32)
        train_log = log_transform_counts(train_counts, mode=args.log_mode)
        train_scaled = apply_minmax(train_log, ymin, ymax)
        train_expr_scaled = torch.from_numpy(train_scaled).float()

        train_emb = compute_resnet_embeddings(
            sample=train_sample,
            cache_path=train_out_dir / "cache" / f"{args.train_sample}_resnet50_emb.pt",
            size=args.size,
            batch_size=args.embed_batch_size,
            workers=args.workers,
            device=device,
            resnet_weight=args.resnet_weight,
        )

        print("[Build original EGN and patch only gene-dim layers]")
        model = build_egn_model(args, gene_dim=gene_dim, device=device)

        print(f"[Load ckpt] {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

        test_sample, test_ds = prepare_test(
            args=args,
            test_sample_name=args.test_sample,
            train_emb=train_emb,
            train_expr_scaled=train_expr_scaled,
            out_dir=out_dir,
            device=device,
        )

        print("[Predict]")
        predict_to_h5(
            model=model,
            test_ds=test_ds,
            out_h5=out_dir / "xenium_predicted_expression.h5",
            genes=genes,
            coords=test_sample["coords"],
            meta=test_sample["meta"],
            ymin=ymin,
            ymax=ymax,
            args=args,
            device=device,
        )

    print("[Done]")


if __name__ == "__main__":
    main()
