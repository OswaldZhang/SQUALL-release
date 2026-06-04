#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # server-safe
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# =====================================================
# Config
# =====================================================
BASE_DIR = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/CESC_inference_vector_lowres"
SURVIVAL_JSON = "/lustre1/zxzeng/bwqin/SQUALL_main/downstream_labels/Survival_five_fold_DSS_COX_outs/Survival_TCGA_CESC.json"
GENE_MAP = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/gene_token_homologs.csv"

OUT_DIR = "pseudobulk_PCA"
os.makedirs(OUT_DIR, exist_ok=True)

# preprocessing
USE_LOG1P = True
Z_SCORE_GENES = True

# cache (强烈推荐)
CACHE_PB = True
PB_MATRIX_NPY = os.path.join(OUT_DIR, "pseudobulk_matrix.npy")
PB_SAMPLES_NPY = os.path.join(OUT_DIR, "pseudobulk_samples.npy")

# =====================================================
# Load survival (for coloring / filtering)
# =====================================================
with open(SURVIVAL_JSON) as f:
    survival = json.load(f)

# =====================================================
# Load gene map (sanity check gene count)
# =====================================================
gene_df = pd.read_csv(GENE_MAP)
gene_df = gene_df.sort_values(gene_df.columns[0])
genes = gene_df["HGNC_symbol"].tolist()
G = len(genes)
print(f"[INFO] Gene count = {G}")

# =====================================================
# Build pseudobulk matrix (with mmap)
# =====================================================
if CACHE_PB and os.path.exists(PB_MATRIX_NPY):
    print("[INFO] Loading cached pseudobulk matrix")
    PB = np.load(PB_MATRIX_NPY)
    samples = np.load(PB_SAMPLES_NPY).tolist()

else:
    print("[INFO] Building pseudobulk matrix from expr.npy")

    samples = []
    PB_list = []

    for sample in tqdm(sorted(os.listdir(BASE_DIR)), desc="Pseudobulk"):
        sample_dir = os.path.join(BASE_DIR, sample)
        if not os.path.isdir(sample_dir):
            continue
        if sample not in survival:
            continue

        expr_path = os.path.join(sample_dir, "expr.npy")
        if not os.path.exists(expr_path):
            continue

        # 🔥 关键：mmap，不整块读入内存
        expr = np.load(expr_path, mmap_mode="r")   # (N_tiles, G)

        if expr.ndim != 2 or expr.shape[1] != G:
            print(f"[WARN] Skip {sample}, bad shape {expr.shape}")
            continue

        pb = expr.mean(axis=0)   # pseudobulk
        PB_list.append(pb)
        samples.append(sample)

    PB = np.vstack(PB_list)     # (n_samples, G)

    if CACHE_PB:
        np.save(PB_MATRIX_NPY, PB)
        np.save(PB_SAMPLES_NPY, np.array(samples))
        print("[INFO] Cached pseudobulk matrix")

print("[INFO] Pseudobulk matrix shape:", PB.shape)

# =====================================================
# Preprocessing
# =====================================================
X = PB.astype(np.float32)

if USE_LOG1P:
    X = np.log1p(X)

if Z_SCORE_GENES:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(X)

# =====================================================
# PCA
# =====================================================
print("[INFO] Running PCA")
pca = PCA(n_components=10, random_state=0)
X_pca = pca.fit_transform(X)

print("[INFO] Explained variance ratio (PC1–PC5):")
for i, v in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"  PC{i}: {v:.4f}")

# =====================================================
# Save PCA table
# =====================================================
df_pca = pd.DataFrame(
    X_pca[:, :5],
    columns=[f"PC{i}" for i in range(1, 6)],
    index=samples
)

df_pca["time"] = [survival[s]["time"] for s in samples]
df_pca["event"] = [survival[s]["status"] for s in samples]

df_pca_path = os.path.join(OUT_DIR, "pseudobulk_PCA_scores.csv")
df_pca.to_csv(df_pca_path)
print("[INFO] Saved:", df_pca_path)

# =====================================================
# Plot PC1 vs PC2
# =====================================================
plt.figure(figsize=(6, 6))

sc = plt.scatter(
    df_pca["PC1"],
    df_pca["PC2"],
    c=df_pca["time"],     # 可换成 event / risk / tumor_fraction
    cmap="viridis",
    s=45,
    alpha=0.85
)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Pseudobulk PCA (PC1 vs PC2)")
plt.colorbar(sc, label="Survival time")

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "pseudobulk_PCA_PC1_PC2.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("[INFO] Saved:", fig_path)
print("✅ DONE")
