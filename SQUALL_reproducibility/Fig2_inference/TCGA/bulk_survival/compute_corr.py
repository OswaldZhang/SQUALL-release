#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr

# =====================================================
# Paths
# =====================================================
BASE_DIR = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/CESC_inference_vector_lowres"
SURVIVAL_JSON = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/Survival_TCGA_CESC.json"
GENE_MAP = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/gene_token_homologs.csv"
TILE_LABEL_DIR = "/lustre1/zxzeng/bwqin/SQUALL/yf_TCGA_label/tile_masks"

OUT_PSEUDOBULK = "gene_pseudobulk_survival_corr.csv"
OUT_TUMOR_RANK = "gene_within_tumor_rank_survival_corr.csv"

# =====================================================
# Load survival
# =====================================================
with open(SURVIVAL_JSON) as f:
    survival = json.load(f)

# =====================================================
# Load gene map
# =====================================================
gene_df = pd.read_csv(GENE_MAP)
gene_df = gene_df.sort_values(gene_df.columns[0])
genes = gene_df["HGNC_symbol"].tolist()
G = len(genes)

# =====================================================
# Containers
# =====================================================
pseudobulk_matrix = {g: [] for g in genes}
tumor_rank_matrix = {g: [] for g in genes}
survival_time = []
samples_used = []

# =====================================================
# Loop samples
# =====================================================
for sample in tqdm(os.listdir(BASE_DIR), desc="Processing samples"):
    sample_dir = os.path.join(BASE_DIR, sample)
    if not os.path.isdir(sample_dir):
        continue
    if sample not in survival:
        continue

    expr_path = os.path.join(sample_dir, "expr.npy")
    rank_path = os.path.join(sample_dir, "expr_rank.npy")
    coord_path = os.path.join(sample_dir, "coords.npy")

    if not (os.path.exists(expr_path) and os.path.exists(rank_path)):
        continue

    # Load arrays
    expr = np.load(expr_path)        # (N, G)
    expr_rank = np.load(rank_path)   # (N, G)
    coords = np.load(coord_path)     # (N, 2)

    # Load tumor labels
    label_csv = os.path.join(TILE_LABEL_DIR, f"{sample}_tile_labels.csv")
    if not os.path.exists(label_csv):
        continue

    df_label = pd.read_csv(label_csv)
    df_label["tile"] = df_label["pos"].apply(lambda x: "_".join(x.split("_")[:4]))
    label_map = dict(zip(df_label["tile"], df_label["label"]))

    tiles = [f"posX_{x}_posY_{y}" for x, y in coords]
    tumor_mask = np.array([label_map.get(t, -1) == 1 for t in tiles])

    if tumor_mask.sum() == 0:
        continue

    # Record survival
    survival_time.append(survival[sample]["time"])
    samples_used.append(sample)

    # -------------------------------------------------
    # Task 1: pseudobulk (whole slide)
    # -------------------------------------------------
    pseudo_vals = expr.mean(axis=0)  # (G,)
    for gi, g in enumerate(genes):
        pseudobulk_matrix[g].append(pseudo_vals[gi])

    # -------------------------------------------------
    # Task 2: within-tumor rank
    # -------------------------------------------------
    tumor_rank_vals = expr_rank[tumor_mask].mean(axis=0)
    for gi, g in enumerate(genes):
        tumor_rank_matrix[g].append(tumor_rank_vals[gi])

# =====================================================
# Convert survival time
# =====================================================
survival_time = np.array(survival_time)

# =====================================================
# Correlation: pseudobulk vs survival
# =====================================================
records_pb = []
for g in genes:
    x = np.array(pseudobulk_matrix[g])
    if len(x) < 5:
        continue
    rho, p = spearmanr(x, survival_time)
    records_pb.append({
        "gene": g,
        "spearman_r": rho,
        "p_value": p,
        "n_samples": len(x)
    })

df_pb = pd.DataFrame(records_pb)
df_pb.to_csv(OUT_PSEUDOBULK, index=False)

# =====================================================
# Correlation: tumor-rank vs survival
# =====================================================
records_tr = []
for g in genes:
    x = np.array(tumor_rank_matrix[g])
    if len(x) < 5:
        continue
    rho, p = spearmanr(x, survival_time)
    records_tr.append({
        "gene": g,
        "spearman_r": rho,
        "p_value": p,
        "n_samples": len(x)
    })

df_tr = pd.DataFrame(records_tr)
df_tr.to_csv(OUT_TUMOR_RANK, index=False)

print("✅ Finished.")
print(f"Saved: {OUT_PSEUDOBULK}")
print(f"Saved: {OUT_TUMOR_RANK}")
