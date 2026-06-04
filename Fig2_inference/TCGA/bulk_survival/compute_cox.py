#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines import CoxPHFitter

# =====================================================
# Paths
# =====================================================
BASE_DIR = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/CESC_inference_vector_lowres"
SURVIVAL_JSON = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/Survival_TCGA_CESC.json"
GENE_MAP = "/lustre1/zxzeng/bwqin/SQUALL/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/gene_token_homologs.csv"
TILE_LABEL_DIR = "/lustre1/zxzeng/bwqin/SQUALL/yf_TCGA_label/tile_masks"

OUT_PB = "cox_gene_pseudobulk.csv"
OUT_TR = "cox_gene_within_tumor_rank.csv"

MIN_SAMPLES = 20   #  Cox

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

print(f"Total genes: {G}")

# =====================================================
# Step 1: build sample × gene matrices
# =====================================================
samples = []
times = []
events = []

PB_list = []   # pseudobulk
TR_list = []   # tumor-rank

for sample in tqdm(os.listdir(BASE_DIR), desc="Building matrices"):
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

    label_csv = os.path.join(TILE_LABEL_DIR, f"{sample}_tile_labels.csv")
    if not os.path.exists(label_csv):
        continue

    # -------- load once per sample --------
    expr = np.load(expr_path)        # (N, G)
    expr_rank = np.load(rank_path)   # (N, G)
    coords = np.load(coord_path)

    df_label = pd.read_csv(label_csv)
    df_label["tile"] = df_label["pos"].apply(lambda x: "_".join(x.split("_")[:4]))
    label_map = dict(zip(df_label["tile"], df_label["label"]))

    tiles = [f"posX_{x}_posY_{y}" for x, y in coords]
    tumor_mask = np.array([label_map.get(t, -1) == 1 for t in tiles])

    if tumor_mask.sum() == 0:
        continue

    # -------- survival --------
    samples.append(sample)
    times.append(survival[sample]["time"])
    events.append(survival[sample]["status"])

    # -------- pseudobulk --------
    PB_list.append(expr.mean(axis=0))

    # -------- within-tumor rank --------
    TR_list.append(expr_rank[tumor_mask].mean(axis=0))

# =====================================================
# Convert to arrays
# =====================================================
PB_matrix = np.vstack(PB_list)    # (n_samples, n_genes)
TR_matrix = np.vstack(TR_list)

times = np.asarray(times)
events = np.asarray(events)

n_samples = PB_matrix.shape[0]
print(f"Samples used: {n_samples}")

# =====================================================
# Step 2: gene-wise Cox (pure compute, no I/O)
# =====================================================
records_pb = []
records_tr = []

for gi, gene in tqdm(enumerate(genes), total=len(genes), desc="Cox per gene"):

    # -----------------------
    # Cox: pseudobulk
    # -----------------------
    x_pb = PB_matrix[:, gi]

    if np.unique(x_pb).size > 1 and n_samples >= MIN_SAMPLES:
        df_pb = pd.DataFrame({
            "time": times,
            "event": events,
            "gene": x_pb
        })

        try:
            cph = CoxPHFitter()
            cph.fit(df_pb, duration_col="time", event_col="event")
            s = cph.summary.loc["gene"]

            records_pb.append({
                "gene": gene,
                "HR": np.exp(s["coef"]),
                "coef": s["coef"],
                "p_value": s["p"],
                "CI_lower": np.exp(s["coef lower 95%"]),
                "CI_upper": np.exp(s["coef upper 95%"]),
                "n_samples": n_samples
            })
        except Exception:
            pass

    # -----------------------
    # Cox: within-tumor rank
    # -----------------------
    x_tr = TR_matrix[:, gi]

    if np.unique(x_tr).size > 1 and n_samples >= MIN_SAMPLES:
        df_tr = pd.DataFrame({
            "time": times,
            "event": events,
            "gene": x_tr
        })

        try:
            cph = CoxPHFitter()
            cph.fit(df_tr, duration_col="time", event_col="event")
            s = cph.summary.loc["gene"]

            records_tr.append({
                "gene": gene,
                "HR": np.exp(s["coef"]),
                "coef": s["coef"],
                "p_value": s["p"],
                "CI_lower": np.exp(s["coef lower 95%"]),
                "CI_upper": np.exp(s["coef upper 95%"]),
                "n_samples": n_samples
            })
        except Exception:
            pass

# =====================================================
# Save
# =====================================================
pd.DataFrame(records_pb).to_csv(OUT_PB, index=False)
pd.DataFrame(records_tr).to_csv(OUT_TR, index=False)

print("✅ Cox finished")
print(f"Saved: {OUT_PB}")
print(f"Saved: {OUT_TR}")
