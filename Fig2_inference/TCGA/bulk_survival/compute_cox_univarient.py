#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

# =====================================================
# Config
# =====================================================
BASE_DIR = "/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/CESC_inference_vector_lowres"
TILE_LABEL_DIR = "/lustre1/zxzeng/bwqin/STORM/yf_TCGA_label/tile_masks"
SURVIVAL_JSON = "/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/Survival_TCGA_CESC.json"
GENE_MAP = "/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/gene_token_homologs.csv"

OUT_PB = "cox_univariate_pseudobulk_FDR.csv"
OUT_TR = "cox_univariate_within_tumor_rank_FDR.csv"

MIN_SAMPLES = 20   # 最少样本数

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
print(f"[INFO] Total genes = {G}")

# =====================================================
# Step 1: build matrices (once)
# =====================================================
samples = []
times = []
events = []

PB_list = []   # pseudobulk
TR_list = []   # within-tumor rank

for sample in tqdm(os.listdir(BASE_DIR), desc="Building matrices"):
    sample_dir = os.path.join(BASE_DIR, sample)
    if not os.path.isdir(sample_dir):
        continue
    if sample not in survival:
        continue

    expr_path = os.path.join(sample_dir, "expr.npy")
    rank_path = os.path.join(sample_dir, "expr_rank.npy")
    coord_path = os.path.join(sample_dir, "coords.npy")

    if not (os.path.exists(expr_path) and os.path.exists(rank_path) and os.path.exists(coord_path)):
        continue

    label_csv = os.path.join(TILE_LABEL_DIR, f"{sample}_tile_labels.csv")
    if not os.path.exists(label_csv):
        continue

    # --- load ---
    expr = np.load(expr_path)        # (N, G)
    expr_rank = np.load(rank_path)   # (N, G)
    coords = np.load(coord_path)     # (N, 2)

    # --- tumor mask ---
    df_label = pd.read_csv(label_csv)
    df_label["tile"] = df_label["pos"].apply(lambda x: "_".join(x.split("_")[:4]))
    label_map = dict(zip(df_label["tile"], df_label["label"]))

    tiles = [f"posX_{x}_posY_{y}" for x, y in coords]
    tumor_mask = np.array([label_map.get(t, -1) == 1 for t in tiles])

    if tumor_mask.sum() == 0:
        continue

    # --- survival ---
    samples.append(sample)
    times.append(float(survival[sample]["time"]))
    events.append(int(survival[sample]["status"]))

    # --- pseudobulk ---
    PB_list.append(expr.mean(axis=0))

    # --- within tumor rank ---
    TR_list.append(expr_rank[tumor_mask].mean(axis=0))

# =====================================================
# Convert to arrays
# =====================================================
PB = np.vstack(PB_list)   # (n_samples, G)
TR = np.vstack(TR_list)

times = np.asarray(times)
events = np.asarray(events)
n_samples = PB.shape[0]

print(f"[INFO] Samples used = {n_samples}")
if n_samples < MIN_SAMPLES:
    raise RuntimeError("Too few samples for Cox")

# =====================================================
# Step 2: univariate Cox
# =====================================================
records_pb = []
records_tr = []

for gi, gene in tqdm(enumerate(genes), total=len(genes), desc="Univariate Cox"):

    # ---------- pseudobulk ----------
    x_pb = PB[:, gi]
    if np.unique(x_pb).size > 1:
        df = pd.DataFrame({
            "time": times,
            "event": events,
            "gene": x_pb
        })
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="time", event_col="event")
            s = cph.summary.loc["gene"]

            records_pb.append({
                "gene": gene,
                "HR": float(np.exp(s["coef"])),
                "coef": float(s["coef"]),
                "p_value": float(s["p"]),
                "CI_lower": float(np.exp(s["coef lower 95%"])),
                "CI_upper": float(np.exp(s["coef upper 95%"])),
                "n_samples": n_samples
            })
        except Exception:
            pass

    # ---------- within-tumor rank ----------
    x_tr = TR[:, gi]
    if np.unique(x_tr).size > 1:
        df = pd.DataFrame({
            "time": times,
            "event": events,
            "gene": x_tr
        })
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="time", event_col="event")
            s = cph.summary.loc["gene"]

            records_tr.append({
                "gene": gene,
                "HR": float(np.exp(s["coef"])),
                "coef": float(s["coef"]),
                "p_value": float(s["p"]),
                "CI_lower": float(np.exp(s["coef lower 95%"])),
                "CI_upper": float(np.exp(s["coef upper 95%"])),
                "n_samples": n_samples
            })
        except Exception:
            pass

# =====================================================
# Step 3: FDR correction
# =====================================================
def add_fdr(df: pd.DataFrame):
    if df.empty:
        return df
    pvals = df["p_value"].values
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    df = df.copy()
    df["q_value"] = qvals
    return df

df_pb = add_fdr(pd.DataFrame(records_pb))
df_tr = add_fdr(pd.DataFrame(records_tr))

# =====================================================
# Save
# =====================================================
df_pb.to_csv(OUT_PB, index=False)
df_tr.to_csv(OUT_TR, index=False)

print("✅ Univariate Cox finished")
print("Saved:", OUT_PB)
print("Saved:", OUT_TR)
