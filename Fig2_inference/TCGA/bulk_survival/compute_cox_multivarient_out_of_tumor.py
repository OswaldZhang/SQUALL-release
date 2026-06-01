#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
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

SURVIVAL_JSON = "/lustre1/zxzeng/bwqin/STORM_main/downstream_labels/Survival_five_fold_DSS_COX_outs/Survival_TCGA_CESC.json"
CLINICAL_CSV = "outcome_stage.csv"
GENE_MAP = "/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/codebase/gene_token_homologs.csv"

# 输出
OUT_PB = "cox_multivariate_PB_log1p_age_stage_tumorfrac_FDR.csv"
OUT_TR = "cox_multivariate_without_tumor_zscore_age_stage_tumorfrac_FDR.csv"

# multivariate 最低样本量
MIN_SAMPLES = 80

# PB / TR 的 gene 变换策略
PB_TRANSFORM = "log1p"   # PB: log1p（仅对非负 gene）
TR_TRANSFORM = "zscore"  # TR: zscore（跨样本逐 gene）

# 协变量（都要加 tumor_fraction）
COVARS = ["age", "stage_num", "tumor_fraction"]

# =====================================================
# Utils
# =====================================================
def stage_to_numeric(stage):
    """
    Robust stage parsing:
    Accepts: 'Stage IB', 'IB', 'IIA', 'III', 'Stage IIIC', 'IVB', etc.
    Only keep leading Roman numeral I/II/III/IV.
    """
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return np.nan
    s = str(stage).upper().replace("STAGE", "").strip()
    m = re.match(r"(IV|III|II|I)", s)
    if not m:
        return np.nan
    roman = m.group(1)
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(roman, np.nan)

def safe_zscore(x: np.ndarray):
    x = x.astype(float)
    sd = float(np.std(x))
    if sd < 1e-12:
        return None
    return (x - float(np.mean(x))) / sd

def load_tile_label_map(label_csv: str):
    """
    label_csv has columns: pos, label
    pos like: posX_10048_posY_10049_*...
    we truncate to first 4 parts: posX_10048_posY_10049
    """
    df_label = pd.read_csv(label_csv)
    df_label["tile"] = df_label["pos"].apply(lambda x: "_".join(str(x).split("_")[:4]))
    return dict(zip(df_label["tile"], df_label["label"]))

def add_fdr(df: pd.DataFrame, p_col="p_value", out_col="q_value"):
    if df.empty:
        df[out_col] = []
        return df
    _, qvals, _, _ = multipletests(df[p_col].values, method="fdr_bh")
    df = df.copy()
    df[out_col] = qvals
    return df

# =====================================================
# Load survival
# =====================================================
with open(SURVIVAL_JSON) as f:
    surv = json.load(f)

# =====================================================
# Load clinical
# =====================================================
df_cli = pd.read_csv(CLINICAL_CSV)
df_cli = df_cli.rename(columns={
    "bcr_patient_barcode": "sample",
    "age_at_initial_pathologic_diagnosis": "age",
    "clinical_stage": "stage"
})

need_cols = ["sample", "age", "stage"]
missing = [c for c in need_cols if c not in df_cli.columns]
if missing:
    raise ValueError(f"[ERROR] clinical csv missing columns: {missing}")

df_cli["stage_num"] = df_cli["stage"].apply(stage_to_numeric)
df_cli = df_cli.dropna(subset=["sample", "age", "stage_num"])
df_cli["sample"] = df_cli["sample"].astype(str).str.strip()
df_cli = df_cli.set_index("sample")

# =====================================================
# Load gene map
# =====================================================
gene_df = pd.read_csv(GENE_MAP)
gene_df = gene_df.sort_values(gene_df.columns[0])
genes = gene_df["HGNC_symbol"].tolist()
G = len(genes)
print(f"[INFO] Total genes = {G}")

# =====================================================
# Sample diagnostics
# =====================================================
expr_samples = {s for s in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, s))}
surv_samples = set(surv.keys())
clin_samples = set(df_cli.index)

print(f"[INFO] expr samples      = {len(expr_samples)}")
print(f"[INFO] survival samples  = {len(surv_samples)}")
print(f"[INFO] clinical samples  = {len(clin_samples)}")
print(f"[INFO] expr ∩ survival    = {len(expr_samples & surv_samples)}")
print(f"[INFO] expr ∩ clinical    = {len(expr_samples & clin_samples)}")
print(f"[INFO] expr ∩ surv ∩ clin = {len(expr_samples & surv_samples & clin_samples)}")

common0 = sorted(expr_samples & surv_samples & clin_samples)

# =====================================================
# Step 1: Build PB + TR + tumor_fraction matrices
#   PB[i, g] = mean(expr[:, g])
#   TR[i, g] = mean(expr_rank[tumor_tiles, g])
#   tumor_fraction = tumor_tiles / all_tiles
# =====================================================
samples = []
times = []
events = []
cov_age = []
cov_stage = []
tumor_fracs = []
PB_list = []
TR_list = []

for sample in tqdm(common0, desc="Building PB/TR/tumor_fraction"):
    sample_dir = os.path.join(BASE_DIR, sample)
    expr_path = os.path.join(sample_dir, "expr.npy")
    rank_path = os.path.join(sample_dir, "expr_rank.npy")
    coord_path = os.path.join(sample_dir, "coords.npy")

    if not (os.path.exists(expr_path) and os.path.exists(rank_path) and os.path.exists(coord_path)):
        continue

    label_csv = os.path.join(TILE_LABEL_DIR, f"{sample}_tile_labels.csv")
    if not os.path.exists(label_csv):
        continue

    expr = np.load(expr_path)        # (N, G)
    expr_rank = np.load(rank_path)   # (N, G)
    coords = np.load(coord_path)     # (N, 2)

    if expr.ndim != 2 or expr_rank.ndim != 2 or coords.ndim != 2:
        continue
    if coords.shape[1] != 2:
        continue
    if expr.shape[0] != expr_rank.shape[0] or expr.shape[0] != coords.shape[0]:
        continue
    if expr.shape[1] != G or expr_rank.shape[1] != G:
        # 基因维度不一致就跳过（防止 gene_map 不匹配）
        continue

    label_map = load_tile_label_map(label_csv)
    tiles = [f"posX_{int(x)}_posY_{int(y)}" for x, y in coords]
    tumor_mask = np.array([label_map.get(t, -1) == 0 for t in tiles], dtype=bool)

    if tumor_mask.sum() == 0:
        continue

    tumor_fraction = float(np.mean(tumor_mask))

    pb = expr.mean(axis=0)                 # (G,)
    tr = expr_rank[tumor_mask].mean(axis=0)  # (G,)

    samples.append(sample)
    times.append(float(surv[sample]["time"]))
    events.append(int(surv[sample]["status"]))
    cov_age.append(float(df_cli.loc[sample, "age"]))
    cov_stage.append(float(df_cli.loc[sample, "stage_num"]))
    tumor_fracs.append(tumor_fraction)

    #PB_list.append(pb)
    TR_list.append(tr)
'''
if len(PB_list) == 0:
    raise RuntimeError("[ERROR] No valid samples after loading expr/rank/survival/clinical/labels.")
'''
#PB = np.vstack(PB_list)  # (n_samples, G)
TR = np.vstack(TR_list)  # (n_samples, G)

meta = pd.DataFrame({
    "sample": samples,
    "time": times,
    "event": events,
    "age": cov_age,
    "stage_num": cov_stage,
    "tumor_fraction": tumor_fracs,
}).set_index("sample")

n_samples = meta.shape[0]
print(f"[INFO] Samples used = {n_samples}")

if n_samples < MIN_SAMPLES:
    raise RuntimeError(
        f"Too few samples for multivariate Cox (n={n_samples} < {MIN_SAMPLES}). "
        f"Fix cohort matching or lower MIN_SAMPLES if you really must."
    )

# =====================================================
# Step 2A: Multivariate Cox per gene (PB)
# Model: Surv ~ log1p(PB_gene) + age + stage_num + tumor_fraction
# =====================================================
'''
records_pb = []

for gi, gene in tqdm(enumerate(genes), total=G, desc="Multivariate Cox per gene (PB)"):
    x = PB[:, gi]

    if np.unique(x).size <= 1:
        continue

    if PB_TRANSFORM == "log1p":
        # log1p requires non-negative
        if (x < 0).any():
            continue
        x_use = np.log1p(x.astype(float))
        gene_transform = "log1p"
    else:
        x_use = x.astype(float)
        gene_transform = "raw"

    df = meta[["time", "event"] + COVARS].copy()
    df["gene"] = x_use

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
            "n_samples": int(n_samples),
            "covariates": "+".join(COVARS),
            "feature_type": "pseudobulk_mean",
            "gene_transform": gene_transform,
        })
    except Exception:
        continue

df_pb = pd.DataFrame(records_pb)
df_pb = add_fdr(df_pb, p_col="p_value", out_col="q_value")
df_pb.to_csv(OUT_PB, index=False)
print("✅ Saved PB multivariate Cox:", OUT_PB)
'''
# =====================================================
# Step 2B: Multivariate Cox per gene (TR)
# Model: Surv ~ zscore(TR_gene) + age + stage_num + tumor_fraction
# =====================================================
records_tr = []

for gi, gene in tqdm(enumerate(genes), total=G, desc="Multivariate Cox per gene (TR)"):
    x = TR[:, gi]

    if np.unique(x).size <= 1:
        continue

    if TR_TRANSFORM == "zscore":
        x_use = safe_zscore(x)
        if x_use is None:
            continue
        gene_transform = "zscore"
    else:
        x_use = x.astype(float)
        gene_transform = "raw"

    df = meta[["time", "event"] + COVARS].copy()
    df["gene"] = x_use

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
            "n_samples": int(n_samples),
            "covariates": "+".join(COVARS),
            "feature_type": "within_tumor_rank_mean",
            "gene_transform": gene_transform,
        })
    except Exception:
        continue

df_tr = pd.DataFrame(records_tr)
df_tr = add_fdr(df_tr, p_col="p_value", out_col="q_value")
df_tr.to_csv(OUT_TR, index=False)
print("✅ Saved TR multivariate Cox:", OUT_TR)

print("✅ All done.")
