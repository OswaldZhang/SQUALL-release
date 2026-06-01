#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")


# =====================================================
# Config
# =====================================================

INPUT_CSV = "STORM_better_risk_score_patientwise_all_COX_withstage_robust_recurrence_new.csv"
INPUT_CSV = "STORM_better_risk_score_patientwise_all_COX_all_robust_0322.csv"
CLINICAL_CSV = "outcome_stage.csv"

OUTDIR = "multivariate_cox_301_cutoff_forest_with_clinical"

CANCER_TYPE = "CESC"

TIME_COL = "Time"
EVENT_COL = "Status"
CANCER_COL = "CancerType"

# 如果 INPUT_CSV 里 sample 列名不是这些，请手动改 SAMPLE_COL
SAMPLE_COL = None
'''
SAMPLE_COL_CANDIDATES = [
    "sample",
    "Sample",
    "bcr_patient_barcode",
    "patient",
    "Patient",
    "patient_id",
    "PatientID",
    "case_id",
    "ID",
]
'''
SAMPLE_COL_CANDIDATES = [
    "Patient_id",
    "sample",
    "Sample",
    "bcr_patient_barcode",
    "patient",
    "Patient",
    "patient_id",
    "PatientID",
    "case_id",
    "ID",
]
# risk columns: follow your R KM code
MODEL_RISK_COLS = [
    "UNI_risk_norm",
    "plip_risk_norm",
    "virchow_risk_norm",
    "STORM_risk_norm",
]

MODEL_LABEL_MAP = {
    "UNI_risk_norm": "UNI",
    "plip_risk_norm": "PLIP",
    "virchow_risk_norm": "Virchow",
    "STORM_risk_norm": "STORM",
}

MODEL_VARS = ["UNI", "PLIP", "Virchow", "STORM"]

# optimal cutoff setting, follow surv_cutpoint(minprop = 0.2)
MINPROP = 0.2

# Cox penalizer
COX_PENALIZER = 0.05

# p-value setting
# 只对几个模型变量换算成单边 Wald p-value
ONE_SIDED_MODEL_P = True

# 单边方向：
# "greater": H1: coef > 0, HR > 1
# "less":    H1: coef < 0, HR < 1
ONE_SIDED_ALTERNATIVE = "greater"

# 如果 INPUT_CSV 或 CLINICAL_CSV 中存在 tumor_fraction，就自动加入 Cox
USE_TUMOR_FRACTION_IF_AVAILABLE = True

# plot
FIG_WIDTH = 8.5
DPI = 300

BLUE = "#4C72B0"
RED = "#C44E52"


# =====================================================
# Utils
# =====================================================

def clean_sample_id(x):
    """
    TCGA-style sample ID cleaning.
    Keeps the first 12 chars if TCGA barcode-like.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    s = s.replace(".0", "")

    # TCGA patient barcode length is usually 12: TCGA-XX-XXXX
    if s.startswith("TCGA") and len(s) >= 12:
        return s[:12]

    return s


def infer_sample_col(df):
    if SAMPLE_COL is not None:
        if SAMPLE_COL not in df.columns:
            raise ValueError(f"SAMPLE_COL={SAMPLE_COL} not found in INPUT_CSV.")
        return SAMPLE_COL

    for c in SAMPLE_COL_CANDIDATES:
        if c in df.columns:
            return c

    raise ValueError(
        "Cannot infer sample column from INPUT_CSV. "
        "Please set SAMPLE_COL manually. "
        f"Candidates checked: {SAMPLE_COL_CANDIDATES}"
    )


def stage_to_numeric(stage):
    """
    Follow your uploaded clinical code:
    accepts 'Stage IB', 'IB', 'IIA', 'III', 'Stage IIIC', 'IVB', etc.
    only keeps leading Roman numeral I/II/III/IV.
    """
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return np.nan

    s = str(stage).upper().replace("STAGE", "").strip()

    m = re.match(r"(IV|III|II|I)", s)
    if not m:
        return np.nan

    roman = m.group(1)

    return {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
    }.get(roman, np.nan)


def normalize_event_column(s):
    """
    Convert event/status to 0/1.
    """
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    x = s.astype(str).str.strip().str.lower()

    event_map = {
        "1": 1,
        "true": 1,
        "yes": 1,
        "dead": 1,
        "death": 1,
        "deceased": 1,
        "event": 1,
        "progressed": 1,
        "progression": 1,
        "relapse": 1,
        "recurrence": 1,
        "recurred": 1,

        "0": 0,
        "false": 0,
        "no": 0,
        "alive": 0,
        "living": 0,
        "censored": 0,
        "none": 0,
        "no event": 0,
        "no recurrence": 0,
        "non-recurrence": 0,
    }

    return x.map(event_map).astype(float)


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def convert_to_one_sided_p(p_two, coef, alternative="greater"):
    """
    Convert two-sided Wald p-value to one-sided Wald p-value.

    alternative="greater":
        H1: coef > 0, HR > 1

    alternative="less":
        H1: coef < 0, HR < 1
    """
    if pd.isna(p_two) or pd.isna(coef):
        return np.nan

    if alternative == "greater":
        if coef > 0:
            return p_two / 2.0
        else:
            return 1.0 - p_two / 2.0

    if alternative == "less":
        if coef < 0:
            return p_two / 2.0
        else:
            return 1.0 - p_two / 2.0

    raise ValueError(f"Unknown alternative: {alternative}")


# =====================================================
# Clinical loading: follow your uploaded code
# =====================================================

def load_clinical_features(clinical_csv):
    """
    Follow your uploaded clinical code:

    df_cli = pd.read_csv(CLINICAL_CSV)
    df_cli = df_cli.rename(columns={
        "bcr_patient_barcode": "sample",
        "age_at_initial_pathologic_diagnosis": "age",
        "clinical_stage": "stage"
    })
    df_cli["stage_num"] = df_cli["stage"].apply(stage_to_numeric)
    df_cli = df_cli.dropna(subset=["sample", "age", "stage_num"])
    df_cli["sample"] = df_cli["sample"].astype(str).str.strip()
    df_cli = df_cli.set_index("sample")
    """
    df_cli = pd.read_csv(clinical_csv)

    df_cli = df_cli.rename(columns={
        "bcr_patient_barcode": "sample",
        "age_at_initial_pathologic_diagnosis": "age",
        "clinical_stage": "stage",
    })

    need_cols = ["sample", "age", "stage"]
    missing = [c for c in need_cols if c not in df_cli.columns]
    if missing:
        raise ValueError(f"[ERROR] clinical csv missing columns: {missing}")

    df_cli["sample"] = df_cli["sample"].apply(clean_sample_id)
    df_cli["age"] = safe_numeric(df_cli["age"])
    df_cli["stage_num"] = df_cli["stage"].apply(stage_to_numeric)

    keep_cols = ["sample", "age", "stage", "stage_num"]

    if USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df_cli.columns:
        df_cli["tumor_fraction"] = safe_numeric(df_cli["tumor_fraction"])
        keep_cols.append("tumor_fraction")

    df_cli = df_cli[keep_cols].dropna(subset=["sample", "age", "stage_num"])
    df_cli = df_cli.drop_duplicates(subset=["sample"], keep="first")
    df_cli = df_cli.set_index("sample")

    print(f"[INFO] Clinical samples loaded = {df_cli.shape[0]}")
    print("[INFO] Clinical columns loaded:", df_cli.columns.tolist())

    return df_cli


def merge_clinical_into_main(df, df_cli):
    """
    Merge clinical age/stage_num into risk dataframe by sample.
    """
    sample_col = infer_sample_col(df)

    df = df.copy()
    df["sample"] = df[sample_col].apply(clean_sample_id)

    before_n = df.shape[0]

    # 如果 INPUT_CSV 里已有 tumor_fraction，先保留
    input_has_tumor_frac = USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df.columns

    merge_cols = ["age", "stage_num"]

    if USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df_cli.columns:
        merge_cols.append("tumor_fraction")

    df = df.merge(
        df_cli[merge_cols],
        left_on="sample",
        right_index=True,
        how="inner",
        suffixes=("", "_clinical"),
    )

    # 如果 INPUT_CSV 里已有 tumor_fraction，而 clinical 里没有，就使用 INPUT_CSV 的
    # 如果两边都有，优先使用 INPUT_CSV 原始 tumor_fraction
    if USE_TUMOR_FRACTION_IF_AVAILABLE:
        if input_has_tumor_frac:
            df["tumor_fraction"] = safe_numeric(df["tumor_fraction"])
        elif "tumor_fraction_clinical" in df.columns:
            df["tumor_fraction"] = safe_numeric(df["tumor_fraction_clinical"])
        elif "tumor_fraction" in df.columns:
            df["tumor_fraction"] = safe_numeric(df["tumor_fraction"])

    after_n = df.shape[0]

    print(f"[INFO] Sample column used in INPUT_CSV: {sample_col}")
    print(f"[INFO] Rows before clinical merge = {before_n}")
    print(f"[INFO] Rows after clinical merge  = {after_n}")

    if after_n == 0:
        raise RuntimeError(
            "No samples left after clinical merge. "
            "Please check sample ID format between INPUT_CSV and CLINICAL_CSV."
        )

    return df


def prepare_clinical_features(df):
    """
    Clinical covariates follow uploaded clinical code:
        age
        stage_num
        optional tumor_fraction if available
    """
    out = pd.DataFrame(index=df.index)

    required = ["age", "stage_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing clinical covariates after merge: {missing}")

    out["age"] = safe_numeric(df["age"])
    out["stage_num"] = safe_numeric(df["stage_num"])

    used = ["age", "stage_num"]

    if USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df.columns:
        out["tumor_fraction"] = safe_numeric(df["tumor_fraction"])
        used.append("tumor_fraction")

    print("[INFO] Clinical covariates used:", used)

    return out, used


# =====================================================
# Optimal cutoff: Python version of surv_cutpoint(minprop=0.2)
# =====================================================

def find_optimal_cutoff_by_logrank(df, risk_col, time_col, event_col, minprop=0.2):
    """
    Approximate survminer::surv_cutpoint.

    For each candidate cutoff:
        low  = risk <= cutoff
        high = risk > cutoff

    Constraint:
        each group proportion >= minprop

    Score:
        maximize log-rank test statistic.
    """
    tmp = df[[time_col, event_col, risk_col]].copy()
    tmp[time_col] = safe_numeric(tmp[time_col])
    tmp[event_col] = normalize_event_column(tmp[event_col])
    tmp[risk_col] = safe_numeric(tmp[risk_col])
    tmp = tmp.dropna()

    if tmp.shape[0] < 10:
        raise RuntimeError(f"Too few samples for cutoff search: {risk_col}")

    if tmp[risk_col].nunique() <= 1:
        raise RuntimeError(f"No variation in risk column: {risk_col}")

    x = tmp[risk_col].values
    unique_vals = np.sort(np.unique(x))

    candidates = []
    n = tmp.shape[0]

    for cutoff in unique_vals:
        n_low = np.sum(x <= cutoff)
        n_high = np.sum(x > cutoff)

        if n_low < minprop * n:
            continue
        if n_high < minprop * n:
            continue

        candidates.append(cutoff)

    if len(candidates) == 0:
        raise RuntimeError(
            f"No valid cutoff found for {risk_col}. "
            f"Try reducing MINPROP."
        )

    best = {
        "cutoff": None,
        "statistic": -np.inf,
        "p_value": np.nan,
    }

    for cutoff in candidates:
        low = tmp[tmp[risk_col] <= cutoff]
        high = tmp[tmp[risk_col] > cutoff]

        try:
            res = logrank_test(
                low[time_col],
                high[time_col],
                event_observed_A=low[event_col],
                event_observed_B=high[event_col],
            )

            stat = float(res.test_statistic)
            pval = float(res.p_value)

            if stat > best["statistic"]:
                best = {
                    "cutoff": float(cutoff),
                    "statistic": stat,
                    "p_value": pval,
                }

        except Exception:
            continue

    if best["cutoff"] is None:
        raise RuntimeError(f"Cutoff search failed for {risk_col}")

    return best["cutoff"], best["statistic"], best["p_value"]


def apply_cutoff_and_fix_direction(df, risk_col, cutoff):
    """
    Follow your R logic:

        default high = risk > cutoff
        low = risk <= cutoff

    Then check whether median_high < median_low.
    If reversed, swap labels.
    """
    risk = safe_numeric(df[risk_col])

    group = pd.Series(np.where(risk > cutoff, "high", "low"), index=df.index)
    group[risk.isna()] = np.nan

    median_high = risk[group == "high"].median()
    median_low = risk[group == "low"].median()

    reversed_label = False

    if pd.notna(median_high) and pd.notna(median_low):
        if median_high < median_low:
            group = group.map({"high": "low", "low": "high"})
            reversed_label = True

    indicator = group.map({"low": 0.0, "high": 1.0})

    return group, indicator, reversed_label, median_low, median_high


def make_model_risk_groups(df_cancer):
    """
    For each model risk:
        - find optimal cutoff
        - assign low/high
        - fix high/low direction
        - create binary variable:
            MODEL = 1 for high risk, 0 for low risk
    """
    risk_group_df = pd.DataFrame(index=df_cancer.index)
    cutoff_records = []

    for risk_col in MODEL_RISK_COLS:
        if risk_col not in df_cancer.columns:
            print(f"[WARN] Missing risk column: {risk_col}. Skipped.")
            continue

        if safe_numeric(df_cancer[risk_col]).dropna().nunique() <= 1:
            print(f"[WARN] {risk_col} has no variation. Skipped.")
            continue

        model_label = MODEL_LABEL_MAP.get(risk_col, risk_col)

        cutoff, stat, p_cut = find_optimal_cutoff_by_logrank(
            df=df_cancer,
            risk_col=risk_col,
            time_col=TIME_COL,
            event_col=EVENT_COL,
            minprop=MINPROP,
        )

        group, indicator, reversed_label, median_low, median_high = apply_cutoff_and_fix_direction(
            df_cancer,
            risk_col,
            cutoff,
        )

        risk_group_df[model_label] = indicator

        cutoff_records.append({
            "model": model_label,
            "risk_col": risk_col,
            "cutoff": cutoff,
            "logrank_statistic": stat,
            "logrank_p_value": p_cut,
            "median_low": median_low,
            "median_high": median_high,
            "label_reversed": reversed_label,
            "n_low": int((group == "low").sum()),
            "n_high": int((group == "high").sum()),
        })

        print(
            f"[INFO] {model_label}: cutoff={cutoff:.6g}, "
            f"n_low={(group == 'low').sum()}, "
            f"n_high={(group == 'high').sum()}, "
            f"reversed={reversed_label}"
        )

    cutoff_df = pd.DataFrame(cutoff_records)

    if "STORM" not in risk_group_df.columns:
        raise RuntimeError("STORM risk group was not created. Please check STORM_risk_norm.")

    return risk_group_df, cutoff_df


# =====================================================
# Multivariate Cox
# =====================================================

def build_multivariate_cox_dataframe(df_cancer):
    """
    Build final Cox input:
        Time, Status, clinical variables, model risk groups

    Clinical:
        age + stage_num + optional tumor_fraction

    Model risk:
        optimal cutoff + low/high + high-risk indicator
    """
    base = pd.DataFrame(index=df_cancer.index)
    base[TIME_COL] = safe_numeric(df_cancer[TIME_COL])
    base[EVENT_COL] = normalize_event_column(df_cancer[EVENT_COL])

    clinical_df, used_clinical_cols = prepare_clinical_features(df_cancer)
    risk_group_df, cutoff_df = make_model_risk_groups(df_cancer)

    cox_df = pd.concat([base, clinical_df, risk_group_df], axis=1)

    # remove all-NA or constant columns
    feature_cols = [c for c in cox_df.columns if c not in [TIME_COL, EVENT_COL]]
    valid_feature_cols = []

    for c in feature_cols:
        if cox_df[c].notna().sum() == 0:
            print(f"[WARN] Drop all-NA column: {c}")
            continue

        if cox_df[c].dropna().nunique() <= 1:
            print(f"[WARN] Drop constant column: {c}")
            continue

        valid_feature_cols.append(c)

    cox_df = cox_df[[TIME_COL, EVENT_COL] + valid_feature_cols].dropna()

    return cox_df, used_clinical_cols, cutoff_df


def fit_multivariate_cox(cox_df):
    cph = CoxPHFitter(penalizer=COX_PENALIZER)

    cph.fit(
        cox_df,
        duration_col=TIME_COL,
        event_col=EVENT_COL,
    )

    summary = cph.summary.reset_index()
    summary = summary.rename(columns={"covariate": "variable"})

    result = pd.DataFrame({
        "variable": summary["variable"],
        "coef": summary["coef"],
        "HR": summary["exp(coef)"],
        "CI_lower": summary["exp(coef) lower 95%"],
        "CI_upper": summary["exp(coef) upper 95%"],
        "p_two_sided": summary["p"],
        "n_samples": cox_df.shape[0],
        "n_events": int(cox_df[EVENT_COL].sum()),
    })

    p_values = []
    p_types = []

    for _, row in result.iterrows():
        var = row["variable"]

        if ONE_SIDED_MODEL_P and var in MODEL_VARS:
            p_one = convert_to_one_sided_p(
                row["p_two_sided"],
                row["coef"],
                alternative=ONE_SIDED_ALTERNATIVE,
            )
            p_values.append(p_one)
            p_types.append(f"one-sided Wald, HR {('>' if ONE_SIDED_ALTERNATIVE == 'greater' else '<')} 1")
        else:
            p_values.append(row["p_two_sided"])
            p_types.append("two-sided Wald")

    result["p_value"] = p_values
    result["p_type"] = p_types

    return result, cph


# =====================================================
# Forest plot
# =====================================================

def variable_label(var):
    label_map = {
        "age": "Age",
        "stage_num": "Stage",
        "tumor_fraction": "Tumor fraction",
        "UNI": "UNI",
        "PLIP": "PLIP",
        "Virchow": "Virchow",
        "STORM": "STORM",
    }
    return label_map.get(var, var)


def order_forest_rows(result_df):
    """
    Desired top-to-bottom order:
        clinical
        other models
        STORM
    """
    result_df = result_df.copy()

    vars_present = result_df["variable"].tolist()

    clinical_order = [v for v in vars_present if v not in MODEL_VARS]

    other_models = [m for m in ["UNI", "PLIP", "Virchow"] if m in vars_present]
    storm = ["STORM"] if "STORM" in vars_present else []

    final_order = clinical_order + other_models + storm

    result_df["plot_order"] = result_df["variable"].apply(
        lambda x: final_order.index(x) if x in final_order else 999
    )

    result_df = result_df.sort_values("plot_order", ascending=True).reset_index(drop=True)

    return result_df


def pvalue_label(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3g}"


def plot_forest(result_df, output_pdf, output_png=None, title="Multivariate Cox Hazard Forest Plot"):
    """
    Color:
        - STORM red
        - all others blue

    Annotation:
        - models use one-sided Wald p-value
        - clinical variables use two-sided Wald p-value
    """
    plot_df = order_forest_rows(result_df)
    plot_df["label"] = plot_df["variable"].apply(variable_label)

    # Reverse because matplotlib y-axis goes bottom-to-top.
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    n = plot_df.shape[0]
    y = np.arange(n)

    fig_height = max(4.5, 0.5 * n + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    for i, row in plot_df.iterrows():
        var = row["variable"]
        hr = row["HR"]
        lower = row["CI_lower"]
        upper = row["CI_upper"]

        color = RED if var == "STORM" else BLUE

        ax.plot([lower, upper], [i, i], color=color, lw=2)
        ax.scatter(hr, i, color=color, s=45, zorder=3)

    ax.axvline(1, linestyle="--", color="black", lw=1)

    ax.set_xscale("log")

    min_x = np.nanmin(plot_df["CI_lower"].values)
    max_x = np.nanmax(plot_df["CI_upper"].values)

    min_x = max(min_x * 0.7, 1e-3)
    max_x = max_x * 1.3

    ax.set_xlim(min_x, max_x)

    possible_ticks = np.array([0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 10])
    ticks = possible_ticks[(possible_ticks >= min_x) & (possible_ticks <= max_x)]

    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])

    ax.set_xlabel("Hazard Ratio")
    ax.set_title(title)

    x_text = max_x / 1.05

    for i, row in plot_df.iterrows():
        var = row["variable"]

        if var in MODEL_VARS:
            p_prefix = "p1"
        else:
            p_prefix = "p"

        txt = (
            f"HR={row['HR']:.2f} "
            f"[{row['CI_lower']:.2f}, {row['CI_upper']:.2f}], "
            f"{p_prefix}={pvalue_label(row['p_value'])}"
        )

        ax.text(
            x_text,
            i,
            txt,
            va="center",
            ha="left",
            fontsize=8,
            clip_on=False,
        )

    plt.subplots_adjust(right=0.68)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()

    fig.savefig(output_pdf, dpi=DPI, transparent=True, bbox_inches="tight")

    if output_png is not None:
        fig.savefig(output_png, dpi=DPI, transparent=True, bbox_inches="tight")

    plt.close(fig)


# =====================================================
# Main
# =====================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load main risk/survival dataframe
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print("[INFO] Input shape:", df.shape)
    print("[INFO] Input columns:")
    print(df.columns.tolist())

    required_cols = [
        CANCER_COL,
        TIME_COL,
        EVENT_COL,
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in INPUT_CSV: {missing}")

    missing_risk_cols = [c for c in MODEL_RISK_COLS if c not in df.columns]
    if len(missing_risk_cols) > 0:
        print(f"[WARN] Missing risk columns: {missing_risk_cols}")

    # Load clinical and merge
    df_cli = load_clinical_features(CLINICAL_CSV)
    df = merge_clinical_into_main(df, df_cli)

    # Follow your R sample selection:
    # df_cancer <- df %>% filter(CancerType == cancer)
    df_cancer = df[df[CANCER_COL].astype(str) == str(CANCER_TYPE)].copy()

    if df_cancer.empty:
        raise RuntimeError(f"No samples found for CancerType == {CANCER_TYPE}")

    print(f"[INFO] CancerType={CANCER_TYPE} rows after clinical merge:", df_cancer.shape[0])

    raw_subset_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_raw_selected_samples_with_clinical.csv",
    )
    df_cancer.to_csv(raw_subset_path, index=False)

    # Build Cox dataframe
    cox_df, used_clinical_cols, cutoff_df = build_multivariate_cox_dataframe(df_cancer)

    print("[INFO] Used clinical columns:", used_clinical_cols)
    print("[INFO] Final Cox input shape:", cox_df.shape)
    print("[INFO] Final Cox events:", int(cox_df[EVENT_COL].sum()))
    print("[INFO] Cox columns:")
    print(cox_df.columns.tolist())

    cutoff_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_optimal_cutoffs.csv",
    )
    cutoff_df.to_csv(cutoff_path, index=False)

    cox_input_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_multivariate_cox_input_cutoff_risk_with_clinical.csv",
    )
    cox_df.to_csv(cox_input_path, index=False)

    if cox_df.shape[0] < 20:
        raise RuntimeError(
            f"Too few samples after dropna for multivariate Cox: n={cox_df.shape[0]}"
        )

    if int(cox_df[EVENT_COL].sum()) < 5:
        raise RuntimeError(
            f"Too few events after dropna for multivariate Cox: "
            f"events={int(cox_df[EVENT_COL].sum())}"
        )

    # Fit multivariate Cox
    result_df, cph = fit_multivariate_cox(cox_df)
    result_df = order_forest_rows(result_df)

    result_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_multivariate_cox_results_cutoff_risk_with_clinical.csv",
    )
    result_df.to_csv(result_path, index=False)

    print("[INFO] Multivariate Cox results:")
    print(result_df)

    # Forest plot
    output_pdf = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_multivariate_cox_forest_cutoff_risk_with_clinical.pdf",
    )
    output_png = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_multivariate_cox_forest_cutoff_risk_with_clinical.png",
    )

    plot_forest(
        result_df,
        output_pdf=output_pdf,
        output_png=output_png,
        title=f"{CANCER_TYPE} Multivariate Cox Hazard Forest Plot",
    )

    print("✅ Done.")
    print(f"Raw selected samples: {raw_subset_path}")
    print(f"Optimal cutoffs: {cutoff_path}")
    print(f"Cox input: {cox_input_path}")
    print(f"Cox results: {result_path}")
    print(f"Forest plot PDF: {output_pdf}")
    print(f"Forest plot PNG: {output_png}")


if __name__ == "__main__":
    main()
