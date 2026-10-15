#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")


# =====================================================
# Config
# =====================================================

INPUT_CSV = "SQUALL_better_risk_score_patientwise_all_COX_all_robust_0322.csv"
OUTDIR = "multivariate_cox_CESC_best_SQUALL_fold"

CANCER_TYPE = "CESC"

# 
TIME_COL = "Time"
EVENT_COL = "Status"

# fold 
FOLD_COL = "Fold"

# cancer type 
CANCER_COL = "CancerType"

# SQUALL  fold  risk 
#  SQUALL_riskscore "SQUALL_riskscore"
SQUALL_RISK_COL = "SQUALL_risk"

#  risk 
#  forest plot “”
MODEL_RISK_COLS = [
    "UNI_risk_norm",
    "plip_risk_norm",
    "virchow_risk_norm",
    "SQUALL_risk_norm",
]

# 
# 
CLINICAL_COLS_RAW = [
    "age",
    "Age",
    "stage",
    "Stage",
    "stage_num",
    "gender",
    "Gender",
    "sex",
    "Sex",
]

#  risk  z-score
#  True risk 
ZSCORE_RISK = True

#  z-score
ZSCORE_CONTINUOUS_CLINICAL = False

# Cox  penalizer 0.05 - 0.2
COX_PENALIZER = 0.05

# forest plot 
FIGSIZE = (7.5, 6.5)
DPI = 300


# =====================================================
# Utils
# =====================================================

def stage_to_numeric(stage):
    """
    Convert clinical stage to numeric.

    Examples:
        Stage I / I / IA / IB       -> 1
        Stage II / IIA / IIB        -> 2
        Stage III / IIIC            -> 3
        Stage IV / IVB              -> 4
    """
    if pd.isna(stage):
        return np.nan

    s = str(stage).upper()
    s = s.replace("STAGE", "")
    s = s.replace("PATHOLOGIC", "")
    s = s.replace("CLINICAL", "")
    s = s.strip()

    m = re.search(r"\b(IV|III|II|I)\b|^(IV|III|II|I)", s)
    if not m:
        return np.nan

    roman = m.group(1) if m.group(1) is not None else m.group(2)
    return {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
    }.get(roman, np.nan)


def safe_zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < 1e-12:
        return s * np.nan
    return (s - mu) / sd


def normalize_event_column(s):
    """
    Convert event/status to 0/1.

    Supports:
        1/0
        True/False
        dead/alive
        deceased/living
        event/censored
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
        "relapse": 1,

        "0": 0,
        "false": 0,
        "no": 0,
        "alive": 0,
        "living": 0,
        "censored": 0,
        "none": 0,
        "no event": 0,
    }

    return x.map(event_map).astype(float)


def find_existing_columns(df, candidates):
    return [c for c in candidates if c in df.columns]


def prepare_clinical_features(df):
    """
    Build clinical covariates.

    Rule:
        - stage/stage_num -> numeric stage_num
        - age/Age -> age
        - gender/sex -> one-hot
        - other numeric clinical columns -> numeric
        - other categorical clinical columns -> one-hot
    """
    out = pd.DataFrame(index=df.index)
    used_raw_cols = []

    # age
    if "age" in df.columns:
        out["age"] = pd.to_numeric(df["age"], errors="coerce")
        used_raw_cols.append("age")
    elif "Age" in df.columns:
        out["age"] = pd.to_numeric(df["Age"], errors="coerce")
        used_raw_cols.append("Age")

    # stage
    if "stage_num" in df.columns:
        out["stage_num"] = pd.to_numeric(df["stage_num"], errors="coerce")
        used_raw_cols.append("stage_num")
    elif "stage" in df.columns:
        out["stage_num"] = df["stage"].apply(stage_to_numeric)
        used_raw_cols.append("stage")
    elif "Stage" in df.columns:
        out["stage_num"] = df["Stage"].apply(stage_to_numeric)
        used_raw_cols.append("Stage")

    # gender / sex
    gender_col = None
    for c in ["gender", "Gender", "sex", "Sex"]:
        if c in df.columns:
            gender_col = c
            break

    if gender_col is not None:
        tmp = df[gender_col].astype(str).str.strip()
        dummies = pd.get_dummies(tmp, prefix="gender", drop_first=True, dtype=float)
        out = pd.concat([out, dummies], axis=1)
        used_raw_cols.append(gender_col)

    # optional extra clinical variables from CLINICAL_COLS_RAW
    for c in CLINICAL_COLS_RAW:
        if c not in df.columns:
            continue
        if c in used_raw_cols:
            continue
        if c in [TIME_COL, EVENT_COL, FOLD_COL, CANCER_COL]:
            continue
        if c in MODEL_RISK_COLS:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            dummies = pd.get_dummies(
                df[c].astype(str).str.strip(),
                prefix=c,
                drop_first=True,
                dtype=float,
            )
            out = pd.concat([out, dummies], axis=1)

        used_raw_cols.append(c)

    if ZSCORE_CONTINUOUS_CLINICAL:
        for c in out.columns:
            n_unique = out[c].dropna().nunique()
            if n_unique > 2:
                out[c] = safe_zscore(out[c])

    return out, used_raw_cols


def fit_univariate_cox_for_fold(df_fold, risk_col):
    """
    Evaluate SQUALL performance for one fold.

    Returns:
        c_index, coef, HR, p_value, n_samples, n_events
    """
    tmp = df_fold[[TIME_COL, EVENT_COL, risk_col]].copy()
    tmp[TIME_COL] = pd.to_numeric(tmp[TIME_COL], errors="coerce")
    tmp[EVENT_COL] = normalize_event_column(tmp[EVENT_COL])
    tmp[risk_col] = pd.to_numeric(tmp[risk_col], errors="coerce")
    tmp = tmp.dropna()

    n_samples = tmp.shape[0]
    n_events = int(tmp[EVENT_COL].sum())

    if n_samples < 10 or n_events < 3:
        return None

    if tmp[risk_col].nunique() <= 1:
        return None

    tmp["risk_z"] = safe_zscore(tmp[risk_col])
    tmp = tmp.dropna()

    if tmp["risk_z"].nunique() <= 1:
        return None

    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(
            tmp[[TIME_COL, EVENT_COL, "risk_z"]],
            duration_col=TIME_COL,
            event_col=EVENT_COL,
        )

        s = cph.summary.loc["risk_z"]
        coef = float(s["coef"])
        hr = float(np.exp(coef))
        p = float(s["p"])

        # lifelines concordance_index 
        # risk  -risk
        c_index = float(concordance_index(
            tmp[TIME_COL],
            -tmp["risk_z"],
            tmp[EVENT_COL],
        ))

        return {
            "c_index": c_index,
            "coef": coef,
            "HR": hr,
            "p_value": p,
            "n_samples": int(n_samples),
            "n_events": int(n_events),
        }

    except Exception as e:
        print(f"[WARN] Fold Cox failed: {e}")
        return None


def choose_best_storm_fold(df):
    records = []

    for fold, df_fold in df.groupby(FOLD_COL):
        res = fit_univariate_cox_for_fold(df_fold, SQUALL_RISK_COL)
        if res is None:
            continue

        res[FOLD_COL] = fold
        records.append(res)

    if len(records) == 0:
        raise RuntimeError(
            "No valid fold can be evaluated. "
            "Please check TIME_COL, EVENT_COL, FOLD_COL and SQUALL_RISK_COL."
        )

    perf = pd.DataFrame(records)

    #  SQUALL C-index  fold
    #  p-value 
    # best_row = perf.sort_values(["p_value", "c_index"], ascending=[True, False]).iloc[0]
    best_row = perf.sort_values(
        ["c_index", "p_value"],
        ascending=[False, True],
    ).iloc[0]

    best_fold = best_row[FOLD_COL]

    return best_fold, perf


def build_multivariate_dataframe(df_best):
    """
    Construct Cox input:
        time, event, clinical covariates, model risks
    """
    base = pd.DataFrame(index=df_best.index)
    base[TIME_COL] = pd.to_numeric(df_best[TIME_COL], errors="coerce")
    base[EVENT_COL] = normalize_event_column(df_best[EVENT_COL])

    clinical_df, used_clinical_cols = prepare_clinical_features(df_best)

    existing_model_cols = find_existing_columns(df_best, MODEL_RISK_COLS)
    if SQUALL_RISK_COL not in existing_model_cols:
        raise ValueError(f"{SQUALL_RISK_COL} not found in dataframe.")

    risk_df = pd.DataFrame(index=df_best.index)

    for c in existing_model_cols:
        risk_df[c] = pd.to_numeric(df_best[c], errors="coerce")
        if ZSCORE_RISK:
            risk_df[c] = safe_zscore(risk_df[c])

    cox_df = pd.concat([base, clinical_df, risk_df], axis=1)

    #  NA 
    feature_cols = [c for c in cox_df.columns if c not in [TIME_COL, EVENT_COL]]
    valid_feature_cols = []

    for c in feature_cols:
        if cox_df[c].notna().sum() == 0:
            continue
        if cox_df[c].dropna().nunique() <= 1:
            continue
        valid_feature_cols.append(c)

    cox_df = cox_df[[TIME_COL, EVENT_COL] + valid_feature_cols].dropna()

    return cox_df, used_clinical_cols, existing_model_cols


def fit_multivariate_cox(cox_df):
    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(
        cox_df,
        duration_col=TIME_COL,
        event_col=EVENT_COL,
    )

    summary = cph.summary.copy()
    summary["variable"] = summary.index
    summary["HR"] = np.exp(summary["coef"])
    summary["CI_lower"] = np.exp(summary["coef lower 95%"])
    summary["CI_upper"] = np.exp(summary["coef upper 95%"])
    summary["p_value"] = summary["p"]
    summary["n_samples"] = cox_df.shape[0]
    summary["n_events"] = int(cox_df[EVENT_COL].sum())

    keep_cols = [
        "variable",
        "coef",
        "HR",
        "CI_lower",
        "CI_upper",
        "p_value",
        "n_samples",
        "n_events",
    ]

    return summary[keep_cols].reset_index(drop=True), cph


def variable_label(var):
    label_map = {
        "age": "Age",
        "stage_num": "Stage",
        "UNI_risk": "UNI",
        "plip_risk": "PLIP",
        "virchow_risk": "Virchow",
        "SQUALL_risk": "SQUALL",
        "UNI_riskscore": "UNI",
        "plip_riskscore": "PLIP",
        "virchow_riskscore": "Virchow",
        "SQUALL_riskscore": "SQUALL",
    }
    return label_map.get(var, var)


def order_forest_rows(result_df, clinical_cols, model_cols):
    """
    Desired order from top to bottom:
        clinical features
        other models
        SQUALL
    """
    vars_present = result_df["variable"].tolist()

    clinical_order = [c for c in result_df["variable"] if c not in model_cols]

    other_models = [
        c for c in model_cols
        if c in vars_present and c != SQUALL_RISK_COL
    ]

    storm = [SQUALL_RISK_COL] if SQUALL_RISK_COL in vars_present else []

    final_order = clinical_order + other_models + storm

    result_df = result_df.copy()
    result_df["plot_order"] = result_df["variable"].apply(
        lambda x: final_order.index(x) if x in final_order else 999
    )
    result_df = result_df.sort_values("plot_order", ascending=True)

    return result_df


def plot_forest(result_df, output_png, output_pdf=None):
    """
    Draw hazard forest plot.

    Top-to-bottom order is kept as result_df order.
    """
    plot_df = result_df.copy()
    plot_df["label"] = plot_df["variable"].apply(variable_label)

    # matplotlib  y 
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    y = np.arange(plot_df.shape[0])

    hr = plot_df["HR"].values
    lower = plot_df["CI_lower"].values
    upper = plot_df["CI_upper"].values

    xerr_low = hr - lower
    xerr_high = upper - hr

    fig_height = max(FIGSIZE[1], 0.42 * plot_df.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], fig_height))

    ax.errorbar(
        hr,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        capsize=3,
        linewidth=1.2,
        markersize=5,
    )

    ax.axvline(1.0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio, log scale")
    ax.set_title("Multivariate Cox Hazard Forest Plot")

    #  HR  p-value
    x_max = np.nanmax(upper)
    x_text = x_max * 1.15

    for i, row in plot_df.iterrows():
        txt = f"HR={row['HR']:.2f} [{row['CI_lower']:.2f}, {row['CI_upper']:.2f}], p={row['p_value']:.3g}"
        ax.text(
            x_text,
            i,
            txt,
            va="center",
            fontsize=8,
        )

    ax.set_xlim(
        max(np.nanmin(lower) * 0.7, 1e-3),
        x_text * 3.0,
    )

    ax.grid(axis="x", linestyle=":", linewidth=0.6)
    plt.tight_layout()

    fig.savefig(output_png, dpi=DPI, bbox_inches="tight")

    if output_pdf is not None:
        fig.savefig(output_pdf, bbox_inches="tight")

    plt.close(fig)


# =====================================================
# Main
# =====================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("[INFO] Input shape:", df.shape)
    print("[INFO] Columns:")
    print(df.columns.tolist())

    required_cols = [
        CANCER_COL,
        FOLD_COL,
        TIME_COL,
        EVENT_COL,
        SQUALL_RISK_COL,
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    #  CESC
    df_cesc = df[df[CANCER_COL].astype(str) == CANCER_TYPE].copy()

    if df_cesc.empty:
        raise RuntimeError(f"No samples found for CancerType == {CANCER_TYPE}")

    print(f"[INFO] {CANCER_TYPE} rows:", df_cesc.shape[0])
    print("[INFO] Fold counts:")
    print(df_cesc[FOLD_COL].value_counts(dropna=False))

    #  SQUALL  fold
    best_fold, fold_perf = choose_best_storm_fold(df_cesc)

    fold_perf_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_SQUALL_fold_performance.csv",
    )
    fold_perf.to_csv(fold_perf_path, index=False)

    print("[INFO] Fold performance:")
    print(fold_perf.sort_values("c_index", ascending=False))

    print(f"[INFO] Best SQUALL fold = {best_fold}")

    #  fold
    df_best = df_cesc[df_cesc[FOLD_COL] == best_fold].copy()

    best_fold_data_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_best_SQUALL_fold_raw_data.csv",
    )
    df_best.to_csv(best_fold_data_path, index=False)

    #  multivariate Cox 
    cox_df, used_clinical_cols, existing_model_cols = build_multivariate_dataframe(df_best)

    print("[INFO] Used raw clinical columns:", used_clinical_cols)
    print("[INFO] Used model risk columns:", existing_model_cols)
    print("[INFO] Cox input shape:", cox_df.shape)
    print("[INFO] Cox events:", int(cox_df[EVENT_COL].sum()))

    cox_input_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_best_fold_multivariate_cox_input.csv",
    )
    cox_df.to_csv(cox_input_path, index=False)

    if cox_df.shape[0] < 20:
        raise RuntimeError(
            f"Too few samples after dropna for multivariate Cox: n={cox_df.shape[0]}"
        )

    if int(cox_df[EVENT_COL].sum()) < 5:
        raise RuntimeError(
            f"Too few events after dropna for multivariate Cox: events={int(cox_df[EVENT_COL].sum())}"
        )

    # multivariate Cox
    result_df, cph = fit_multivariate_cox(cox_df)

    #  ->  -> SQUALL
    result_df = order_forest_rows(
        result_df,
        clinical_cols=used_clinical_cols,
        model_cols=existing_model_cols,
    )

    result_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_best_SQUALL_fold_multivariate_cox_results.csv",
    )
    result_df.to_csv(result_path, index=False)

    print("[INFO] Multivariate Cox results:")
    print(result_df)

    # forest plot
    output_png = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_best_SQUALL_fold_multivariate_cox_forest.png",
    )
    output_pdf = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_best_SQUALL_fold_multivariate_cox_forest.pdf",
    )

    plot_forest(result_df, output_png, output_pdf)

    print("✅ Done.")
    print(f"Best fold: {best_fold}")
    print(f"Fold performance: {fold_perf_path}")
    print(f"Cox input: {cox_input_path}")
    print(f"Cox results: {result_path}")
    print(f"Forest plot PNG: {output_png}")
    print(f"Forest plot PDF: {output_pdf}")


if __name__ == "__main__":
    main()
