#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'DejaVu Sans'


# =====================================================
# Config
# =====================================================

INPUT_CSV = "SQUALL_better_risk_score_patientwise_all_COX_all_robust_0322.csv"
CLINICAL_CSV = "outcome_stage.csv"

OUTDIR = "multivariate_cox_each_model_adjusted_clinical_stage_categorical"

CANCER_TYPE = "CESC"

TIME_COL = "Time"
EVENT_COL = "Status"
CANCER_COL = "CancerType"

#  INPUT_CSV  sample  SAMPLE_COL
SAMPLE_COL = None

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

# risk columns
MODEL_RISK_COLS = [
    "UNI_risk_norm",
    "plip_risk_norm",
    "virchow_risk_norm",
    "SQUALL_risk_norm",
]

MODEL_LABEL_MAP = {
    "UNI_risk_norm": "UNI",
    "plip_risk_norm": "PLIP",
    "virchow_risk_norm": "Virchow",
    "SQUALL_risk_norm": "SQUALL",
}

MODEL_ORDER = ["UNI", "PLIP", "Virchow", "SQUALL"]
MODEL_VARS = set(MODEL_ORDER)

# optimal cutoff setting, follow surv_cutpoint(minprop = 0.2)
MINPROP = 0.2

# Cox penalizer
COX_PENALIZER = 0.05

#  Wald p-value
ONE_SIDED_MODEL_P = True

# 
# "greater": H1: coef > 0, HR > 1
# "less":    H1: coef < 0, HR < 1
ONE_SIDED_ALTERNATIVE = "greater"

#  INPUT_CSV  CLINICAL_CSV  tumor_fraction Cox
USE_TUMOR_FRACTION_IF_AVAILABLE = True

# clinical-only rows shown in forest plot
# clinical rows are estimated from clinical-only Cox:
# Surv ~ age + stage dummies + optional tumor_fraction
SHOW_CLINICAL_FROM_CLINICAL_ONLY_MODEL = True

# plot
FIG_WIDTH = 9.2
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
    Accepts:
        Stage I / I / IA / IB       -> 1
        Stage II / IIA / IIB        -> 2
        Stage III / IIIC            -> 3
        Stage IV / IVB              -> 4
    """
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return np.nan

    s = str(stage).upper()
    s = s.replace("STAGE", "")
    s = s.replace("PATHOLOGIC", "")
    s = s.replace("CLINICAL", "")
    s = s.strip()

    # numeric string
    try:
        val = float(s)
        if val in [1, 2, 3, 4]:
            return val
    except Exception:
        pass

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


def stage_num_to_roman(x):
    if pd.isna(x):
        return "NA"

    x = int(float(x))

    return {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
    }.get(x, str(x))


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


def pvalue_label(p):
    if pd.isna(p):
        return "NA"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.3g}"


# =====================================================
# Clinical loading
# =====================================================

def load_clinical_features(clinical_csv):
    """
    Follow uploaded clinical code:

    df_cli = pd.read_csv(CLINICAL_CSV)
    df_cli = df_cli.rename(columns={
        "bcr_patient_barcode": "sample",
        "age_at_initial_pathologic_diagnosis": "age",
        "clinical_stage": "stage"
    })
    df_cli["stage_num"] = df_cli["stage"].apply(stage_to_numeric)
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

    # For clinical-adjusted Cox, samples missing age/stage cannot be used.
    df_cli = df_cli[keep_cols].dropna(subset=["sample", "age", "stage_num"])
    df_cli = df_cli.drop_duplicates(subset=["sample"], keep="first")
    df_cli = df_cli.set_index("sample")

    print(f"[INFO] Clinical samples loaded = {df_cli.shape[0]}")
    print("[INFO] Clinical columns loaded:", df_cli.columns.tolist())
    print("[INFO] stage_num counts in clinical:")
    print(df_cli["stage_num"].value_counts(dropna=False).sort_index())

    return df_cli


def merge_clinical_into_main(df, df_cli):
    """
    Merge clinical age/stage_num into risk dataframe by sample.
    """
    sample_col = infer_sample_col(df)

    df = df.copy()
    df["sample"] = df[sample_col].apply(clean_sample_id)

    before_n = df.shape[0]

    input_has_tumor_frac = USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df.columns

    merge_cols = ["age", "stage", "stage_num"]

    if USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df_cli.columns:
        merge_cols.append("tumor_fraction")

    df = df.merge(
        df_cli[merge_cols],
        left_on="sample",
        right_index=True,
        how="inner",
        suffixes=("", "_clinical"),
    )

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


# =====================================================
# Stage categorical covariates
# =====================================================

def make_stage_dummies(df):
    """
    Make categorical stage dummy variables.

    Reference stage:
        smallest stage_num present in current cohort.

    Example:
        if stages are I, II, III, IV:
            stage_II_vs_I
            stage_III_vs_I
            stage_IV_vs_I

        if stages are III, IV:
            stage_IV_vs_III
    """
    stage = safe_numeric(df["stage_num"])

    present_stages = sorted(stage.dropna().unique().tolist())

    if len(present_stages) <= 1:
        print("[WARN] Only one stage present. Stage categorical variables are not estimable.")
        return pd.DataFrame(index=df.index), [], present_stages, None
    ref_stage = present_stages[0]
    ref_label = stage_num_to_roman(ref_stage)

    out = pd.DataFrame(index=df.index)
    stage_cols = []

    for st in present_stages[1:]:
        '''
        if int(float(st)) == 4:
            print("[INFO] Skip Stage IV dummy: Stage IV will not be compared/plotted.")
            continue
        '''
        st_label = stage_num_to_roman(st)
        col = f"stage_{st_label}_vs_{ref_label}"
        out[col] = (stage == st).astype(float)
        stage_cols.append(col)

    print(f"[INFO] Stage reference = Stage {ref_label}")
    print("[INFO] Stage dummy columns:", stage_cols)

    return out, stage_cols, present_stages, ref_stage


def prepare_clinical_features(df):
    """
    Clinical covariates:
        age
        stage categorical dummies
        optional tumor_fraction
    """
    out = pd.DataFrame(index=df.index)

    required = ["age", "stage_num"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing clinical covariates after merge: {missing}")

    out["age"] = safe_numeric(df["age"])

    stage_df, stage_cols, present_stages, ref_stage = make_stage_dummies(df)
    out = pd.concat([out, stage_df], axis=1)

    used = ["age"] + stage_cols

    if USE_TUMOR_FRACTION_IF_AVAILABLE and "tumor_fraction" in df.columns:
        out["tumor_fraction"] = safe_numeric(df["tumor_fraction"])
        used.append("tumor_fraction")

    print("[INFO] Clinical covariates used:", used)

    return out, used


# =====================================================
# Optimal cutoff
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

    if "SQUALL" not in risk_group_df.columns:
        raise RuntimeError("SQUALL risk group was not created. Please check SQUALL risk column.")

    return risk_group_df, cutoff_df


# =====================================================
# Cox helpers
# =====================================================

def drop_invalid_feature_columns(cox_df, feature_cols):
    """
    Drop all-NA or constant features from a specific Cox design.
    """
    valid_cols = []
    dropped_cols = []

    for c in feature_cols:
        if cox_df[c].notna().sum() == 0:
            print(f"[WARN] Drop all-NA column from Cox: {c}")
            dropped_cols.append(c)
            continue

        if cox_df[c].dropna().nunique() <= 1:
            print(f"[WARN] Drop constant column from Cox: {c}")
            dropped_cols.append(c)
            continue

        valid_cols.append(c)

    return valid_cols, dropped_cols


def summarize_cph(cph, variables, n_samples, n_events, p_one_sided_for_models=True):
    summary = cph.summary.copy()

    records = []

    for var in variables:
        if var not in summary.index:
            continue

        s = summary.loc[var]

        coef = float(s["coef"])
        p_two = float(s["p"])

        if p_one_sided_for_models and var in MODEL_VARS:
            p_value = convert_to_one_sided_p(
                p_two,
                coef,
                alternative=ONE_SIDED_ALTERNATIVE,
            )
            p_type = f"one-sided Wald, HR {('>' if ONE_SIDED_ALTERNATIVE == 'greater' else '<')} 1"
        else:
            p_value = p_two
            p_type = "two-sided Wald"

        records.append({
            "variable": var,
            "coef": coef,
            "HR": float(s["exp(coef)"]),
            "CI_lower": float(s["exp(coef) lower 95%"]),
            "CI_upper": float(s["exp(coef) upper 95%"]),
            "p_two_sided": p_two,
            "p_value": p_value,
            "p_type": p_type,
            "n_samples": int(n_samples),
            "n_events": int(n_events),
        })

    return pd.DataFrame(records)


def fit_cox_for_variables(df, variables):
    """
    Fit Cox:
        Surv ~ variables
    """
    cols = [TIME_COL, EVENT_COL] + variables
    tmp = df[cols].copy()

    for c in cols:
        tmp[c] = safe_numeric(tmp[c])

    feature_cols = variables
    valid_feature_cols, dropped_cols = drop_invalid_feature_columns(tmp, feature_cols)

    if len(valid_feature_cols) == 0:
        raise RuntimeError(f"No valid features left for Cox. Dropped: {dropped_cols}")

    tmp = tmp[[TIME_COL, EVENT_COL] + valid_feature_cols].dropna()

    if tmp.shape[0] < 10:
        raise RuntimeError(f"Too few samples for Cox after dropna: n={tmp.shape[0]}")

    if int(tmp[EVENT_COL].sum()) < 3:
        raise RuntimeError(f"Too few events for Cox: events={int(tmp[EVENT_COL].sum())}")

    cph = CoxPHFitter(penalizer=COX_PENALIZER)
    cph.fit(
        tmp,
        duration_col=TIME_COL,
        event_col=EVENT_COL,
    )

    result = summarize_cph(
        cph,
        variables=valid_feature_cols,
        n_samples=tmp.shape[0],
        n_events=int(tmp[EVENT_COL].sum()),
        p_one_sided_for_models=True,
    )

    return result, cph, tmp, valid_feature_cols, dropped_cols


# =====================================================
# Main analysis:
#   clinical-only rows + each model adjusted by clinical
# =====================================================

def build_analysis_dataframe(df_cancer):
    """
    Build dataframe with:
        Time, Status, clinical covariates, model high-risk indicators.
    """
    base = pd.DataFrame(index=df_cancer.index)
    base[TIME_COL] = safe_numeric(df_cancer[TIME_COL])
    base[EVENT_COL] = normalize_event_column(df_cancer[EVENT_COL])

    clinical_df, clinical_cols = prepare_clinical_features(df_cancer)
    risk_group_df, cutoff_df = make_model_risk_groups(df_cancer)

    analysis_df = pd.concat([base, clinical_df, risk_group_df], axis=1)

    return analysis_df, clinical_cols, list(risk_group_df.columns), cutoff_df


def fit_each_model_adjusted_by_clinical(analysis_df, clinical_cols, model_cols):
    """
    Clinical rows:
        from clinical-only Cox:
            Surv ~ clinical

    Model rows:
        for each model:
            Surv ~ clinical + model

    This gives each model's HR adjusted by clinical variables,
    without competing against other AI models.
    """
    all_results = []

    clinical_result = pd.DataFrame()

    if SHOW_CLINICAL_FROM_CLINICAL_ONLY_MODEL and len(clinical_cols) > 0:
        print("[INFO] Fit clinical-only Cox:")
        print("       Surv ~ " + " + ".join(clinical_cols))

        try:
            clinical_result, clinical_cph, clinical_input, valid_clinical_cols, dropped_clinical_cols = fit_cox_for_variables(
                analysis_df,
                variables=clinical_cols,
            )
            clinical_result["fit_type"] = "clinical_only"
            clinical_result["model_adjusted"] = "Clinical"
            all_results.append(clinical_result)
        except Exception as e:
            print(f"[WARN] Clinical-only Cox failed: {e}")

    model_result_records = []

    for model in MODEL_ORDER:
        if model not in model_cols:
            print(f"[WARN] Model {model} not available, skipped.")
            continue

        variables = clinical_cols + [model]

        print(f"[INFO] Fit model-adjusted Cox for {model}:")
        print("       Surv ~ " + " + ".join(variables))

        try:
            res, cph, cox_input, valid_cols, dropped_cols = fit_cox_for_variables(
                analysis_df,
                variables=variables,
            )

            # keep only the model's row for final comparison
            model_row = res[res["variable"] == model].copy()

            if model_row.empty:
                print(f"[WARN] {model} was dropped or not estimable.")
                continue

            model_row["fit_type"] = "clinical_plus_model"
            model_row["model_adjusted"] = model
            model_row["adjusted_for"] = "+".join([c for c in valid_cols if c != model])
            model_row["n_fit_features"] = len(valid_cols)

            all_results.append(model_row)

            # Also save full model summaries for diagnostics
            res_full = res.copy()
            res_full["fit_type"] = "clinical_plus_model_full"
            res_full["model_adjusted"] = model
            model_result_records.append(res_full)

            print(
                f"[INFO] {model}: n={int(model_row['n_samples'].iloc[0])}, "
                f"events={int(model_row['n_events'].iloc[0])}, "
                f"HR={float(model_row['HR'].iloc[0]):.3f}, "
                f"p={float(model_row['p_value'].iloc[0]):.3g}"
            )

        except Exception as e:
            print(f"[WARN] Cox failed for {model}: {e}")

    if len(all_results) == 0:
        raise RuntimeError("No Cox results generated.")

    final_result = pd.concat(all_results, axis=0, ignore_index=True)

    if len(model_result_records) > 0:
        full_model_results = pd.concat(model_result_records, axis=0, ignore_index=True)
    else:
        full_model_results = pd.DataFrame()

    return final_result, full_model_results


# =====================================================
# Forest plot
# =====================================================

def variable_label(var):
    if var == "age":
        return "Age"

    if var == "tumor_fraction":
        return "Tumor fraction"

    # stage dummy labels:
    # stage_II_vs_I -> Stage II vs I
    if var.startswith("stage_") and "_vs_" in var:
        tmp = var.replace("stage_", "")
        a, b = tmp.split("_vs_")
        return f"Stage {a} vs {b}"

    label_map = {
        "UNI": "UNI",
        "PLIP": "PLIP",
        "Virchow": "Virchow",
        "SQUALL": "SQUALL",
    }

    return label_map.get(var, var)


def order_forest_rows(result_df):
    """
    Desired top-to-bottom order:
        clinical
        other models
        SQUALL
    """
    result_df = result_df.copy()

    vars_present = result_df["variable"].tolist()

    clinical_vars = [v for v in vars_present if v not in MODEL_VARS]

    # preferred clinical order: age, stage dummies, tumor_fraction
    stage_vars = [v for v in clinical_vars if v.startswith("stage_")]
    other_clinical = [
        v for v in clinical_vars
        if v not in ["age", "tumor_fraction"] and v not in stage_vars
    ]

    clinical_order = []
    if "age" in clinical_vars:
        clinical_order.append("age")

    clinical_order += stage_vars

    if "tumor_fraction" in clinical_vars:
        clinical_order.append("tumor_fraction")

    clinical_order += other_clinical

    other_models = [m for m in ["UNI", "PLIP", "Virchow"] if m in vars_present]
    storm = ["SQUALL"] if "SQUALL" in vars_present else []

    final_order = clinical_order + other_models + storm

    result_df["plot_order"] = result_df["variable"].apply(
        lambda x: final_order.index(x) if x in final_order else 999
    )

    result_df = result_df.sort_values("plot_order", ascending=True).reset_index(drop=True)

    return result_df


def plot_forest(result_df, output_pdf, output_png=None, title="Clinical-adjusted Cox Hazard Forest Plot"):
    """
    Color:
        - SQUALL red
        - all others blue

    Annotation:
        - model rows use one-sided Wald p-value p1
        - clinical rows use two-sided Wald p-value p
    """
    plot_df = order_forest_rows(result_df)
    plot_df["label"] = plot_df["variable"].apply(variable_label)

    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    n = plot_df.shape[0]
    y = np.arange(n)

    fig_height = max(4.5, 0.52 * n + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    min_x = np.nanmin(plot_df["CI_lower"].values)
    max_x = np.nanmax(plot_df["CI_upper"].values)

    min_x = max(min_x * 0.7, 1e-3)
    max_x = max_x * 1.3

    for i, row in plot_df.iterrows():
        var = row["variable"]
        hr = row["HR"]
        lower = row["CI_lower"]
        upper = row["CI_upper"]

        color = RED if var == "SQUALL" else BLUE

        ax.plot([lower, upper], [i, i], color=color, lw=2)
        ax.scatter(hr, i, color=color, s=45, zorder=3)

    ax.axvline(1, linestyle="--", color="black", lw=1)

    ax.set_xscale("log")
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

    plt.subplots_adjust(right=0.67)

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

    df_cli = load_clinical_features(CLINICAL_CSV)
    df = merge_clinical_into_main(df, df_cli)

    # Follow your R sample selection:
    # df_cancer <- df %>% filter(CancerType == cancer)
    df_cancer = df[df[CANCER_COL].astype(str) == str(CANCER_TYPE)].copy()

    if df_cancer.empty:
        raise RuntimeError(f"No samples found for CancerType == {CANCER_TYPE}")

    print(f"[INFO] CancerType={CANCER_TYPE} rows after clinical merge:", df_cancer.shape[0])
    print("[INFO] stage_num counts after cancer filtering:")
    print(df_cancer["stage_num"].value_counts(dropna=False).sort_index())

    raw_subset_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_raw_selected_samples_with_clinical.csv",
    )
    df_cancer.to_csv(raw_subset_path, index=False)

    analysis_df, clinical_cols, model_cols, cutoff_df = build_analysis_dataframe(df_cancer)

    print("[INFO] Clinical columns:", clinical_cols)
    print("[INFO] Model columns:", model_cols)
    print("[INFO] Analysis dataframe shape:", analysis_df.shape)
    print("[INFO] Analysis columns:")
    print(analysis_df.columns.tolist())

    cutoff_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_optimal_cutoffs.csv",
    )
    cutoff_df.to_csv(cutoff_path, index=False)

    analysis_input_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_cox_analysis_input_each_model_adjusted.csv",
    )
    analysis_df.to_csv(analysis_input_path, index=False)

    result_df, full_model_results = fit_each_model_adjusted_by_clinical(
        analysis_df,
        clinical_cols=clinical_cols,
        model_cols=model_cols,
    )

    result_df = order_forest_rows(result_df)

    result_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_cox_results_each_model_adjusted_by_clinical.csv",
    )
    result_df.to_csv(result_path, index=False)

    full_result_path = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_cox_full_results_each_model_with_clinical.csv",
    )
    full_model_results.to_csv(full_result_path, index=False)

    print("[INFO] Final forest plot results:")
    print(result_df)

    output_pdf = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_forest_each_model_adjusted_by_clinical.pdf",
    )
    output_png = os.path.join(
        OUTDIR,
        f"{CANCER_TYPE}_forest_each_model_adjusted_by_clinical.png",
    )

    plot_forest(
        result_df,
        output_pdf=output_pdf,
        output_png=output_png,
        title=f"{CANCER_TYPE} Clinical-adjusted Cox Hazard Forest Plot",
    )

    print("✅ Done.")
    print(f"Raw selected samples: {raw_subset_path}")
    print(f"Optimal cutoffs: {cutoff_path}")
    print(f"Analysis input: {analysis_input_path}")
    print(f"Forest results: {result_path}")
    print(f"Full per-model Cox results: {full_result_path}")
    print(f"Forest plot PDF: {output_pdf}")
    print(f"Forest plot PNG: {output_png}")


if __name__ == "__main__":
    main()
