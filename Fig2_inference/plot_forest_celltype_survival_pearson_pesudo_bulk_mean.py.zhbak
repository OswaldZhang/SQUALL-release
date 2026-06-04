from matplotlib.ticker import FixedLocator, ScalarFormatter
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

def stage_to_numeric(stage):
    match = re.match(r"(IV|V|I{1,3})([A-C]?[0-9]?)", str(stage).upper())
    if match:
        roman = match.group(1)
        roman_map = { 'IV': 4, 'V': 5, 'I': 1, 'II': 2, 'III': 3}
        return roman_map.get(roman, np.nan)
    return np.nan
print("IVC",stage_to_numeric("IVC"))
def find_optimal_cutoff(values, time, event):
    values = values.dropna()
    percentiles = np.linspace(0.2, 0.8, 100)
    min_p = 1.0
    best_cutoff = values.median()
    for perc in percentiles:
        cutoff = np.percentile(values, perc * 100)
        group = values > cutoff
        ix = values.index
        try:
            result = logrank_test(time[ix][group], time[ix][~group],
                                  event_observed_A=event[ix][group],
                                  event_observed_B=event[ix][~group])
            if result.p_value < min_p:
                min_p = result.p_value
                best_cutoff = cutoff
        except:
            continue
    return best_cutoff

# === 文件路径 ===
json_paths = {
    "Tumor": "CESC_gene_slide_mean_expression_treg_tumor_only.json",
    "Other": "CESC_gene_slide_mean_other_region.json",
    "All": "CESC_gene_slide_mean_expression_all_region.json"
}
clinical_csv = "outcome_stage.csv"
survival_json = "/lustre1/zxzeng/bwqin/SQUALL_main/downstream_labels/Survival_five_fold_DSS_COX_outs/Survival_TCGA_CESC.json"
output_dir = "Forest_CD8_manual_combined_optimal"
os.makedirs(output_dir, exist_ok=True)

cd8_manual_markers = ["CD3G", "CD8A", "GZMK"]

# === 读取临床数据 ===
df_clinical = pd.read_csv(clinical_csv)
df_clinical = df_clinical.rename(columns={"bcr_patient_barcode": "sample", "age_at_initial_pathologic_diagnosis": "age", "clinical_stage": "stage"})
df_clinical["stage"] = df_clinical["stage"].str.replace("Stage ", "", regex=False).str.strip()
df_clinical["stage"] = df_clinical["stage"].apply(stage_to_numeric)
print(df_clinical["stage"])
print("df_clinical stage",list(set(df_clinical["stage"].tolist())))
df_clinical = df_clinical.dropna(subset=["age", "stage"])

# === 读取生存数据 ===
with open(survival_json) as f:
    survival_data = json.load(f)

# === 读取表达数据 ===
region_scores = {}
valid_samples = None
for region_key, path in json_paths.items():
    with open(path) as f:
        gene_expr = json.load(f)

    samples = sorted({s for g in gene_expr for s in gene_expr[g]})
    genes = list(gene_expr.keys())
    df_expr = pd.DataFrame(index=samples, columns=genes, dtype=float)
    for g in gene_expr:
        for s, v in gene_expr[g].items():
            df_expr.at[s, g] = v

    valid_genes = [g for g in cd8_manual_markers if g in df_expr.columns]
    df_expr[f"{region_key}_CD8_manual_score"] = df_expr[valid_genes].mean(axis=1)

    df_expr = df_expr[df_expr.index.isin(survival_data.keys())]
    region_scores[region_key] = df_expr[[f"{region_key}_CD8_manual_score"]]

    if valid_samples is None:
        valid_samples = set(df_expr.index)
    else:
        valid_samples &= set(df_expr.index)

# === 合并数据 ===
final_df = pd.DataFrame(index=sorted(valid_samples))
for region in region_scores:
    final_df = final_df.join(region_scores[region], how="left")
final_df["time"] = final_df.index.map(lambda x: survival_data[x]["time"])
final_df["status"] = final_df.index.map(lambda x: survival_data[x]["status"])
final_df = final_df.reset_index().rename(columns={"index": "sample"})
final_df = final_df.merge(df_clinical, on="sample", how="inner")

# === Optimal cutoff for CD8 scores ===
for region in ["Tumor", "Other", "All"]:
    score_col = f"{region}_CD8_manual_score"
    cutoff = find_optimal_cutoff(final_df[score_col], final_df["time"], final_df["status"])
    final_df[score_col] = (final_df[score_col] > cutoff).astype(int)

# === 其他二值化 ===
final_df["age"] = (final_df["age"] > 65).astype(int)
final_df["stage1"] = (final_df["stage"] == 1).astype(int)
final_df["stage2"] = (final_df["stage"] == 2).astype(int)
final_df["stage3"] = (final_df["stage"] == 3).astype(int)
final_df["stage4"] = (final_df["stage"] == 4).astype(int)

# === Cox 分析 ===
binary_vars = ["Tumor_CD8_manual_score", "Other_CD8_manual_score", "All_CD8_manual_score", "age", "stage1", "stage2", "stage3","stage4"]
rename_map = {
    "Tumor_CD8_manual_score": "Tumor CD8 score",
    "Other_CD8_manual_score": "Other CD8 score",
    "All_CD8_manual_score": "All CD8 score",
    "age": "Age > 65",
    "stage1": "Stage I",
    "stage2": "Stage II",
    "stage3": "Stage III",
    "stage4": "Stage IV"
    }
desired_order = [
    "Age > 65", "Stage I", "Stage II", "Stage III","Stage IV",
    "Other CD8 score", "All CD8 score", "Tumor CD8 score"
]

results = []
for var in binary_vars:
    df_model = final_df[[var, "time", "status"]].dropna()
    try:
        cph = CoxPHFitter()
        cph.fit(df_model, duration_col="time", event_col="status")
        summary = cph.summary.loc[var]
        hr = np.exp(summary["coef"])
        ci_lower = np.exp(summary["coef lower 95%"])
        ci_upper = np.exp(summary["coef upper 95%"])
        p = summary["p"]
        coef = summary["coef"]
        p_one_side = p# / 2 if coef < 0 else 1 - p / 2
        n_positive = int(df_model[var].sum())
    except Exception as e:
        print(f"❗ {var} Cox failed: {e}")
        hr, ci_lower, ci_upper, p_one_side, n_positive = [np.nan] * 5

    results.append({
        "Variable": rename_map[var],
        "HR": hr,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "p": p_one_side,
        "N": n_positive
    })

# === 绘图准备 ===
df_forest = pd.DataFrame(results).dropna()
df_forest = df_forest.set_index("Variable").loc[desired_order].reset_index()

# === 联合绘图 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [2, 1]})

# --- 森林图 ---
y_pos = np.arange(len(df_forest))
ax1.errorbar(
    df_forest["HR"], y_pos,
    xerr=[df_forest["HR"] - df_forest["CI_lower"], df_forest["CI_upper"] - df_forest["HR"]],
    fmt="s", color="black", markersize=8, capsize=4
)
ax1.axvline(x=1, color="gray", linestyle="--")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_forest["Variable"])
ax1.set_xlabel("Hazard Ratio (HR)")
ax1.set_title("Univariable Cox Forest Plot")
ax1.set_xscale("log")
ax1.set_xticks([0.25, 0.5, 1, 2, 5, 10])  # 可根据 HR 范围自定义
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.tick_params(axis='x', which='major', labelsize=10)

# --- 表格文本 ---
hr_ci_text = [f"{hr:.2f} ({low:.2f}-{high:.2f})" for hr, low, high in zip(df_forest["HR"], df_forest["CI_lower"], df_forest["CI_upper"])]
p_text = [f"{p:.3f}" for p in df_forest["p"]]
n_text = [str(n) for n in df_forest["N"]]

for i, (hrci, pval, n) in enumerate(zip(hr_ci_text, p_text, n_text)):
    ax2.text(0.00, i, hrci, va='center', ha='left', fontsize=10)
    ax2.text(0.62, i, pval, va='center', ha='left', fontsize=10)
    ax2.text(0.85, i, n, va='center', ha='left', fontsize=10)

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, len(df_forest)-0.5)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("HR (95% CI)     p-value     N")

for spine in ax2.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{output_dir}/forest_combined_optimal.pdf", dpi=300)
plt.savefig(f"{output_dir}/forest_combined_optimal.png", dpi=300)

