import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu

# ===  ===
metric_csv = "clustering_comparison_metrics_summary_v2.csv"
hmid_csv = "HMID_infer.csv"
output_path = "overall_performance_boxplot_sig_fixed.png"

models = ["ExprRaw", "RGBRaw", "MISO", "SpatialGlue", "SQUALL"]
metrics = ["NMI", "AMI", "ARI"]
custom_palette = {
    "SQUALL": "#D64242",
    "MISO": "#6E90C2",
    "RGBRaw": "#729FCF",
    "ExprRaw": "#DADADA",
    "SpatialGlue": "#99B5CA"
}

def get_stars(pval):
    if pval <= 0.0001:
        return '****'
    elif pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    return None

# ===  ===
df_metric = pd.read_csv(metric_csv)
df_metric = df_metric[df_metric["SQUALL_NMI"] != 1]
df_meta = pd.read_csv(hmid_csv)

# ===  HMID → Collection ===
hmid_to_cohort = {str(row['hm_offset']): row['Collection'] for _, row in df_meta.iterrows()}
df_metric['Collection'] = df_metric["HMID"].map(hmid_to_cohort)

# ===  ===
opt_list = ['4195ab4c-20bd-4cd3-8b3d-65601277e731', 'GSE238264', 'E-MTAB-13530',
            'GSE200310', 'GSE212526', 'GSE224411', 'VISDP000144']
df_metric = df_metric[df_metric['Collection'].isin(opt_list)]

# ===  ===
long_data = []
for metric in metrics:
    for model in models:
        col_name = f"{model}_{metric}"
        if col_name in df_metric.columns:
            scores = df_metric[col_name].dropna()
            for score in scores:
                long_data.append({
                    "Metric": metric,
                    "Model": model,
                    "Score": score
                })

df_long = pd.DataFrame(long_data)

# ===  ===
sns.set(style="white")
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for i, metric in enumerate(metrics):
    ax = axes[i]
    df_plot = df_long[df_long["Metric"] == metric]
    storm_scores = df_plot[df_plot["Model"] == "SQUALL"]["Score"]

    sns.boxplot(
        data=df_plot,
        x="Model", y="Score", palette=custom_palette,
        order=models, ax=ax
    )

    ax.set_title(f"{metric} (Overall)")
    ax.set_xlabel("")
    if i == 0:
        ax.set_ylabel("Score")
    else:
        ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45)

    # === 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ===  ===
    counter_up, counter_down = 0, 0
    base_gap = 0.05

    for j, model in enumerate(models):
        if model == "SQUALL":
            continue
        model_scores = df_plot[df_plot["Model"] == model]["Score"]
        if len(model_scores) == len(storm_scores):
            stat, p = mannwhitneyu(model_scores, storm_scores, alternative="less")
            print(metric,"SQUALL",model,p)
            stars = get_stars(p)
            if stars:
                y_base = max(model_scores.max(), storm_scores.max()) + 0.02
                use_upper = (j % 2 == 0)

                if use_upper:
                    h = y_base + counter_up * base_gap
                    counter_up += 1
                else:
                    h = y_base - (counter_down + 1) * base_gap
                    counter_down += 1

                x1, x2 = j, models.index("SQUALL")
                ax.plot([x1, x1, x2, x2], [h - 0.01, h, h, h - 0.01], color='black', lw=1)
                ax.text((x1 + x2) / 2, h + 0.005, stars, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.savefig(output_path.replace(".png",".pdf"), dpi=300,transparent = True)
plt.close()
print(f"✅ Saved boxplot with significance: {output_path}")

