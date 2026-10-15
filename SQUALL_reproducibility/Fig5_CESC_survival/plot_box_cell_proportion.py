import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ===  ===
df = pd.read_csv("cluster_celltype_counts_from_cells_json.csv")
df = df[df["cell_type"].isin(["Connective","Inflammatory", "Neoplastic", "Epithelial","Dead"])].copy()
df["cluster"] = df["cluster"].astype(str)
df["Group"] = df["cluster"].apply(lambda x: "Cluster 15" if x == "14" else "Other Clusters")

# ===  ===
def compute_fraction(group):
    total = group["count"].sum()
    return group.groupby("cell_type")["count"].sum() / total

fractions = []
for pid in df["pathology_id"].unique():
    df_pid = df[df["pathology_id"] == pid]
    for group, df_grp in df_pid.groupby("Group"):
        frac = compute_fraction(df_grp).reset_index()
        frac["pathology_id"] = pid
        frac["Group"] = group
        fractions.append(frac)

frac_df = pd.concat(fractions, ignore_index=True).rename(columns={"count": "Fraction"})

# ===  ===
cell_types = ["Connective","Inflammatory", "Neoplastic", "Epithelial","Dead"]
colors = {"Other Clusters": "gray", "Cluster 15": "#B22222"}

fig, ax = plt.subplots(figsize=(10, 6))
positions = [0, 1, 2,3,4]
width = 0.3

y_max = 0
import numpy as np  #  import
for i, ct in enumerate(cell_types):
    for j, group in enumerate(["Other Clusters", "Cluster 15"]):
        values = frac_df[(frac_df["cell_type"] == ct) & (frac_df["Group"] == group)]["Fraction"]
        pos = i - width/2 if group == "Other Clusters" else i + width/2

        # 
        bp = ax.boxplot(
            values,
            positions=[pos],
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=colors[group], color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black")
        )
        jittered_x = pos + np.random.normal(0, 0.05, size=len(values))  # 0.05 
        ax.scatter(
                jittered_x,
                values,
                color=colors[group],
                edgecolor="black",
                alpha=0.8,
                s=30,
                zorder=3
                )
        '''
        # 
        ax.scatter(
            [pos] * len(values),
            values,
            color=colors[group],
            edgecolor="none",
            alpha=0.8,
            s=30,
            zorder=3
        )
        '''
        y_max = max(y_max, values.max())

# === MWU  ===
for i, ct in enumerate(cell_types):
    g1 = frac_df[(frac_df["cell_type"] == ct) & (frac_df["Group"] == "Other Clusters")]["Fraction"]
    g2 = frac_df[(frac_df["cell_type"] == ct) & (frac_df["Group"] == "Cluster 15")]["Fraction"]
    stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
    
    if p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = "ns"

    ax.plot([i - 0.2, i + 0.2], [y_max + 0.02] * 2, color="black", lw=1)
    ax.text(i, y_max + 0.03, stars, ha="center", fontsize=13)

# ===  ===
ax.set_xticks(positions)
ax.set_xticklabels(cell_types, fontsize=12)
ax.set_ylabel("Cell Fraction", fontsize=12)
ax.set_title("Cell Quantification per Sample", fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', rotation=15)

# 
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="gray", label="Other Clusters", edgecolor="black"),
    Patch(facecolor="#B22222", label="Cluster 15", edgecolor="black"),
]
ax.legend(handles=legend_elements,title="", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
# ===  ===
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("cell_quantification_boxplot_color_corrected.pdf", dpi=300)
plt.savefig("cell_quantification_boxplot_color_corrected.png", dpi=300)
plt.close()

