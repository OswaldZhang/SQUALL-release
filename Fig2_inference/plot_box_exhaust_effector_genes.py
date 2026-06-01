import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import mannwhitneyu

# === 配置路径 ===
expression_base = "/lustre1/zxzeng/bwqin/SQUALL_main/inference/CESC_expression"
tile_label_base = "/lustre1/zxzeng/bwqin/SQUALL/yf_TCGA_label/tile_masks"
output_dir = "final_tcell_treg_analysis"
os.makedirs(output_dir, exist_ok=True)

# === Gene set ===
geneset_dict = {
    "T cell": ["CD3D", "CD3E", "CD3G"],
    "CD8 T Effector": ["CD8A", "GZMA", "GZMK", "IFNG", "PRF1", "NKG7", "TNFRSF9"],
    "CD8 T Exhaustion": ["CD8A", "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "LAYN", "ENTPD1"]
}
all_genes = sorted(set(g for gs in geneset_dict.values() for g in gs))

# === 工具函数 ===
def normalize_rank(ranks):
    arr = np.array(ranks, dtype=float)
    if len(arr) == 0:
        return arr
    sorted_idx = np.argsort(arr)
    norm = np.zeros_like(arr, dtype=float)
    norm[sorted_idx] = np.linspace(0, 1, len(arr))
    return norm

def pval_to_star(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# === 统计表达 ===
records = []
for sample in tqdm(os.listdir(expression_base), desc="Processing Samples"):
    sample_path = os.path.join(expression_base, sample)
    if not os.path.isdir(sample_path):
        continue

    # 加载 label
    label_file = os.path.join(tile_label_base, f"{sample}_tile_labels.csv")
    if not os.path.exists(label_file):
        continue
    df_label = pd.read_csv(label_file)
    df_label["tile"] = df_label["pos"].apply(lambda x: "_".join(x.split("_")[:4]))
    label_map = dict(zip(df_label["tile"], df_label["label"]))

    # 加载 gene 表达
    gene_expr = {}
    for g in all_genes:
        f = os.path.join(sample_path, f"{g}.json")
        if os.path.exists(f):
            with open(f) as jf:
                gene_expr[g] = json.load(jf)

    tiles = list({t for g in gene_expr for t in gene_expr[g]})
    if not tiles:
        continue

    for gene in all_genes:
        if gene not in gene_expr:
            continue
        values = [gene_expr[gene].get(t, 0) for t in tiles]
        normed = normalize_rank(values)
        for t, v in zip(tiles, normed):
            region = "Tumor" if label_map.get(t, -1) == 1 else "Other"
            records.append({
                "gene": gene,
                "rank_score": v,
                "region": region,
                "sample": sample,
                "tile": t
            })

df_all = pd.DataFrame(records)

# === 绘图（带显著性标注） ===
plt.figure(figsize=(len(all_genes) * 0.5 + 2, 6))
ax = sns.boxplot(
    data=df_all,
    x="gene",
    y="rank_score",
    hue="region",
    hue_order=["Other", "Tumor"],
    palette={"Other": "gray", "Tumor": "#D64242"}
)

# 添加星号显著性标注
for i, gene in enumerate(all_genes):
    dfg = df_all[df_all["gene"] == gene]
    tumor_vals = dfg[dfg["region"] == "Tumor"]["rank_score"]
    other_vals = dfg[dfg["region"] == "Other"]["rank_score"]
    print(gene,tumor_vals.median(),other_vals.median())
    if len(tumor_vals) > 0 and len(other_vals) > 0:
        try:
            stat, p = mannwhitneyu(tumor_vals, other_vals, alternative="two-sided")
            print(p)
            ymax = max(tumor_vals.max(), other_vals.max())
            ax.text(i, ymax + 0.07, pval_to_star(p), ha='center', fontsize=10, fontweight='bold')
        except:
            continue

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.ylabel("Normalized Expression Rank")
plt.xlabel("")
plt.legend(title="Region", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_gene_boxplot_tumor_vs_other.pdf"), dpi=300, bbox_inches="tight", transparent=True)
plt.close()

