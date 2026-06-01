import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from gseapy import enrichr

# === Step 1: Load datasets ===
adata_cluster = sc.read_h5ad("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/OV/cluster_outputs/oc_with_hc_25_annotation.h5ad")
adata_expr = sc.read_h5ad("combined_annotated_pseudobulk.h5ad")

# === Step 2: 标记 cluster 6 和 42 ===
adata_cluster.obs["hc_clusters"] = adata_cluster.obs["hc_clusters_25"].astype(str)
adata_cluster.obs["cluster_label"] = adata_cluster.obs["hc_clusters"].apply(lambda x: x if x in ["14"] else "other")

# === Step 3: 提取 cluster 6 / 42 对应的 spot_id ===
cluster_spots = adata_cluster.obs.copy()
cluster_spots["spot_id"] = adata_cluster.obs['spot_id']
cluster_spots = cluster_spots[cluster_spots["cluster_label"].isin(["14"])]
cluster_14_ids = cluster_spots[cluster_spots["cluster_label"] == "14"]["spot_id"]
# === Step 4: 匹配表达数据中的 spot_id（通过模糊匹配）===
matched_14 = adata_expr.obs[adata_expr.obs["spot_id"].str.contains('|'.join(cluster_14_ids))]

# === Step 5: 创建 group label 并添加到表达矩阵 ===
adata_expr.obs["cluster_group"] = "other"
adata_expr.obs.loc[matched_14.index, "cluster_group"] = "14"
# === Step 6: DEG 分析 ===
def compute_deg(adata, group, reference="other", top_pct=0.05, prefix="None"):
    all_groups = adata.obs["cluster_group"].unique().tolist()
    sc.pp.highly_variable_genes(adata, n_top_genes=8000, flavor='seurat')  # 可根据需要调整 flavor
    adata = adata[:, adata.var['highly_variable']].copy()
    reference_groups = [g for g in all_groups if g != group]
    sc.tl.rank_genes_groups(
        adata,
        groupby="cluster_group",
        groups=[group],
        reference=reference,
        method="wilcoxon",
        pts=True
    )
    result = adata.uns["rank_genes_groups"]
    genes = pd.DataFrame({
        "gene_symbol": result["names"][group],
        "scores": result["scores"][group],
        "logfoldchanges": result["logfoldchanges"][group],
        "pvals": result["pvals"][group],
        "pvals_adj": result["pvals_adj"][group]
    })

    genes_sorted = genes.sort_values("logfoldchanges", ascending=False)
    n_top = max(1, int(len(genes_sorted) * top_pct))
    top_genes = genes_sorted.head(n_top)["gene_symbol"].tolist()
    bottom_genes = genes_sorted.tail(n_top)["gene_symbol"].tolist()

    # 保存所有 DEG 和 top/bottom gene 列表
    os.makedirs("deg_results", exist_ok=True)
    genes_sorted.to_csv(f"deg_results/{prefix}_full_DEG_norm2.csv", index=False)
    pd.DataFrame(top_genes, columns=["gene_symbol"]).to_csv(f"deg_results/{prefix}_top_genes_norm2.csv", index=False)
    pd.DataFrame(bottom_genes, columns=["gene_symbol"]).to_csv(f"deg_results/{prefix}_bottom_genes_norm2.csv", index=False)

    return top_genes, bottom_genes

adata_deg = adata_expr.copy()
# 获取所有 gene 名（Assume in .var_names）
genes = adata_deg.var_names

# 定义剔除条件（参考你对 deg_df 的处理逻辑）
to_remove = (
    genes.str.startswith(("RPL", "RPS", "MT-", "MT.", "MTRNR")) |
    genes.str.lower().str.startswith(("mir", "snr", "lnc")) |
    genes.isin(["GAPDH", "ACTB", "ACTG1", "TUBB", "B2M"])
)

# 保留基因的布尔掩码
keep_genes = ~to_remove

# 子集保留基因
adata_deg = adata_deg[:, keep_genes].copy()
sc.pp.normalize_total(adata_deg, target_sum=1e4)

# Step 2: Log-transform
sc.pp.log1p(adata_deg)
top_14, bottom_14 = compute_deg(adata_deg, group="14", prefix="Cluster14_HVG")
adata_deg.write("deg_results/adata_with_deg_norm2_tumoronly_HVG_hc45_14.h5ad")

# === Step 7: 富集分析 ===
def run_enrichment(gene_list, description, gene_set_library="GO_Biological_Process_2021"):
    enr = enrichr(
        gene_list=gene_list,
        gene_sets=gene_set_library,
        organism="Human",
        outdir=None,
        no_plot=True
    )
    enr.results['Cluster'] = description
    return enr.results

enrich_top_14 = run_enrichment(top_14, description="Cluster14_Top")
enrich_bottom_14 = run_enrichment(bottom_14, description="Cluster14_Bottom")
# === Step 8: 保存富集结果 ===
os.makedirs("enrichment_results", exist_ok=True)
enrich_top_14.to_csv("enrichment_results/cluster14_top_enrichment_norm2_HVG.csv", index=False)
enrich_bottom_14.to_csv("enrichment_results/cluster14_bottom_enrichment_norm2_HVG.csv", index=False)
# === Step 9: 绘制富集条形图 ===
def plot_enrichment_barplot(enrich_df, title, out_path, top_n=10):
    plot_df = enrich_df.sort_values("Adjusted P-value").head(top_n)
    plot_df["-log10(Padj)"] = -np.log10(plot_df["Adjusted P-value"])
    plt.figure(figsize=(8, 6))
    sns.barplot(data=plot_df, y="Term", x="-log10(Padj)", palette="viridis")
    plt.title(title)
    plt.xlabel("-log10 Adjusted P-value")
    plt.ylabel("Enriched Term")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

plot_enrichment_barplot(enrich_top_14, "Cluster 14 - Top 5% Genes", "enrichment_results/cluster14_top_barplot_norm2.png")
plot_enrichment_barplot(enrich_bottom_14, "Cluster 14 - Bottom 5% Genes", "enrichment_results/cluster14_bottom_barplot_norm2.png")
