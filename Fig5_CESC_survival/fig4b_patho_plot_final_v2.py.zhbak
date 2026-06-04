import os
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 读取数据 ===
adata = sc.read_h5ad("cluster_outputs/oc_with_hc_25_annotation.h5ad")

# === 清洗原始 annotation 字段 ===
adata.obs["annotation"] = adata.obs["annotation"].astype(str)
adata.obs["annotation"] = adata.obs["annotation"].str.replace(r"^Debris.*", "Debris", regex=True)
adata.obs["annotation"] = adata.obs["annotation"].replace({"Unannotated": "Unknown"})

# === 合并类别 ===
merge_map = {
    "Calcification”": "Other pathological conditions",
    "Paramesonephric duct cyst": "Other pathological conditions",
    "Benign cyst": "Other pathological conditions",
    "Normal ovary": "Normal parenchyma",
    "Normal glandular epithelium": "Normal parenchyma",
    "Normal fallopian tube": "Normal parenchyma",
    "Simple stroma": "Stroma",
    "Complex stroma": "Stroma",
}
adata.obs["annotation_merged"] = adata.obs["annotation"].replace(merge_map).astype("category")
adata.write_h5ad
# === 获取所有合并后的类别 ===
all_classes = adata.obs["annotation_merged"].cat.categories.tolist()

# === 指定关键类别的颜色
fixed_color_map = {
    "Debris": "#1f77b4",                    # 蓝
    "Stroma": "#ff7f0e",                    # 橙
    "Normal parenchyma": "#2ca02c",         # 绿
    "Tumor": "#9467bd",                     # 紫
    "Adipose tissue": "#8c564b",            # 棕
    "Other pathological conditions": "#e377c2"  # 粉
}

# === 其他类自动配色
other_classes = [c for c in all_classes if c not in fixed_color_map]
auto_colors = sns.color_palette("tab20", len(other_classes))
auto_color_map = dict(zip(other_classes, auto_colors))

# === 合并颜色映射
highlight_color_map = {**fixed_color_map, **auto_color_map}

# === 创建输出目录
output_dir = "figures/annotation_individual_umaps_scanpy_merged_fixed_hc25"
os.makedirs(output_dir, exist_ok=True)
# === 额外绘制 hc_clusters_25 的整体 UMAP 分布 ===
cluster_palette = sns.color_palette("tab20", len(adata.obs["hc_clusters_25"].cat.categories))

sc.pl.umap(
    adata,
    color="hc_clusters_25",
    palette=cluster_palette,
    size=20,
    frameon=False,
    title="hc_clusters_25",
    legend_loc='on data',
    show=False
)

fname = os.path.join(output_dir, "UMAP_hc_clusters_25.png")
plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
sc.pl.umap(
    adata,
    color="hc_clusters_25",
    palette=cluster_palette,
    size=20,
    frameon=False,
    title="hc_clusters_25",
    legend_loc=None,
    show=False
)

fname = os.path.join(output_dir, "UMAP_hc_clusters_25_wonumber.png")
plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
print(f"✅ Saved cluster UMAP: {fname}")
# === 绘图循环 ===
# === 绘制按 pathology_id 着色的 UMAP ===

# === 为每个 cluster 单独绘图 ===
cluster_col = "hc_clusters_25"
unique_clusters = adata.obs[cluster_col].cat.categories

for i,cluster in enumerate(unique_clusters):
    temp_col = f"highlight_cluster_{cluster}"

    # 标记当前簇
    adata.obs[temp_col] = ["Highlight" if x == cluster else "Other" for x in adata.obs[cluster_col]]

    # 排序让 Highlight 后画（显示在上方）
    adata_temp = adata.copy()
    sort_idx = adata_temp.obs.sort_values(by=temp_col, ascending=False).index
    adata_temp = adata_temp[sort_idx, :]
    # 用与整体 UMAP 相同的颜色
    highlight_color = cluster_palette[i]
    palette = {
        "Highlight": highlight_color,
        "Other": "#B0B0B0"
    }

    # 显示编号从 1 开始
    cluster_label = f"Cluster {int(cluster) + 1}"

    # 绘图
    sc.pl.umap(
        adata_temp,
        color=temp_col,
        palette=palette,
        size=8,
        alpha=0.3,
        edgecolor='none',
        frameon=False,
        legend_loc='on data',
        title=cluster_label,
        show=False
    )

    # 保存图像
    fname = os.path.join(output_dir, f"UMAP_cluster_{int(cluster)+1}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved: {fname}")

    # 清理临时列
    del adata.obs[temp_col]




pathology_palette = sns.color_palette("husl", len(adata.obs["pathology_id"].unique()))
sc.pl.umap(
    adata,
    color="pathology_id",
    size=20,
    frameon=False,
    title="pathology_id",
    legend_loc='right margin',
    show=False
)
fname = os.path.join(output_dir, "UMAP_pathology_id_withnumber.png")
plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
plt.close()
print(f"✅ Saved pathology_id UMAP: {fname}")
for target_class in all_classes:
    temp_col = f"highlight_{target_class.replace(' ', '_')}"

    # 标记高亮类
    adata.obs[temp_col] = ["Highlight" if x == target_class else "Other" for x in adata.obs["annotation_merged"]]

    # 创建副本，按 temp_col 排序（Highlight 后画）
    adata_temp = adata.copy()
    sort_idx = adata_temp.obs.sort_values(by=temp_col, ascending=False).index
    adata_temp = adata_temp[sort_idx, :]

    # 设置绘图颜色：高亮类为固定/自动颜色，其余为灰色
    palette = {
        "Highlight": highlight_color_map[target_class],
        "Other": "#B0B0B0"
    }

    # === 绘图 ===
    sc.pl.umap(
        adata_temp,
        color=temp_col,
        palette=palette,
        size=8,
        alpha=0.3,          # ✅ 透明
        edgecolor='none',   # ✅ 去边框
        frameon=False,
        legend_loc=None,
        title=target_class,
        show=False
    )

    # 保存图像
    fname = os.path.join(output_dir, f"UMAP_{target_class.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved: {fname}")

    # 删除临时列
    del adata.obs[temp_col]

