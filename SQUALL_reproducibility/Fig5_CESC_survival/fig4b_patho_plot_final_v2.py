import os
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===  ===
adata = sc.read_h5ad("cluster_outputs/oc_with_hc_25_annotation.h5ad")

# ===  annotation  ===
adata.obs["annotation"] = adata.obs["annotation"].astype(str)
adata.obs["annotation"] = adata.obs["annotation"].str.replace(r"^Debris.*", "Debris", regex=True)
adata.obs["annotation"] = adata.obs["annotation"].replace({"Unannotated": "Unknown"})

# ===  ===
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
# ===  ===
all_classes = adata.obs["annotation_merged"].cat.categories.tolist()

# === 
fixed_color_map = {
    "Debris": "#1f77b4",                    # 
    "Stroma": "#ff7f0e",                    # 
    "Normal parenchyma": "#2ca02c",         # 
    "Tumor": "#9467bd",                     # 
    "Adipose tissue": "#8c564b",            # 
    "Other pathological conditions": "#e377c2"  # 
}

# === 
other_classes = [c for c in all_classes if c not in fixed_color_map]
auto_colors = sns.color_palette("tab20", len(other_classes))
auto_color_map = dict(zip(other_classes, auto_colors))

# === 
highlight_color_map = {**fixed_color_map, **auto_color_map}

# === 
output_dir = "figures/annotation_individual_umaps_scanpy_merged_fixed_hc25"
os.makedirs(output_dir, exist_ok=True)
# ===  hc_clusters_25  UMAP  ===
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
# ===  ===
# ===  pathology_id  UMAP ===

# ===  cluster  ===
cluster_col = "hc_clusters_25"
unique_clusters = adata.obs[cluster_col].cat.categories

for i,cluster in enumerate(unique_clusters):
    temp_col = f"highlight_cluster_{cluster}"

    # 
    adata.obs[temp_col] = ["Highlight" if x == cluster else "Other" for x in adata.obs[cluster_col]]

    #  Highlight 
    adata_temp = adata.copy()
    sort_idx = adata_temp.obs.sort_values(by=temp_col, ascending=False).index
    adata_temp = adata_temp[sort_idx, :]
    #  UMAP 
    highlight_color = cluster_palette[i]
    palette = {
        "Highlight": highlight_color,
        "Other": "#B0B0B0"
    }

    #  1 
    cluster_label = f"Cluster {int(cluster) + 1}"

    # 
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

    # 
    fname = os.path.join(output_dir, f"UMAP_cluster_{int(cluster)+1}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved: {fname}")

    # 
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

    # 
    adata.obs[temp_col] = ["Highlight" if x == target_class else "Other" for x in adata.obs["annotation_merged"]]

    #  temp_col Highlight 
    adata_temp = adata.copy()
    sort_idx = adata_temp.obs.sort_values(by=temp_col, ascending=False).index
    adata_temp = adata_temp[sort_idx, :]

    # /
    palette = {
        "Highlight": highlight_color_map[target_class],
        "Other": "#B0B0B0"
    }

    # ===  ===
    sc.pl.umap(
        adata_temp,
        color=temp_col,
        palette=palette,
        size=8,
        alpha=0.3,          # ✅ 
        edgecolor='none',   # ✅ 
        frameon=False,
        legend_loc=None,
        title=target_class,
        show=False
    )

    # 
    fname = os.path.join(output_dir, f"UMAP_{target_class.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"✅ Saved: {fname}")

    # 
    del adata.obs[temp_col]

