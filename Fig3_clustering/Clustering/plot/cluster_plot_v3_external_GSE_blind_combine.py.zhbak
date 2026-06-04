import os
import json
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from sklearn.preprocessing import LabelEncoder

# ========== 路径配置 ==========
annotation_dir = "external_GSE/annotation_csv_output"
miso_dir = "external_GSE/clustering_miso_external_assemble"
storm_dir = "external_GSE/results_clusters_GSE_external"
expr_leiden_dir = "external_GSE/results_clusters_external_assemble_expr_leiden"
rgb_leiden_dir = "external_GSE/results_clusters_external_assemble_rgb_leiden"
spatialglue_dir = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/SpatialGlue/spatialglue_multi_external_assemble"
raw_dir = "external_GSE/external_GSE"
folder = "external_GSE/intergrate_GSE_offset"
hmid_list = [fname.split("_embeddings")[0] for fname in os.listdir(folder) if fname.endswith("_embeddings.pt")]

def plot_clusters_comparison(hmid, save_path):
    def load_he_and_locs(sample_path):
        spatial_path = os.path.join(sample_path, "spatial")
        hires_imgs = [f for f in os.listdir(spatial_path) if 'hires' in f.lower() and f.lower().endswith(('.png', '.jpg', '.tif'))]
        if not hires_imgs:
            raise FileNotFoundError(f"No hires image found in {spatial_path}")
        img_path = os.path.join(spatial_path, hires_imgs[0])
        img = np.array(Image.open(img_path))

        csv_files = [f for f in os.listdir(spatial_path) if f.endswith('.csv') and "tissue" in f]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file in {spatial_path}")
        locs_path = os.path.join(spatial_path, csv_files[0])

        try:
            with open(os.path.join(spatial_path, 'scalefactors_json.json')) as f:
                scale_info = json.load(f)
            if not scale_info:
                raise ValueError
        except:
            with open(os.path.join(spatial_path, 'sub_scalefactors_json.json')) as f:
                scale_info = json.load(f)
        scale = scale_info["tissue_hires_scalef"]

        locs = pd.read_csv(locs_path, header=None)
        if "barcode" in str(locs.iloc[0, 0]).lower():
            locs = pd.read_csv(locs_path)
            locs = locs.set_index("barcode")
        else:
            locs.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
            locs = locs.set_index("barcode")
        locs = locs[locs["in_tissue"] == 1]
        return img, locs.copy(), scale

    def draw(ax, locs, clusters, title, scale, spot_size=30):
        coords = locs.copy()
        print("coords",coords)
        try:
            coords[["pxl_row", "pxl_col"]] = coords[["pxl_row", "pxl_col"]] * scale
        except:
            coords[["pxl_row", "pxl_col"]] = coords[["pxl_row_in_fullres", "pxl_col_in_fullres"]] * scale
        coords = coords.round().astype(int)
        cmap = plt.get_cmap("tab20")
        norm = mcolors.Normalize(vmin=0, vmax=np.max(clusters))
        ax.scatter(coords["pxl_col"], coords["pxl_row"], c=clusters, cmap=cmap, norm=norm,
                   s=spot_size, alpha=0.9, edgecolors='none')
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_aspect('equal')

    print(f"📌 Drawing {hmid}...",flush =True)
    sample_path = os.path.join(raw_dir, hmid)
    he_img, locs, scale = load_he_and_locs(sample_path)

    fig, axs = plt.subplots(2, 4, figsize=(22, 12))
    axs = axs.flatten()

    axs[0].imshow(he_img)
    axs[0].set_title("H&E Image")
    axs[0].axis("off")

    shared_barcodes = set(locs.index)

    # Expression Raw
    #print(pd.read_csv(os.path.join(expr_leiden_dir, f"{hmid}_clusters.csv")))
    expr_df = pd.read_csv(os.path.join(expr_leiden_dir, f"{hmid}_clusters.csv")).set_index("barcode")
    expr_barcodes = list(shared_barcodes & set(expr_df.index))
    #print("expr_barcodes",expr_barcodes)
    draw(axs[1], locs.loc[expr_barcodes], expr_df.loc[expr_barcodes]["cluster"].values, "Expression Raw", scale)
    shared_barcodes &= set(expr_barcodes)

    # RGB Raw
    rgb_df = pd.read_csv(os.path.join(rgb_leiden_dir, f"{hmid}_clusters.csv")).set_index("barcode")
    rgb_barcodes = list(shared_barcodes & set(rgb_df.index))
    draw(axs[2], locs.loc[rgb_barcodes], rgb_df.loc[rgb_barcodes]["cluster"].values, "RGB Raw", scale)
    shared_barcodes &= set(rgb_barcodes)

    # Miso
    miso_df = pd.read_csv(os.path.join(miso_dir, f"{hmid}_cluster.csv")).set_index("barcode")
    miso_barcodes = list(shared_barcodes & set(miso_df.index))
    draw(axs[4], locs.loc[miso_barcodes], miso_df.loc[miso_barcodes]["cluster"].values, "Miso", scale)
    shared_barcodes &= set(miso_barcodes)

    # SpatialGlue
    spg_df = pd.read_csv(os.path.join(spatialglue_dir, hmid, f"{hmid}_clusters.csv"), index_col=0)
    spg_barcodes = list(shared_barcodes & set(spg_df.index))
    draw(axs[5], locs.loc[spg_barcodes], spg_df.loc[spg_barcodes]["SpatialGlue"].values, "SpatialGlue", scale)
    shared_barcodes &= set(spg_barcodes)

    # SQUALL
    storm_df = pd.read_csv(os.path.join(storm_dir, f"{hmid}_combine_clusters.csv"), names=["barcode", "cluster"], header=0)
    storm_df = storm_df.set_index("barcode")

    le_storm = LabelEncoder()
    storm_df["cluster"] = le_storm.fit_transform(storm_df["cluster"])
    storm_barcodes = list(shared_barcodes & set(storm_df.index))
    draw(axs[6], locs.loc[storm_barcodes], storm_df.loc[storm_barcodes]["cluster"].values, "SQUALL", scale)
    shared_barcodes &= set(storm_barcodes)
    #print("shared_barcodes",len(shared_barcodes))
    # Annotation (最后绘制)
    anno_path = os.path.join(annotation_dir, f"{hmid}_annotation.csv")
    if os.path.exists(anno_path):
        anno_df = pd.read_csv(anno_path)
        # 去重：保留每个 barcode 的第一条记录
        anno_df = anno_df.drop_duplicates(subset="barcode", keep="first")
        anno_df["annotation"] = anno_df["annotation"].fillna("other").astype(str)
        anno_df = anno_df.set_index("barcode")
        final_barcodes = sorted(set(shared_barcodes) & set(anno_df.index) & set(locs.index))
        anno_df = anno_df.loc[final_barcodes]
        coords = locs.loc[final_barcodes]
        annos = anno_df["annotation"].values
        #print("coords", len(coords))
        #print("annos", len(annos))
        if final_barcodes:
            annos = anno_df.loc[final_barcodes]["annotation"].values
            le = LabelEncoder()
            labels_numeric = le.fit_transform(annos)
            label_names = le.classes_

            coords = locs.loc[final_barcodes].copy()
            coords[["pxl_row", "pxl_col"]] = coords[["pxl_row", "pxl_col"]] * scale
            coords = coords.round().astype(int)

            cmap = get_cmap("tab20", len(label_names))
            norm = mcolors.Normalize(vmin=0, vmax=len(label_names) - 1)
            #print(labels_numeric)
            sc = axs[3].scatter(coords["pxl_col"], coords["pxl_row"], c=labels_numeric,
                                cmap=cmap, norm=norm, s=30, edgecolors='none')
            axs[3].invert_yaxis()
            axs[3].axis("off")
            axs[3].set_aspect('equal')
            axs[3].set_title("Annotation")
            cbar = plt.colorbar(sc, ax=axs[3], fraction=0.046, pad=0.01)
            cbar.set_ticks(np.arange(len(label_names)))
            cbar.set_ticklabels(label_names)
            cbar.ax.tick_params(labelsize=8)
        else:
            axs[3].axis("off")
            axs[3].set_title("Annotation (no shared barcodes)")
    else:
        axs[3].axis("off")
        axs[3].set_title("Annotation Missing")

    axs[7].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, transparent=True)
    plt.close()

# ========== 执行 ==========
os.makedirs("comparison_plots_raw_external_GSE_blind_7panel", exist_ok=True)
for hmid in hmid_list:
    out_path = f"comparison_plots_raw_external_GSE_blind_7panel/{hmid}_compare_all_methods_7panel.pdf"
    try:
        plot_clusters_comparison(hmid, out_path)
    except Exception as e:
        print(f"❌ Error processing {hmid}: {e}",flush=True)
        traceback.print_exc()
        continue

print("✅ All done.")

