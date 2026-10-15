import os
import numpy as np
import pandas as pd
from scipy import io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ====  ====
'''
prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OC_all/"
output_prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/hist2tscript_OC/OV/OCXenium/OCXenium_1"
#tif_path = "/lustre1/zxzeng/bwqin/SQUALL/Xenium/HCC/Xenium/HCC/HE_rescaled_0.5mpp.tiff"
tif_path = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/hist2tscript_OC/OV/OCXenium/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif"
gene_list_path = "/lustre1/zxzeng/bwqin/SQUALL_main/downstream_labels/expr_get_embedding_Xenium_all.csv"
'''
prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/CC_all/"
output_prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/hist2tscript_CC/OV/CCXenium/CCXenium_1"
#tif_path = "/lustre1/zxzeng/bwqin/SQUALL/Xenium/HCC/Xenium/HCC/HE_rescaled_0.5mpp.tiff"
tif_path = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/hist2tscript_CC/OV/CCXenium/Xenium_Prime_Cervical_Cancer_FFPE_he_image.ome.tif"
gene_list_path = "/lustre1/zxzeng/bwqin/SQUALL_main/downstream_labels/expr_get_embedding_Xenium_all.csv"



print("📥 ...", flush=True)
selected_genes = pd.read_csv(gene_list_path)["HGNC_symbol"].dropna().unique().tolist()

print("📥 ...", flush=True)
cnts = io.mmread(f"{prefix}cnts.mtx").tocsr()
genes = pd.read_csv(f"{prefix}genes.tsv", header=None, sep="\t")[0].tolist()
barcodes = pd.read_csv(f"{prefix}barcodes.tsv", header=None, sep="\t")[0].tolist()

print("📥 ...", flush=True)
locs_df = pd.read_csv(f"{prefix}locs-raw.tsv", sep="\t")
locs_df["barcode"] = barcodes

print("✅ ", flush=True)

# ====  ====
print("🧬 ...", flush=True)
valid_genes = [g for g in selected_genes if g in genes]
gene_indices = [genes.index(g) for g in valid_genes]
cnts_filtered = cnts[:, gene_indices]
filtered_gene_names = [genes[i] for i in gene_indices]

# ====  tile  ====
print("📐  tile ...", flush=True)
tile_size = 224
locs_df["tile_x"] = (locs_df["x"] // tile_size).astype(int)
locs_df["tile_y"] = (locs_df["y"] // tile_size).astype(int)
locs_df["tile_id"] = locs_df["tile_x"].astype(str) + "_" + locs_df["tile_y"].astype(str)

# ====  tile ====
print("📊  tile...", flush=True)
expr_df = pd.DataFrame.sparse.from_spmatrix(cnts_filtered, index=barcodes, columns=filtered_gene_names)
expr_df["tile_id"] = locs_df["tile_id"].values
expr_df_grouped = expr_df.groupby("tile_id").sum().reset_index()

# ====  tile  ====
print("📍  tile ...", flush=True)
tile_coords = locs_df.groupby("tile_id")[["tile_x", "tile_y"]].first().reset_index()
tile_coords["x"] = (tile_coords["tile_x"] + 0.5) * tile_size
tile_coords["y"] = (tile_coords["tile_y"] + 0.5) * tile_size

# ====  ====
print("🔗 ...", flush=True)
merged_df = pd.merge(tile_coords, expr_df_grouped, on="tile_id")

# ====  ====
print("💾  .tsv.gz...", flush=True)
expr_out = merged_df.drop(columns=["tile_id", "tile_x", "tile_y", "x", "y"])
expr_out.insert(0, "barcode", merged_df["tile_id"])
expr_out.to_csv(f"{output_prefix}.tsv.gz", sep="\t", index=False, compression="gzip")

# ====  spot  ====
print("💾  spot  .spots.txt...", flush=True)
spots_df = merged_df[["tile_id", "x", "y"]].copy()
spots_df.columns = ["barcode", "pixel_x", "pixel_y"]
spots_df.to_csv(f"{output_prefix}.spots.txt", sep="\t", index=False)

# ====  Coords  ====
print("💾  tumor  .Coords.tsv...", flush=True)
coords_df = spots_df.copy()
coords_df.rename(columns={"barcode": "id", "pixel_x": "x", "pixel_y": "y"}, inplace=True)
coords_df["label"] = "tumor"
coords_df.to_csv(f"{output_prefix}_Coords.tsv", sep="\t", index=False)

# ====  JPG ====
print("🖼️  tiff  jpg...", flush=True)
img = Image.open(tif_path)
img.save(f"{output_prefix}.jpg")

print("✅  ST-Net ", flush=True)

