import os
import numpy as np
import pandas as pd
from scipy import io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ==== 路径设置 ====
prefix = "/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all/"
output_prefix = "/lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript_HCC/HCC/HCCVisiumHD/HCCVisiumHD_1"
tif_path = "/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all/HE_rescaled_0.5mpp.tiff"
gene_list_path = "/lustre1/zxzeng/bwqin/STORM_main/downstream_labels/expr_get_embedding_Xenium_all.csv"

print("📥 加载目标基因列表...", flush=True)
selected_genes = pd.read_csv(gene_list_path)["HGNC_symbol"].dropna().unique().tolist()

print("📥 加载表达矩阵和基因信息...", flush=True)
cnts = io.mmread(f"{prefix}cnts.mtx").tocsr()
genes = pd.read_csv(f"{prefix}genes.tsv", header=None, sep="\t")[0].tolist()
barcodes = pd.read_csv(f"{prefix}barcodes.tsv", header=None, sep="\t")[0].tolist()

print("📥 加载空间坐标（微米单位）...", flush=True)
locs_df = pd.read_csv(f"{prefix}locs-raw.tsv", sep="\t")
locs_df["barcode"] = barcodes

print("✅ 初始数据加载完成", flush=True)

# ==== 筛选基因 ====
print("🧬 筛选目标基因...", flush=True)
valid_genes = [g for g in selected_genes if g in genes]
gene_indices = [genes.index(g) for g in valid_genes]
cnts_filtered = cnts[:, gene_indices]
filtered_gene_names = [genes[i] for i in gene_indices]

# ==== 分配 tile 坐标 ====
print("📐 分配 tile 坐标...", flush=True)
tile_size = 224
locs_df["tile_x"] = (locs_df["x"] // tile_size).astype(int)
locs_df["tile_y"] = (locs_df["y"] // tile_size).astype(int)
locs_df["tile_id"] = locs_df["tile_x"].astype(str) + "_" + locs_df["tile_y"].astype(str)

# ==== 表达合并到 tile ====
print("📊 合并表达值到 tile...", flush=True)
expr_df = pd.DataFrame.sparse.from_spmatrix(cnts_filtered, index=barcodes, columns=filtered_gene_names)
expr_df["tile_id"] = locs_df["tile_id"].values
expr_df_grouped = expr_df.groupby("tile_id").sum().reset_index()

# ==== 获取 tile 中心坐标 ====
print("📍 计算 tile 中心坐标...", flush=True)
tile_coords = locs_df.groupby("tile_id")[["tile_x", "tile_y"]].first().reset_index()
tile_coords["x"] = (tile_coords["tile_x"] + 0.5) * tile_size
tile_coords["y"] = (tile_coords["tile_y"] + 0.5) * tile_size

# ==== 合并表达和坐标 ====
print("🔗 合并表达和坐标数据...", flush=True)
merged_df = pd.merge(tile_coords, expr_df_grouped, on="tile_id")

# ==== 保存表达矩阵 ====
print("💾 保存表达矩阵到 .tsv.gz...", flush=True)
expr_out = merged_df.drop(columns=["tile_id", "tile_x", "tile_y", "x", "y"])
expr_out.insert(0, "barcode", merged_df["tile_id"])
expr_out.to_csv(f"{output_prefix}.tsv.gz", sep="\t", index=False, compression="gzip")

# ==== 保存 spot 文件 ====
print("💾 保存 spot 信息到 .spots.txt...", flush=True)
spots_df = merged_df[["tile_id", "x", "y"]].copy()
spots_df.columns = ["barcode", "pixel_x", "pixel_y"]
spots_df.to_csv(f"{output_prefix}.spots.txt", sep="\t", index=False)

# ==== 保存 Coords 文件 ====
print("💾 保存 tumor 标注到 .Coords.tsv...", flush=True)
coords_df = spots_df.copy()
coords_df.rename(columns={"barcode": "id", "pixel_x": "x", "pixel_y": "y"}, inplace=True)
coords_df["label"] = "tumor"
coords_df.to_csv(f"{output_prefix}_Coords.tsv", sep="\t", index=False)

# ==== 保存原始图像为 JPG ====
print("🖼️ 保存原始 tiff 为 jpg（不缩放）...", flush=True)
img = Image.open(tif_path)
img.save(f"{output_prefix}.jpg")

print("✅ 所有 ST-Net 所需文件已生成！", flush=True)

