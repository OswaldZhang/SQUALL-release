import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gzip

# === 参数 ===
tile_size = 224
input_prefix = "/lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript/OV/OV1/OV1_1"
output_prefix = "/lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript/OV/OV1/OV1_1"
output_dir = "/lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/hist2tscript/OV/OV1"

os.makedirs(output_dir, exist_ok=True)

# === 读取数据 ===
coords = pd.read_csv(f"{input_prefix}_Coords.tsv", sep="\t")
spots = pd.read_csv(f"{input_prefix}.spots.txt", sep="\t")
with gzip.open(f"{input_prefix}.tsv.gz", "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)

# === 合并坐标与表达 ===
coords = coords.rename(columns={"id": "barcode"})
spots = spots.rename(columns={"barcode": "barcode"})
merged = coords.merge(spots, on="barcode")
merged = merged[merged["barcode"].isin(expr.index)]  # 只保留在表达矩阵里的行

# === 为每个点加 tile 网格坐标 ===
merged["tile_x"] = (merged["pixel_x"] // tile_size).astype(int)
merged["tile_y"] = (merged["pixel_y"] // tile_size).astype(int)
merged["tile_id"] = merged["tile_x"].astype(str) + "_" + merged["tile_y"].astype(str)

# === 聚合表达矩阵 ===
tile_expr = []
tile_info = []
tile_barcodes = []

for tile_id, group in tqdm(merged.groupby("tile_id")):
    barcodes = group["barcode"].values
    tile_count = expr.loc[barcodes].sum(axis=0)
    center_x = group["pixel_x"].mean()
    center_y = group["pixel_y"].mean()
    tumor_label = group["tumor"].mode()[0] if "tumor" in group.columns else "unknown"
    label = f"L{len(tile_info)+1}"

    tile_expr.append(tile_count)
    tile_info.append({
        "id": f"tile_{tile_id}",
        "x": center_x,
        "y": center_y,
        "lab": label,
        "tumor": tumor_label
    })
    tile_barcodes.append(f"tile_{tile_id}")

# === 保存新的 Coords.tsv ===
coords_df = pd.DataFrame(tile_info)
coords_df.to_csv(os.path.join(output_dir, f"{output_prefix}_Coords.tsv"), sep="\t", index=False)

# === 保存新的 spots.txt ===
spots_df = coords_df.rename(columns={"id": "barcode", "x": "pixel_x", "y": "pixel_y"})
spots_df["x"] = spots_df["pixel_x"]
spots_df["y"] = spots_df["pixel_y"]
spots_df = spots_df[["barcode", "x", "y", "pixel_x", "pixel_y"]]
spots_df.to_csv(os.path.join(output_dir, f"{output_prefix}.spots.txt"), sep="\t", index=False)

# === 保存新的表达矩阵 .tsv.gz ===
tile_expr_df = pd.DataFrame(tile_expr, index=tile_barcodes, columns=expr.columns)
tile_expr_df.index.name = "barcode"
with gzip.open(os.path.join(output_dir, f"{output_prefix}.tsv.gz"), "wt") as f:
    tile_expr_df.to_csv(f, sep="\t")

print(f"✅ 生成完成，共生成 {len(tile_expr_df)} 个 tile（spot）")

