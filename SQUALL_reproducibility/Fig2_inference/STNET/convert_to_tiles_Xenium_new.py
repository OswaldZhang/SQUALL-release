import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gzip

# ===  ===
tile_pixel_size = 224
resolution = np.array([0.22073106, 0.22072619])  # µm/pixel
tile_size_um_x = tile_pixel_size * resolution[0]
tile_size_um_y = tile_pixel_size * resolution[1]

input_prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/OV_Xenium/OVXenium_1"
output_prefix = input_prefix
output_dir = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/hist2tscript_new/OV/OVXenium/OVXenium_1"

os.makedirs(output_dir, exist_ok=True)

# ===  ===
coords = pd.read_csv(f"{input_prefix}_Coords.tsv", sep="\t")
spots = pd.read_csv(f"{input_prefix}.spots.txt", sep="\t")
with gzip.open(f"{input_prefix}.tsv.gz", "rt") as f:
    expr = pd.read_csv(f, sep="\t", index_col=0)

# ===  ===
coords = coords.rename(columns={"id": "barcode"})
spots = spots.rename(columns={"barcode": "barcode"})
merged = coords.merge(spots, on="barcode")
print(merged)
merged = merged[merged["barcode"].isin(expr.index)]

# ===  µm  tile  ===
merged["tile_x"] = (merged["pixel_x"] // tile_size_um_x).astype(int)
merged["tile_y"] = (merged["pixel_y"] // tile_size_um_y).astype(int)
merged["tile_id"] = merged["tile_x"].astype(str) + "_" + merged["tile_y"].astype(str)

# ===  ===
tile_expr = []
tile_info = []
tile_barcodes = []

for tile_id, group in tqdm(merged.groupby("tile_id")):
    barcodes = group["barcode"].values
    tile_count = expr.loc[barcodes].sum(axis=0)
    center_x = group["pixel_x"].mean()
    center_y = group["pixel_y"].mean()
    tumor_label = group["tumor"].mode()[0] if "tumor" in group.columns else "tumor"
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

# ===  Coords.tsv ===
coords_df = pd.DataFrame(tile_info)
coords_df.to_csv(os.path.join(output_dir, f"{output_prefix}_Coords.tsv"), sep="\t", index=False)

# ===  spots.txt ===
spots_df = coords_df.rename(columns={"id": "barcode", "x": "pixel_x", "y": "pixel_y"})
spots_df["x"] = spots_df["pixel_x"]
spots_df["y"] = spots_df["pixel_y"]
spots_df = spots_df[["barcode", "x", "y", "pixel_x", "pixel_y"]]
spots_df.to_csv(os.path.join(output_dir, f"{output_prefix}.spots.txt"), sep="\t", index=False)

# ===  ===
tile_expr_df = pd.DataFrame(tile_expr, index=tile_barcodes, columns=expr.columns)
tile_expr_df.index.name = "barcode"
with gzip.open(os.path.join(output_dir, f"{output_prefix}.tsv.gz"), "wt") as f:
    tile_expr_df.to_csv(f, sep="\t")

print(f"✅  {len(tile_expr_df)}  tile tile  {tile_size_um_x:.2f} × {tile_size_um_y:.2f} µm")

