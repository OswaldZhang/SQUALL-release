import os
import numpy as np
import pandas as pd
from scipy import io
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# ==== 路径设置 ====
prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_VisiumHD/"  # 路径末尾务必带 /
output_prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/OV_VisiumHD/OV1"    # 输出文件前缀
tif_path = f"{prefix}20240709-LC_cut.tif"

prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/OV_Xenium/"  # 路径末尾务必带 /
output_prefix = "/lustre1/zxzeng/bwqin/SQUALL_main/clustering/ST-NET/data/OV_Xenium/OVXenium_1"    # 输出文件前缀
tif_path = f"{prefix}HE_registered.tif"


# ==== 读取表达矩阵 ====
cnts = io.mmread(f"{prefix}cnts.mtx").tocsr()
genes = pd.read_csv(f"{prefix}genes.tsv", header=None, sep="\t")[0].tolist()
barcodes = pd.read_csv(f"{prefix}barcodes.tsv", header=None, sep="\t")[0].tolist()

# ==== 读取感兴趣基因 ====
selected_genes = [
        "COL5A2","THY1","AEBP1","COL5A1","PRRX1","CCDC80","THBS2","NBL1","CFH","CTSK","SULF1","HTRA1","CTHRC1","FSTL1","PALLD","MRC2","MZB1","DERL3","FKBP11","CD79A","TENT5C","POU2AF1","PIM2","ST6GAL1","CD38","FCRL5","SLAMF7","CEP128","XBP1","GAB1","TNFRSF17","P2RX1","GZMA","CD2","CD3G","CD3E","CD96","GZMB","GZMH","CD8A","GZMK","ETS1","KLRK1","CXCR4","CST7","THEMIS","IFNG","CRTAM","BCL11B","RUNX3","FYN","IKZF3","ZNF683","LCK","PTPN22","ITK","EPCAM","CLDN7","MAL2","LAPTM4B","DMKN","ABHD11","CSNK2A1","HMGB3","ID4","JUP","IGF2BP2","TNNI3","MEIS1","HMGA1","MUC16","EHF","PARD3","MUC1","CXADR","IGFBP2","CD68","MSR1","FCGR3A","FCGR2A","CD14","CTSH","OLR1","CD163","LILRB4","C5AR1","SLC11A1","CTSL","SNX10","CD86","CYBB","ITGB2","TNFSF13B","NINJ1","FPR1","BCL2A1","CD83","C3AR1","SPI1","LAIR1","CSF1R","FCGR1A","HBEGF","XCL2","TRDC","KLRD1","KLRC1","KLRK1","GZMB","FGFBP2","PRF1","SH2D1B","NCR1","KLRF1","GZMA","GZMH","CD7","KIR2DL4","EOMES","ETS1","CST7","S1PR5","CD96","CXCR4","TMIGD2","RUNX3","CD2","CD247","FYN","CD2","ETS1","BATF","CD3E","CD3G","CTLA4","TIGIT","TNFRSF18","ICOS","CD247","LCK","KLRB1","FOXP3","IL2RG","CXCR4","CD7","BCL11B","IL12RB2","LAIR2","CD96","SLAMF1","CCR6","FCGR3B","AQP9","CSF3R","KCNJ15","CXCR2","CXCR1","HCAR3","CEACAM3","TNFRSF10C","TREM1","BCL2A1","FPR1","EPHB1","FCAR","FPR2","SELL","OSM","CD300E","CFP","LILRB2","LILRA5","FPR1","BCL2A1","CYBB","NCF2","LILRA1","FGR","SLC11A1","OLR1","AQP9","TREM1","LILRB1","RETN","ADGRE2","FPR2","SPI1","C5AR1","HBEGF","MARCO","LILRA2","EREG","CLEC12A","SPIB","LILRA4","DNASE1L3","CLEC4C","CD1C","P2RY14","FLT3","CCL19","SMPD3","CCL22","CALCRL","IRF8","CCR7","CLEC9A","SIGLEC6","CD80","CSF2RA","RASSF2","CSF2RB","TLR10","MS4A2","HDC","SLC18A2","IL1RL1","KIT","GATA2","SIGLEC6","CTSG","ADCYAP1","TAL1","GATA1","HPGDS","MLPH","DRD2","PZP","HPGD","IL9R","TIE1","ADGRL4","PCDH17","CDH5","CLEC14A","MMRN2","ROBO4","BCL6B","APLNR","SELE","CD34","PLVAP","F2RL3","FLT4","NOSTRIN","KDR","CALCRL","FOLH1","CLDN5","PODXL","ERG","S1PR1","ESM1"
        ]

# ==== 过滤基因 ====
valid_genes = [g for g in selected_genes if g in genes]
gene_indices = [genes.index(g) for g in valid_genes]
cnts_filtered = cnts[:, gene_indices]
filtered_gene_names = [genes[i] for i in gene_indices]

# ==== 保存表达矩阵（tsv.gz） ====
cnts_df = pd.DataFrame.sparse.from_spmatrix(cnts_filtered, index=barcodes, columns=filtered_gene_names)
cnts_df.reset_index(inplace=True)
cnts_df.rename(columns={"index": "barcode"}, inplace=True)
print(cnts_df)
cnts_df.to_csv(f"{output_prefix}.tsv.gz", sep="\t", index=False, compression="gzip")

# ==== 保存 spot 信息 ====
locs_df = pd.read_csv(f"{prefix}locs-raw.tsv", sep="\t")
locs_df["barcode"] = cnts_df["barcode"]
locs_df["pixel_x"] = locs_df["x"]
locs_df["pixel_y"] = locs_df["y"]
locs_df.to_csv(f"{output_prefix}.spots.txt", sep="\t", index=False)

# ==== 保存 tumor 标注（全部标 tumor） ====
coords_df = locs_df.copy()
coords_df["id"] = cnts_df["barcode"]#coords_df["barcode"]
coords_df["label"] = "tumor"
coords_df = coords_df[["id", "x", "y", "label"]]
coords_df.to_csv(f"{output_prefix}_Coords.tsv", sep="\t", index=False)

# ==== 转换大图（原始 TIF）为 JPG 并压缩尺寸 ====
img = Image.open(tif_path)
downsampled_img = img.resize((img.width // 4, img.height // 4))  # 下采样比例可调
downsampled_img.save(f"{output_prefix}.jpg")

print("✅ 所有 ST-Net 所需文件已生成")

