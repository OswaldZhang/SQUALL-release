import scanpy as sc
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
#he_img=io.imread("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/20240709-LIVER_cut.tif")
#io.imsave('/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/he-raw.jpg',he_img)
he_img=io.imread("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all/HE_rescaled_0.5mpp.tiff")
io.imsave('/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/he-raw.jpg',he_img)
print("image OK!",flush = True)
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import io
from scipy import sparse
import os

# 读取数据
#data = sc.read_h5ad("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/adata.h5ad")
data = sc.read_h5ad("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all/adata.h5ad")# 获取数据维度
n_obs, n_vars = data.shape

# 生成虚拟的 barcode 名称
barcodes = [f"cell_{i}" for i in range(n_obs)]

# 获取 gene 名（变量名）
gene_names = data.var_names.tolist()

# 将稀疏矩阵转为 dense，只取前 20 行和前 20 列
cnts_matrix = data.X.tocoo() if sparse.issparse(data.X) else sparse.coo_matrix(data.X)
io.mmwrite(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/cnts.mtx", cnts_matrix)

# 2. 保存 gene 名称（列名）
pd.Series(data.var_names).to_csv(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/genes.tsv", sep="\t", index=False, header=False)

# 3. 保存 barcode 名称（行名）
pd.Series(data.obs_names).to_csv(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/barcodes.tsv", sep="\t", index=False, header=False)

print("Counts OK!",flush = True)
'''
cnts_array = data.X[:20, :20].toarray() if hasattr(data.X, 'toarray') else data.X[:20, :20]
cnts_df = pd.DataFrame(cnts_array, index=barcodes[:20], columns=gene_names[:20])
cnts_df.to_csv("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/cnts.tsv", sep="\t")
'''
# 导出 locs-raw.tsv
spatial = data.obsm["spatial"]
locs_df = pd.DataFrame(spatial, columns=["x", "y"])/0.5
locs_df.insert(0, "barcode", barcodes)
locs_df.to_csv("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/locs-raw.tsv", sep="\t", index=False)
print("locs OK!",flush = True)
# 导出 pixel-size 和 radius
with open("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/pixel-size-raw.txt", "w") as f:
    f.write(str(0.5))

with open("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/HCC_VisiumHD_all_new/radius-raw.txt", "w") as f:
    f.write(str(4))
