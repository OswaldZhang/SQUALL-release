import scanpy as sc
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
#he_img=io.imread("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/HE_rescaled_0.5mpp.tiff")
he_img = io.imread("/lustre1/zxzeng/bwqin/STORM/Xenium/Lung_Xenium_public/Xenium_Prime_Human_Lung_Cancer_FFPE_he_image_downsampled.tif")
io.imsave('/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/he-raw.jpg',he_img)
print("image OK!",flush = True)
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import io
from scipy import sparse
import os

# 读取数据
data = sc.read_h5ad("lung_all/processed_spotwise_sum_adata_lowres.h5ad")

# 获取数据维度
n_obs, n_vars = data.shape

# 生成虚拟的 barcode 名称
barcodes = [f"cell_{i}" for i in range(n_obs)]

# 获取 gene 名（变量名）
gene_names = data.var_names.tolist()

# 将稀疏矩阵转为 dense，只取前 20 行和前 20 列
cnts_matrix = data.X.tocoo() if sparse.issparse(data.X) else sparse.coo_matrix(data.X)
io.mmwrite(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/cnts.mtx", cnts_matrix)

# 2. 保存 gene 名称（列名）
pd.Series(data.var_names).to_csv(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/genes.tsv", sep="\t", index=False, header=False)

# 3. 保存 barcode 名称（行名）
pd.Series(data.obs_names).to_csv(f"/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/barcodes.tsv", sep="\t", index=False, header=False)

print("Counts OK!",flush = True)
'''
cnts_array = data.X[:20, :20].toarray() if hasattr(data.X, 'toarray') else data.X[:20, :20]
cnts_df = pd.DataFrame(cnts_array, index=barcodes[:20], columns=gene_names[:20])
cnts_df.to_csv("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/cnts.tsv", sep="\t")
'''
# 导出 locs-raw.tsv
spatial = data.obsm["spatial"]
locs_df = pd.DataFrame(spatial, columns=["x", "y"])#/57382*2000
locs_df.insert(0, "barcode", barcodes)
locs_df.to_csv("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/locs-raw.tsv", sep="\t", index=False)
'''
spatial=pd.read_csv("/lustre1/zxzeng/bwqin/STORM/Xenium/Breast_Xenium_public/pesudo_visium/spatial/tissue_positions_list.csv",sep=",",na_filter=False,index_col=0,names=["barcode","in_tissue","spot_x","spot_y","x","y"])
for column in spatial.columns:
    spatial[column] = pd.to_numeric(spatial[column], errors='coerce')
while(spatial.iloc[0,0]!=0 and spatial.iloc[0,0]!=1):
    spatial=spatial[1:]
spatial=spatial[spatial["in_tissue"]==1]
df=spatial.loc[:,["x","y"]]
offset_x_min=df['x'].min()
offset_y_min=df['y'].min()
offset_x_max=df['x'].max()
offset_y_max=df['y'].max()
df["x"]-=offset_x_min
df["y"]-=offset_y_min
df.to_csv("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/locs-raw.tsv",sep='\t')

df.index.unique()
'''
print("locs OK!",flush = True)
# 导出 pixel-size 和 radius
with open("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/pixel-size-raw.txt", "w") as f:
    f.write(str(5.922959))

with open("/lustre1/zxzeng/bwqin/STORM_main/clustering/istar/data/lung_all/radius-raw.txt", "w") as f:
    f.write(str(9.2859))

