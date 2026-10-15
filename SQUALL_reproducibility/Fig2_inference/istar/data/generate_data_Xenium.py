import scanpy as sc
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
Image.MAX_IMAGE_PIXELS = None
he_img=io.imread("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/HE_rescaled_0.5mpp.tiff")
#he_img = io.imread("/lustre1/zxzeng/bwqin/SQUALL/Xenium/CC_Xenium_public/Xenium_Prime_Cervical_Cancer_FFPE_he_image.ome.tif")
io.imsave('/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/he-raw.jpg',he_img)
print("image OK!",flush = True)
import scanpy as sc
import pandas as pd
import numpy as np
from scipy import io
from scipy import sparse
import os

# 
data = sc.read_h5ad("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/adata.h5ad")

# 
n_obs, n_vars = data.shape

#  barcode 
barcodes = [f"cell_{i}" for i in range(n_obs)]

#  gene 
gene_names = data.var_names.tolist()

#  dense 20  20 
cnts_matrix = data.X.tocoo() if sparse.issparse(data.X) else sparse.coo_matrix(data.X)
io.mmwrite(f"/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/cnts.mtx", cnts_matrix)

# 2.  gene 
pd.Series(data.var_names).to_csv(f"/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/genes.tsv", sep="\t", index=False, header=False)

# 3.  barcode 
pd.Series(data.obs_names).to_csv(f"/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/barcodes.tsv", sep="\t", index=False, header=False)

print("Counts OK!",flush = True)
'''
cnts_array = data.X[:20, :20].toarray() if hasattr(data.X, 'toarray') else data.X[:20, :20]
cnts_df = pd.DataFrame(cnts_array, index=barcodes[:20], columns=gene_names[:20])
cnts_df.to_csv("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/cnts.tsv", sep="\t")
'''
#  locs-raw.tsv
spatial = data.obsm["spatial"]
locs_df = pd.DataFrame(spatial, columns=["x", "y"])/0.5
locs_df.insert(0, "barcode", barcodes)
locs_df.to_csv("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/locs-raw.tsv", sep="\t", index=False)
print("locs OK!",flush = True)
#  pixel-size  radius
with open("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/pixel-size-raw.txt", "w") as f:
    f.write(str(0.5))

with open("/lustre1/zxzeng/bwqin/SQUALL_main/clustering/istar/data/HCC_Xenium_all_new/radius-raw.txt", "w") as f:
    f.write(str(4.0))

