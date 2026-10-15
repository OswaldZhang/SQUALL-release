import traceback
import scanpy as sc
import pandas as pd
import os
import glob
import matplotlib.colors as mcolors
folder = "/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference_result"
hmid_list = [
    fname.split("_")[0]
    for fname in os.listdir(folder)
    if fname.endswith("_embeddings.pt")
]
hmid_list = ['HM0555']
print(hmid_list)

def cluster_expr_leiden(adata_expr, hmid):
    sc.pp.neighbors(adata_expr, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_expr, resolution=1.0, key_added='leiden_expr')

    df_expr = pd.DataFrame({
        "barcode": adata_expr.obs_names,
        "cluster": adata_expr.obs['leiden_expr'].astype(int).values
    })
    output_path = '/lustre1/zxzeng/bwqin/STORM_main/single_section/results_clusters_expr_leiden'
    os.makedirs(output_path, exist_ok=True)
    df_expr.to_csv(os.path.join(output_path, f"{hmid}_clusters.csv"), index=False)

def cluster_rgb_leiden(adata_rgb, hmid):
    sc.pp.neighbors(adata_rgb, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_rgb, resolution=1.0, key_added='leiden_rgb')

    df_rgb = pd.DataFrame({
        "barcode": adata_rgb.obs_names,
        "cluster": adata_rgb.obs['leiden_rgb'].astype(int).values
    })
    output_path = '/lustre1/zxzeng/bwqin/STORM_main/single_section/results_clusters_rgb_leiden'
    os.makedirs(output_path, exist_ok=True)
    df_rgb.to_csv(os.path.join(output_path, f"{hmid}_clusters.csv"), index=False)
def cluster_raw_expr_leiden(adata, hmid):
    adata_expr = adata.copy()
    sc.pp.normalize_total(adata_expr, target_sum=1e4)
    sc.pp.log1p(adata_expr)
    sc.pp.scale(adata_expr)
    sc.pp.pca(adata_expr, n_comps=30)
    sc.pp.neighbors(adata_expr, n_neighbors=15, use_rep='X_pca')
    sc.tl.leiden(adata_expr, resolution=1.0, key_added='leiden_raw_expr')

    df_expr = pd.DataFrame({
        "barcode": adata_expr.obs_names,
        "cluster": adata_expr.obs['leiden_raw_expr'].astype(int).values
    })
    out_path = '/lustre1/zxzeng/bwqin/STORM_main/single_section/results_clusters_expr_leiden'
    os.makedirs(out_path, exist_ok=True)
    df_expr.to_csv(os.path.join(out_path, f"{hmid}_clusters.csv"), index=False)

def cluster_rgb_leiden_from_obs(adata, hmid):
    # RGB 特征应在 adata.obs 中有 3列：'r', 'g', 'b'，你可自行调整列名
    rgb_matrix = adata.obs[['r', 'g', 'b']].to_numpy()
    rgb_adata = sc.AnnData(X=rgb_matrix)
    rgb_adata.obs_names = adata.obs_names

    sc.pp.pca(rgb_adata, n_comps=3)
    sc.pp.neighbors(rgb_adata, n_neighbors=15, use_rep='X_pca')
    sc.tl.leiden(rgb_adata, resolution=1.0, key_added='leiden_rgb')

    df_rgb = pd.DataFrame({
        "barcode": rgb_adata.obs_names,
        "cluster": rgb_adata.obs['leiden_rgb'].astype(int).values
    })
    out_path = '/lustre1/zxzeng/bwqin/STORM_main/single_section/results_clusters_rgb_leiden'
    os.makedirs(out_path, exist_ok=True)
    df_rgb.to_csv(os.path.join(out_path, f"{hmid}_clusters.csv"), index=False)

for hmid in hmid_list:
    print("Processing ",hmid,flush = True)
    try:
        import torch
        from pathlib import Path

        def load_embeddings(file_path):
            # 加载.pt文件
            embeddings = torch.load(file_path)
            
            # 提取embeddings数据
            expr_embedding = embeddings['expr_embedding']  # shape: [H, W, 1024]
            rgb_embedding = embeddings['rgb_embedding']    # shape: [H, W, 1024]
            dimensions = embeddings['dimensions']          # 原始图像尺寸 (width, height)
            
            print(f"Expression embedding shape: {expr_embedding.shape}")
            print(f"RGB embedding shape: {rgb_embedding.shape}")
            print(f"Original dimensions: {dimensions}")
            
            return expr_embedding, rgb_embedding, dimensions, embeddings
            # return embeddings

        file_path = f"/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference_result/{(hmid).upper()}_embeddings.pt"
        expr_emb, rgb_emb, dims, resized_embeddings = load_embeddings(file_path)
        # resized_embeddings = load_embeddings(file_path)

        import matplotlib.pyplot as plt
        for i in range(3):
            # 进一步可视化合成的特征图
            # 显示 RGB 特征图
            max_x, max_y  = resized_embeddings['dimensions']
            target_height = int(max_y/16)  # 计算目标图的高度
            target_width = int(max_x/16)  # 计算目标图的宽度
            
            plt.imshow(rgb_emb.cpu().numpy()[:target_height,:target_width,i])  # 使用 CPU 显示
            plt.title("org RGB Embedding")
            plt.savefig("results_embedding_GSE250521/"+hmid+"_org_RGB_Embedding.png")
            plt.close()
            # 显示表达数据特征图
            plt.imshow(expr_emb.cpu().numpy()[:target_height,:target_width,i])  # 使用 CPU 显示
            plt.title("org Expression Embedding")
            plt.savefig("results_embedding_GSE250521/"+hmid+"_org_Expression_Embedding.png")
            plt.close()

        # 打印结果检查尺寸
        # print(f"org expr embedding shape: {org_expr_embedding.shape}")
        # print(f"org RGB embedding shape: {org_rgb_embedding.shape}")

        from VisiumCopy1 import fileReader, Processer
        import yaml
        import torch
        import os
        import json
        import math
        import numpy as np
        import logging
        from datetime import datetime
        from functools import partial
        from concurrent.futures import ThreadPoolExecutor
        from torch.utils.data import Dataset, DataLoader
        from torch.cuda.amp import autocast, GradScaler
        from pathlib import Path
        from skimage import io
        from models.Storm import Storm
        from scipy.sparse import csr_matrix
        from tqdm import tqdm
        import time
        from numba import njit, prange
        import warnings
        import itertools
        import gc
        import anndata as ad
        import warnings
        import pandas as pd
        import scanpy as sc
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        import os
        import json
        import pandas as pd
        import numpy as np
        import scanpy as sc
        import anndata as ad
        from skimage import io

        def load_tissue_positions(tpl_path):
            """Load and process tissue positions from CSV file."""
            # Custom header
            custom_columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'tl_xn', 'tl_yn']
            
            print(f"Loading tissue positions from '{tpl_path}'...")
            tissue_grid = pd.read_csv(tpl_path, header=None, nrows=2)
            original_columns = tissue_grid.iloc[0].to_list()

            if original_columns == custom_columns:
                tissue_grid = pd.read_csv(tpl_path)
            elif original_columns[0] == custom_columns[0]:
                tissue_grid = pd.read_csv(tpl_path, header=None)
                tissue_grid.columns = custom_columns
                tissue_grid = tissue_grid.iloc[1:].reset_index(drop=True)
            else:
                tissue_grid = pd.read_csv(tpl_path, header=None)
                tissue_grid.columns = custom_columns

            return tissue_grid

        def build_adata(base_path: str, h5file_path: str = None, h5ad_path: str = None) -> ad.AnnData:
            """
            Build an AnnData object from various input formats (10x mtx, h5, or h5ad).
            
            Parameters:
            - base_path (str): Base path to the data directory
            - h5file_path (str, optional): Path to h5 file
            - h5ad_path (str, optional): Path to h5ad file
            
            Returns:
            - adata (AnnData): The initialized AnnData object with spatial information
            """
            try:
                # Load the count matrix based on available format
                if h5ad_path and os.path.exists(h5ad_path):
                    print(f"Loading AnnData from h5ad: '{h5ad_path}'...")
                    adata = sc.read_h5ad(h5ad_path)
                elif h5file_path and os.path.exists(h5file_path):
                    print(f"Loading AnnData from h5: '{h5file_path}'...")
                    adata = sc.read_10x_h5(h5file_path)
                else:
                    print(f"Loading count matrix from 10x mtx: '{base_path}'...")
                    adata = sc.read_10x_mtx(
                        base_path,
                        var_names='gene_symbols',
                        cache=True
                    )
                print("Data loaded successfully.")

                # Load the high-resolution tissue image
                img_path = os.path.join(base_path, 'spatial/tissue_hires_image.png')
                print(f"Loading tissue image from '{img_path}'..")
                spatial_path =  os.path.join(base_path, "spatial")
                he_path = os.path.join(spatial_path, 'tissue_hires_image.png')
                if not os.path.exists(he_path):
                    print(f"Warning: {img_path} missing tissue_hires_image.png, skipping.", flush=True)
                    img_files = [f for f in os.listdir(spatial_path) if 'hires' in f.lower() and f.lower().endswith(('.png', '.jpg', '.tif'))]
                    if not img_files:
                        print(f"Warning: No HE image with 'hires' found for {img_path}, skipping.")
                    img_path = os.path.join(spatial_path, img_files[0])
                if not os.path.exists(img_path):
                    print(f"❌ {img_path} no HE image")
                he_img = io.imread(img_path)
                print("Tissue image loaded successfully.")

                # Load the tissue positions list
                tpl_path = os.path.join(base_path, 'spatial/tissue_positions_list.csv')
                csv_files = [f for f in os.listdir(spatial_path) if f.endswith('.csv') and 'tissue' in f]
                if not csv_files:
                    print(f"Warning: No CSV file found for {spatial_path}, skipping.")
                tpl_path = os.path.join(spatial_path, csv_files[0])
                tissue_grid = load_tissue_positions(tpl_path)

                # Add barcode column if not present
                if 'barcode' not in adata.obs.columns:
                    adata.obs['barcode'] = adata.obs.index.astype(str)

                # Ensure matching types
                adata.obs['barcode'] = adata.obs['barcode'].astype(str)
                tissue_grid['barcode'] = tissue_grid['barcode'].astype(str)

                # Align barcodes
                adata = adata[adata.obs['barcode'].isin(tissue_grid['barcode']), :]
                adata.obs = adata.obs.merge(tissue_grid, on='barcode', how='left')
                adata.obs_names = adata.obs['barcode']
                #cluster_raw_expr_leiden(adata, hmid)
                # 如果 RGB 在 adata.obs 中
                #cluster_rgb_leiden_from_obs(adata, hmid)

                # Set spatial coordinates
                if {'tl_xn', 'tl_yn'}.issubset(adata.obs.columns):
                    adata.obsm['spatial'] = adata.obs[['tl_xn', 'tl_yn']].values
                else:
                    missing_cols = {'tl_xn', 'tl_yn'} - set(adata.obs.columns)
                    raise KeyError(f"Missing columns in tissue_grid CSV: {missing_cols}")

                # Load JSON scalefactors
                json_path = os.path.join(base_path, 'spatial/scalefactors_json.json')
                sub_json_path = os.path.join(base_path, 'spatial/sub_scalefactors_json.json')
                
                scalef = None
                try:
                    with open(json_path, 'r') as f:
                        scalef = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    try:
                        with open(sub_json_path, 'r') as f:
                            scalef = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        json_candidates = glob.glob(os.path.join(base_path, 'spatial', '*.json'))
                        if len(json_candidates) == 1:
                            with open(json_candidates[0], 'r') as f:
                                scalef = json.load(f)
                        else:
                            raise FileNotFoundError("❌ Failed to load any valid JSON file in spatial/, or multiple candidates found.")

                tissue_hires_scalef = scalef['tissue_hires_scalef']
                spot_diameter_fullres = scalef['spot_diameter_fullres']
                
                adata.uns['spatial'] = {
                    'combine': {
                        'images': {
                            'hires': he_img
                        },
                        'scalefactors': {
                            'tissue_hires_scalef': tissue_hires_scalef,
                            'spot_diameter_fullres': spot_diameter_fullres
                        }
                    }
                }

                adata.obs['radius'] = tissue_hires_scalef * spot_diameter_fullres / 2

                # Add downscaled and offset coordinates
                offset_path = '/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/offset.json'
                try:
                    with open(offset_path, 'r') as file:
                        offset_dict = json.load(file)
                        # Convert hmid to uppercase for matching
                        hmid = os.path.basename(base_path).upper()
                        if hmid in offset_dict:
                            x_offset, y_offset = offset_dict[hmid]
                            x_offset, y_offset = float(x_offset), float(y_offset)
                            adata.obs['tl_xn'] = pd.to_numeric(adata.obs['tl_xn'], errors='raise')
                            adata.obs['tl_yn'] = pd.to_numeric(adata.obs['tl_yn'], errors='raise')
                            adata.obs['tl_xn_down'] = (adata.obs['tl_xn'] * tissue_hires_scalef).astype(int)
                            adata.obs['tl_yn_down'] = (adata.obs['tl_yn'] * tissue_hires_scalef).astype(int)
                            adata.obs['xn_org'] = (adata.obs['tl_xn'] * tissue_hires_scalef - y_offset).astype(int)
                            adata.obs['yn_org'] = (adata.obs['tl_yn'] * tissue_hires_scalef - x_offset).astype(int)
                        else:
                            print(f"No offset found for {hmid} in offset.json")
                            print("Skipping offset coordinates calculation")
                except FileNotFoundError:
                    print(f"Offset file not found: {offset_path}")
                    print("Skipping offset coordinates calculation")
                except json.JSONDecodeError:
                    print("Error decoding offset.json")
                    print("Skipping offset coordinates calculation")

                return adata

            except Exception as e:
                print(f"An error occurred: {e}")
                raise

        def initialize_adata(hmid: str, base_dir: str = '/data200T/STORM') -> ad.AnnData:
            """
            Unified function to initialize AnnData object from available data formats.
            Automatically detects and uses the appropriate data format (h5ad, h5, or 10x mtx).
            
            Parameters:
            - hmid (str): Sample ID (e.g. 'hm2946')
            - base_dir (str): Base directory containing all data
            
            Returns:
            - adata (AnnData): The initialized AnnData object
            """
            # Define all possible paths
            hmdb_path = os.path.join("/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb", hmid)
            hmdb_raw_path = os.path.join("/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb",hmid)
            # Check for h5ad and h5 files in hmdb directory
            h5ad_path = os.path.join(hmdb_path, 'total.h5ads')
            h5_path = os.path.join(hmdb_raw_path, 'filtered_feature_bc_matrix.h5')
            raw_h5_path = os.path.join(hmdb_raw_path, 'raw_feature_bc_matrix.h5')
            expression_h5_path = os.path.join(hmdb_raw_path, 'expression.h5')
            expression_h5ad_path = os.path.join(hmdb_raw_path, 'expression.h5ad')
            # Determine which format to use
            if os.path.exists(h5ad_path):
                print(f"Using h5ad file from: {h5ad_path}")
                return build_adata(hmdb_path, h5ad_path=h5ad_path)
            elif os.path.exists(h5_path):
                print(f"Using h5 file from: {h5_path}")
                return build_adata(hmdb_path, h5file_path=h5_path)
            elif os.path.exists(raw_h5_path):
                print(f"Using h5 file from: {h5_path}")
                return build_adata(hmdb_path, h5file_path=raw_h5_path)
            elif os.path.exists(expression_h5_path):
                print(f"Using h5 file from: {expression_h5_path}")
                return build_adata(hmdb_path, h5file_path=expression_h5_path)
            elif os.path.exists(expression_h5ad_path):
                print(f"Using h5 file from: {expression_h5ad_path}")
                return build_adata(hmdb_path, h5ad_path=expression_h5ad_path)
            else:
                print(f"Using 10x mtx format from: {hmdb_raw_path}")
                return build_adata(hmdb_raw_path)
                
        base_path = f'/data200T/STORM/hmdb/{hmid}'
        # 使用默认基础目录
        # hmid = 'hm2946'
        adata = initialize_adata(hmid)
        #cluster_raw_expr_leiden(adata, hmid)
        # 如果 RGB 在 adata.obs 中
        #cluster_rgb_leiden_from_obs(adata, hmid)
        print("check adata",adata,flush = True)
        # 如果 'feature' 不存在，先创建它
        if 'feature' not in adata.uns:
            adata.uns['feature'] = {}

        max_x, max_y  = resized_embeddings['dimensions']
        target_height = int(max_y/16)  # 计算目标图的高度
        target_width = int(max_x/16)  # 计算目标图的宽度

        # 然后将数据添加到其中
        adata.uns['feature']['expr'] = expr_emb.cpu().numpy()[:target_height,:target_width,:]
        adata.uns['feature']['rgb'] = rgb_emb.cpu().numpy()[:target_height,:target_width,:]

        from scipy.spatial import KDTree
        # ================================================== yunfeng 修改
        def get_spots_feature(adata, N_points=10000, scale = 4):
            
            units_kdtree, units_center = _build_unit_kdtree(adata, scale)
            # Generate random points in the unit circle once
            pts = _random_pts_in_a_spot(N_points, seed=0)

            spot_features_adatas = []
            for feature in adata.uns['feature']:
                # Process all circles
                feature_vectors = _process_spots(
                    adata, N_points, pts, units_kdtree, units_center, feature
                )

                print(feature_vectors.shape)
                # Build new AnnData object
                # spot_features = vstack(feature_vectors)
                # spot_features_adata = anndata.AnnData(X=spot_features)
                spot_features_adata = ad.AnnData(X=feature_vectors)
                # spot_features_adata.obs = restored_spots.obs.copy()
                # spot_features_adata.obs['x'] = restored_spots.obs['x']
                # spot_features_adata.obs['y'] = restored_spots.obs['y']
                print("adata",adata,flush =True)
                print("spot_features_adata",spot_features_adata,flush = True)
                spot_features_adata.obs_names = adata.obs_names.copy()
                spot_features_adata.obs['x'] = adata.obs['tl_xn'].values
                spot_features_adata.obs['y'] = adata.obs['tl_yn'].values
                spot_features_adata.obs['x_down'] = adata.obs['tl_xn_down'].values
                spot_features_adata.obs['y_down'] = adata.obs['tl_yn_down'].values
                spot_features_adata.obs['x_org'] = adata.obs['xn_org'].values
                spot_features_adata.obs['y_org'] = adata.obs['yn_org'].values
                spot_features_adata.obs['radius'] = adata.obs['radius'].values
                # spot_features_adata.obsm['spatial'] = restored_spots.obs[['x', 'y', 'radius']].values
                spot_features_adata.obsm['spatial'] = adata.obs[['tl_yn', 'tl_xn']].values
                # spot_features_adata.obsm['spatial'] = adata.obs[['yn_org', 'xn_org']].values
                spot_features_adata.uns['spatial'] = adata.uns['spatial']
                spot_features_adatas.append(spot_features_adata)
            return spot_features_adatas

        # ==== KDTree for unit centers
        def _build_unit_kdtree(adata, scale):
            # y_coord = adata.obs['xn_org'].values.astype(int) * scale
            # x_coord = adata.obs['yn_org'].values.astype(int) * scale
            y_coord = adata.obs['xn_org'].values.astype(int) * 1.0 #scale
            x_coord = adata.obs['yn_org'].values.astype(int) * 1.0 #scale
            # y_coord = adata.obs['tl_xn'].values.astype(int) * scale
            # x_coord = adata.obs['tl_yn'].values.astype(int) * scale
            units_center = np.column_stack((x_coord, y_coord))
            # 检查是否仍有 NaN 或 inf 值
            if np.any(np.isnan(units_center)) or np.any(np.isinf(units_center)):
                raise ValueError("units_center contains NaN or inf values, which are not allowed for KDTree.")
            
            units_kdtree = KDTree(units_center)
            return units_kdtree, units_center

        # ==== Query possible overlapping feature units
        def _query_intersect_units(x_spot_center, y_spot_center, spot_radius, units_kdtree, units_center, unit_size=8.07646):
            half_size = unit_size / 2
            half_diag = half_size * np.sqrt(2)
            search_radius = spot_radius + half_diag
            # search_radius = 12.7
            indices = units_kdtree.query_ball_point([x_spot_center, y_spot_center], r=search_radius)
          
            intersect_units = []
            for idx in indices:
                xi, yi = units_center[idx]
                x_min = xi - half_size
                x_max = xi + half_size
                y_min = yi - half_size
                y_max = yi + half_size
                # x_min = xi - search_radius
                # x_max = xi + search_radius
                # y_min = yi - search_radius
                # y_max = yi + search_radius
                intersect_units.append((idx, x_min, x_max, y_min, y_max))
            return intersect_units

        # ==== Generate random points in spots
        def _random_pts_in_a_spot(N_pts, seed=None): # unit circle
            if seed is not None:
                np.random.seed(seed)
            theta = np.random.uniform(0, 2 * np.pi, N_pts)
            # ************************************************* check
            r = np.sqrt(np.random.uniform(0, 1, N_pts))
            # ************************************************* check
            
            x_pts = r * np.cos(theta)
            y_pts = r * np.sin(theta)
            pts = np.column_stack((x_pts, y_pts))
            return pts

        # ==== Count points in units
        @njit
        def _cnt_pts_in_units(x_pts, y_pts, x_mins, x_maxs, y_mins, y_maxs):
            N_pts = x_pts.shape[0]
            n_units = x_mins.shape[0]
            cnts_arr = np.zeros(n_units, dtype=np.int64)

            for i in range(N_pts):
                x_pt = x_pts[i]
                y_pt = y_pts[i]
                for j in range(n_units):
                    if x_pt >= x_mins[j] and x_pt <= x_maxs[j] and y_pt >= y_mins[j] and y_pt <= y_maxs[j]:
                        cnts_arr[j] += 1
                        break  # A point counts for only one unit
            return cnts_arr

        # ==== Process all spots
        def _process_spots(adata, N_pts, pts, units_kdtree, units_center, feature):
            adata_X = adata.X.copy()
            adata_X_shape = adata_X.shape
            feature_vectors = []
            n_spots = adata.shape[0]
            print("_process_spots adata_X",adata_X,flush = True)
            print("_process_spots n_spots",n_spots,flush = True)
            for spot_idx in range(n_spots):
                # Ensure to extract the coordinates and radius correctly
                x_spot_center, y_spot_center, spot_radius = adata.obs.iloc[spot_idx][['yn_org', 'xn_org', 'radius']]
                #print("x_spot_center, y_spot_center, spot_radius",x_spot_center, y_spot_center, spot_radius)# noise_level = 12.0  # 根据需要调整噪声大小
                # x_spot_center += np.random.uniform(-noise_level, noise_level)
                # y_spot_center += np.random.uniform(-noise_level, noise_level)
                # Query possible overlapping units
                intersect_units = _query_intersect_units(x_spot_center, y_spot_center, spot_radius, units_kdtree, units_center, unit_size = spot_radius)
                
                if not intersect_units:
                    # If no possible units, return zero vector
                    print('sry, there was no intersect_units!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # feature_vector = csr_matrix((1, adata_X_shape[1]))
                    feature_vector = np.zeros(1024)
                else:
                    # Scale and translate single spot points
                    points = pts * spot_radius + np.array([x_spot_center, y_spot_center])
                    
                    # Convert intersect_units to arrays for numba
                    intersect_units_arr = np.array(intersect_units)
                    #print("intersect_units_arr",intersect_units_arr.shape)
                    if intersect_units_arr.ndim == 2 and intersect_units_arr.shape[0] > 1:
                        intersect_units_arr = intersect_units_arr[:1]
                    idxs = intersect_units_arr[:, 0].astype(np.int64)
                    x_mins = intersect_units_arr[:, 1]
                    x_maxs = intersect_units_arr[:, 2]
                    y_mins = intersect_units_arr[:, 3]
                    y_maxs = intersect_units_arr[:, 4]
                    #print("x_mins\n",x_mins,"x_maxs\n",x_maxs,"y_mins\n",y_mins,"y_maxs\n",y_maxs)
                    # Count points in squares
                    cnts_arr = _cnt_pts_in_units(
                        points[:, 0], points[:, 1], x_mins, x_maxs, y_mins, y_maxs
                    )
                    
                    # Build counts dict
                    cnts = {}
                    for j in range(len(cnts_arr)):
                        if cnts_arr[j] > 0:
                            cnts[idxs[j]] = cnts_arr[j]
                    
                    # Calculate weights
                    weights = {idx: count / N_pts for idx, count in cnts.items()}
                    feature_vector = np.zeros(1024)
                    # For each unit, extract the relevant feature values from expr
                    for idx, weight in weights.items():
                        # if idx >= len(x_mins):
                        #         continue  # Skip if idx is out of bounds
                            
                        # Get the bounding box of the unit
                        # ymin, ymax = int(x_mins[idx]/16), int(x_maxs[idx]/16)
                        # xmin, xmax = int(y_mins[idx]/16), int(y_maxs[idx]/16)
                        if feature == 'expr' or feature == 'rgb':
                            xmin, xmax = int(x_mins/16), int(x_maxs/16)
                            ymin, ymax = int(y_mins/16), int(y_maxs/16)
                        else:
                            xmin, xmax = int(x_mins), int(x_maxs)
                            ymin, ymax = int(y_mins), int(y_maxs)
                        ymax = max(ymin + 1, ymax)
                        xmax = max(xmin + 1, xmax)
                        # print(idx, weight,ymin, ymax,xmin, xmax)
                        # Extract the corresponding slice from the expression array
                        expr_slice = adata.uns['feature'][feature][ymin:ymax, xmin:xmax, :]
                        # expr_slice = adata.uns['feature']['rgb'][ymin:ymax, xmin:xmax, :]
                        # expr_slice = adata.uns['feature']['resized_expr'][ymin:ymax, xmin:xmax, :]
                        # expr_slice = adata.uns['feature']['resized_rgb'][ymin:ymax, xmin:xmax, :]
                        # Compute the weighted sum of the expression values within the bounding box
                        weighted_expr = expr_slice.sum(axis=(0, 1)) * weight
                        feature_vector += weighted_expr
                feature_vectors.append(feature_vector)
                # print(xmin, ymin, xmax, ymax, weights)
            # print("Total number of NaN values:", np.isnan(feature_vectors).sum())  # 如果是稀疏矩阵
            # print(np.max(feature_vectors), np.min(feature_vectors))
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0)
            # # 稠密的 numpy 数组，转换为稀疏矩阵
            feature_vectors = csr_matrix(feature_vectors)
            # print(type(feature_vectors))  # 查看类型
            # print(feature_vectors.shape)  # 查看形状

            # Return the list of feature vectors as a numpy array
            # return np.array(feature_vectors)
            return feature_vectors

        tissue_hires_scalef = adata.uns['spatial']['combine']['scalefactors']['tissue_hires_scalef']
        spot_features_adatas = get_spots_feature(adata, N_points=10000, scale = tissue_hires_scalef)

        import anndata
        import numpy as np
        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        import time
        from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        import scanpy as sc

        def cluster_and_plot_spots(
                spot_features_adata, n_clusters, method='leiden', location_weight=None,
                sort=True, plot=False):
            """
            Perform clustering and plot results for spots.
            
            Parameters:
                spot_features_adata (AnnData): Input AnnData object containing features for spots.
                n_clusters (int): Number of clusters for applicable methods.
                method (str): Clustering method ('leiden', 'louvain', 'kmeans', etc.).
                location_weight (array-like, optional): Spatial weight for clustering.
                sort (bool): Whether to sort the cluster labels.
                plot (bool): Whether to generate a plot for the results.
            
            Returns:
                AnnData: Updated AnnData object with cluster labels for spots.
            """
            print(f"Clustering using `{method}`...")
            t0 = time.time()

            # Preprocess data if location_weight is given
            if location_weight is not None:
                print("Applying location weights...")
                spot_features_adata.obsm['X_weighted'] = spot_features_adata.X * location_weight
                data = spot_features_adata.obsm['X_weighted']
            else:
                data = spot_features_adata.X

            # Convert sparse to dense if needed
            if sp.issparse(data):
                print("Converting sparse matrix to dense for clustering...")
                data = data.toarray()

            # Perform clustering based on the method
            if method in ['leiden', 'louvain']:
                sc.pp.neighbors(spot_features_adata, n_neighbors=15, use_rep='X')  # Neighbor graph
                if method == 'leiden':
                    sc.tl.leiden(spot_features_adata, resolution=1.0, key_added="clusters")
                elif method == 'louvain':
                    sc.tl.louvain(spot_features_adata, resolution=1.0, key_added="clusters")
                labels = spot_features_adata.obs["clusters"].astype(int).to_numpy()
            elif method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=0)
                labels = model.fit_predict(data)
            elif method == 'mbkm':
                print("Performing MiniBatchKMeans clustering...")
                model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1024)
                labels = model.fit_predict(data)
            elif method == 'gm':
                # Reduce dimensionality for GaussianMixture
                print("Reducing dimensionality for GaussianMixture...")
                sc.tl.pca(spot_features_adata, n_comps=5)  # Reduce to 30 principal components
                data = spot_features_adata.obsm['X_pca']

                model = GaussianMixture(
                    n_components=n_clusters,
                    covariance_type='diag',  # Diagonal covariance for efficiency
                    max_iter=100,
                    n_init=1,
                    reg_covar=1e-4,  # 增加正则化项
                    random_state=0
                )
                labels = model.fit_predict(data)
            elif method == 'agglomerative':
                print("Performing agglomerative clustering...")
                # Reduce dimensionality for AgglomerativeClustering
                sc.tl.pca(spot_features_adata, n_comps=5)  # Reduce to 30 principal components
                data = spot_features_adata.obsm['X_pca']
                
                model = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage='ward'
                )
                labels = model.fit_predict(data)
            else:
                raise ValueError(f"Method `{method}` not recognized.")

            print(f"Clustering completed in {int(time.time() - t0)} seconds.")
            print(f"Number of clusters: {np.unique(labels).size}")

            # Optional: sort cluster labels
            if sort:
                labels = sort_labels(labels)[0]

            # Add cluster labels to AnnData
            spot_features_adata.obs[f'{method}_clusters'] = labels

            # Store clustering results in `uns`
            spot_features_adata.uns[f'{method}_clusters'] = {
                'labels': labels,
                'method': method,
                'n_clusters': n_clusters
            }

            # Plot results
            if plot:
                plot_clusters(spot_features_adata, labels, method)

            return spot_features_adata


        def sort_labels(labels):
            """
            Sort labels for consistent ordering.
            """
            unique_labels = np.unique(labels)
            sorted_mapping = {old: new for new, old in enumerate(unique_labels)}
            sorted_labels = np.array([sorted_mapping[label] for label in labels])
            return sorted_labels, sorted_mapping


        def plot_clusters(spot_features_adata, labels, method):
            """
            Plot clusters using UMAP or t-SNE for spot features.
            
            Parameters:
                spot_features_adata (AnnData): Input AnnData object containing spot features.
                labels (array-like): Cluster labels for each spot.
                method (str): Clustering method used.
            """
            # Ensure neighbors are computed for UMAP
            if 'neighbors' not in spot_features_adata.uns:
                print("Computing neighbors for UMAP...")
                sc.pp.neighbors(spot_features_adata, n_neighbors=15, use_rep='X_pca')
            print("Performing dimensionality reduction for visualization...")
            sc.tl.umap(spot_features_adata)
            
            # Plot clusters
            plt.figure(figsize=(8, 6))
            sc.pl.umap(spot_features_adata, color=[f'{method}_clusters'], legend_loc='on data', show=True, title=f"Clusters ({method})")

        # Test different clustering methods
        # methods = ['leiden']#, 'mbkm', 'kmeans', 'gm', 'agglomerative'] # 'louvain', ]
        methods = ['leiden', 'mbkm', 'kmeans', 'gm', 'agglomerative','louvain']
        methods = ['kmeans']
        for method in methods:
            cluster_and_plot_spots(spot_features_adatas[0], n_clusters=10, method=method) #, plot=True)
            cluster_and_plot_spots(spot_features_adatas[1], n_clusters=10, method=method) #, plot=True)


        import matplotlib.pyplot as plt

        def plot_spatial_clusters_with_radius(spot_features_adata, cluster_key='leiden_clusters',save_key = "f"):
            """
            Plot the spatial distribution of clusters with the radius of each point, overlaying the scatter plot on the image.
            
            Parameters:
                spot_features_adata (AnnData): AnnData object containing spatial data, cluster labels, and radius.
                cluster_key (str): The key in `obs` containing the cluster labels (e.g., 'leiden_clusters').
            """
            # Extract spatial coordinates and radius from the AnnData object
            spatial_coords = spot_features_adata.obs[['x_down', 'y_down']].values
            radius = spot_features_adata.obs['radius'].values
            clusters = spot_features_adata.obs[cluster_key].values
            cluster_col = f"{method}_combined_clusters"
            print("spot_features_adata",spot_features_adata,flush = True)
            if cluster_col in spot_features_adata.obs and "combine" in save_key:
                save_df = pd.DataFrame({
                    'barcode': spot_features_adata.obs_names,
                    'cluster': spot_features_adata.obs[cluster_col].values
                    })
                save_df.to_csv(f"results_clusters_hmdb/{hmid}_clusters.csv", index=False)
            background_image = spot_features_adata.uns['spatial']['combine']['images']['hires']

            # Create the figure and axes
            plt.figure(figsize=(10, 8))
            
            # Display the background image
            plt.imshow(background_image, extent=[0, background_image.shape[1], background_image.shape[0], 0], cmap='gray')
            
            # Overlay the scatter plot
            scatter = plt.scatter(
                spatial_coords[:, 1],  # X-coordinates
                spatial_coords[:, 0],  # Y-coordinates
                c=clusters,
                cmap='tab20',
                s=radius,  # Adjust size based on radius (optional scaling factor)
                alpha=0.8,  # Transparency for scatter points
                # edgecolors="w",
                # linewidth=0.5
            )
            
            # Add a colorbar and labels
            #plt.colorbar(scatter, label='Cluster')
            from matplotlib.lines import Line2D
            unique_clusters = np.sort(
                np.unique(clusters)
            )
            
            cmap = plt.get_cmap("tab20")
            norm = mcolors.Normalize(
                vmin=0,
                vmax=np.max(unique_clusters),
            )
            
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=cmap(norm(cluster)),
                    markeredgecolor="none",
                    markersize=10,
                    label=f"c{int(cluster) + 1}",
                )
                for cluster in unique_clusters
            ]
            
            plt.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=5,
                frameon=False,
                fontsize=11,
                handletextpad=0.35,
                columnspacing=0.9,
)
            plt.title(f"Spatial Distribution of Clusters with Radius ({cluster_key})")
            #plt.xlabel('X Coordinate')
            #plt.ylabel('Y Coordinate')
            # plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            
            # Show the plot
            plt.savefig(save_key+"_cluster.pdf",dpi = 500,transparent = True)


        """Clustering functionality."""

        import scanpy as sc
        import anndata as ad
        import numpy as np
        from scipy import sparse
        from sklearn.mixture import GaussianMixture
        from typing import Tuple
        import config

        def cluster_combined_features(spot_features_adata_expr: ad.AnnData, 
                                    spot_features_adata_rgb: ad.AnnData,
                                    n_clusters: int = config.N_CLUSTERS,
                                    method: str = 'kmeans') -> Tuple[ad.AnnData, ad.AnnData]:
            """Combine expression and RGB features and perform clustering.
            
            Args:
                spot_features_adata_expr: Expression features AnnData
                spot_features_adata_rgb: RGB features AnnData
                n_clusters: Number of clusters
                method: Clustering method ('gm' for Gaussian Mixture)
                
            Returns:
                Tuple of updated expression and RGB AnnData objects
            """
            # Create a temporary AnnData for combined features
            combined_features = sparse.hstack([
                spot_features_adata_expr.X,
                spot_features_adata_rgb.X
            ])
            combined_adata = ad.AnnData(X=combined_features)
            
            if method == 'gm':
                # Use scanpy's PCA with specified components
                sc.pp.pca(combined_adata, n_comps=config.N_PCA_COMPONENTS)
                # Add neighbors computation
                sc.pp.neighbors(combined_adata, n_neighbors=config.N_NEIGHBORS, use_rep='X_pca')
                
                model = GaussianMixture(**config.CLUSTERING_PARAMS['gm'])
                labels = model.fit_predict(combined_adata.obsm['X_pca'])
            elif method == 'kmeans':
                sc.pp.pca(combined_adata, n_comps=config.N_PCA_COMPONENTS)
                # Add neighbors computation
                sc.pp.neighbors(combined_adata, n_neighbors=config.N_NEIGHBORS, use_rep='X_pca')
                model = KMeans(n_clusters=n_clusters, random_state=0)
                labels = model.fit_predict(combined_adata.obsm['X_pca'])
            else:
                raise ValueError(f"Method {method} not recognized.")

            # Ensure labels are consecutive integers starting from 0
            unique_labels = np.unique(labels)
            sorted_mapping = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([sorted_mapping[label] for label in labels])
            
            # Add labels to both AnnData objects
            spot_features_adata_expr.obs[f'{method}_combined_clusters'] = labels
            spot_features_adata_rgb.obs[f'{method}_combined_clusters'] = labels
            
            return spot_features_adata_expr, spot_features_adata_rgb
        spot_features_adatas_combine,spot_features_adatas_combine_2 = cluster_combined_features(spot_features_adatas[0],spot_features_adatas[1])
        for method in methods:
            # Example usage
            print(spot_features_adatas_combine)
            #plot_spatial_clusters_with_radius(spot_features_adatas[0], cluster_key=f'{method}_clusters',save_key = "results_hmdb/"+hmid+"_expr")
            #plot_spatial_clusters_with_radius(spot_features_adatas[1], cluster_key=f'{method}_clusters',save_key = "results_hmdb/"+hmid+"_rgb")
            plot_spatial_clusters_with_radius(spot_features_adatas_combine, cluster_key=f'{method}_combined_clusters',save_key = "results_hmdb/"+hmid+"_combine")
    except:
        print("Failed !!! ",hmid,flush = True)
        traceback.print_exc()
        continue