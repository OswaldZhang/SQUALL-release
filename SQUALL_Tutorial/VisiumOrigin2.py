import os
import json
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
from skimage import io
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix,vstack
from skimage.segmentation import expand_labels
from scipy.interpolate import NearestNDInterpolator
from utils.Visium import *
class fileReader:
    '''
    read files for visium spatial
    standard data
    '''
    def read_all(self,folder_path,method,key,gene_token,medium_token_path=None):
        self.key=key
        self.method=method
        raw_img_path,raw_tpl_path,json_path,h5_path=get_paths(folder_path)
        # print(raw_img_path,raw_tpl_path,json_path,h5_path)
        self.read_img(raw_img_path)
        self.read_tissue_position(raw_tpl_path,json_path)
        self.read_h5(h5_path,method,key,gene_token,medium_token_path)

    def read_img(self,raw_img_path):
        '''
        Reads files from a given folder path and stores relevant data in class attributes.
        Args:
            folder_path (str): The path to the folder containing the necessary files.
        '''
        self.raw_he = io.imread(raw_img_path)
        if self.raw_he.ndim == 3 and self.raw_he.shape[2] == 4:
            self.raw_he = self.raw_he[:, :, :3]
    def read_tissue_position(self,raw_tpl_path,json_path):
        tpl=pd.read_csv(raw_tpl_path, header=None)
        if (len(tpl.columns)==7):
            self.tissue_position_list = pd.read_csv(raw_tpl_path, header=None, names=['number','barcode','in_tissue','array_row', 'array_col',
                                                                  'pxl_row_in_fullres',  'pxl_col_in_fullres'])
        else:
            self.tissue_position_list = pd.read_csv(raw_tpl_path, header=None, names=['barcode','in_tissue','array_row', 'array_col',
                                                                  'pxl_row_in_fullres',  'pxl_col_in_fullres'])
        self.tissue_position_list.set_index('barcode', inplace=True)
        while(str(self.tissue_position_list.iloc[0,0])!="0" and str(self.tissue_position_list.iloc[0,0])!="1"):
            self.tissue_position_list=self.tissue_position_list[1:]
        self.tissue_position_list["in_tissue"]=self.tissue_position_list["in_tissue"].astype('int')
        self.tissue_position_list["pxl_row_in_fullres"]=self.tissue_position_list["pxl_row_in_fullres"].astype('float')
        self.tissue_position_list["pxl_col_in_fullres"]=self.tissue_position_list["pxl_col_in_fullres"].astype('float')
        self.tissue_position_list["array_row"]=self.tissue_position_list["array_row"].astype('float')
        self.tissue_position_list["array_col"]=self.tissue_position_list["array_col"].astype('float')
        with open(json_path, 'r') as f:
            self.scaleJson = json.load(f)
        self.dia = self.scaleJson['spot_diameter_fullres']*self.scaleJson['tissue_hires_scalef']

    def read_h5(self,h5_path,method,key,gene_token,medium_token_path=None):
        self.method=method
        self.medium_token_path=medium_token_path
        self.key=key
        if(h5_path.count("processed_spotwise_sum_adata.h5ad")):
            adata = sc.read_h5ad(h5_path)
        if(h5_path.count(".h5")):
            try:
                adata = sc.read_10x_h5(h5_path)
            except:
                adata = sc.read_h5ad(h5_path)
        elif(h5_path.count(".mtx")):
            print("reading mtx")
            h5_path=os.path.dirname(h5_path)
            adata=sc.read_10x_mtx(path=h5_path,make_unique=False,var_names="gene_symbols")
        print("self.adata",adata)
        print("self.adata.var['gene_ids']",adata.var['gene_ids'])
        self.adata=set_gene_token(adata,self.key,gene_token)
        
        if(self.method=="binary"):
            self.adata.X=binary_matrix(self.adata.X)
        elif(self.method=="raw"):
            pass
        # elif(type(method)==int):
        #     self.adata.X=np.array([kbins(row,method) for row in self.adata.X])
        elif(self.method=="norm"):
            sc.pp.normalize_total(self.adata, inplace=True)
            sc.pp.log1p(self.adata)
        elif(self.method=="medium"):
            with open (self.medium_token_path,"r") as f:
                gene_dict=json.load(f)
            def minMax(col,num):
                try:
                    return col/gene_dict[str(num)]["q50"]
                except:
                    return col
            self.adata.X=np.array([minMax(col,num+1) for col,num in zip(self.adata.X.T,range(1,15758))]).T

class Processer:
    def __init__(self,Reader,patch_size):
        self.patch_size=patch_size
        self.raw_he=Reader.raw_he
        self.tissue_position_list=Reader.tissue_position_list
        self.scaleJson = Reader.scaleJson
        self.dia = Reader.dia
        self.adata=Reader.adata
    def process_tpl(self):
        '''
        Process the tissue position data in a CSV-like DataFrame.
        Args:
            None (This is an instance method and does not require any arguments.)
        Returns:
            None (The function modifies the internal DataFrame 'tissue_position_list' directly.)
        Description:
            1. Sets the index of 'tissue_position_list' to 'barcode'.
            2. Removes rows from the beginning until the first row has a barcode starting with '0' or '1'.
            3. Converts all columns in 'tissue_position_list' to float type.
            4. Scales the values in 'pxl_row_in_hires' and 'pxl_col_in_hires' using the scale factor from 'json['tissue_hires_scalef']'.
            5. Renames certain columns for clarity.
            6. Adds two new columns, 'top_left_before' and 'top_left_after', both initialized as None.
        '''
        print(self.tissue_position_list.columns.tolist(), flush=True)
        print(self.scaleJson['tissue_hires_scalef'], flush=True)
        print("scale:", self.scaleJson['tissue_hires_scalef'], type(self.scaleJson['tissue_hires_scalef']), flush=True)
        print("Columns in tissue_position_list:", self.tissue_position_list.columns.tolist(), flush=True)
        scale = self.scaleJson['tissue_hires_scalef'].iloc[0] if isinstance(self.scaleJson['tissue_hires_scalef'], pd.Series) else self.scaleJson['tissue_hires_scalef']#.iloc[0]
        print(f"Scale value: {scale}", flush=True)
        self.tissue_position_list[['pxl_row_in_hires','pxl_col_in_hires']] = self.tissue_position_list[['pxl_row_in_fullres','pxl_col_in_fullres']]* scale
        self.tissue_position_list.rename(columns={
            'pxl_row_in_fullres': 'pxl_y_fullres',
            'pxl_col_in_fullres': 'pxl_x_fullres',
            'pxl_row_in_hires': 'pxl_y_hires',
            'pxl_col_in_hires': 'pxl_x_hires',
            'array_row': 'spot_y',
            'array_col': 'spot_x'
        }, inplace=True)
        print("origin !!! max(pxl_x_hires),max(pxl_y_hires)",max(self.tissue_position_list["pxl_x_hires"]),max(self.tissue_position_list["pxl_y_hires"]),flush=True)
        self.tissue_position_list['top_left_before'] = None
        self.tissue_position_list['top_left_after'] = None

    def round_spot(self):
        '''
        Round the coordinates of spots to grid lines.
        This method calculates the radius of a circle based on tissue scale factor and spot diameter,
        rounds the top-left corner coordinates of each tissue position to the nearest grid line (with 4 as the grid size),
        crops the H&E image using the bounding box of the rounded tissue positions with an additional spot area,
        and finally transforms the tissue position list by normalizing the rounded top-left coordinates.
        '''
        circ_radius = 1 * self.scaleJson['tissue_hires_scalef'] * self.scaleJson['spot_diameter_fullres'] * 0.5
        self.circ_radius=circ_radius
        print("max(pxl_x_hires),max(pxl_y_hires)",max(self.tissue_position_list["pxl_x_hires"]),max(self.tissue_position_list["pxl_y_hires"]),flush=True)
        for idx, row in self.tissue_position_list.iterrows():
            top_left_before = (row['pxl_x_hires'] - circ_radius, row['pxl_y_hires'] - circ_radius)
            self.tissue_position_list.at[idx, 'top_left_before'] = top_left_before
            top_left_after_x = round_to_tl(row['pxl_x_hires'] - circ_radius, 4)
            top_left_after_y = round_to_tl(row['pxl_y_hires'] - circ_radius, 4)
            top_left_after = (top_left_after_x, top_left_after_y)
            
            self.tissue_position_list.at[idx, 'top_left_after'] = top_left_after

        # ==========================================
        # Cropping H&E via tissue bounding box
        x_values, y_values = zip(*self.tissue_position_list['top_left_after'])
        a_spot = np.ceil(circ_radius*2).astype(int)
        x_min=max(0,min(x_values))
        y_min=max(0,min(y_values))
        ih,iw,_=self.raw_he.shape
        self.x_max=min(iw,max(x_values))
        self.y_max=min(ih,max(y_values))
        print("iw ih",iw,ih,flush = True)
        print("max(x_values) max(y_values)",max(x_values),max(y_values),flush = True)
        print("self.raw_he first time ",self.raw_he.shape)
        self.prop_he = self.raw_he[y_min:self.y_max+a_spot, x_min:self.x_max+a_spot] # add a spot
        print("coords in round_spot",x_min,y_min,self.x_max,self.y_max)
        # ==========================================
        final_tpl = self.tissue_position_list[['in_tissue', 'spot_y', 'spot_x', 'top_left_after']].copy()
        final_tpl[['tl_x', 'tl_y']] = pd.DataFrame(final_tpl['top_left_after'].tolist(),
                                                index=final_tpl.index)
        final_tpl.drop(columns=['top_left_after'], inplace=True)
        # normalize coordinate
        self.offset=(min(final_tpl['tl_x']),min(final_tpl['tl_y']))
        final_tpl['tl_x'] -= min(final_tpl['tl_x'])
        final_tpl['tl_y'] -= min(final_tpl['tl_y'])
        self.final_tpl=final_tpl
        print("self.prop_he first time",self.prop_he.shape,flush = True)

    def cal_cor(self):
        '''
        Generate tissue grid by tissue bounding box
        '''
        # 
        # We get raw and filtered h5 here, not sure whether raw will be filtered


        ###############################Ceveat: some tissue doesn't have bounding box
        # Please note here are the top left point
        filtered_tpf = self.final_tpl[self.final_tpl['in_tissue'] == 1]
        #x_max, x_min = min(filtered_tpf['tl_x'].max(),self.x_max), filtered_tpf['tl_x'].min()
        #y_max, y_min = min(filtered_tpf['tl_y'].max(),self.y_max), filtered_tpf['tl_y'].min()
        x_max = filtered_tpf['tl_x'].max()
        x_min = filtered_tpf['tl_x'].min()
        y_max = filtered_tpf['tl_y'].max()
        y_min = filtered_tpf['tl_y'].min()

        ###############################Ensure dividable
        # Divideable by patch size
        x_max = x_min + (((x_max - x_min) + self.patch_size - 1) // self.patch_size) * self.patch_size
        y_max = y_min + (((y_max - y_min) + self.patch_size - 1) // self.patch_size) * self.patch_size
        print("x_max, x_min,y_max, y_min",flush =True)
        print(x_max, x_min,y_max, y_min,flush =True)
        return x_min,x_max,y_min,y_max
    def generate_grid(self):
        # Generate tissue grid
        x_min,x_max,y_min,y_max=self.cal_cor()
        grid_spacing = 4
        tl_xs_val = np.arange(x_min, x_max+grid_spacing, grid_spacing)
        tl_ys_val = np.arange(y_min, y_max+grid_spacing, grid_spacing)

        tl_xn, tl_yn = np.meshgrid(tl_xs_val, tl_ys_val)# , indexing='xy')
        tissue = {
            'tl_xn': tl_xn.ravel(),
            'tl_yn': tl_yn.ravel()
        }

        tissue_grid = pd.DataFrame(tissue)
        tissue_grid['index'] = tissue_grid.apply(lambda row: f"s_004_{row['tl_xn']}_{row['tl_yn']}-n", axis=1)

        tissue_grid.set_index('index', inplace=True)
        self.tissue_grid=tissue_grid
        # ==========================================
        # Find barcode correspondence
    def mark_bc_label(self,row):
        self.array_forBarcode[int(row["center_x"]/4),int(row["center_y"]/4)]=row.num
    def mark_tissue_label(self,row):
        try:
            self.array_forIntissue[int(row["tl_xn"]/4),int(row["tl_yn"]/4)]=row["in_tissue"]
        except:
            return
    def mark_tissue_grid(self,row):
        index=f"s_004_{int(row['center_x'])}_{int(row['center_y'])}-n"
        try:
            self.tissue_grid.loc[index,"in_tissue"]=row["in_tissue"]
        except:
            self.tissue_grid[index]=[row["center_x"],row["center_y"],row["in_tissue"]]

    def expand_barcode(self):
        self.final_tpl["center_x"]=self.final_tpl["tl_x"]+8
        self.final_tpl["center_y"]=self.final_tpl["tl_y"]-8
        self.final_tpl["num"]=np.arange(1,len(self.final_tpl)+1)
        self.array_forBarcode=np.zeros(shape=(int(self.final_tpl["tl_x"].max()/4+5),int(self.final_tpl["tl_y"].max()/4+5)))
        self.array_forIntissue=np.zeros(shape=(int(self.final_tpl["tl_x"].max()/4+5),int(self.final_tpl["tl_y"].max()/4+5)))
        self.final_tpl.apply(self.mark_bc_label,axis=1)
        self.final_tpl.apply(self.mark_tissue_grid,axis=1)
        kd_tree = KDTree(self.final_tpl[['tl_x', 'tl_y']])
        def find_nearest_in_tissue(row):
            if pd.isna(row['in_tissue']):
                _ , ind = kd_tree.query([[row['tl_xn'], row['tl_yn']]], k=1)
                nearest_in_tissue = self.final_tpl.iloc[ind[0][0]]['in_tissue']
                return nearest_in_tissue
            else:
                return row['in_tissue']
        self.tissue_grid['in_tissue'] = self.tissue_grid.apply(find_nearest_in_tissue, axis=1)
        self.tissue_grid.apply(self.mark_tissue_label,axis=1)
        self.array_forBarcode=expand_labels(self.array_forBarcode,distance=int(self.dia//4))
        # self.array_forIntissue=expand_labels(self.array_forIntissue,distance=int(self.unit*45//4))
        self.array_forBarcode=self.array_forBarcode*self.array_forIntissue
        barcode_list=[0]
        barcode_list.extend(self.final_tpl.index.tolist())
        pd_barcode=[]
        for i, barcode in np.ndenumerate(self.array_forBarcode):
            x=i[0]
            y=i[1]
            pd_barcode.append([f"s_004_{int(x*4)}_{int(y*4)}-n",barcode_list[int(barcode)]])
        pd_barcode=pd.DataFrame(pd_barcode,columns=["position","barcode_55"])
        pd_barcode.index=pd_barcode["position"]
        self.barcode=pd_barcode
    def map_tissue(self):
        # ==========================================
        # in_tissue mapping
        self.expand_barcode()
        self.tissue_grid = self.tissue_grid.join(self.barcode, how='left', rsuffix='_cor') #06-19-2024 Zongxu change join
        self.tissue_grid['barcode_55'] = self.tissue_grid['barcode_55'].fillna('new')
        self.tissue_grid['spot_x'] = self.tissue_grid['tl_xn'] // 4
        self.tissue_grid['spot_y'] = self.tissue_grid['tl_yn'] // 4
    
    def find_avg_grid(self):  
        self.avg_grid = self.tissue_grid[  
            self.tissue_grid['barcode_55'].isin(self.adata.obs_names) &   
            (self.tissue_grid['in_tissue'] == 1)  
        ]

        avg_bc = self.avg_grid['barcode_55']
        # avg_X = self.adata[avg_bc].X / 16
        groups=self.avg_grid.groupby("barcode_55")
        barcode_lis=avg_bc.unique()
        avg_X = np.zeros(shape=self.adata[avg_bc].X.shape)
        cur_length=0
        obs_index=[]
        for barcode in barcode_lis:
            group=groups.get_group(barcode)
            length=len(group)
            avg_X[cur_length:cur_length+length,:]=(self.adata[barcode].X.sum(axis=0)/length)*np.ones(shape=(length,1))
            obs_index.extend(group.index)
            cur_length+=length
        avg_X=csr_matrix(avg_X)
        self.avg_adata = ad.AnnData(X=avg_X, var=self.adata.var)  
        self.avg_adata.obs.index = obs_index

    ## origin 
    # def insert_grid(self):
    #     self.interp_grid = self.tissue_grid[
    #         ~self.tissue_grid.index.isin(self.avg_grid.index) & 
    #         (self.tissue_grid['in_tissue'] == 1)
    #     ].dropna(subset=["tl_xn"],how='any',axis=0)

    #     orig_coords = self.final_tpl[self.final_tpl["in_tissue"]==1][["center_x","center_y"]].values
    #     interp_coords = self.interp_grid[['tl_xn', 'tl_yn']].values
    #     orig_expr=self.final_tpl[self.final_tpl["in_tissue"]==1].index
    #     interpolator = NearestNDInterpolator(orig_coords, orig_expr)
    #     interp_expr_before = interpolator(interp_coords)
    #     wrong_barcode=list(set([x for x in interp_expr_before if x not in self.adata.obs_names.values]))
    #     print(len([str(x) for x in interp_expr_before if str(x) in self.adata.obs_names.values]))
    #     interp_expr = self.adata[[str(x) for x in interp_expr_before if str(x) in self.adata.obs_names.values],:].X
    #     if(len(interp_expr) != len(interp_expr_before)):
    #         interp_expr = np.vstack((interp_expr,np.zeros(shape=(len(interp_expr_before)-len(interp_expr),len(self.adata.var_names)))))
    #     interp_expr = csr_matrix(interp_expr)
    #     self.interp_adata = ad.AnnData(X=interp_expr, var=self.adata.var)
    #     self.interp_adata.obs.index = [x for x in self.interp_grid.index.values if x not in wrong_barcode]
    # map the barcode

    def insert_grid(self):
        # Filter grid points for interpolation
        self.interp_grid = self.tissue_grid[
            ~self.tissue_grid.index.isin(self.avg_grid.index) &
            (self.tissue_grid['in_tissue'] == 1)
            ].dropna(subset=["tl_xn"], how='any', axis=0)

        # Get original spot coordinates and their corresponding barcodes
        orig_spots = self.final_tpl[self.final_tpl["in_tissue"] == 1]
        orig_coords = orig_spots[["center_x", "center_y"]].values
        orig_barcodes = orig_spots.index.tolist()

        # Filter to only include barcodes that exist in the expression data
        valid_indices = [i for i, bc in enumerate(orig_barcodes) if bc in self.adata.obs_names]
        if not valid_indices:
            print("Warning: No valid barcodes found for interpolation")
            # Create empty data
            interp_expr = csr_matrix((self.interp_grid.shape[0], self.adata.shape[1]))
            self.interp_adata = ad.AnnData(X=interp_expr, var=self.adata.var)
            self.interp_adata.obs.index = self.interp_grid.index
            return

        # Get only valid coordinates and their corresponding barcodes
        valid_coords = orig_coords[valid_indices]
        valid_barcodes = [orig_barcodes[i] for i in valid_indices]

        # Get expression data for valid barcodes
        valid_expr = self.adata[valid_barcodes].X
        if issparse(valid_expr):
            valid_expr = valid_expr.toarray()

        # Check dimensions match before interpolation
        print(f"Valid coordinates: {len(valid_coords)}, Valid expression values: {valid_expr.shape[0]}")
        assert len(valid_coords) == valid_expr.shape[0], "Mismatch between coordinates and expression values"

        # Get coordinates for points to interpolate
        interp_coords = self.interp_grid[['tl_xn', 'tl_yn']].values

        # Create an interpolator for each gene (column in expression matrix)
        interp_expr = np.zeros((len(interp_coords), valid_expr.shape[1]))

        # Perform a separate interpolation for each gene/feature
        for gene_idx in range(valid_expr.shape[1]):
            gene_values = valid_expr[:, gene_idx]
            interpolator = NearestNDInterpolator(valid_coords, gene_values)
            interp_expr[:, gene_idx] = interpolator(interp_coords)

        # Create AnnData with interpolated values
        self.interp_adata = ad.AnnData(X=csr_matrix(interp_expr), var=self.adata.var)
        self.interp_adata.obs.index = self.interp_grid.index


    def concat_data(self):
        zero_grid = self.tissue_grid[
            ~(self.tissue_grid.index.isin(self.avg_grid.index)) 
            & ~(self.tissue_grid.index.isin(self.interp_grid.index))
        ]
        zero_X = csr_matrix((zero_grid.shape[0], self.adata.shape[1]))
        zero_adata = ad.AnnData(X=zero_X, var=self.adata.var)
        zero_adata.obs.index = zero_grid.index
        # ==========================================
        # Construct final adata obj
        # fnl_adata = avg_adata.concatenate(interp_adata)

        fnl_adata_X = vstack([self.avg_adata.X, self.interp_adata.X, zero_adata.X])
        fnl_adata_obs = pd.concat([self.avg_adata.obs, self.interp_adata.obs, zero_adata.obs])
        fnl_adata_var = self.avg_adata.var.copy()
        fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
        self.fnl_adata=fnl_adata
        assert set(fnl_adata.obs_names) == set(self.tissue_grid.index), "Indices do not match"
        # ==========================================
        # Padding H&E image with white space
        # add a unit size
    def crop_img(self):
        self.process_tpl()
        self.round_spot()
        self.generate_grid()
        self.map_tissue()
        print("self.prop_he Cropping",self.prop_he.shape,flush = True)
        tg_xmax, tg_ymax = self.tissue_grid['tl_xn'].max()+4, self.tissue_grid['tl_yn'].max()+4

        padding_y, padding_x= max(0, tg_ymax - self.prop_he.shape[0]), max(0, tg_xmax - self.prop_he.shape[1])

        mask=exc_tissue(self.prop_he,method='otsu')
        print("mask ",mask.shape,flush = True)
        ih, iw, ic = self.prop_he.shape
        pdh, pdw = ih + padding_y, iw + padding_x
        he_fg, he_bg = self.prop_he[mask], self.prop_he[~mask]
        raw_he_feature = np.median(he_fg, axis=0).astype(int)
        raw_bg_feature = np.median(he_bg, axis=0).astype(int)
        raw_he=self.prop_he
        self.prop_he=white_balance_using_white_point(self.prop_he,~mask)
        print("self.prop_he.shape",self.prop_he.shape,flush =True)
        pad_w = np.array([255,255,255]).astype(np.uint8)
        pd_he = np.ones((int(pdh), int(pdw), int(ic)), dtype=np.uint8) * pad_w
        #pd_he = np.ones((int(pdw), int(pdh), int(ic)), dtype=np.uint8) * pad_w
        pdy1, pdy2 = 0, ih
        pdx1, pdx2 = 0, iw
        print("pdy1, pdy2",pdy1, pdy2,flush =True)
        print("pdx1, pdx2",pdx1, pdx2,flush =True)
        pd_he[pdy1:pdy2, pdx1:pdx2] = self.prop_he
        he_fg, he_bg = self.prop_he[mask], self.prop_he[~mask]
        after_he_feature = np.median(he_fg, axis=0).astype(int)
        after_bg_feature = np.median(he_bg, axis=0).astype(int)
        if padding_y > 0:
            pd_he[pdy2:, :] = pad_w
        if padding_x > 0:
            pd_he[:, pdx2:] = pad_w
        self.pd_he=pd_he
        return raw_he_feature,raw_bg_feature,after_he_feature,after_bg_feature,raw_he,self.pd_he
    def generate_adata(self):
        
        self.find_avg_grid()
        self.insert_grid()
        self.concat_data()

    def save(self,final_grid_path=None,final_h5ad_path=None,final_png_path=None,final_color_path=None):
        raw_he_color,raw_bg_color,after_he_color,after_bg_color,raw_he,_=self.crop_img()
        self.after_bg_color=after_bg_color
        self.tissue_grid=self.tissue_grid[~self.tissue_grid["tl_xn"].isna()]
        if(final_color_path):
            with open(final_color_path,"w") as f:
                f.write(str(raw_he_color)+" "+str(raw_bg_color))
            with open(final_color_path.replace("raw_color","offset"),"w") as f:
                f.write(str(self.offset))
            with open(final_color_path.replace("raw","after"),"w") as f:
                f.write(str(after_he_color)+" "+str(after_bg_color))
        if(final_png_path):
            io.imsave(final_png_path,self.pd_he)
            io.imsave(final_png_path.replace("tissue","raw_tissue"),raw_he)
        if(final_grid_path):  
            self.tissue_grid.to_csv(final_grid_path, sep=',', index=True, header=True)
        if(final_h5ad_path):
            self.generate_adata()
            self.fnl_adata.X=csr_matrix(self.fnl_adata.X)
            self.fnl_adata.write(final_h5ad_path)
