#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# here put the import lib
import os
import math
import gzip
import shutil
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KDTree
from skimage import io,color, filters
from skimage.segmentation import expand_labels
from scipy.interpolate import NearestNDInterpolator
from scipy.sparse import vstack, issparse, csr_matrix,hstack
from sklearn.preprocessing import MinMaxScaler,StandardScaler
patch_size = 224
def exc_tissue(img, method='lumin', l_threshold=0.8):
    '''
    Generate a tissue mask from an RGB image using either luminance thresholding or Otsu's method.
    Args:
        img (ndarray): An RGB image represented as a NumPy array with shape (height, width, 3).
        method (str, optional): The method to use for generating the tissue mask. Options are 'lumin' and 'otsu'. Defaults to 'lumin'.
        l_threshold (float, optional): The luminance threshold value used when `method` is set to 'lumin'. Values range from 0 to 1. Defaults to 0.8.
    Returns:
        ndarray: A binary mask representing the detected tissue region in the input image. It has the same height and width as the input image.
    '''
    if method not in ['lumin', 'otsu']:
        raise ValueError("Method should be 'lumin' or 'otsu'")
    
    if method == 'lumin':
        img_lab = color.rgb2lab(img)
        light_float = img_lab[:, :, 0] / 100.0  # L* channel range is [0, 100], normalize to [0, 1]
        mask = light_float < l_threshold
    elif method == 'otsu':
        gray_img = color.rgb2gray(img)
        otsu_threshold = filters.threshold_otsu(gray_img)
        mask = gray_img < otsu_threshold
    
    # Check it's not empty
    # assert mask.sum() == 0, "Empty tissue mask computed"
    
    return mask

def gz_file(file_path,save_path):
    '''  
    Compress a file using gzip.  
    such as matrix.mtx -> matrix.mtx.gz
            features.tsv -> features.tsv.gz

    This function takes a file path as input, reads its contents, and then compresses  
    those contents using the gzip format. The compressed file is saved to the specified  
    save path.  

    Parameters:  
    file_path (str): The path to the file that needs to be compressed.  
    save_path (str): The path where the compressed file should be saved.  
    '''  
    f = open(file_path, "rb")
    value = f.read()
    f.close()

    Gz_file = gzip.GzipFile(filename="", mode="wb",
                compresslevel=9,
                fileobj=open(save_path,"wb"))
    Gz_file.write(value)
    Gz_file.close()

def white_balance_using_white_point(img, mask):
    '''  
    Apply white balance to an image using a white point mask.  
  
    This function adjusts the white balance of an image by calculating gains for each  
    color channel (red, green, blue) based on the 90th percentile values of the  
    corresponding channels within a masked region of the image. The masked region is  
    assumed to contain the white point or a near-white area, which is used as a  
    reference for balancing the colors.  
  
    Parameters:  
    img (numpy.ndarray): The input image, expected to be in the RGB color space.  
    mask (numpy.ndarray): A binary mask of the same shape as the image, where 1s  
                          indicate the region of interest (white point) and 0s  
                          indicate the background.  
  
    Returns:  
    numpy.ndarray: The white-balanced image, with the same shape and data type as  
                   the input image.  
    ''' 
    img_float = img.astype(np.float32) / 255.0
    img_mask=img*(np.expand_dims(mask,axis=2)*np.ones(shape=(1,1,3)))
    wr, wg, wb = np.percentile(img_mask[:,:,0],90)/255,np.percentile(img_mask[:,:,1],90)/255,np.percentile(img_mask[:,:,2],90)/255
    wr = wr if wr != 0 else 1  
    wg = wg if wg != 0 else 1  
    wb = wb if wb != 0 else 1
    gain_r = 1.0 / wr
    gain_g = 1.0 / wg
    gain_b = 1.0 / wb
    balanced_img = img_float.copy()
    balanced_img[:, :, 0] *= gain_r
    balanced_img[:, :, 1] *= gain_g
    balanced_img[:, :, 2] *= gain_b
    balanced_img = np.clip(balanced_img, 0, 1)
    balanced_img = (balanced_img * 255).astype(np.uint8)
    return balanced_img
def set_gene_token_by_id(adata):
    """  
    Sets gene tokens for the given annotation data (adata) based on gene IDs.  
  
    Parameters:  
    adata (anndata.AnnData): Annotated data object containing gene expression information.  
  
    Returns:  
    anndata.AnnData: Updated annotated data object with gene tokens set.  
    """  
    adata.var_names_make_unique()
    gene_names =  adata.var["gene_ids"].str.replace("__","_").apply(split_gene_id).values
    tkn_list=pd.read_csv("/home/daijt/deal_data/gene_token_homologs.csv")
    human_matches = tkn_list['ENSG_ID'].isin(gene_names).sum()
    mouse_matches = tkn_list['ENSMUSG_ID'].isin(gene_names).sum()
    print(f"gene id matches:\nhuman:{human_matches} \nmouse:{mouse_matches}")
    if human_matches>10000 and mouse_matches>10000:
        gene_token = set(tkn_list['ENSG_ID'])
        tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
        tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        adata_1= adata_token_by_id(adata,gene_token,tkn_info)
        gene_token = set(tkn_list['ENSMUSG_ID'])
        tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
        tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        adata_2=adata_token_by_id(adata,gene_token,tkn_info)
        return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
    elif human_matches > mouse_matches:
        gene_token = set(tkn_list['ENSG_ID'])
        tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
        tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        return adata_token_by_id(adata,gene_token,tkn_info)
    elif mouse_matches > human_matches:
        gene_token = set(tkn_list['ENSMUSG_ID'])
        tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
        tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        return adata_token_by_id(adata,gene_token,tkn_info)
    else:
        raise ValueError("Cannot determine the species from gene names")
def split_gene_id(gene_id):
    return gene_id.split("_")[-1].split(".")[0]
def adata_token_by_id(adata,gene_token,tkn_info):
    '''
    Set gene tokens by ID and update the annotated data (adata) with corresponding gene symbols.  
      
    Parameters:  
    adata (ad.AnnData): Annotated data object containing gene expression data.  
      
    Returns:  
    ad.AnnData: Updated annotated data object with gene tokens and corresponding symbols.  
    '''
    # Extract ID
    adata.var['gene_ids'] = adata.var['gene_ids'].apply(split_gene_id).values
    adata.var.index=adata.var["gene_ids"].values
    adata.var_names_make_unique()
    # Drop .
    intkn_adata = adata[:, adata.var.index.isin(gene_token)].copy()
    intkn_adata.var = intkn_adata.var.iloc[:, 0:0]

    intkn_genes = set(intkn_adata.var_names)
    miss_gene = [gene for gene in gene_token if gene not in intkn_genes]

    # var should be a df
    miss_gene = pd.DataFrame(miss_gene, columns=['gene_ids']).set_index('gene_ids')

    miss_X = csr_matrix((adata.n_obs, len(miss_gene)))
    miss_adata = ad.AnnData(X=miss_X, var=miss_gene)
    miss_adata.obs.index = adata.obs.index
    print(len(miss_gene),len(intkn_genes))
    print(miss_adata.X.shape,intkn_adata.X.shape)
    fnl_adata_X = hstack([intkn_adata.X, miss_adata.X])
    fnl_adata_obs = intkn_adata.obs.copy()
    fnl_adata_var = pd.concat([intkn_adata.var, miss_adata.var])
    fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
    token_order = [fnl_adata.var_names.get_loc(name) for name in tkn_info['gene_ids'].values]
    fnl_adata.var = fnl_adata.var.iloc[token_order]
    if issparse(fnl_adata.X):
        fnl_adata.X = fnl_adata.X.toarray()[:, token_order]
    else:
        fnl_adata = fnl_adata[:, token_order]
    try:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info.reset_index(), left_on='index', right_on='gene_ids', how='left')
        fnl_adata.var = fnl_adata.var.drop(['index'], axis=1)
    except:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info.reset_index(), left_on='gene_ids', right_on='gene_ids', how='left')
    fnl_adata.var.set_index('symbol', inplace=True)
    return fnl_adata

def set_gene_token_by_symbol(adata):
    '''
    Sets gene tokens for the given annotation data (adata) based on symbols.  
  
    Parameters:  
    adata (anndata.AnnData): Annotated data object containing gene expression information.  
  
    Returns:  
    anndata.AnnData: Updated annotated data object with gene tokens set.  
    ''' 
    gene_names =  pd.Series(adata.var_names).str.replace("__","_").apply(split_gene_id).values
    tkn_list=pd.read_csv("/home/daijt/deal_data/gene_token_homologs.csv")
    human_matches = tkn_list['HGNC_symbol'].isin(gene_names).sum()
    mouse_matches = tkn_list['MGI_symbol'].isin(gene_names).sum()
    print(f"symbol matches:\nhuman:{human_matches} \nmouse:{mouse_matches}")
    if human_matches>10000 and mouse_matches>10000:
        gene_token = set(tkn_list['HGNC_symbol'])
        tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
        tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        adata_1= adata_token_by_symbol(adata,gene_token,tkn_info)
        gene_token = tkn_list['MGI_symbol']
        tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
        tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        adata_2=adata_token_by_symbol(adata,gene_token,tkn_info)
        return ad.AnnData(X=adata_1.X+adata_2.X,obs=adata_1.obs,var=adata_1.var)
    elif human_matches > mouse_matches:
        gene_token = set(tkn_list['HGNC_symbol'])
        tkn_info = tkn_list[['ENSG_ID', 'HGNC_symbol']].copy()
        tkn_info.rename(columns={'ENSG_ID': 'gene_ids', 'HGNC_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        return adata_token_by_symbol(adata,gene_token,tkn_info)
    elif mouse_matches > human_matches:
        gene_token = tkn_list['MGI_symbol']
        tkn_info = tkn_list[['ENSMUSG_ID', 'MGI_symbol']].copy()
        tkn_info.rename(columns={'ENSMUSG_ID': 'gene_ids', 'MGI_symbol': 'symbol'}, inplace=True)
        tkn_info.set_index('symbol', inplace=True)
        return adata_token_by_symbol(adata,gene_token,tkn_info)
    else:
        raise ValueError("Cannot determine the species from gene names")
def adata_token_by_symbol(adata,gene_token,tkn_info):
    '''
    Set gene tokens by symbols and update the annotated data (adata) with corresponding gene symbols.  
      
    Parameters:  
    adata (ad.AnnData): Annotated data object containing gene expression data.  
      
    Returns:  
    ad.AnnData: Updated annotated data object with gene tokens and corresponding symbols.  
    '''
    # Extract ID
    adata.var["symbol"]=adata.var.index.values
    adata.var_names = pd.Series(adata.var_names).apply(split_gene_id).values
    # Drop 
    adata.var_names_make_unique()
    intkn_adata = adata[:, adata.var_names.isin(gene_token)].copy()
    intkn_adata.var_names = intkn_adata.var_names
    intkn_adata.var = intkn_adata.var.iloc[:, 0:0]
    intkn_genes = set(intkn_adata.var_names)
    miss_gene = [gene for gene in set(gene_token) if gene not in intkn_genes]

    # var should be a df
    miss_gene = pd.DataFrame(miss_gene, columns=['symbol']).set_index('symbol')
    miss_X = csr_matrix((adata.n_obs, len(miss_gene)))
    miss_adata = ad.AnnData(X=miss_X, var=miss_gene)
    miss_adata.obs.index = adata.obs.index
    fnl_adata_X = hstack([intkn_adata.X, miss_adata.X])
    fnl_adata_obs = intkn_adata.obs.copy()
    fnl_adata_var = pd.concat([intkn_adata.var, miss_adata.var])
    fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
    fnl_adata.var_names_make_unique()
    token_order = [fnl_adata.var_names.get_loc(name) for name in tkn_info.index.values]
    try:
        fnl_adata.var = fnl_adata.var.iloc[token_order]
    except:
        fnl_adata.var = pd.DataFrame(fnl_adata.var.index.values[token_order],index=fnl_adata.var.index.values[token_order])
    if issparse(fnl_adata.X):
        fnl_adata.X = fnl_adata.X.toarray()[:, token_order]
    else:
        fnl_adata = fnl_adata[:, token_order]
    try:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info.reset_index(), left_on='index', right_on='symbol', how='left')
        fnl_adata.var = fnl_adata.var.drop(['index'], axis=1)
    except:
        fnl_adata.var = fnl_adata.var.reset_index().merge(tkn_info.reset_index(), left_on='symbol', right_on='symbol', how='left')
    
    return fnl_adata

def round_to_tl(value,grid_spacing):
    '''  
    Round a given value to the nearest multiple of grid_spacing towards top and left.  
  
    Parameters:  
    value (float or int): The value to be rounded.  
    grid_spacing (float or int): The interval to which the value should be rounded.  
  
    Returns:  
    float or int: The rounded value.  
    '''
    return math.floor(value / grid_spacing) * grid_spacing

def binary_matrix(matrix):
    '''  
    Convert a matrix to a binary matrix where all non-zero elements are set to 1.  
  
    Parameters:  
    matrix (numpy.ndarray): The input matrix to be converted.  
  
    Returns:  
    numpy.ndarray: A binary matrix where all non-zero elements from the input matrix are set to 1.  
    '''  
    matrix[matrix!= 0] = 1
    return matrix.astype(int)

def kbins(matrix,n):
    '''  
    This function bins the values in the input matrix into `n` equal-width bins based on the matrix's range.  
      
    Parameters:  
    matrix (numpy.ndarray): The input matrix containing numerical values.  
    n (int): The number of bins to divide the matrix values into.  
      
    Returns:  
    numpy.ndarray: An array of integers where each element represents the bin index for the corresponding element in the input matrix.  
    '''
    bins=range(matrix.min(),matrix.max(),(matrix.max()-matrix.min())//n)
    return np.digitize(matrix,bins)
class process_satandard:
    '''
    deal for one sample in a folder
    '''
    def get_paths(self,folder_path):
        '''
        Get detail path for img , json , csv and h5(mtx)

        Args:
        folder_path:the sample folder path for sample
            example:
                path
                ├──spatial
                │    ├──GSM5026146_S4_tissue_hires_image.png
                │    ├──GSM5026146_scalefactors_json.json
                │    └──GSM5026146_tissue_positions_list.csv
                └──filtered_feature_bc_matrix.h5 / matrix.mtx
        Returns:
        raw_img_path:path to deal img
        raw_tpl_path:path to read tissue position
        json_path:path to read scale factors JSON file
        h5_path:path to read 
        '''
        folder_path=folder_path.replace("\n","")
        raw_img_path , raw_tpl_path , json_path , h5_path="","","",""
        for file in os.listdir(folder_path):
            file_path=os.path.join(folder_path,file)
            if(file_path.lower().count("raw") and (file_path.count("h5") or file_path.count(".mtx")) and (not file_path.count("total"))):
                h5_path=file_path
                break  
            elif((file_path.count("h5") or file_path.count(".mtx")) and (not file_path.count("total"))):
                h5_path=file_path
            
        for file in os.listdir(os.path.join(folder_path,"spatial")):
            file_path=os.path.join(folder_path+"/spatial/",file)
            file=file.lower()
            if(file.count("position") and file.count(".csv")):
                raw_tpl_path=file_path
                continue
            if(file.count("scalefactor") and file.count(".json")):
                json_path=file_path
                continue
            if(file_path.lower().count("hires") and (file_path.count(".png") or file_path.count(".tif") or file_path.count(".jpg")) and (not file_path.count("lowres"))):
                raw_img_path=file_path
                continue
        if (not (raw_img_path and raw_tpl_path and json_path and h5_path)):
            raise ValueError(os.path.basename(raw_img_path)+"  "+os.path.basename(raw_tpl_path)+"  "+os.path.basename(json_path)+"  "+os.path.basename(h5_path))
        return raw_img_path.replace("._",""),raw_tpl_path.replace("._",""),json_path.replace("._",""),h5_path.replace("._","")
    def read_all(self,folder_path,method):
        '''  
        This method reads all relevant data from the specified folder path using the given method.  
        
        Parameters:  
        - folder_path : The path to the folder containing the data files.  
        - method : Can be "binary", "raw", an integer for binning,   
                        "minmax" for Min-Max scaling, or "z-score" for Z-score normalization.  
        '''
        self.folder_path=folder_path
        raw_img_path,raw_tpl_path,json_path,h5_path=self.get_paths(folder_path)
        self.read_img(raw_img_path)
        self.read_tissue_position(raw_tpl_path,json_path)
        self.read_h5(h5_path,method)
        
    def read_img(self,raw_img_path):
        '''
        Reads files from a given folder path and stores relevant data in class attributes.
        Args:
            folder_path (str): The path to the folder containing the necessary files.
        '''
        if(raw_img_path.count(".gz")):
            with gzip.open(raw_img_path, 'rb') as f_in:
                raw_img_path=raw_img_path.replace(".gz","")
                with open(raw_img_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        self.raw_he = io.imread(raw_img_path)
        if self.raw_he.ndim == 3 and self.raw_he.shape[2] == 4:
            self.raw_he = self.raw_he[:, :, :3]
    def read_tissue_position(self,raw_tpl_path,json_path):
        if(raw_tpl_path.count(".gz")):
            with gzip.open(raw_tpl_path, 'rb') as f_in:
                raw_tpl_path=raw_tpl_path.replace(".gz","")
                with open(raw_tpl_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        if(json_path.count(".gz")):
            with gzip.open(json_path, 'rb') as f_in:
                json_path=json_path.replace(".gz","")
                with open(json_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        self.tissue_position_list = pd.read_csv(raw_tpl_path, header=None, names=['barcode','in_tissue','array_row', 'array_col',
                                                                  'pxl_row_in_fullres',  'pxl_col_in_fullres'])
        self.tissue_position_list.set_index('barcode', inplace=True)
        while(str(self.tissue_position_list.iloc[0,0])!="0" and str(self.tissue_position_list.iloc[0,0])!="1"):
            self.tissue_position_list=self.tissue_position_list[1:]
        self.tissue_position_list["in_tissue"]=self.tissue_position_list["in_tissue"].astype(int)
        self.json = pd.read_json(json_path, lines=True)
        self.dia = self.json['spot_diameter_fullres']*self.json['tissue_hires_scalef']
    def read_h5(self,h5_path,method):
        '''  
    This method reads an H5 file or a directory containing 10x Genomics mtx files, processes the data,   
    and scales or binarizes it based on the specified method.  
  
    Parameters:  
    self (object): The instance of the class that this method belongs to.  
    h5_path (str): The path to the H5 file or the directory containing mtx, features.tsv, and barcodes.tsv files.  
    method (str or int):  Can be "binary", "raw", an integer for binning,   
                         "minmax" for Min-Max scaling, or "z-score" for Z-score normalization.  
    '''
        if(h5_path.count(".h5")):
            try:
                adata = sc.read_10x_h5(h5_path)
            except:
                adata = sc.read_h5ad(h5_path)
        elif(h5_path.count(".mtx")):
            h5_path=os.path.dirname(h5_path)
            for file in os.listdir(h5_path):
                file_path=os.path.join(h5_path,file)
                if(file.count("matrix.mtx") and (not file.count(".gz"))):
                    gz_file(file_path,os.path.join(h5_path,"matrix.mtx.gz"))
                if(file.count("features.tsv") and (not file.count(".gz"))):
                    gz_file(file_path,os.path.join(h5_path,"features.tsv.gz"))
                if(file.count("barcodes.tsv") and (not file.count(".gz"))):
                    gz_file(file_path,os.path.join(h5_path,"barcodes.tsv.gz"))
            adata=sc.read_10x_mtx(path=h5_path,make_unique=False)
        adata.var_names_make_unique()
        try:
            self.adata=set_gene_token_by_id(adata)
        except:
            self.adata=set_gene_token_by_symbol(adata)
        if(method=="binary"):
            self.adata.X=binary_matrix(self.adata.X)
        elif(method=="raw"):
            pass
        elif(type(method)==int):
            self.adata.X=kbins(self.adata.X,method)
        elif(method=="minmax"):
            M=MinMaxScaler()
            self.adata.X=M.fit_transform(self.adata.X)
        elif(method=="z-score"):
            S=StandardScaler()
            self.adata.X=S.fit_transform(self.adata.X)
        self.adata.obs_names_make_unique()
    def process_tpl(self):
        '''
        Process the tissue position data in a CSV-like DataFrame.
        Args:
            None (This is an instance method and does not require any arguments.)
        Returns:
            None (The function modifies the internal DataFrame 'tissue_position_list' directly.)
        '''
        self.tissue_position_list=self.tissue_position_list.astype(float)
        self.tissue_position_list[['pxl_row_in_hires','pxl_col_in_hires']] = self.tissue_position_list[['pxl_row_in_fullres','pxl_col_in_fullres']]* self.json['tissue_hires_scalef'].iloc[0]
        self.tissue_position_list.rename(columns={
            'pxl_row_in_fullres': 'pxl_y_fullres',
            'pxl_col_in_fullres': 'pxl_x_fullres',
            'pxl_row_in_hires': 'pxl_y_hires',
            'pxl_col_in_hires': 'pxl_x_hires',
            'array_row': 'spot_y',
            'array_col': 'spot_x'
        }, inplace=True)
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
        circ_radius = 1 * self.json['tissue_hires_scalef'].iloc[0] * self.json['spot_diameter_fullres'].iloc[0] * 0.5
        self.circ_radius=circ_radius
        
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
        self.prop_he = self.raw_he[y_min:self.y_max+a_spot, x_min:self.x_max+a_spot] # add a spot
        # ==========================================
        # Original tpl
        final_tpl = self.tissue_position_list[['in_tissue', 'spot_y', 'spot_x', 'top_left_after']].copy()
        final_tpl[['tl_x', 'tl_y']] = pd.DataFrame(final_tpl['top_left_after'].tolist(),
                                                index=final_tpl.index)
        final_tpl.drop(columns=['top_left_after'], inplace=True)
        # normalize coordinate
        self.offset=(min(final_tpl['tl_x']),min(final_tpl['tl_y']))
        final_tpl['tl_x'] -= min(final_tpl['tl_x'])
        final_tpl['tl_y'] -= min(final_tpl['tl_y'])
        self.final_tpl=final_tpl
        
    
    def cal_cor(self):
        '''  
        Calculate the corrected tissue grid boundaries based on the tissue bounding box.  
        This method ensures that the boundaries are adjustable to be divisible by a specific patch size.  
        '''
        # 
        # We get raw and filtered h5 here, not sure whether raw will be filtered
        ###############################Ceveat: some tissue doesn't have bounding box
        # Please note here are the top left point
        filtered_tpf = self.final_tpl[self.final_tpl['in_tissue'] == 1]
        x_max, x_min = min(filtered_tpf['tl_x'].max(),self.x_max), filtered_tpf['tl_x'].min()
        y_max, y_min = min(filtered_tpf['tl_y'].max(),self.y_max), filtered_tpf['tl_y'].min()
        ###############################Ensure dividable
        # Divideable by patch size
        x_max = x_min + (((x_max - x_min) + patch_size - 1) // patch_size) * patch_size
        y_max = y_min + (((y_max - y_min) + patch_size - 1) // patch_size) * patch_size
        return x_min,x_max,y_min,y_max
    def generate_grid(self):
        '''  
        Generate a grid of tissue patches based on the calculated boundaries.  
        This grid is used to index and organize the tissue patches.  
        '''
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
    def mark_bc_label(self,row):
        ''' 
        Marks the barcode label on a 2D array.  

        Parameters:  
        row (pd.Series): A pandas Series containing the row data, with 'center_x' and 'center_y' as coordinates.  
        '''
        self.array_forBarcode[int(row["center_x"]/4),int(row["center_y"]/4)]=row.num
    def mark_tissue_label(self,row):
        '''
        Marks the tissue label on a 2D array.  
          
        Parameters:  
        row (pd.Series): A pandas Series containing the row data, with 'tl_xn', 'tl_yn', and 'in_tissue' as attributes.  
        '''
        try:
            self.array_forIntissue[int(row["tl_xn"]/4),int(row["tl_yn"]/4)]=row["in_tissue"]
        except:
            return

    def mark_tissue_grid(self,row):
        '''
        Marks the tissue grid with in-tissue information.  
          
        Parameters:  
        row (pd.Series): A pandas Series containing the row data, with 'center_x', 'center_y', and 'in_tissue' as attributes.  
        '''
        index=f"s_004_{int(row['center_x'])}_{int(row['center_y'])}-n"
        try:
            self.tissue_grid.loc[index,"in_tissue"]=row["in_tissue"]
        except:
            self.tissue_grid[index]=[row["center_x"],row["center_y"],row["in_tissue"]]

    def expand_barcode(self):
        ''' 
        Expands the barcode array, integrates it with tissue information, and generates a final barcode DataFrame.  
        '''
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
        '''  
        Map barcodes to the tissue grid and calculate spot positions.  
        '''
        self.expand_barcode()
        self.tissue_grid = self.tissue_grid.join(self.barcode, how='left', rsuffix='_cor') #06-19-2024 Zongxu change join
        self.tissue_grid['barcode_55'] = self.tissue_grid['barcode_55'].fillna('new')
        self.tissue_grid['spot_x'] = self.tissue_grid['tl_xn'] // 4
        self.tissue_grid['spot_y'] = self.tissue_grid['tl_yn'] // 4
    
    def find_avg_grid(self):  
        '''
        This function is used to find and calculate the average representation matrix for each barcode in the organization grid.  
        First, it filters out the organization grid entries that are located within the organization and where the barcode is present in the ADATA observation name.  
        Then, for each barcode, it calculates the average of the representations of all the points under that barcode and generates a new AnnData object,  
        It contains these average expressions and the corresponding variable information.
        '''
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
    
    def insert_grid(self):
        '''
        This function is used to insert an interpolated representation matrix for those organizational grid entries that are not in the average grid but are within the organization.  
        It uses the nearest neighbor interpolation method to assign barcodes and expressions to unknown points based on their coordinates and barcodes for known expressions.  
        It then creates a new AnnData object that contains these interpolated expressions and the corresponding variable information.  
        
        '''
        self.interp_grid = self.tissue_grid[
            ~self.tissue_grid.index.isin(self.avg_grid.index) & 
            (self.tissue_grid['in_tissue'] == 1)
        ]

        orig_coords = self.final_tpl[self.final_tpl["in_tissue"]==1][["center_x","center_y"]].values
        interp_coords = self.interp_grid[['tl_xn', 'tl_yn']].values
        orig_expr=self.final_tpl[self.final_tpl["in_tissue"]==1].index
        interpolator = NearestNDInterpolator(orig_coords, orig_expr)
        interp_expr_before = interpolator(interp_coords)
        wrong_barcode=list(set([x for x in interp_expr_before if x not in self.adata.obs_names.values]))
        interp_expr = self.adata[[x for x in interp_expr_before if x in self.adata.obs_names.values],:].X
        if(len(interp_expr) != len(interp_expr_before)):
            interp_expr = np.vstack((interp_expr,np.zeros(shape=(len(interp_expr_before)-len(interp_expr),len(self.adata.var_names)))))
        interp_expr = csr_matrix(interp_expr)
        self.interp_adata = ad.AnnData(X=interp_expr, var=self.adata.var)
        self.interp_adata.obs.index = [x for x in self.interp_grid.index.values if x not in wrong_barcode]

    def concat_data(self):
        '''
        This function is used to combine average, interpolated, and zero-grid data into a single complete AnnData object.  
        Zero grid data refers to those organizational grid entries that are neither in the average grid nor in the interpolated grid, and their expression is set to zero.  
        Finally, it ensures that the observed index of the merged data exactly matches the organizational grid index.  
        '''
        zero_grid = self.tissue_grid[
            ~(self.tissue_grid.index.isin(self.avg_grid.index)) 
            & ~(self.tissue_grid.index.isin(self.interp_grid.index))
        ]
        zero_X = csr_matrix((zero_grid.shape[0], self.adata.shape[1]))
        zero_adata = ad.AnnData(X=zero_X, var=self.adata.var)
        zero_adata.obs.index = zero_grid.index
        fnl_adata_X = vstack([self.avg_adata.X, self.interp_adata.X, zero_adata.X])
        fnl_adata_obs = pd.concat([self.avg_adata.obs, self.interp_adata.obs, zero_adata.obs])
        fnl_adata_var = self.avg_adata.var.copy()
        fnl_adata = ad.AnnData(fnl_adata_X, obs=fnl_adata_obs, var=fnl_adata_var)
        self.fnl_adata=fnl_adata
        assert set(fnl_adata.obs_names) == set(self.tissue_grid.index), "Indices do not match"

    def crop_img(self):
        '''
        crop the tissue area, correct the background, and padding the edges of the cut image so that the image is divisible by patch size
        Returns: specific parameters in the processing process
        '''
        self.process_tpl()
        self.round_spot()
        self.generate_grid()
        self.map_tissue()
        tg_xmax, tg_ymax = self.tissue_grid['tl_xn'].max()+4, self.tissue_grid['tl_yn'].max()+4

        padding_y, padding_x= max(0, tg_ymax - self.prop_he.shape[0]), max(0, tg_xmax - self.prop_he.shape[1])

        mask=exc_tissue(self.prop_he,method='otsu')
        
        ih, iw, ic = self.prop_he.shape
        pdh, pdw = ih + padding_y, iw + padding_x
        he_fg, he_bg = self.prop_he[mask], self.prop_he[~mask]
        raw_he_feature = np.median(he_fg, axis=0).astype(int)
        raw_bg_feature = np.median(he_bg, axis=0).astype(int)
        raw_he=self.prop_he
        self.prop_he=white_balance_using_white_point(self.prop_he,~mask)
        pad_w = np.array([255,255,255]).astype(np.uint8)
        pd_he = np.ones((int(pdh), int(pdw), int(ic)), dtype=np.uint8) * pad_w
        pdy1, pdy2 = 0, ih
        pdx1, pdx2 = 0, iw
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
        '''
        handle adata objects
        '''
        self.tissue_grid=self.tissue_grid[~self.tissue_grid["tl_xn"].isna()]
        self.find_avg_grid()
        self.insert_grid()
        self.concat_data()
    def save(self,final_grid_path,final_png_path,final_color_path):
        '''
        save the intermediate file for processing
        '''
        raw_he_color,raw_bg_color,after_he_color,after_bg_color,raw_he,_=self.crop_img()
        self.after_bg_color=after_bg_color
        with open(final_color_path,"w") as f:
            f.write(str(raw_he_color)+" "+str(raw_bg_color))
        with open(final_color_path.replace("raw_color","offset"),"w") as f:
            f.write(str(self.offset))
        with open(final_color_path.replace("raw","after"),"w") as f:
            f.write(str(after_he_color)+" "+str(after_bg_color))
        self.generate_adata()
        io.imsave(final_png_path,self.pd_he)
        io.imsave(final_png_path.replace("tissue","raw_tissue"),raw_he)
        self.tissue_grid.to_csv(final_grid_path, sep=',', index=True, header=True)
