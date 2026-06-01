'''
Author: zhang.zongxu
Version: 1.0
Date: 2024-07-26 05:00:17
LastEditors: Do not edit
LastEditTime: 2024-07-30 19:21:51
License: (C)Copyright 2022-2027, CC BY-NC-ND 4.0
Description: 
FilePath: /undefined/Users/zhangzongxu/Project/SQUALL/datasets/pkl2data.py
'''
import os
import pandas as pd
import numpy as np

from skimage import io
from einops import rearrange
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

import threading
import multiprocessing
import pickle
import argparse
import time
import torch

import warnings


def load_pkl_data(file_path):
    '''
    load data from pkl
    '''
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    if len(data) != 3:
        raise ValueError("Incomplete data!")
    
    # print("Total patch:", len(data[0]))

    return data

def _ft_intissue(ptissue, threshold=0.99):
    '''
    Determine whether the proportion of areas within the organization is greater than the threshold
    '''
    tissue_prop = np.sum(ptissue['in_tissue']) / len(ptissue['in_tissue'])
    return tissue_prop >= threshold

def _bd_idx(ptissue, ptissue_size):
    '''
    get tile location
    '''
    posX = int((ptissue['spot_x'].min()+1) / ptissue_size)+1
    posY = int((ptissue['spot_y'].min()+1) / ptissue_size)+1
    return posX, posY
    
def _scipy_to_torch_csr(scipy_csr):
    '''Converts a SciPy CSR matrix to a PyTorch CSR tensor with float16 values and int32 indices.'''
    values = torch.tensor(scipy_csr.data, dtype=torch.float16)
    indices = torch.tensor(scipy_csr.indices, dtype=torch.int32)
    indptr = torch.tensor(scipy_csr.indptr, dtype=torch.int32)
    shape = torch.Size(scipy_csr.shape)
    return torch.sparse_csr_tensor(indptr, indices, values, shape)

def _to_mtx(pmtx, ptissue_size=56, norm=None, use_sparse=True):
    '''
    Convert a given matrix (dense or sparse) to a specified format for further processing.  
      
    Parameters:  
    - pmtx: The input matrix, which can be a scipy sparse matrix or a numpy array.  
    - ptissue_size: (int, default=56) The size of an issue or tile in the matrix. Only relevant for numpy arrays.  
    - norm: (str or None, default=None) The normalization type to apply. If 'l2', L2 normalization is performed.  
    - use_sparse: (bool, default=True) Indicates whether the input matrix should be treated as sparse or dense.  
      
    Returns:  
    - A PyTorch CSR sparse tensor if the input is a sparse matrix and use_sparse is True.  
    - A reshaped and optionally normalized numpy array if the input is a numpy array.  
    '''
    if use_sparse:
        data = pmtx
    else:
        data = pmtx.toarray().astype(np.float16)
        
    if isinstance(data, csr_matrix):
        # For sparse matrices, handle non-zero elements only
        values = data.data.astype(np.float16)
        indptr = data.indptr.astype(np.int32)
        indices = data.indices.astype(np.int32)

        if norm == 'l2':
            row_sums = np.array(data.power(2).sum(axis=1)).flatten().astype(np.float16)
            norms = np.sqrt(row_sums)
            norms[norms == 0] = 1e-6  # Avoid division by zero
            values = values / norms[np.repeat(np.arange(norms.size), np.diff(indptr))]

        # Create PyTorch CSR sparse tensor
        torch_csr_tensor = _scipy_to_torch_csr(csr_matrix((values, indices, indptr), shape=data.shape))
        return torch_csr_tensor

    elif isinstance(data, np.ndarray):
        expr = rearrange(data, '(h w) c -> h w c', h=ptissue_size, w=ptissue_size).astype(np.float16)
        if norm == 'l2':
            norms = np.linalg.norm(expr, axis=2, keepdims=True).astype(np.float16)
            norms[norms == 0] = 1e-6
            expr = expr / norms
        return expr
    else:
        raise TypeError("Unsupported data type for _to_mtx function.")

def process_patch(data, idx, sample='test', save_path='./test', pHE_size=256, pmtx_size=64, threshold=0.3, norm='l2'):
    '''
    Process a single patch of data, including normalization, extraction of relevant regions,  
    and saving of processed images and expression matrices.  
  
    Parameters:  
    - data: A tuple containing three lists: images, tissue masks, and expression matrices.  
    - idx: The index of the patch to process.  
    - sample: (str, default='test') A label or identifier for the sample.  
    - save_path: (str, default='./test') The directory where processed files will be saved.  
    - pHE_size: (int, default=256) The size of the High-resolution image (HE) to be saved.  
    - pmtx_size: (int, default=64) The size of the tissue and expression matrix patch.  
    - threshold: (float, default=0.3) The threshold used to determine if a patch contains sufficient tissue.  
    - norm: (str, default='l2') The normalization type to apply to the expression matrix.  
  
    Returns:  
    - A tuple containing:  
        - The original image patch.  
        - The processed expression matrix.  
        - The file path of the saved HE image.  
        - The file path of the saved expression matrix.  
        - A status string indicating if the files were saved ('saved') or skipped ('skipped').  
    '''
    ptissue_size = pmtx_size
    scale_factor = pHE_size / ptissue_size
    
    img = data[0][idx]
    if(img.max()>2):
        img=img/255
    ptissue = data[1][idx]
    pmtx = data[2][idx]
    
    posX, posY = _bd_idx(ptissue, ptissue_size)

    spl_prefix = sample
    
    idx_prefix = f"posX_{posX}_posY_{posY}_{spl_prefix}"
    
    if not _ft_intissue(ptissue, threshold):
        return None, None, idx_prefix, 'skipped'
    
    expr = _to_mtx(pmtx, ptissue_size, norm)
    # expr = pmtx
    img_file = f"{save_path}/{idx_prefix}_HE.tif"
    expr_file = f"{save_path}/{idx_prefix}_expr.pt"

    return img, expr, img_file, expr_file, 'saved'

def save_results(results, lock):
    '''
    Save the processed results to disk in a thread-safe manner.  
  
    Parameters:  
    - results: A list of tuples, where each tuple contains the processing result for a single patch.  
               Each tuple should include the image, expression matrix, image file path, expression file path,  
               and a status string indicating whether the patch was saved ('saved') or skipped ('skipped').  
    - lock: A threading lock object used to ensure that the counters for skipped and saved patches are updated safely.  
  
    Returns:  
    - A tuple containing the number of patches that were skipped and the number of patches that were saved.  
    '''
    skipped = 0
    saved = 0
    
    for result in results:
        if result[-1] == 'skipped':
            with lock:
                skipped += 1
            continue
        
        img, expr, img_file, expr_file, status = result
        
        io.imsave(img_file, img, check_contrast=False)
        # np.save(expr_file, expr)
        torch.save(expr, expr_file)
        
        with lock:
            saved += 1

    return skipped, saved

def save_patches(data, sample, save_path, pHE_size=256, pmtx_size=64, threshold=0.3, norm=None, max_workers=32, batch_size=300):
    '''
    Save processed image and expression matrix patches to disk in batches.  
  
    Parameters:  
    - data: A tuple containing three lists of image patches, tissue masks, and expression matrices.  
    - sample: A string identifier for the sample.  
    - save_path: The directory where the processed patches will be saved.  
    - pHE_size: The size of the High-resolution image (HE) patches to be saved.  
    - pmtx_size: The size of the tissue and expression matrix patches.  
    - threshold: The threshold used to filter out patches with insufficient tissue.  
    - norm: The normalization type to apply to the expression matrices (optional).  
    - max_workers: The maximum number of threads to use for processing patches in parallel.  
    - batch_size: The number of patches to process in each batch.  
  
    Returns:  
    - None. The function prints the total number of patches, the number of filtered patches, and the number of saved patches.  
    '''
    num_patches = len(data[0])
    lock = threading.Lock()
    total_skipped = 0
    total_saved = 0

    for i in range(0, num_patches, batch_size):
        batch_data = [data[0][i:i + batch_size], data[1][i:i + batch_size], data[2][i:i + batch_size]]
        batch_indices = range(i, min(i + batch_size, num_patches))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_patch, batch_data, idx - i, sample, save_path, pHE_size, pmtx_size, threshold, norm)
                for idx in batch_indices
            ]
            results = [future.result() for future in as_completed(futures)]
        
        skipped, saved = save_results(results, lock)
        total_skipped += skipped
        total_saved += saved
    print(f"{sample} total patches: {num_patches}, filtered patches: {total_skipped}, saved patches: {total_saved}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message="Sparse CSR tensor support is in beta state.")
    parser = argparse.ArgumentParser(description="Transform pickle data into tiff and npy.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output data.')
    parser.add_argument('--norm', type=str, default='l2', help='Normalization method.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Minimum tissue area requirement to save a patch.')
    parser.add_argument('--max_workers', type=int, default=int(multiprocessing.cpu_count()*0.75), help='Number of max workers to use.')
    parser.add_argument('--batch_size', type=int, default=300, help='Number of patches to process in each batch.')

    args = parser.parse_args()
    start_time = time.time() 
    data = load_pkl_data(args.file_path)
    csv_sample=pd.read_csv("/home/daijt/hmID.csv")
    folder_name=os.path.basename(os.path.dirname(args.file_path))
    sample=csv_sample.loc["hmID",folder_name]
    save_patches(data, 
                sample=sample,
                save_path=args.save_path,
                norm=args.norm,
                threshold=args.threshold,
                max_workers=args.max_workers,
                batch_size=args.batch_size
            )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processing {args.sample} took {elapsed_time:.2f} seconds.")
