from VisiumOrigin2 import fileReader, Processer
import yaml
import torch
import os
import json
import math
import numpy as np
import logging
from datetime import datetime, timedelta
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from skimage import io
from models.Squall import Squall
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from numba import njit, prange
import warnings
import itertools
import gc
import multiprocessing
from contextlib import contextmanager
import pandas as pd
import socket
import fcntl
import random
Storm = Squall
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志
log_format = '%(asctime)s - %(levelname)s - [%(hostname)s] %(message)s'

class HostnameFilter(logging.Filter):
    def filter(self, record):
        record.hostname = socket.gethostname()
        return True

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.addFilter(HostnameFilter())

# Reset dict key as class attribute
class AttrDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def get_encoder(config_path, ckpt_path, device='cpu'):
    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # to attribute
    model_config = AttrDict(config['model'])

    # build model
    model = Storm(model_config)

    # load ckpt
    state_dict = torch.load(ckpt_path, map_location="cpu")["base_model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    # set eval mode & to GPU
    model.eval()
    model.to(device)
    return model

@contextmanager
def timer(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"[{name}] finished in {elapsed_time:.2f} seconds")

@njit(parallel=True)
def process_patch_mask(x_coords, y_coords, x_st, y_st, patch_size):
    """使用numba加速掩码计算"""
    mask = np.zeros(len(x_coords), dtype=np.bool_)
    for i in prange(len(x_coords)):
        if (x_coords[i] >= x_st and x_coords[i] < x_st + patch_size and
                y_coords[i] >= y_st and y_coords[i] < y_st + patch_size):
            mask[i] = True
    return mask

class IntegratedDataset(Dataset):
    def __init__(self, processed_patches, patch_coordinates, sample_id: str,
                 resolution: float = 4.0, target_size=(14, 14)):
        self.patches = processed_patches
        self.coordinates = patch_coordinates
        self.sample_id = sample_id
        self.resolution = resolution
        self.target_size = target_size
        logger.info(f"Dataset initialized with {len(processed_patches)} patches")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        try:
            he_patch, expr_matrix = self.patches[idx]
            x, y = self.coordinates[idx]

            # 输出调试信息
            if idx == 0:  # 只对第一个样本输出详细信息
                logger.info(f"Sample {idx} details:")
                logger.info(f"HE patch shape: {he_patch.shape}")
                logger.info(f"Expression matrix shape: {expr_matrix.shape}")
                logger.info(f"Coordinates: ({x}, {y})")

            # 图像预处理
            if not isinstance(he_patch, np.ndarray):
                raise ValueError(f"HE patch is not a numpy array: {type(he_patch)}")

            if len(he_patch.shape) != 3:
                raise ValueError(f"HE patch has wrong dimensions: {he_patch.shape}")

            if he_patch.max() > 2:
                he_patch = he_patch / 255.0
            rgb_tensor = torch.from_numpy(he_patch).float()

            # 处理表达式矩阵
            if isinstance(expr_matrix, csr_matrix):
                expr = torch.from_numpy(expr_matrix.toarray()).float()
            else:
                expr = torch.from_numpy(expr_matrix).float()

            size = int(math.sqrt(expr_matrix.shape[0]))
            expr = expr.reshape(size, size, -1)

            # 检查最终张量的形状
            if idx == 0:  # 只对第一个样本输出详细信息
                logger.info(f"Final tensor shapes:")
                logger.info(f"RGB tensor: {rgb_tensor.shape}")
                logger.info(f"Expression tensor: {expr.shape}")

            data = {
                "coords": torch.tensor([x, y], dtype=torch.int),
                "expr": expr,
                "rgb": rgb_tensor
            }

            res = torch.full((1,), self.resolution, dtype=torch.float32)
            return data, self.sample_id, res

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            raise e

def process_patches(x_st, y_st, patch_size, processer):
    """处理单个patch"""
    try:
        # 检查边界条件
        if x_st + patch_size > processer.pd_he.shape[1] or y_st + patch_size > processer.pd_he.shape[0]:
            logger.warning(f"Skip patch at ({x_st}, {y_st}): would exceed image bounds")
            return None

        # 提取patch
        patch_he = processer.pd_he[y_st:y_st + patch_size, x_st:x_st + patch_size].copy()

        # 检查patch尺寸
        if patch_he.shape[:2] != (patch_size, patch_size):
            logger.error(f"Invalid patch shape at ({x_st}, {y_st}): {patch_he.shape}")
            return None

        # 计算mask
        x_coords = processer.tissue_grid['tl_xn'].values
        y_coords = processer.tissue_grid['tl_yn'].values
        tissue_mask = process_patch_mask(x_coords, y_coords, x_st, y_st, patch_size)

        patch_tissue = processer.tissue_grid[tissue_mask]

        if len(patch_tissue) > 0:
            patch_indices = patch_tissue.index
            patch_mtx = processer.fnl_adata[patch_indices].X
            return (patch_he, patch_mtx), (x_st, y_st)

    except Exception as e:
        logger.error(f"Error processing patch at ({x_st}, {y_st}): {str(e)}")

    return None

def collect_patches(processer, patch_size, stride, grid_bounds, batch_size=256):
    """分批收集patches以控制内存使用"""
    grid_xmin, grid_xmax, grid_ymin, grid_ymax = grid_bounds
    processed_patches = []
    patch_coordinates = []
    start_time = time.time()

    # 输出 image 和 grid 信息
    logger.info(f"Image shape: {processer.pd_he.shape}")
    logger.info(f"Grid bounds: xmin={grid_xmin}, xmax={grid_xmax}, ymin={grid_ymin}, ymax={grid_ymax}")
    logger.info(f"Patch size: {patch_size}, Stride: {stride}")

    total_patches = ((grid_xmax - grid_xmin - patch_size) // stride + 1) * \
                    ((grid_ymax - grid_ymin - patch_size) // stride + 1)

    logger.info(f"Expected total patches: {total_patches}")

    with tqdm(total=total_patches, desc="Collecting patches", position=0) as pbar:
        for x_st in range(grid_xmin, grid_xmax - patch_size + 1, stride):
            for y_st in range(grid_ymin, grid_ymax - patch_size + 1, stride):
                result = process_patches(x_st, y_st, patch_size, processer)
                if result is not None:
                    patch_data, coords = result
                    he_patch, expr_matrix = patch_data

                    # 检查 patch 形状
                    if he_patch.shape[:2] != (patch_size, patch_size):
                        logger.warning(f"Skipping patch with wrong shape: {he_patch.shape}")
                        continue

                    processed_patches.append(patch_data)
                    patch_coordinates.append(coords)

                    # 当收集到足够的patches时，返回当前 batch
                    if len(processed_patches) >= batch_size:
                        elapsed = time.time() - start_time
                        logger.info(f"\nProcessed {len(processed_patches)} patches in {elapsed:.1f}s")
                        yield processed_patches, patch_coordinates
                        processed_patches = []
                        patch_coordinates = []

                pbar.update(1)

    # 返回最后一批数据
    if processed_patches:
        yield processed_patches, patch_coordinates

@torch.no_grad()
def process_and_infer_optimized(processer, model, patch_size=224, stride=16,
                                batch_size=32, sample_id="default",
                                target_size=(14, 14), device='cuda'):
    """优化版本：直接将推理结果放入大图对应位置"""
    start_time = time.time()
    patch_count = 0
    batch_count = 0

    # 计算网格边界
    grid_bounds = (
        int(processer.tissue_grid['tl_xn'].min()),
        int(processer.tissue_grid['tl_xn'].max()),
        int(processer.tissue_grid['tl_yn'].min()),
        int(processer.tissue_grid['tl_yn'].max())
    )

    # 计算大图尺寸
    feature_size = target_size[0]  # 假设target_size是正方形
    max_x = grid_bounds[1]
    max_y = grid_bounds[3]
    # target_height = int(max_y / stride + feature_size)
    # target_width = int(max_x / stride + feature_size)
    target_height = int(max_y / stride)
    target_width = int(max_x / stride)

    # 初始化大图和计数器
    large_expr_embedding = torch.zeros((target_height, target_width, 1024),
                                       dtype=torch.float32, device=device)
    large_rgb_embedding = torch.zeros((target_height, target_width, 1024),
                                      dtype=torch.float32, device=device)
    count_image = torch.zeros((target_height, target_width, 1),
                              dtype=torch.float32, device=device)

    logger.info(f"Initialized target embeddings of size: {target_height}x{target_width}")

    # 分批处理patches
    for processed_patches, patch_coordinates in collect_patches(
            processer, patch_size, stride, grid_bounds, batch_size=256
    ):
        if not processed_patches:
            continue

        dataset = IntegratedDataset(
            processed_patches=processed_patches,
            patch_coordinates=patch_coordinates,
            sample_id=sample_id,
            target_size=target_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        for batch_data in dataloader:
            data_batch, _, res = batch_data
            coords = data_batch["coords"]

            # 准备数据并推理
            res = res.unsqueeze(-1).unsqueeze(-1).to(device)
            rgb = data_batch["rgb"].to(device, non_blocking=True).permute(0, 3, 1, 2)
            expr = data_batch["expr"].to(device).permute(0, 3, 1, 2)

            all_embedding = model.forward_all(rgb, expr, res)

            # 分割embeddings
            mid = all_embedding.shape[1] // 2
            rgb_emb = all_embedding[:, :mid, :].view(-1, *target_size, 1024)
            expr_emb = all_embedding[:, mid:, :].view(-1, *target_size, 1024)

            # 直接将结果放入大图对应位置
            for i in range(len(coords)):
                x, y = coords[i]
                start_x = int(x / stride)
                start_y = int(y / stride)
                end_x = start_x + feature_size
                end_y = start_y + feature_size

                large_expr_embedding[start_y:end_y, start_x:end_x, :] += expr_emb[i].to(device)
                large_rgb_embedding[start_y:end_y, start_x:end_x, :] += rgb_emb[i].to(device)
                count_image[start_y:end_y, start_x:end_x, 0] += 1

            patch_count += len(coords)
            batch_count += 1

            # 清理内存
            del rgb_emb, expr_emb, rgb, expr, res, all_embedding
            torch.cuda.empty_cache()

        del dataset, dataloader
        torch.cuda.empty_cache()

        if batch_count % 10 == 0:
            logger.info(f"Processed {patch_count} patches in {batch_count} batches")

    # 平均化重叠区域
    mask = count_image > 0
    large_expr_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)
    large_rgb_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)

    # 汇总处理信息
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(
        f"\nProcessing Summary:\n"
        f"Total patches: {patch_count}\n"
        f"Total batches: {batch_count}\n"
        f"Time: {total_time:.2f}s\n"
        f"Speed: {patch_count / total_time:.2f} patches/s\n"
        f"GPU memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB\n"
        f"Completed at: {datetime.now()}"
    )

    return {
        sample_id: {
            "expr_embedding": large_expr_embedding,
            "rgb_embedding": large_rgb_embedding,
            "dimensions": (max_x, max_y)
        }
    }

class TaskManager:
    def __init__(self, shared_dir="/lustre1/zxzeng/bwqin/STORM_main/single_section/hmdb_inference_offset"):
        self.shared_dir = Path(shared_dir)
        self.task_file = self.shared_dir / "task_status.json"
        self.lock_file = self.shared_dir / "task_lock"
        self.hostname = socket.gethostname()
        
        # Ensure shared directory exists
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        
    @contextmanager
    def file_lock(self):
        """File lock context manager"""
        lock_file = open(str(self.lock_file), 'w')
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def get_default_task_info(self):
        """返回默认的任务信息字典"""
        return {
            'status': 'pending',
            'server': None,
            'start_time': None,
            'end_time': None,
            'error_type': None,
            'error_message': None,
            'attempts': 0
        }

    def init_task_status(self, sample_list):
        """Initialize task status file with more detailed status tracking"""
        with self.file_lock():
            if self.task_file.exists():
                # 如果文件存在，读取并更新缺失的字段
                with open(self.task_file, 'r') as f:
                    task_status = json.load(f)
                
                # 为每个任务添加缺失的字段
                default_info = self.get_default_task_info()
                for sample in task_status['tasks']:
                    for key, value in default_info.items():
                        if key not in task_status['tasks'][sample]:
                            task_status['tasks'][sample][key] = value
                
                # 添加新的样本
                for sample in sample_list:
                    if sample not in task_status['tasks']:
                        task_status['tasks'][sample] = self.get_default_task_info()
            else:
                # 如果文件不存在，创建新的
                task_status = {
                    'tasks': {
                        sample: self.get_default_task_info() for sample in sample_list
                    },
                    'last_update': datetime.now().isoformat()
                }
            
            # 保存更新后的状态
            with open(self.task_file, 'w') as f:
                json.dump(task_status, f, indent=2)

    def get_next_task(self, max_attempts=3):
        """Get next task, prioritizing failed tasks that haven't exceeded max attempts"""
        with self.file_lock():
            if not self.task_file.exists():
                return None
                
            with open(self.task_file, 'r') as f:
                task_status = json.load(f)
            
            # 确保每个任务都有 attempts 字段
            for task_info in task_status['tasks'].values():
                if 'attempts' not in task_info:
                    task_info['attempts'] = 0
            
            # First look for failed tasks that can be retried
            failed_tasks = [
                sample for sample, info in task_status['tasks'].items()
                if info['status'] in ['failed', 'notfound'] 
                and info.get('attempts', 0) < max_attempts
            ]
            
            # Then look for pending tasks
            pending_tasks = [
                sample for sample, info in task_status['tasks'].items()
                if info['status'] == 'pending'
            ]
            
            available_tasks = failed_tasks + pending_tasks
            
            if not available_tasks:
                return None
                
            # Randomly select a task
            task = random.choice(available_tasks)
            
            # Update task status
            task_status['tasks'][task].update({
                'status': 'processing',
                'server': self.hostname,
                'start_time': datetime.now().isoformat(),
                'attempts': task_status['tasks'][task]['attempts'] + 1
            })
            task_status['last_update'] = datetime.now().isoformat()
            
            with open(self.task_file, 'w') as f:
                json.dump(task_status, f, indent=2)
            
            return task

    def update_task_status(self, task, status, error_type=None, error_message=None):
        """Update task status with detailed error information"""
        with self.file_lock():
            with open(self.task_file, 'r') as f:
                task_status = json.load(f)
            
            task_status['tasks'][task].update({
                'status': status,
                'end_time': datetime.now().isoformat(),
                'error_type': error_type,
                'error_message': error_message
            })
            task_status['last_update'] = datetime.now().isoformat()
            
            with open(self.task_file, 'w') as f:
                json.dump(task_status, f, indent=2)

    def get_processing_stats(self):
        """Get processing statistics with more detailed status breakdown"""
        with self.file_lock():
            with open(self.task_file, 'r') as f:
                task_status = json.load(f)
            
            stats = {
                'pending': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0,
                'notfound': 0
            }
            
            server_stats = {}
            error_stats = {}
            
            for task, info in task_status['tasks'].items():
                stats[info['status']] += 1
                if info['server']:
                    server_stats[info['server']] = server_stats.get(info['server'], 0) + 1
                if info['error_type']:
                    error_stats[info['error_type']] = error_stats.get(info['error_type'], 0) + 1
            
            return stats, server_stats, error_stats

def get_homo_sapiens_samples():
    """读取CSV文件并获取Homo sapiens样本列表"""
    df = pd.read_csv('histMol_final_updated.csv')
    homo_samples = df[df['Species'] == 'Homo sapiens']['hmid_offset'].tolist()
    return homo_samples

def process_single_sample(hmid, model, device):
    """Process single sample with enhanced error handling"""
    output_path = Path(f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}_embeddings.pt")
    #input_path = Path(f"/lustre1/zxzeng/bwqin/STORM_main/single_section/hmdb_others/{hmid}")
    input_path = Path(f"/lustre1/zxzeng/bwqin/STORM_main/single_section/GSE230098/{hmid}")
    gene_token_path = Path("/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/gene_token_homologs.csv")
    
    try:
        logger.info(f"Starting to process sample {hmid}")
        
        # Check if input files exist
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        if not gene_token_path.exists():
            raise FileNotFoundError(f"Gene token file not found: {gene_token_path}")
            
        start_time = time.time()
        
        # Initialize reader and processor
        reader = fileReader()
        reader.read_all(
            folder_path=str(input_path),
            gene_token=str(gene_token_path),
            method="binary",
            key="symbol"
        )

        processer = Processer(reader, 224)
        #processer.crop_img()
        #processer.generate_adata()
        os.makedirs(f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}",exist_ok=True)
        processer.save(final_grid_path=f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}/{hmid}_grid.csv",
                final_h5ad_path=f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}/{hmid}.h5ad",
                final_png_path=f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}/tissue_{hmid}.png",
                final_color_path=f"/lustre1/zxzeng/bwqin/STORM_main/single_section/intergrate_GSE230098_offset/{hmid}/raw_color_{hmid}_color.txt")
        # Process and get embeddings
        
        embeddings = process_and_infer_optimized(
            processer=processer,
            model=model,
            patch_size=224,
            stride=16,
            batch_size=128,
            sample_id=hmid,
            target_size=(14, 14),
            device=device
        )

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings[hmid], output_path)
        process_time = time.time() - start_time
        logger.info(f"Successfully processed sample {hmid}, time: {process_time:.1f}s")
        return True, None, None

    except FileNotFoundError as e:
        logger.error(f"File not found for sample {hmid}: {str(e)}")
        return False, "notfound", str(e)
    except Exception as e:
        logger.error(f"Error processing sample {hmid}: {str(e)}", exc_info=True)
        return False, "failed", str(e)
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    config_path = '/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/large_ddp_rpb_lowres_benchmark.yaml'
    ckpt_path = '/lustre1/zxzeng/bwqin/STORM/disk_5TB-3/hmdb_for_bad_block/hmdb_inference/ckpt-epoch-300.pth'
    model = get_encoder(config_path, ckpt_path, device=device)
    
    # Get sample list and initialize task manager
    #samples = get_homo_sapiens_samples()
    folder_path = "/lustre1/zxzeng/bwqin/STORM_main/single_section/GSE230098"
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    samples = subdirs
    print("samples",samples)
    task_manager = TaskManager()
    task_manager.init_task_status(samples)
    
    logger.info(f"Server {socket.gethostname()} started processing")
    
    while True:
        # Get next task
        task = task_manager.get_next_task()
        
        if task is None:
            logger.info("All tasks completed")
            break
            
        # Process task
        success, error_type, error_message = process_single_sample(task, model, device)
        
        # Update task status
        status = 'completed' if success else error_type or 'failed'
        task_manager.update_task_status(task, status, error_type, error_message)
        
        # Display processing statistics
        stats, server_stats, error_stats = task_manager.get_processing_stats()
        logger.info("\nCurrent processing statistics:")
        logger.info(f"Pending: {stats['pending']}")
        logger.info(f"Processing: {stats['processing']}")
        logger.info(f"Completed: {stats['completed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Not Found: {stats['notfound']}")
        logger.info("\nServer statistics:")
        for server, count in server_stats.items():
            logger.info(f"{server}: {count} tasks")
        logger.info("\nError statistics:")
        for error_type, count in error_stats.items():
            logger.info(f"{error_type}: {count} occurrences")

if __name__ == "__main__":
    main()