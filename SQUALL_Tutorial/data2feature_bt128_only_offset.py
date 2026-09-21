from VisiumOrigin2 import fileReader, Processer
import argparse
import yaml
import torch
import math
import numpy as np
import logging
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from models.Squall import Squall
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from numba import njit, prange
import warnings
import gc
from contextlib import contextmanager
import socket
Storm = Squall
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "base_model" in checkpoint:
        state_dict = checkpoint["base_model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("The checkpoint does not contain a valid model state dictionary")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)

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

            if idx == 0:
                logger.info(f"Sample {idx} details:")
                logger.info(f"HE patch shape: {he_patch.shape}")
                logger.info(f"Expression matrix shape: {expr_matrix.shape}")
                logger.info(f"Coordinates: ({x}, {y})")

            if not isinstance(he_patch, np.ndarray):
                raise ValueError(f"HE patch is not a numpy array: {type(he_patch)}")

            if len(he_patch.shape) != 3:
                raise ValueError(f"HE patch has wrong dimensions: {he_patch.shape}")

            if he_patch.max() > 2:
                he_patch = he_patch / 255.0
            rgb_tensor = torch.from_numpy(he_patch).float()

            if isinstance(expr_matrix, csr_matrix):
                expr = torch.from_numpy(expr_matrix.toarray()).float()
            else:
                expr = torch.from_numpy(expr_matrix).float()

            size = int(math.sqrt(expr_matrix.shape[0]))
            expr = expr.reshape(size, size, -1)

            if idx == 0:
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
    try:
        if x_st + patch_size > processer.pd_he.shape[1] or y_st + patch_size > processer.pd_he.shape[0]:
            logger.warning(f"Skip patch at ({x_st}, {y_st}): would exceed image bounds")
            return None

        patch_he = processer.pd_he[y_st:y_st + patch_size, x_st:x_st + patch_size].copy()

        if patch_he.shape[:2] != (patch_size, patch_size):
            logger.error(f"Invalid patch shape at ({x_st}, {y_st}): {patch_he.shape}")
            return None

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
    grid_xmin, grid_xmax, grid_ymin, grid_ymax = grid_bounds
    processed_patches = []
    patch_coordinates = []
    start_time = time.time()

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

                    if he_patch.shape[:2] != (patch_size, patch_size):
                        logger.warning(f"Skipping patch with wrong shape: {he_patch.shape}")
                        continue

                    processed_patches.append(patch_data)
                    patch_coordinates.append(coords)

                    if len(processed_patches) >= batch_size:
                        elapsed = time.time() - start_time
                        logger.info(f"\nProcessed {len(processed_patches)} patches in {elapsed:.1f}s")
                        yield processed_patches, patch_coordinates
                        processed_patches = []
                        patch_coordinates = []

                pbar.update(1)

    if processed_patches:
        yield processed_patches, patch_coordinates

@torch.no_grad()
def process_and_infer_optimized(processer, model, patch_size=224, stride=16,
                                batch_size=32, sample_id="default",
                                target_size=(14, 14), resolution=4.0,
                                num_workers=0, device='cuda'):
    start_time = time.time()
    patch_count = 0
    batch_count = 0

    grid_bounds = (
        int(processer.tissue_grid['tl_xn'].min()),
        int(processer.tissue_grid['tl_xn'].max()),
        int(processer.tissue_grid['tl_yn'].min()),
        int(processer.tissue_grid['tl_yn'].max())
    )

    feature_size = target_size[0]
    max_x = grid_bounds[1]
    max_y = grid_bounds[3]
    # target_height = int(max_y / stride + feature_size)
    # target_width = int(max_x / stride + feature_size)
    target_height = int(max_y / stride)
    target_width = int(max_x / stride)

    large_expr_embedding = torch.zeros((target_height, target_width, 1024),
                                       dtype=torch.float32, device=device)
    large_rgb_embedding = torch.zeros((target_height, target_width, 1024),
                                      dtype=torch.float32, device=device)
    count_image = torch.zeros((target_height, target_width, 1),
                              dtype=torch.float32, device=device)

    logger.info(f"Initialized target embeddings of size: {target_height}x{target_width}")

    for processed_patches, patch_coordinates in collect_patches(
            processer, patch_size, stride, grid_bounds, batch_size=256
    ):
        if not processed_patches:
            continue

        dataset = IntegratedDataset(
            processed_patches=processed_patches,
            patch_coordinates=patch_coordinates,
            sample_id=sample_id,
            resolution=resolution,
            target_size=target_size
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(str(device).startswith("cuda"))
        )

        for batch_data in dataloader:
            data_batch, _, res = batch_data
            coords = data_batch["coords"]

            res = res.unsqueeze(-1).unsqueeze(-1).to(device)
            rgb = data_batch["rgb"].to(device, non_blocking=True).permute(0, 3, 1, 2)
            expr = data_batch["expr"].to(device).permute(0, 3, 1, 2)

            all_embedding = model.forward_all(rgb, expr, res)

            mid = all_embedding.shape[1] // 2
            rgb_emb = all_embedding[:, :mid, :].view(-1, *target_size, 1024)
            expr_emb = all_embedding[:, mid:, :].view(-1, *target_size, 1024)

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

            del rgb_emb, expr_emb, rgb, expr, res, all_embedding
            torch.cuda.empty_cache()

        del dataset, dataloader
        torch.cuda.empty_cache()

        if batch_count % 10 == 0:
            logger.info(f"Processed {patch_count} patches in {batch_count} batches")

    mask = count_image > 0
    large_expr_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)
    large_rgb_embedding[mask.repeat(1, 1, 1024)] /= count_image[mask].repeat_interleave(1024)

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

def process_single_sample(
        sample_id, model, device, data_root, output_dir, gene_token_path,
        patch_size=224, stride=16, batch_size=2, target_size=(14, 14),
        resolution=4.0, num_workers=0):
    """Process single sample with enhanced error handling"""
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    gene_token_path = Path(gene_token_path).expanduser().resolve()
    input_path = data_root / sample_id
    sample_output_dir = output_dir / sample_id
    output_path = output_dir / f"{sample_id}_embeddings.pt"
    
    try:
        logger.info(f"Starting to process sample {sample_id}")
        
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

        processer = Processer(reader, patch_size)
        #processer.crop_img()
        #processer.generate_adata()
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        processer.save(
            final_grid_path=str(sample_output_dir / f"{sample_id}_grid.csv"),
            final_h5ad_path=str(sample_output_dir / f"{sample_id}.h5ad"),
            final_png_path=str(sample_output_dir / f"tissue_{sample_id}.png"),
            final_color_path=str(sample_output_dir / f"raw_color_{sample_id}_color.txt")
        )
        # Process and get embeddings
        
        embeddings = process_and_infer_optimized(
            processer=processer,
            model=model,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            sample_id=sample_id,
            target_size=target_size,
            resolution=resolution,
            num_workers=num_workers,
            device=device
        )

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings[sample_id], output_path)
        process_time = time.time() - start_time
        logger.info(
            f"Successfully processed sample {sample_id}, time: {process_time:.1f}s; "
            f"saved to {output_path}"
        )
        return True, None, None

    except FileNotFoundError as e:
        logger.error(f"File not found for sample {sample_id}: {str(e)}")
        return False, "notfound", str(e)
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {str(e)}", exc_info=True)
        return False, "failed", str(e)
    finally:
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()

def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create SQUALL embeddings from one or more Visium sample directories."
    )
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Directory containing one subdirectory per sample."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory in which processed files and embeddings will be written."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="SQUALL checkpoint file."
    )
    parser.add_argument(
        "--model-config", type=Path, default=script_dir / "config.yaml",
        help="Model YAML file (default: config.yaml beside this script)."
    )
    parser.add_argument(
        "--gene-token", type=Path, default=script_dir / "gene_token_homologs.csv",
        help="Gene-token CSV (default: gene_token_homologs.csv beside this script)."
    )
    parser.add_argument(
        "--sample-id", action="append", dest="sample_ids",
        help="Sample subdirectory to process; repeat for multiple samples. "
             "If omitted, every subdirectory under --data-root is processed."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-size", type=int, default=14)
    parser.add_argument("--resolution", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(name)


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_config = args.model_config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    gene_token = args.gene_token.expanduser().resolve()

    required_paths = {
        "data root": data_root,
        "model config": model_config,
        "checkpoint": checkpoint,
        "gene-token CSV": gene_token,
    }
    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")
    model = get_encoder(model_config, checkpoint, device=device)

    samples = args.sample_ids
    if not samples:
        samples = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    if not samples:
        raise ValueError(f"No sample directories found under {data_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for sample_id in samples:
        success, error_type, error_message = process_single_sample(
            sample_id=sample_id,
            model=model,
            device=device,
            data_root=data_root,
            output_dir=output_dir,
            gene_token_path=gene_token,
            patch_size=args.patch_size,
            stride=args.stride,
            batch_size=args.batch_size,
            target_size=(args.target_size, args.target_size),
            resolution=args.resolution,
            num_workers=args.num_workers,
        )
        if not success:
            failures.append((sample_id, error_type, error_message))

    if failures:
        for sample_id, error_type, error_message in failures:
            logger.error(f"{sample_id}: {error_type}: {error_message}")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
