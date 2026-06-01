import os
import re
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
import argparse
import yaml

# ======================================================
# Args
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--tile_dir", type=str,
                    default="/lustre1/zxzeng/bwqin/SQUALL/TCGA_tiles_0_5/CESC")
parser.add_argument("--output_dir", type=str,
                    default="CESC_inference_vector")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--ckpt", type=str, default="SQUALL_hires.pth")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--pool", type=str, choices=["sum", "mean"], default="sum")
args = parser.parse_args()

# ======================================================
# Model loader
# ======================================================
def get_encoder(config_path, ckpt_path, device):
    class AttrDict(dict):
        def __getattr__(self, name):
            return self[name]

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    from models.Storm import Storm
    model = Storm(AttrDict(config["model"]))
    state_dict = torch.load(ckpt_path, map_location="cpu")["base_model"]
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    return model

model = get_encoder(args.config, args.ckpt, args.device)

# ======================================================
# Inference function
# ======================================================
@torch.no_grad()
def infer_batch(model, rgb_batch, res_batch, pool="sum"):
    """
    rgb_batch: [B, H, W, 3]
    return:    [B, G]
    """
    rgb_batch = rgb_batch.to(args.device).permute(0, 3, 1, 2)
    res_batch = res_batch.to(args.device).view(-1, 1, 1, 1)

    expr_map = model.forward_rgb_to_expr(rgb_batch, res_batch)
    # expr_map: [B, 56, 56, G]

    if pool == "sum":
        expr_vec = expr_map.sum(dim=(1, 2))
    else:
        expr_vec = expr_map.mean(dim=(1, 2))

    return expr_vec.cpu().numpy()

# ======================================================
# Slides to skip
# ======================================================
ERROR_SLIDE = {
    "B1025775", "B1175732", "B1327300", "B1366920", "B1417452", "B1581497",
    "B1026643", "B1208372", "B1335042", "B1381179", "B1446654",
    "B1047395", "B1313939", "B1345808", "B1391471", "B1547511",
    "B1072177", "B1319348", "B1346289", "B1400759", "B1559222",
    "B1092165", "B1323138", "B1354216", "B1417125", "B1564186"
}

# ======================================================
# Main loop
# ======================================================
os.makedirs(args.output_dir, exist_ok=True)
os.listdir(args.tile_dir)

for slide in sorted(os.listdir(args.tile_dir)):
    slide_dir = os.path.join(args.tile_dir, slide)
    if slide in ERROR_SLIDE or not os.path.isdir(slide_dir):
        print(f"⛔ skip {slide}")
        continue

    out_dir = os.path.join(args.output_dir, slide)
    os.makedirs(out_dir, exist_ok=True)

    tif_files = sorted(f for f in os.listdir(slide_dir) if f.endswith("_HE.tif"))
    if len(tif_files) == 0:
        continue

    all_expr = []
    all_coords = []

    batch_rgb = []
    batch_coords = []

    for tif in tqdm(tif_files, desc=f"{slide}"):
        m = re.search(r'posX_(\d+)_posY_(\d+)_', tif)
        if m is None:
            continue

        x, y = int(m.group(1)), int(m.group(2))

        try:
            rgb = io.imread(os.path.join(slide_dir, tif)).astype("float32")
        except:
            continue

        img = torch.from_numpy(rgb)
        if img.max() > 1:
            img = img / 255.0

        batch_rgb.append(img)
        batch_coords.append((x, y))

        if len(batch_rgb) == args.batch_size:
            rgb_batch = torch.stack(batch_rgb)
            res_batch = torch.full((len(batch_rgb),), 0.5)
            expr_vec = infer_batch(model, rgb_batch, res_batch, pool=args.pool)

            all_expr.append(expr_vec)
            all_coords.extend(batch_coords)

            batch_rgb, batch_coords = [], []

    # last batch
    if batch_rgb:
        rgb_batch = torch.stack(batch_rgb)
        res_batch = torch.full((len(batch_rgb),), 0.5)
        expr_vec = infer_batch(model, rgb_batch, res_batch, pool=args.pool)
        all_expr.append(expr_vec)
        all_coords.extend(batch_coords)

    # ======================================================
    # Save
    # ======================================================
    expr = np.concatenate(all_expr, axis=0)   # [N, G]
    coords = np.asarray(all_coords)            # [N, 2]

    np.save(os.path.join(out_dir, "expr.npy"), expr)
    np.save(os.path.join(out_dir, "coords.npy"), coords)

    print(f"✅ {slide} saved: expr {expr.shape}, coords {coords.shape}")
