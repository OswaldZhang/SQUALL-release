# +
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys, time, platform
import openslide
from PIL import Image
import pickle

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Import custom utility functions
from func.utils_preprocessing import *
from func import utils_color_norm

# Initialize color normalization
color_norm = utils_color_norm.macenko_normalizer()

# =============================================================================
# Check available device (CPU/GPU)
device = torch.device('cpu')
print("Device:", device)

# Set model name
model_name = "ctrans"

# Set random seed for reproducibility
init_random_seed(random_seed=42)

# =============================================================================
# Parse input arguments or set default paths
try:
    path2input = sys.argv[1]
    path2output = sys.argv[2]
    slide_name = sys.argv[3]
except:
    path2input = '../../../input_data/slides/TCGA-OL-A5S0-01Z-00-DX1.49A7AC9D-C186-406C-BA67-2D73DE82E13B.svs'
    path2output = '../../../output_data/features_test/'
    slide_name = 'OL-A5S0-01Z-00-DX1'

# =============================================================================
# Configuration settings
unit_tile_size = "px"  # Options: 'px' for pixels or 'micron' for micrometers
st_tile_size = 224
mag_assumed = 40  # Default assumed magnification
evaluate_edge = True
evaluate_color = False
save_tile_file = False
extract_pretrained_features = True

mag_selected = 20
mask_downsampling = 16
batch_size = 128

# Output directory for mask
path2mask = f"{path2output}mask/"
os.makedirs(path2mask, exist_ok=True)

# =============================================================================
# Read the slide using OpenSlide
slide = openslide.OpenSlide(f"{path2input}")

# Get slide properties and magnification
try:
    MPP_X = slide.properties[openslide.PROPERTY_NAME_MPP_X]
except:
    MPP_X = 0.2485  # Default microns per pixel value

MPP_X = float(MPP_X)

# =============================================================================
# Define functions to calculate tile size

def tile_size_from_pxl(MPP_X, st_tile_size):
    target = st_tile_size * 0.71 / MPP_X
    return round(target / 16) * 16  # Round to nearest multiple of 16

def tile_size_from_micron(MPP_X, st_tile_size):
    target = st_tile_size / MPP_X
    return round(target / 16) * 16  # Round to nearest multiple of 16

# Determine tile size based on the unit
if unit_tile_size == "px":
    tile_size0 = tile_size_from_pxl(MPP_X, st_tile_size)
elif unit_tile_size == "micron":
    tile_size0 = tile_size_from_micron(MPP_X, st_tile_size)
else:
    print("Error: unit_tile_size should be 'px' for pixels or 'micron' for micrometers")
    sys.exit()

# =============================================================================
# Get magnification levels
if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
    mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    mag_original = mag_max
    print("Magnification Max:", mag_max)
else:
    print(f"[WARNING] Magnification not found, assuming: {mag_assumed}")
    mag_max = mag_assumed
    mag_original = 0

# Determine downsampling level
downsampling = int(int(mag_max) / mag_selected)
print(f"Downsampling factor: {downsampling}")

# =============================================================================
# Get slide dimensions and calculate tiling parameters
px0, py0 = slide.level_dimensions[0]
tile_size = int(tile_size0 / downsampling)
print(f"Slide dimensions (level 0): {px0} x {py0}, Tile size: {tile_size}")

n_rows, n_cols = int(py0 / tile_size0), int(px0 / tile_size0)
n_tiles_total = n_rows * n_cols
print(f"Rows: {n_rows}, Columns: {n_cols}, Total tiles: {n_tiles_total}")

mask_tile_size = int(np.ceil(tile_size / mask_downsampling))

# =============================================================================
# Initialize empty masks
img_mask = np.full((n_rows * mask_tile_size, n_cols * mask_tile_size, 3), 255, dtype=np.uint8)
mask = np.full((n_rows * mask_tile_size, n_cols * mask_tile_size, 3), 255, dtype=np.uint8)

# =============================================================================
# Process each tile
tiles_list = []
tile_names_list = []
# -

for row in range(n_rows):
    print(f"Processing row: {row}/{n_rows}")
    for col in range(n_cols):
        tile = slide.read_region((col * tile_size0, row * tile_size0), level=0, size=[tile_size0, tile_size0])
        tile = tile.convert("RGB")

        if tile.size == (tile_size0, tile_size0):
            tile = tile.resize((tile_size, tile_size))
            mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))
            
            img_mask[row * mask_tile_size:(row + 1) * mask_tile_size, 
                     col * mask_tile_size:(col + 1) * mask_tile_size, :] = mask_tile

            # Evaluate tile
            if evaluate_tile(np.array(tile), edge_mag_thrsh=15, edge_fraction_thrsh=0.5):
                tile_norm = Image.fromarray(color_norm.transform(np.array(tile)))

                mask_tile_norm = np.array(tile_norm.resize((mask_tile_size, mask_tile_size)))
                mask[row * mask_tile_size:(row + 1) * mask_tile_size,
                     col * mask_tile_size:(col + 1) * mask_tile_size, :] = mask_tile_norm    

                x_min, y_min = col * tile_size0, row * tile_size0
                tile_name = f'{slide_name}_{x_min}_{y_min}'
                
                tiles_list.append(tile_norm)
                tile_names_list.append(tile_name)

# +
# =============================================================================
# Generate and save mask visualization
line_color = [0, 255, 0]

img_mask[:, ::mask_tile_size, :] = line_color
img_mask[::mask_tile_size, :, :] = line_color
mask[:, ::mask_tile_size, :] = line_color
mask[::mask_tile_size, :, :] = line_color

fig, ax = plt.subplots(1, 2, figsize=(40, 20))
ax[0].imshow(img_mask)
ax[1].imshow(mask)

ax[0].set_title(f"{slide_name}, Original Magnification: {mag_original}, Assumed: {mag_assumed}")
ax[1].set_title(f"Rows: {n_rows}, Cols: {n_cols}, Tiles: {n_tiles_total}, Selected: {len(tiles_list)}")

plt.tight_layout()

plt.savefig(f"{path2mask}{slide_name}.pdf", format="pdf", dpi=50)
plt.show()

# +
print("Completed processing.")

# =============================================================================
# Extract features and save as pickle
features = tiles2features(tiles_list, model_name, batch_size)
slide_file = f"{path2output}{slide_name}.pkl"

tile_features_list = list(zip(tile_names_list, features))

with open(slide_file, 'wb') as file:
    pickle.dump(tile_features_list, file)

print(f"Features saved to {slide_file}")
# -


