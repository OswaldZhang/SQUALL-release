#!/usr/bin/env python
# +
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
import pickle
import torch
# Custom utility imports
from func.utils_preprocessing import *  # Custom preprocessing utilities
import func.utils_color_norm as utils_color_norm  # Custom color normalization functions

# Set device to GPU if available, else CPU
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

# Set a random seed for reproducibility
init_random_seed(random_seed=42)

##======================================================================================================

# Read input and output paths and slide index from command-line arguments
path2input = sys.argv[1]
path2output = sys.argv[2]
i_slide = int(sys.argv[3])

print(f"i_slide: {i_slide}")

# Force device to CPU for this processing
device = torch.device('cpu')

# Set the batch size for processing
batch_size = 128

# Initialize the color normalization using Macenko's method
color_normalizer = utils_color_norm.macenko_normalizer()

# Define directories for slides, spots, features, and masks
path_to_slides = f'{path2input}slides/'
path_to_spots = f'{path2input}spots/'
path_to_features = f"{path2output}features/"
path_to_masks = f"{path2output}mask/"

# Thresholds for edge evaluation
EDGE_MAG_THRESHOLD = 15
EDGE_FRACTION_THRESHOLD = 0.5

# Define tile size for processing
TILE_SIZE = 224

# Create output directories if they do not exist
os.makedirs(path_to_features, exist_ok=True)
os.makedirs(path_to_masks, exist_ok=True)

# Load metadata for slides
metadata = pd.read_csv(f"{path2input}metadata.csv")

# Extract slide names and corresponding file names
slide_names = metadata['slide_name'].values
slide_file_names = metadata['slide_file_name'].values

# Process the slide specified by `i_slide`
slide_name = slide_names[i_slide]
slide_file_name = slide_file_names[i_slide]

# Read the slide image
slide_image_path = f'{path_to_slides}{slide_file_name}.tif'
slide_img = skimage.io.imread(slide_image_path)

# Read spot data for the slide
spots_path = f'{path_to_spots}{slide_file_name}.csv'
spots = pd.read_csv(spots_path)

# Filter selected spots and sort by x, y coordinates
spots = spots[spots['selected'] == 1].sort_values(['x', 'y'])

# Extract tile and grid coordinates
tile_coords = spots[['pixel_x', 'pixel_y']].round().astype(int).values
grid_coords = spots[['x', 'y']].values

# Calculate the number of rows and columns in the grid
n_rows, n_cols = spots['x'].max() - spots['x'].min() - 1, spots['y'].max() - spots['y'].min() - 1

# Initialize masks for the slide
img_mask = np.full(slide_img.shape, 255, dtype=np.uint8)
processed_mask = np.full(slide_img.shape, 255, dtype=np.uint8)

# Lists to store processed tiles and their names
tiles_list = []
tiles_names = []

# Loop through each grid coordinate and tile coordinate
for i, (grid_coord, tile_coord) in enumerate(zip(grid_coords, tile_coords)):
    grid_x, grid_y = grid_coord
    pixel_x, pixel_y = tile_coord
    spot_name = f'{slide_name}_{grid_x}x{grid_y}'

    # Calculate start and end coordinates for the tile
    start_x = pixel_x - TILE_SIZE // 2
    end_x = pixel_x + TILE_SIZE // 2
    start_y = pixel_y - TILE_SIZE // 2
    end_y = pixel_y + TILE_SIZE // 2

    # Skip tiles that are out of bounds
    if (start_x < 0 or end_x > slide_img.shape[0] or
        start_y < 0 or end_y > slide_img.shape[1]):
        continue

    # Extract the tile from the slide image
    tile = slide_img[start_x:end_x, start_y:end_y, :]
    img_mask[start_x:end_x, start_y:end_y, :] = tile

    # Evaluate the tile based on edge thresholds
    if evaluate_tile(tile, EDGE_MAG_THRESHOLD, EDGE_FRACTION_THRESHOLD):
        # Apply color normalization
        normalized_tile = Image.fromarray(color_normalizer.transform(tile))
        tiles_list.append(normalized_tile)
        tiles_names.append(spot_name)
        processed_mask[start_x:end_x, start_y:end_y, :] = np.array(normalized_tile)

# Add grid lines to masks
line_color = [0, 255, 0]
img_mask[:, ::TILE_SIZE, :] = line_color
img_mask[::TILE_SIZE, :, :] = line_color
processed_mask[:, ::TILE_SIZE, :] = line_color
processed_mask[::TILE_SIZE, :, :] = line_color

# Display masks for visualization
fig, axes = plt.subplots(1, 2, figsize=(40, 20))
axes[0].imshow(img_mask)
axes[1].imshow(processed_mask)
axes[0].set_title(slide_name)
plt.tight_layout(h_pad=0.4, w_pad=0.5)

# Save processed mask to a PDF
plt.savefig(f"{path_to_masks}{slide_name}.pdf", format="pdf", dpi=50)
plt.show()
plt.close()

# Extract features using a pre-trained model
features = tiles2features(tiles_list, "ctrans", batch_size)

# Save extracted features to pickle files
for tile_name, tile_features in zip(tiles_names, features):
    with open(f"{path_to_features}{tile_name}.pkl", 'wb') as file:
        pickle.dump(tile_features, file)
