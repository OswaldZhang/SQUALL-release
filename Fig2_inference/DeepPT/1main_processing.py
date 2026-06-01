##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## @ Danh-Tai HOANG
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,platform
import openslide
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

from utils_preprocessing import *
import utils_color_norm
color_norm = utils_color_norm.macenko_normalizer()

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

init_random_seed(random_seed=42)
##======================================================================================================

path2storage = sys.argv[1]
project = sys.argv[2]
i_slide = int(sys.argv[3])

print(f"project: {project}, i_slide: {i_slide}")

mag_assumed = 40
evaluate_edge = True
evaluate_color = False
save_tile_file = False
extract_pretrained_features = True

mag_selected = 20
tile_size = 512
mask_downsampling = 16

mask_tile_size = int(np.ceil(tile_size/mask_downsampling))
#print("mask_tile_size:", mask_tile_size)

##---------------------------------------
path2slide = path2storage + f"{project}_slides_data/"
print("path2slide:", path2slide)
path2meta = "../10metadata/"
    
path2mask = f"{project}_mask/"
path2features = f"{project}_features/"

#metadata = pd.read_csv(f"../10match_slide_rna/{project}_slide_matched.csv")
metadata = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")

## evaluate tile
edge_mag_thrsh = 15
edge_fraction_thrsh = 0.5

##--------------------------
if not os.path.exists(path2mask):
    os.makedirs(path2mask)

if extract_pretrained_features:
    if not os.path.exists(path2features):
        os.makedirs(path2features)

##======================================================================================================
slide_file_names = metadata.slide_file_name.values
slide_names = metadata.slide_name.values

start_time = time.time()

slide_file_name = slide_file_names[i_slide]
slide_name = slide_names[i_slide]
print(f"slide_file_name: {slide_file_name}, slide_name: {slide_name}")

if save_tile_file:
    ## create tile_folder:
    tile_folder = f"{project}_tiles/" + slide_name
    print(f"tile_folder: {tile_folder}")

    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)

##====================================================================================================== 
slide = openslide.OpenSlide(f"{path2slide}{slide_file_name}")

## magnification max
if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
    mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    print("mag_max:", mag_max)
    mag_original = mag_max
else:
    print("[WARNING] mag not found, assuming: {mag_assumed}")
    mag_max = mag_assumed
    mag_original = 0

## downsample_level
downsampling = int(int(mag_max)/mag_selected)
print(f"downsampling: {downsampling}")

##====================================================================================================== 
## slide size at largest level (level=0)
px0, py0 = slide.level_dimensions[0]
tile_size0 = int(tile_size*downsampling)
print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

n_rows,n_cols = int(py0/tile_size0), int(px0/tile_size0)
print(f"n_rows: {n_rows}, n_cols: {n_cols}")

n_tiles_total = n_rows*n_cols
print(f"n_tiles_total: {n_tiles_total}")

##====================================================================================================== 
img_mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)
mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)

i_tile = 0
tiles_list = []
for row in range(n_rows):
    print(f"row: {row}/{n_rows}")
    for col in range(n_cols):
        
        tile = slide.read_region((col*tile_size0, row*tile_size0),\
                                 level=0, size=[tile_size0, tile_size0]) ## RGBA image
        tile = tile.convert("RGB")
        
        if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:
            # downsample to target tile size
            tile = tile.resize((tile_size, tile_size))

            mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))
            
            img_mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile

            tile = np.array(tile)
            #print(tile.shape)

            ## evaluate tile
            select = evaluate_tile(tile, edge_mag_thrsh, edge_fraction_thrsh)

            if select == 1:
                ## 2022.09.08: color normalization:
                tile_norm = Image.fromarray(color_norm.transform(tile))

                mask_tile_norm = np.array(tile_norm.resize((mask_tile_size, mask_tile_size)))

                mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile_norm    

                #tiles_list.append(np.array(tile_norm).astype(np.uint8))
                tiles_list.append(tile_norm)

                if save_tile_file:
                    tile_name = "tile_" + str(row).zfill(5)+"_" + str(col).zfill(5) + "_" \
                             + str(i_tile).zfill(5) + "_" + str(downsampling).zfill(3)

                    tile_norm.save(f"{tile_folder}/{tile_name}.png")

        i_tile += 1

##====================================================================================================== 
## plot: draw color lines on the mask
line_color = [0,255,0]

n_tiles = len(tiles_list)

img_mask[:,::mask_tile_size,:] = line_color
img_mask[::mask_tile_size,:,:] = line_color
mask[:,::mask_tile_size,:] = line_color
mask[::mask_tile_size,:,:] = line_color

fig, ax = plt.subplots(1,2,figsize=(40,20))
ax[0].imshow(img_mask)
ax[1].imshow(mask)

ax[0].set_title(f"{slide_name}, mag_original: {mag_original}, mag_assumed: {mag_assumed}")
ax[1].set_title(f"n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}")

plt.tight_layout(h_pad=0.4, w_pad=0.5)
plt.savefig(f"{path2mask}{slide_name}.pdf", format="pdf", dpi=50)
plt.close()

img_mask = 0 ; mask = 0

print("completed cleaning")

##======================================================================================================
##======================================================================================================
## load ResNet pretrained model
model = Feature_Extraction(model_type="load_from_saved_file")
model.to(device)
model.load_state_dict(torch.load("../ResNet50_IMAGENET1K_V2.pt",map_location=device))
model.eval()

batch_size = 64
##----------
## resize:
torch_resize = transforms.Resize(224)

tiles_list_resized = []
for i in range(n_tiles):
    tiles_list_resized.append(torch_resize(tiles_list[i]))
tiles_list = tiles_list_resized

##----------
## normalize by ImageNet mean and std:
data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])
                                    ])

##----------------------------------------
def extract_features_from_tiles(tiles_list):

    ## transfor to torch and normalize
    tiles = []
    for i in range(n_tiles):
        tiles.append(data_transform(tiles_list[i]).unsqueeze(0))
    tiles = torch.cat(tiles, dim=0)
    print("tiles.shape:", tiles.shape)   ## [n_tiles, 3, 224, 224]
    #tiles_list = 0

    ##------------------------------------
    ## extract feature from tile image
    features = []
    for idx_start in range(0, n_tiles, batch_size):
        idx_end = idx_start + min(batch_size, n_tiles - idx_start)

        feature = model(tiles[idx_start:idx_end])
        
        features.append(feature.detach().cpu().numpy())

    features = np.concatenate(features)

    return features

##----------------------------------------
if extract_pretrained_features:
    features = extract_features_from_tiles(tiles_list)
    print("features.shape:", features.shape)
    np.save(f"{path2features}{slide_name}.npy", features)


##======================================================================================================
##======================================================================================================
print(f"finished -- i_slide: {i_slide}, total time: {int(time.time() - start_time)}")


