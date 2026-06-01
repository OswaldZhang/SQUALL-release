# +
import numpy as np
import cv2
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
from func.ctrans_model import CTransPath  # Custom model import

# Set the device to GPU if available; otherwise, CPU
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

# Force device to CPU for this script
device = torch.device('cpu')

# Function to apply transformations to a list of tiles
def tile_transform(tiles_list, data_mean, data_std):
    """
    Applies a series of transformations (resize, normalization, etc.) to a list of image tiles.
    """
    data_transform = transforms.Compose([
        transforms.Resize(224),  # Resize each tile to 224x224
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize(mean=data_mean, std=data_std)  # Normalize with given mean and std
    ])

    n_tiles = len(tiles_list)
    print("n_tiles:", n_tiles)

    tiles = []
    for i in range(n_tiles):
        # Apply transformation to each tile and add a batch dimension
        transformed_tile = data_transform(tiles_list[i]).unsqueeze(0).to(device)
        tiles.append(transformed_tile)

    # Concatenate all tiles into a single tensor
    tiles = torch.cat(tiles, dim=0)
    print("tiles.shape:", tiles.shape)

    return tiles

##======================================================================================================
class Feature_Extraction(nn.Module):
    """
    Feature extraction module using ResNet50.
    """
    def __init__(self, model_type="load_from_saved_file"):
        super().__init__()
        if model_type == "load_from_internet":
            # Load ResNet50 with pre-trained weights
            self.resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_type == "load_from_saved_file":
            # Initialize ResNet50 without pre-trained weights
            self.resnet = resnet50(weights=None)
        else:
            print("Invalid model_type: should be 'load_from_internet' or 'load_from_saved_file'")
        
    def forward(self, x):
        """
        Pass input through ResNet layers to extract features.
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten features for output
        return x

##======================================================================================================
def evaluate_tile(img_np, edge_mag_thrsh, edge_fraction_thrsh):
    """
    Evaluate if a tile meets certain edge-based criteria.
    """
    select = 1  # Initial selection status

    # Convert the RGB image to grayscale
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Compute Sobel gradients to detect edges
    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    # Compute the magnitude of edges
    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)
    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    # Calculate the fraction of pixels with edge magnitudes below the threshold
    unique, counts = np.unique(mag, return_counts=True)
    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum() / (img_np.shape[0] * img_np.shape[1])

    # If the fraction of weak edges is greater than the threshold, reject the tile
    if edge_mag > edge_fraction_thrsh:
        select = 0

    return select

##======================================================================================================
def init_random_seed(random_seed=42):
    """
    Set a consistent random seed for reproducibility.
    """
    np.random.seed(random_seed)  # Python RNG
    torch.manual_seed(random_seed)  # Torch RNG
    torch.cuda.manual_seed(random_seed)  # CUDA RNG
    torch.cuda.manual_seed_all(random_seed)  # All CUDA devices
    torch.backends.cudnn.deterministic = True  # Deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for consistent results

##======================================================================================================
def tiles2features(tiles_list, model_name, batch_size):
    """
    Extract features from tiles using a specified model.
    """
    # Load model based on the specified model_name
    if model_name == "vit":
        path2model = "../vit-base-patch16-224-in21k"  # Path to ViT model
        model = ViTModel.from_pretrained(path2model).to(device)
        data_mean = [0.5, 0.5, 0.5]
        data_std = [0.5, 0.5, 0.5]

    elif model_name == "dino":
        path2model = "../dino_vit_small_patch16_ep200.pt"  # Path to DINO model
        model = VisionTransformer(img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0).to(device)
        model.load_state_dict(torch.load(path2model, map_location=device))  # Load weights
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

    elif model_name == "ctrans":
        path2model = "/data/Ruppin_AI/st/11features_extraction_tai/ctranspath.pth"  # Path to CTransPath
        model = CTransPath(num_classes=0).to(device)
        model.load_state_dict(torch.load(path2model)['model'])  # Load weights
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

    model.eval()  # Set the model to evaluation mode

    # Transform tiles using the specified mean and standard deviation
    tiles = tile_transform(tiles_list, data_mean, data_std)

    # Extract features in batches
    n_tiles = tiles.shape[0]
    features = []
    for idx_start in range(0, n_tiles, batch_size):
        idx_end = min(idx_start + batch_size, n_tiles)
        with torch.no_grad():  # Disable gradient computation
            y = model(tiles[idx_start:idx_end])
            if model_name == "vit":
                y = y.last_hidden_state[:, 0]  # Extract CLS token for ViT

        features.append(y.detach().cpu().numpy())  # Convert features to NumPy

    features = np.concatenate(features)  # Combine all batches
    print("features.shape:", features.shape)
    return features

