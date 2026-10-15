##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Name: Danh-Tai HOANG
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import cv2
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50

##======================================================================================================
class Feature_Extraction(nn.Module):
    def __init__(self, model_type="load_from_saved_file"):
        super().__init__()

        if model_type == "load_from_internet":
            self.resnet = resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        elif model_type == "load_from_saved_file":
            self.resnet = resnet50(weights=None)
        else:
            print("cannot find model_type can only be load_from_internet or load_from_saved_file")

        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

##======================================================================================================
def evaluate_tile(img_np, edge_mag_thrsh, edge_fraction_thrsh):

    select = 1  ## initial value
    
    #img_np = np.array(img_RGB)
    tile_size = img_np.shape[0]
        
    ##---------------------------------------
    ## 0) exclude if edge_mag > 0.5
    img_gray=cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Remove noise using a Gaussian filter
    #img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)

    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    unique, counts = np.unique(mag, return_counts=True)

    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum()/(tile_size*tile_size)

    if edge_mag > edge_fraction_thrsh:
        select = 0
    
    return select
        
##================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
