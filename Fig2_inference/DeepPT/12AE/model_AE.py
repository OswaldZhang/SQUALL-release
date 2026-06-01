##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## @ Danh-Tai HOANG 
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import Subset,Dataset
import os,sys,time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#from tqdm import tqdm

##================================================================================================
##================================================================================================
class SlideDataset(Dataset):
    ## input: features_list[n_slides](slide_name, features[n_tiles,512])    
    def __init__(self, features):
        self.features = features
        self.dim = self.features[0][1].shape[1] ## 512

    def __getitem__(self, index):
        sample = torch.Tensor(self.features[index][1]).float()
        return sample

    def __len__(self):
        return len(self.features)

##================================================================================================
class AutoEncoder(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs,n_hiddens),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hiddens,n_outputs),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

##================================================================================================
def training_epoch(model, optimizer, train_set, batch_size, device):
    model.train()
    loss_fn = nn.MSELoss()

    n_slides_train = len(train_set)
    #print("n_slides_train:", n_slides_train)

    ## shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)

    loss_list = []
    for i_batch in range(0,n_slides_train, batch_size):    
        #print(i_slide)

        n_slides_batch = min(batch_size, n_slides_train - i_batch)
        #print(n_slides_batch)

        ##---------------------------
        ## for each batch
        loss = 0
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]

            x = train_set[idx]

            #print(x.shape)            ## [512, n_tiles]

            pred = model(x.float().to(device))       
            #print("pred.shape:", pred.shape)   ## [n_genes]

            loss += loss_fn(pred, x.float().to(device))

        loss /= n_slides_batch

        loss_list += [loss.detach().cpu().numpy()] ## add loss of each batch to a list

        ## reset gradients to zero
        optimizer.zero_grad()

        ## compute gradients
        loss.backward()

        ## update parameters using gradients
        optimizer.step()
    
    return np.mean(loss_list)

##================================================================================================
def predict(model, valid_set, device):    
    model.eval()
    loss_fn = nn.MSELoss()
    
    #labels = []
    #preds = []
    loss_list = []
    with torch.no_grad():
        for x in valid_set:      

            #labels += [x]
            pred = model(x.float().to(device))   ## y_pred = model(x)

            loss = loss_fn(pred, x.float().to(device))
            loss_list += [loss.detach().cpu().numpy()] ## convert to numpy

            #pred = nn.ReLU()(pred)                      ## y_pred
            #preds += [pred.detach().cpu().numpy()]      ## convert to numpy

    ## convert list to 2D array
    #labels = np.concatenate(labels)
    #preds = np.concatenate(preds)

    return np.mean(loss_list)

##================================================================================================
def fit(model, optimizer, train_set, test_set, max_epochs, batch_size, device):
    print(" ")
    print(" ----- fit ----- ")
    #### train_model
    start_time = time.time()
    
    train_loss_list = []
    test_loss_list = []

    for e in range(max_epochs):

        ## train
        train_loss = training_epoch(model, optimizer, train_set, batch_size, device)
        
        ## predict
        test_loss = predict(model, test_set, device)

        print(f"epoch: {e}/{max_epochs}, time: {int(time.time() - start_time)}s, \
            train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        
    ## train set
    #train_loss = predict(model, train_set, device)
    
    #return model, train_loss_list, train_labels, train_preds, test_loss_list, test_labels, test_preds
    return model, train_loss_list, test_loss_list

##================================================================================================
def features_compression(model,path2features, project, device):
    
    features_list = np.load(f"{path2features}_{project}/{project}_features.npy", allow_pickle=True)
    #print("len(features_list):", len(features_list))

    n_slides = len(features_list)
    slide_names = np.array([features_list[i][0] for i in range(n_slides)])
    #print("slide_names:", slide_names)
    
    ##-------------------------------
    model.eval()
    #features_AE2d = []
    with torch.no_grad():
        features_AE = []

        for i in range(n_slides):
            x = features_list[i][1]             
            #print(x.shape)  ## [n_tiles, 2048]
            y = model.encoder(torch.from_numpy(x).float().to(device))
            y = y.detach().cpu().numpy()
            #print(y.shape) ##[n_tiles, 512]

            features_AE.append((slide_names[i], y))

            #features_AE2d += [y]

    print("len(features_AE):", len(features_AE))

    return features_AE #, np.concatenate(features_AE2d)

##================================================================================================
def convert_features_list_to2d(features_list):
    ## out: features2d[n_tiles_total, 512]

    n_slides = len(features_list)
    print(f"n_slides: {n_slides}")
    features2d = []
    for i in range(n_slides):
        #print("i_slide:", i)
        features2d.append(features_list[i][1])    

    features2d = np.concatenate(features2d)
    
    print(f"features2d.shape: {features2d.shape}")

    return features2d

##================================================================================================
def analyze_result(features,features_AE):

    ## convert features_list to 2d:
    features2d = convert_features_list_to2d(features)
    features_AE2d = convert_features_list_to2d(features_AE)

    ##  
    median = np.median(features2d,axis=0)
    std = np.std(features2d,axis=0)

    median_ae = np.median(features_AE2d,axis=0)
    std_ae = np.std(features_AE2d,axis=0)

    ##------------
    nx,ny = 2,2
    fig, ax = plt.subplots(ny,nx,figsize=(nx*4,ny*3))

    ax[0,0].hist(median,bins=20,histtype='bar',color="lightblue",edgecolor="black",rwidth=0.85)
    ax[0,1].hist(std,bins=20,histtype='bar',color="lightblue",edgecolor="black",rwidth=0.85)

    ax[1,0].hist(median_ae,bins=20,histtype='bar',color="lightblue",edgecolor="black",rwidth=0.85)
    ax[1,1].hist(std_ae,bins=20,histtype='bar',color="lightblue",edgecolor="black",rwidth=0.85)

    ax[0,0].set_xlabel("ResNet feature median")
    ax[0,1].set_xlabel("ResNet feature std")

    ax[1,0].set_xlabel("AE feature median")
    ax[1,1].set_xlabel("AE feature std")

    ax[0,0].set_ylabel("number of features")
    ax[0,1].set_ylabel("number of features")
    ax[1,0].set_ylabel("number of features")
    ax[1,1].set_ylabel("number of features")

    plt.tight_layout(h_pad=1, w_pad= 1.0)
    plt.savefig(f"hist.pdf", format="pdf", dpi=50)
##================================================================================================
def project_dictionary(project_key):

    project_dict = {"BRCA": ["BRCA"], 
                    "Lung": ["LUAD", "LUSC"], 
                    "Kidney": ["KIRC", "KIRP", "KICH"],
                    "UCEC": ["UCEC"],
                    "Brain": ["LGG", "GBM"],
                    "Gastric": ["STAD", "ESCA"],
                    "Colorectal": ["COAD", "READ"],
                    "Skin": ["SKCM"],
                    "BC": ["BC"]}

    project_keys = list(project_dict.keys())

    if project_key in project_keys:
        projects = project_dict[project_key]
    else:
        projects = [project_key]

    return projects

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
