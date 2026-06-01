##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## @ Danh-Tai HOANG
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset,Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.stats.multitest as smt
##===================================================================================================
#### Build dataset
class SlideRNADataset(Dataset):
    ## input: features_list[n_slides](slide_name, features[n_tiles,512])
    ## target[n_slides, n_genes]
    
    def __init__(self, features, targets):
        
        self.features = features
        self.targets = targets            ## gene_expression_values
        self.dim = self.features[0][1].shape[1] ## 512

    def __getitem__(self, index):
        #sample = torch.Tensor(self.features[index][1]).float().t()
        sample = torch.Tensor(self.features[index][1]).float()
        target = torch.Tensor(self.targets[index]).float()
        
        return sample, target

    def __len__(self):
        return len(self.features)

##===================================================================================================
def load_dataset(path2features, path2target, path2split, rna_type, ik_fold, il_fold, genes, project):
    ## load image feature
    features_file = f"{project}_features_AE.npy"
    print(f"features_file: {features_file}")

    features = np.load(f"{path2features}{features_file}", allow_pickle=True)
    print("len(features):", len(features))    

    ## load RNA
    rna_file = f"{path2target}{project}_{rna_type}.csv"
    print(f"rna_file: {rna_file}")

    df = pd.read_csv(rna_file, index_col=None, usecols=genes)[genes]
    targets = df[genes].values

    ## load_train_valid_test_idx:
    train_valid_test_idx = np.load(f"{path2split}{project}_train_valid_test_idx.npz", allow_pickle=True)

    train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    test_idx = train_valid_test_idx["test_idx"][ik_fold]

    ## shuffle training set
    #np.random.shuffle(train_idx)

    dataset = SlideRNADataset(features, targets)

    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, valid_set, test_set

##===================================================================================================
def compute_coefs(labels, preds):
    return np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])

def compute_slope(labels, preds):
    return np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])

def compute_coef_slope(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    
    return coef, slope

##------------------------------------------------------------------
## R and p_1side values
def pearson_r_and_p(label, pred):
    R,p = pearsonr(label, pred)
    if R> 0:
        p_1side = p/2.
    else:
        p_1side = 1-p/2

    return p_1side

##------------------------------------------------------------------
## number of genes with Holm-Sidak correlated p-val<0.05
def compute_coef_slope_padj(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    p_adj = smt.multipletests(p_value, alpha=0.05, method='hs', is_sorted=False, returnsorted=False)[1]

    return coef, slope, p_adj
##===================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##===================================================================================================


