# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset,Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.stats.multitest as smt
    



# %%
def compute_coefs(labels, preds, variance_threshold=1e-8):
    """
    Compute Pearson correlation coefficients for each feature while ignoring NaN values
    and ensuring that variance is greater than a small threshold.

    Parameters:
    labels (numpy.ndarray): Ground truth values.
    preds (numpy.ndarray): Predicted values.
    variance_threshold (float): Minimum variance threshold to compute correlation.

    Returns:
    numpy.ndarray: Correlation coefficients for each feature.
    """
    coefs = []
    for i in range(labels.shape[1]):
        # Mask to ignore NaN values in both labels and predictions
        valid_idx = ~np.isnan(labels[:, i]) & ~np.isnan(preds[:, i])

        # Check if there are at least two valid values to compute correlation
        if np.sum(valid_idx) > 1:
            # Check variance threshold to avoid division by zero issues in correlation
            label_var = np.var(labels[valid_idx, i])
            pred_var = np.var(preds[valid_idx, i])

            if label_var > variance_threshold and pred_var > variance_threshold:
                coef, _ = pearsonr(labels[valid_idx, i], preds[valid_idx, i])
            else:
                coef = np.nan  # Assign NaN if variance is too low
        else:
            coef = np.nan  # Assign NaN if not enough valid values

        coefs.append(coef)

    return np.array(coefs)


# %%
class SlideRNADataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.dim = self.features[0][1].shape[0]  # Assuming [n_tiles, 512]

    def __getitem__(self, index):
        # Get the sample and convert it to a float tensor
        sample = torch.Tensor(self.features[index][1]).float()
        
        # Add an extra dimension to make the shape [1, 768]
        sample = sample.unsqueeze(0)  # Now, sample has shape [1, 768]

        # Convert the target to a float tensor
        target = torch.Tensor(self.targets[index]).float()
        
        return sample, target

    def __len__(self):
        return len(self.features)

##===================================================================================================
def load_dataset(path2features, path2target, path2split, ik_fold, il_fold, genes):
    ## load image feature
    features = np.load(f"{path2features}", allow_pickle=True)
    print("len(features):", len(features))    

    ## load RNA
    rna_file = f"{path2target}"
    print(f"rna_file: {rna_file}")

    df = pd.read_pickle(rna_file)[genes]
    targets = df[genes].values

    ## load_train_valid_test_idx:
    train_valid_test_idx = np.load(f"{path2split}", allow_pickle=True)

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
# def compute_coefs(labels, preds):
#     return np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])

from numpy.polynomial import polynomial as P

def polyfit(x,y, deg = 1):
    c, stats = P.polyfit(x,y,deg,full=True)
    return c[1]



def compute_slope(labels, preds):
    return np.array([polyfit(labels[:,i], preds[:,i], 1) for i in range(labels.shape[1])])

def compute_coef_slope(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i]) for i in range(labels.shape[1])])
    slope = np.array([polyfit(labels[:,i], preds[:,i], 1) for i in range(labels.shape[1])])
    
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
    slope = np.array([polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    p_adj = smt.multipletests(p_value, alpha=0.05, method='hs', is_sorted=False, returnsorted=False)[1]

    return coef, slope, p_adj

def compute_coef_padj(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    p_adj = smt.multipletests(p_value, alpha=0.05, method='hs', is_sorted=False, returnsorted=False)[1]

    return coef, p_adj

##===================================================================================================
def norm_percentiles(y_train, y_test):
    ## mean and std over samples for each gene
    mean = np.mean(y_train,axis=0)
    std = np.std(y_train,axis=0)
    
    ## y_test[n_samples, n_genes]
    perc_test = np.zeros_like(y_test).astype(float)
    for i in range(y_test.shape[1]):
        ## for each gene
        perc_test[:,i] = norm.cdf(y_test[:,i], mean[i], std[i])
    
    return perc_test
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

# #===================================================================================================


