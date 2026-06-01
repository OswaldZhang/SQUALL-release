# +
## import numpy as np
import pandas as pd
import torch
#from torch import optim
from torch.utils.data import DataLoader, ConcatDataset  # Utilities for data handling
import os, sys, time

from model_MLP import *  # Custom MLP model implementation
from utils import *  # Custom utility functions

# Check the available device (GPU or CPU)
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

# Initialize the random seed for reproducibility
init_random_seed(random_seed=42)
# -

##-------------------------------------------------
# Define model and training hyperparameters
n_inputs = 768  # Number of input features (e.g., from embeddings)
n_hiddens = 768  # Number of hidden units in the MLP
dropout = 0.2  # Dropout rate to prevent overfitting
batch_size = 32  # Batch size for training
learning_rate = 0.0001  # Learning rate for the optimizer

# +
try:
    path2input = sys.argv[1]  # Input directory path
    path2output = sys.argv[2]  # Output directory path
    ik_fold = int(sys.argv[3])  # Outer k-fold index
    il_fold = int(sys.argv[4])  # Inner k-fold index
    max_epochs= int(sys.argv[5])

    
except:
    path2input = '../../../input_data/'
    path2output = '../../../output_data/'
    ik_fold = 0
    il_fold = 0
    max_epochs = 50
# -

patience = max_epochs//10  # Maximum epochs and patience for early stopping

# +
##-------------------------------------------------
# Define paths to the required data files
path2features = f"{path2output}features.pkl"  # Path to precomputed features
path2target = f'{path2input}ge_actual.pkl'  # Path to target gene expression values
path2split = f"{path2input}train_valid_test_idx.npz"  # Path to train/valid/test splits
path2gene_file = f"{path2input}gene_file.pkl"  # Path to the file containing gene names

# Load the gene names within the specified range
genes = pd.read_pickle(path2gene_file)
genes = genes["gene"].values

# Print the selected genes and their count
print("genes:", genes)
print("len(genes):", len(genes))

## Create a result directory for saving outputs
result_dir = f"{path2output}/result_{ik_fold}_{il_fold}"
print(result_dir)
os.makedirs(result_dir, exist_ok=True)  # Ensure the directory exists

##================================================================================================
# Load datasets for training, validation, and testing
train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split, 
                                              ik_fold, il_fold, genes)


# Compute the initial bias for the model's output layer based on the mean of the training labels
bias_init = torch.nn.Parameter(torch.Tensor(np.mean([sample[1].detach().cpu().numpy() 
                         for sample in train_set], axis=0)).to(device))

# Determine the number of output neurons based on the number of genes
n_outputs = len(genes)

# Initialize the MLP model with ReLU activation and two layers
model = MLP_regression_relu_two(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
model.to(device)  # Move the model to the specified device (GPU/CPU)

# Define the optimizer (Adam optimizer) with the specified learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##================================================================================================
print(" ")
print(" --- fit --- ")

# Train the model using the `fit` function with early stopping
model, train_loss, train_coef, \
valid_loss, valid_coef, valid_labels, valid_preds = \
    fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size)

# +
# Analyze the results, save outputs, and generate visualizations
analyze_result(result_dir, genes, model, train_loss, train_coef, 
               valid_loss, valid_coef, valid_labels, valid_preds, test_set)

print("--- completed ---")
# -


