##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## @ Danh-Tai HOANG
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,ConcatDataset
import os,sys,time
from model_MLP import *
from utils import *

## check available device
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("device:", device)

init_random_seed(random_seed=42)

##================================================================================================
rna_type = "log"
print("rna_type:", rna_type)

##-------------------------------------------------
n_inputs = 512
n_hiddens = 512
dropout = 0.2
batch_size = 32
learning_rate = 0.0001           ## 0.001
print("dropout:", dropout)
print("batch_size:", batch_size)
print("learning_rate:", learning_rate)

project = sys.argv[1]
ik_fold = int(sys.argv[2])
il_fold = int(sys.argv[3])
i_gene_min = int(sys.argv[4])
i_gene_step = int(sys.argv[5])

max_epochs,patience = 500,50

print("ik_fold:", ik_fold)
print("il_fold:", il_fold)
print("i_gene_min:", i_gene_min)
print("i_gene_step:", i_gene_step)
print("max_epochs: {}, patience: {}".format(max_epochs, patience))

##-------------------------------------------------
path2features = f"../12AE/"
path2target = "../10metadata"
path2split = "../10metadata/"

gene_file = f"{path2target}{project}_genes.txt"
print("gene_file:", gene_file)

i_gene_max = int(i_gene_min + i_gene_step)
genes = np.loadtxt(gene_file, dtype="str")
genes = genes[i_gene_min:i_gene_max][:,0]
print("genes:", genes)
print("len(genes):", len(genes))

## create result directory
result_dir = "results/result_%s_%s_%s"%(ik_fold, il_fold, i_gene_min)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

##================================================================================================
train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split, rna_type, \
                                              ik_fold, il_fold, genes, project)

bias_init = torch.nn.Parameter(torch.Tensor(np.mean([sample[1].detach().cpu().numpy()\
                         for sample in train_set], axis=0)).to(device))
n_outputs = len(genes)

model = MLP_regression(n_inputs, n_hiddens, n_outputs, dropout, bias_init)
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

##================================================================================================
print(" ")
print(" --- fit --- ")
start_time = time.time()

model,train_loss,train_coef,train_slope,\
valid_loss,valid_coef,valid_slope,valid_labels,valid_preds = \
fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size)

print("fit -- completed -- time: {:.2f}".format(time.time() - start_time))

##================================================================================================
#print(" ")
#print(" --- analyze_result --- ")
start_time = time.time()

analyze_result(result_dir,genes,model,train_loss,train_coef,train_slope, \
               valid_loss,valid_coef,valid_slope,valid_labels, valid_preds, test_set)

print(f"analyze_result -- completed -- time: {(time.time() - start_time):.2f}s")
##================================================================================================

print("--- completed ---")
