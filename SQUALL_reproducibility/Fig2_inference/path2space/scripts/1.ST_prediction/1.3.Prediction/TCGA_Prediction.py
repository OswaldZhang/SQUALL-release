# +
from torch.utils.data import Dataset
from torch import nn
import torch
import numpy as np
import torch
import pandas as pd

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

class SlideDataset(Dataset):
    """ A Dataset class for holding slide data """

    def __init__(self, features):
        self.features = features
        self.dim = self.features[0][1].shape[0]  # assuming 768

    def __getitem__(self, index):
        # Retrieve the sample and add a batch dimension
        sample = torch.Tensor(self.features[index][1]).float().unsqueeze(0)
        return sample

    def __len__(self):
        return len(self.features)

class MLP_regression_ReLUs_in_both_layers(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super(MLP_regression_ReLUs_in_both_layers, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU()
        )
        
        # Custom bias initialization for self.layer1
        if bias_init is not None:
            self.layer1[0].bias = bias_init

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)
        return x

def predict(model, dataset, device):
    """Function to make prediction using model on the given dataset."""
    
    model.eval()  # Set the model to evaluation mode
    preds = []
    with torch.no_grad():  # No need to track gradients
        for x in dataset:
            # Forward pass through the model
            y = model(x.float().to(device))
            # Ensure y is 2D (1, num_columns)
            if y.dim() == 1:
                y = y.unsqueeze(0)
            # Detach the output, move it to CPU, and convert to numpy array
            preds.append(y.detach().cpu().numpy())

    # Concatenate along axis 0 to stack the predictions vertically
    return np.concatenate(preds, axis=0)


# -

def predict_st_from_trained_model(ik_folds, il_folds, 
                                  features_list, path2models, genes, device = device,
                                  n_inputs=768, n_hiddens=768):
    """
    Train and predict using MLP regression with dropout, then aggregate the results.

    Parameters:
    n_inputs (int): Number of input features.
    n_hiddens (int): Number of hidden units in the MLP.
    ik_folds (list): Outer cross-validation folds.
    il_folds (list): Inner cross-validation folds.
    features_list (list): List of feature sets.
    dataset (torch.utils.data.Dataset): Dataset for prediction.
    path2models (str): Path to the directory where trained models are stored.
    genes (pd.DataFrame): DataFrame containing gene information.
    device (torch.device): PyTorch device (e.g., 'cpu' or 'cuda').

    Returns:
    pd.DataFrame: DataFrame containing the mean predictions.
    """
    dataset = SlideDataset(features_list)
    n_genes = len(genes)
    # Initialize the model
    model = MLP_regression_ReLUs_in_both_layers(n_inputs=n_inputs,
                                                n_hiddens=n_hiddens, 
                                                n_outputs=n_genes,
                                                dropout=0.2, bias_init=None)

    # Initialize storage for predictions across ik folds
    preds_ik = np.zeros((len(ik_folds), len(features_list), n_genes))

    # Loop through outer folds (ik_folds)
    for ik_fold in ik_folds:
        print(f"\nik_fold: {ik_fold}")
        preds_il = np.zeros((len(il_folds), len(features_list), n_genes))

        # Loop through inner folds (il_folds)
        for il_fold in il_folds:
            path2model = f"{path2models}result_{ik_fold}_{il_fold}/model_trained.pth"
            
            # Load trained model weights
            state_dict = torch.load(path2model, map_location=device) #, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            
            # Perform predictions
            pred = predict(model, dataset, device)
            preds_il[il_fold, :, :] = pred

        # Compute mean across inner folds
        preds_ik[ik_fold, :, :] = np.mean(preds_il, axis=0)

    # Calculate and save mean predictions across outer folds
    preds_mean = np.mean(preds_ik, axis=0)

    # Extract patient/sample names from features_list
    pt_names = [x for x, y in features_list]

    # Create DataFrame for predictions
    df_pred = pd.DataFrame(preds_mean, 
                           columns=genes, 
                           index=pt_names)

    return df_pred



