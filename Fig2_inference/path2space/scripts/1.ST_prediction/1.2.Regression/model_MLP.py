# +
# #!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch import nn
from tqdm import tqdm  # Progress bar for iterations
from utils import *  # Import custom utility functions

# Check the available device (GPU or CPU)
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#================================================================================================
class MLP_regression(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for regression tasks.
    This model consists of a single hidden layer followed by a dropout and a final output layer.
    """
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init):
        super(MLP_regression, self).__init__()

        # Define the first layer with optional dropout
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout)
        )
        
        # Define the second layer (output layer)
        self.layer1 = nn.Linear(n_hiddens, n_outputs)

        # Optionally initialize the bias of the last layer
        if bias_init is not None:
            self.layer1.bias = bias_init

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.layer0(x)  # Pass through the first layer
        x = self.layer1(x)  # Pass through the second layer
        x = torch.mean(x, dim=0)  # Take the mean across tiles
        return x  # Return predicted values

#------------------------------------------------------------------------------------------------
class MLP_regression_relu_two(nn.Module):
    """
    MLP with two layers and ReLU activation after each layer.
    """
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super(MLP_regression_relu_two, self).__init__()

        # First layer with ReLU activation
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Second layer with ReLU activation
        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU()
        )
        
        # Optional bias initialization
        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias = nn.Parameter(bias_init)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)  # Aggregate predictions (across tiles)
        return x

#------------------------------------------------------------------------------------------------
class MLP_regression_relu(nn.Module):
    """
    MLP with a single ReLU activation at the output layer.
    """
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init=None):
        super(MLP_regression_relu, self).__init__()

        # First layer without activation
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout)
        )
        
        # Second layer with ReLU activation
        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU()
        )
        
        # Optional bias initialization
        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias = nn.Parameter(bias_init)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)  # Aggregate predictions
        return x

#================================================================================================
def training_epoch(model, optimizer, train_set, batch_size):
    """
    Trains the model for one epoch using the training dataset.
    """
    model.train()
    loss_fn = nn.MSELoss()  # Mean Squared Error as the loss function

    n_slides_train = len(train_set)
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)  # Shuffle the training data

    loss_list = []
    labels = []
    preds = []
    for i_batch in range(0, n_slides_train, batch_size):
        n_slides_batch = min(batch_size, n_slides_train - i_batch)

        loss = 0  # Initialize batch loss
        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]
            x, y = train_set[idx]

            # Forward pass
            pred = model(x.float().to(device))
            loss += loss_fn(pred, y.float().to(device))  # Accumulate loss

            # Store labels and predictions
            labels.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

        loss /= n_slides_batch
        loss_list.append(loss.detach().cpu().numpy())

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    labels = np.array(labels)
    preds = np.array(preds)
    coef = compute_coefs(labels, preds)  # Compute R^2 coefficient

    return np.mean(loss_list), np.nanmean(coef)

#================================================================================================
def predict(model, valid_set):
    """
    Predicts the output and evaluates the performance on the validation set.
    """
    model.eval()
    loss_fn = nn.MSELoss()

    labels = []
    preds = []
    loss_list = []

    with torch.no_grad():  # Disable gradient computation
        for x, y in valid_set:
            pred = model(x.float().to(device))  # Forward pass
            loss = loss_fn(pred, y.float().to(device))
            loss_list.append(loss.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

    labels = np.array(labels)
    preds = np.array(preds)
    coef = compute_coefs(labels, preds)  # Compute R^2 coefficient

    return np.mean(loss_list), np.nanmean(coef), labels, preds

#================================================================================================
def fit(model, optimizer, train_set, valid_set, max_epochs, patience, batch_size):
    """
    Fits the model using training and validation datasets with early stopping.
    """
    train_loss_list = []
    train_coef_list = []

    valid_loss_list = []
    valid_coef_list = []

    epoch_since_best = 0
    valid_coef_old = -1.0
    for e in range(max_epochs):
        epoch_since_best += 1

        # Train for one epoch
        train_loss, train_coef = training_epoch(model, optimizer, train_set, batch_size)

        # Validate the model
        valid_loss, valid_coef, valid_labels, valid_preds = predict(model, valid_set)

        print(f"{e}, train_loss: {train_loss:.4f}, coef: {train_coef:.4f},\
            valid_loss: {valid_loss:.4f}, coef: {valid_coef:.4f}")

        train_loss_list.append(train_loss)
        train_coef_list.append(train_coef)
        valid_loss_list.append(valid_loss)
        valid_coef_list.append(valid_coef)

        # Check for improvement in validation coefficient
        if valid_coef > valid_coef_old:
            epoch_since_best = 0
            valid_coef_old = valid_coef

        # Early stopping condition
        if epoch_since_best == patience:
            print('Early stopping at epoch {}'.format(e + 1))
            break

    return model, train_loss_list, train_coef_list, valid_loss_list, valid_coef_list, valid_labels, valid_preds

#================================================================================================
def analyze_result(result_dir, genes, model, train_loss, train_coef, \
                   valid_loss, valid_coef, valid_labels, valid_preds, test_set):
    """
    Analyze and save model results, including predictions and coefficients.
    """
    torch.save(model.state_dict(), f"{result_dir}/model_trained.pth")  # Save model

    # Save training and validation losses
    train_valid_loss = np.array((train_loss, valid_loss, train_coef, valid_coef)).T
    np.savetxt(f"{result_dir}/train_valid_loss.txt", train_valid_loss, fmt="%.6f")

    # Evaluate on the test set
    test_loss, test_coef, test_labels, test_preds = predict(model, test_set)

    # Sort results based on validation and test set coefficients
    valid_coef1 = compute_coefs(valid_labels, valid_preds)
    test_coef1 = compute_coefs(test_labels, test_preds)

    np.savetxt(f"{result_dir}/coef.txt", np.array((valid_coef1, test_coef1)).T, fmt="%8s")

    # Save coefficients sorted by validation performance
    i1 = np.argsort(valid_coef1)[::-1]  # Sort indices in descending order of validation coefficients
    np.savetxt(f"{result_dir}/coef_sorted_based_valid.txt", 
               np.array((i1, genes[i1], valid_coef1[i1], test_coef1[i1])).T, 
               fmt="%8s %22s %8s %8s")

    # Save coefficients sorted by test performance
    i2 = np.argsort(test_coef1)[::-1]  # Sort indices in descending order of test coefficients
    np.savetxt(f"{result_dir}/coef_sorted_based_test.txt", 
               np.array((i2, genes[i2], valid_coef1[i2], test_coef1[i2])).T, 
               fmt="%8s %22s %8s %8s")

    # Save the validation and test predictions/labels for further analysis
    np.savetxt(f"{result_dir}/valid_labels.txt", valid_labels, fmt="%.8f")
    np.savetxt(f"{result_dir}/valid_preds.txt", valid_preds, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_labels.txt", test_labels, fmt="%.8f")
    np.savetxt(f"{result_dir}/test_preds.txt", test_preds, fmt="%.8f")

    ##-----------------------------------------------------------------------
    ## Visualization Section

    nx, ny = 2, 2  # Grid dimensions for the plots
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 4, ny * 3))

    # 1st column: Training and validation loss
    ax[0, 0].plot(train_loss, 'k--', label="train")
    ax[0, 0].plot(valid_loss, 'b-', label="valid")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].legend()

    # 2nd column: Training and validation coefficients
    ax[1, 0].plot(train_coef, 'k--', label="train")
    ax[1, 0].plot(valid_coef, 'b-', label="valid")
    ax[1, 0].set_ylabel("coef")
    ax[1, 0].legend()

    # Additional formatting for x-axis labels
    for j in range(ny):
        ax[j, 0].set_xlabel("n_epochs")

    ## 2nd column: Test set predictions vs labels
    i = 0  # Index for the best-performing gene
    label = test_labels[:, i2[i]]
    pred = test_preds[:, i2[i]]

    ax[0, 1].plot(label, pred, "o", label="test")
    ax[0, 1].plot([min(label), max(label)], [min(label), max(label)], "--")
    ax[0, 1].set_xlabel("label")
    ax[0, 1].set_ylabel("pred")
    ax[0, 1].set_title(f"{genes[i2[i]]}, R={test_coef1[i2[i]]:3.2f}")

    # Histogram of test coefficients
    bins = np.linspace(min(test_coef1), max(test_coef1), 10, endpoint=False)
    ax[1, 1].hist(test_coef1, bins, histtype='bar', rwidth=0.8)
    ax[1, 1].set_xlabel("coef")
    ax[1, 1].set_ylabel("#genes")

    # Adjust layout and save loss and coefficient plots
    plt.tight_layout(h_pad=1, w_pad=0.5)
    plt.savefig(f"{result_dir}/loss.pdf", format='pdf', dpi=50)

    ##---------------------------

    ## Best predictable genes: Scatter plots for the top genes
    nx, ny = 5, 2  # Grid dimensions for scatter plots
    fig, ax = plt.subplots(ny, nx, figsize=(nx * 4, ny * 3))

    ij = 0  # Counter for the top genes
    for j in range(ny):
        for i in range(nx):
            # Extract labels and predictions for the current gene
            label = test_labels[:, i2[ij]]
            pred = test_preds[:, i2[ij]]

            # Scatter plot of label vs prediction
            ax[j, i].plot(label, pred, "o", label="test")
            ax[j, i].plot([min(label), max(label)], [min(label), max(label)], "--")
            ax[j, i].set_xlabel("label")
            ax[j, i].set_ylabel("pred")
            ax[j, i].set_title(f"{genes[i2[ij]]}, R={test_coef1[i2[ij]]:3.2f}")

            ij += 1  # Move to the next gene

    # Final layout adjustments and saving the plot
    plt.tight_layout(h_pad=1, w_pad=0.5)
    plt.savefig(f"{result_dir}/best_preds.pdf", format='pdf', dpi=50)

#================================================================================================

# -


