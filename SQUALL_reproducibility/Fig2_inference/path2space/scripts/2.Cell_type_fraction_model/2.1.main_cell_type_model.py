# %%
# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import sys
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from collections import Counter

# Setting a random seed for reproducibility
np.random.seed(42)

# Define paths for input and output files
ik_fold = int(sys.argv[1])
path2input = sys.argv[2]
path2output = sys.argv[3]



# %%
os.makedirs(path2output, exist_ok=True)

# %%
path2genes = f"{path2input}gene_file.pkl"
split_file_path = f"{path2input}train_test_idx_split.npz"
path_to_features_labels = f"{path2input}features_labels.pkl"



# %%
# =============================================================================
# Function to split the dataset into training and testing sets
def split_data(split_file, X, y, ik_fold):
    """
    Splits data into training and testing sets based on indices.

    Parameters:
    split_file (str): Path to the file containing train-test indices.
    X (pd.DataFrame): Feature dataset.
    y (pd.DataFrame): Label dataset.
    ik_fold (int): Index of the fold for splitting.

    Returns:
    tuple: Training and testing sets for features and labels.
    """
    train_test_idx = np.load(f"{split_file}", allow_pickle=True)
    train_idx = train_test_idx["train_idx"][ik_fold]
    test_idx = train_test_idx["test_idx"][ik_fold]

    X_train, y_train = X.iloc[:,train_idx], y.iloc[:,train_idx]
    X_test, y_test = X.iloc[:,test_idx], y.iloc[:,test_idx]

    # Handle case when test data has only one sample
    if X_test.ndim == 1:
        X_test = X_test[np.newaxis, :]
        y_test = np.array([y_test])

    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test


# %%
# Function to select k best features using f_regression scores
def select_k_best_features_overall(X, y, k, cell_types):
    """
    Selects the top k features based on maximum scores across all targets.

    Parameters:
    X (pd.DataFrame): Feature dataset.
    y (pd.DataFrame): Label dataset.
    k (int): Number of features to select.
    cell_types (list): List of target columns.

    Returns:
    tuple: DataFrame with selected features and their column names.
    """
    n_features = X.shape[1]
    max_scores = np.zeros(n_features)

    # Compute scores for each target variable
    for col in cell_types:
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y.loc[:, col])
        max_scores = np.maximum(max_scores, selector.scores_)

    # Select top k features based on maximum scores
    top_k_indices = np.argsort(max_scores)[-k:]
    return X.iloc[:, top_k_indices], X.iloc[:, top_k_indices].columns

# =============================================================================
# Main script execution


k = 1000  # Number of top features to select
cell_types = ['TILs', 'Stromal', 'Epithelial']



# %%
# Load feature and label data
X = pd.read_pickle(path_to_features_labels)
genes = pd.read_pickle(path2genes)["gene"].values



# %%
# Separate features and labels
y = X.loc[:, cell_types].copy()

X = X.loc[:, genes].copy()



# %%
# Split data into train and test sets
X_train, y_train, X_test, y_test = split_data(split_file_path, 
                                              X.T, 
                                              y.T, 
                                              ik_fold)

# %%
# Select the best features
X_train, best_feat = select_k_best_features_overall(X_train.T, y_train.T, k, cell_types)
np.savetxt(f"{path2output}best_features_fold_{ik_fold}.txt", best_feat, fmt="%s")
X_test = X_test.T.loc[:, best_feat].copy()



# %%
# =============================================================================
# Hyperparameter grid and model training

# Define hyperparameter grid for MLPRegressor



# %%
hyper_parameters = {
    'mlpregressor__hidden_layer_sizes': [(32,)],
    'mlpregressor__activation': ['relu'],
    'mlpregressor__solver': ['adam'],
    'mlpregressor__alpha': [ 0.8],
    'mlpregressor__learning_rate_init': [0.001],
    'mlpregressor__learning_rate': ['constant'],
    'mlpregressor__max_iter': [1000],
}



# %%
# hyperparameters used in the manuscript:

# hyper_parameters = {
#     'mlpregressor__hidden_layer_sizes': [(32,), (64,), (128,), (256,), (512,),
#                                          (256, 128, 32, 16), (256, 128),
#                                          (512, 256), (512, 256, 128), 
#                                          (512, 256, 128, 32), (512, 256, 128, 32, 16)],
#     'mlpregressor__activation': ['tanh', 'relu'],
#     'mlpregressor__solver': ['sgd', 'adam'],
#     'mlpregressor__alpha': [0.0001, 0.001, 0.1, 0.5, 0.8],
#     'mlpregressor__learning_rate_init': [0.001, 0.0001, 0.00001],
#     'mlpregressor__learning_rate': ['constant', 'adaptive'],
#     'mlpregressor__max_iter': [1000000],
# }



# %%
# Define the model
mlp_model = MLPRegressor(early_stopping=True, validation_fraction=0.2, n_iter_no_change=50)

# Create the pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),  # Scale the data
    ('mlpregressor', mlp_model)  # Add the MLP regressor
])

# Perform grid search with cross-validation
grid = GridSearchCV(estimator=pipeline, param_grid=hyper_parameters, 
                    cv=5, n_jobs=-1, scoring='neg_mean_squared_error', refit=True)
grid_result = grid.fit(X_train, y_train.T)

# Save the best model
best_model = grid_result.best_estimator_
joblib.dump(best_model, f'{path2output}/model_fold_{ik_fold}.joblib')

# =============================================================================
# Predictions and result saving

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Prepare results DataFrame
df_pred = pd.DataFrame(y_pred, index=X_test.index, 
                       columns=[f'{x}_predicted' for x in cell_types])

label = y_test.copy().T
label.columns = [f'{x}_label' for x in cell_types]
label.index = X_test.index

df_pred = label.join(df_pred)

# Save the results
df_pred.to_pickle(f"{path2output}predictions_test_fold_{ik_fold}.pkl")


