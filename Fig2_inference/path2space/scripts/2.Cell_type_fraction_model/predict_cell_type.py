# +
import joblib
import numpy as np
import pandas as pd

def predict_cell_type_frac(path2inputs, X, num_folds=5, cell_types=['TILs', 'Stromal', 'Epithelial']):
    """
    Aggregates predictions from multiple folds of a trained model.

    Parameters:
    - path2inputs (str): Path to the directory containing model and feature files.
    - X (pd.DataFrame): Input feature dataset.
    - num_folds (int): Number of folds (default is 5).
    - cell_types (list): List of target cell types for prediction.

    Returns:
    - pd.DataFrame: DataFrame containing the averaged predictions across folds.
    """
    
    if cell_types is None:
        cell_types = ['TILs', 'Stromal', 'Epithelial']  # Default cell types

    predictions = []  # Initialize an array to store predictions

    for fold in range(num_folds):
        # Load the best model for the current fold
        model_path = f"{path2inputs}/model_fold_{fold}.joblib"
        best_model = joblib.load(model_path)

        # Load the relevant features for the current fold
        feature_path = f"{path2inputs}/best_features_fold_{fold}.txt"
        with open(feature_path, 'r') as f:
            best_features = [line.strip() for line in f.readlines()]

        # Subset the input data to only include the selected features
        X_subset = X.loc[:, best_features]

        # Predict using the current fold's model
        y_pred = best_model.predict(X_subset)

        # Append the predictions to the list
        predictions.append(y_pred)

    # Calculate the mean prediction across all folds
    mean_prediction = np.mean(predictions, axis=0)

    # Prepare the results DataFrame
    df_mean_pred = pd.DataFrame(mean_prediction, index=X.index,
                                columns=[f'{x}' for x in cell_types])
    
    return df_mean_pred

# -


