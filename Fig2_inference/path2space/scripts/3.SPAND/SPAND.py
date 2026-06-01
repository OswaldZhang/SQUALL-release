# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: python/3.10
#     language: python
#     name: py3.10
# ---

# +
# Import necessary modules
import os  # Provides functions for interacting with the operating system
import sys  # Provides access to system-specific parameters and functions
import time  # Used for tracking time (though not explicitly used here)
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from libpysal import weights  # For creating spatial weights matrices
from esda import Moran  # For performing Moran's I analysis
from pysal.explore import esda  # Additional spatial data analysis tools (partially redundant import)
import pickle  # For loading and saving serialized Python objects

# Function to calculate Moran's I (Global) for a 2D gene expression array
def calculate_morans_i_from_df(values, positions_df):
    """
    Calculates Global Moran's I for a 2D spatial dataset.

    Parameters:
    - values: A Pandas Series containing the values to be analyzed (e.g., gene expression).
    - positions_df: A DataFrame with columns ['x', 'y'] representing spatial coordinates.

    Returns:
    - morans_i_global: The calculated Global Moran's I statistic.
    """
    # Combine values and positions into one DataFrame
    df = pd.concat([values, positions_df], axis=1)
    # Reshape the DataFrame into a 2D array based on 'x' and 'y' coordinates
    df = df.pivot(index='x', columns='y', values=df.columns[0])
    
    # Convert the reshaped DataFrame to a NumPy array and ensure it is of type float64
    gene_array = df.to_numpy().astype(np.float64)

    # Flatten the array and create a mask to identify non-NaN values
    flat_array = gene_array.flatten()
    nans_mask = ~np.isnan(flat_array)

    # Extract only the valid (non-NaN) values
    valid_array = flat_array[nans_mask]
    
    # Check if there are valid (non-NaN) values
    if len(valid_array) > 0:
        # Create a spatial weights matrix based on the 2D grid structure
        w_full = weights.lat2W(gene_array.shape[0], gene_array.shape[1], silence_warnings=True)

        # Filter the weights matrix to exclude NaN positions
        valid_ids = np.where(nans_mask)[0]
        valid_neighs = {i: [j for j in w_full.neighbors[i] if nans_mask[j]] for i in valid_ids}

        # Create a new spatial weights object for the valid points
        w_valid = weights.W(valid_neighs)

        # Calculate Global Moran's I statistic
        morans_i_global = esda.Moran(valid_array, w_valid).I
        return morans_i_global
    else:
        # If no valid values exist, return NaN
        return np.nan

# Get command-line arguments
path2input = sys.argv[1]  # Path to the input data file
gene = sys.argv[2]  # Gene to analyze

# Load spatial transcriptomics predictions for one slide from a serialized (pickled) DataFrame
df_preds = pd.read_pickle(f'{path2input}')

# Calculate Moran's I and normalize it by the mean of the gene's values
spand = calculate_morans_i_from_df(df_preds[gene], df_preds[['x', 'y']]) / df_preds[gene].mean()

# Print the normalized Moran's I value
print(spand)

