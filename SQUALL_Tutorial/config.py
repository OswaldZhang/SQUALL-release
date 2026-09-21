"""Configuration settings for the STORM analysis pipeline."""

import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Base directories
BASE_DIR = '/data200T/STORM'

# HMDB_DIR = os.path.join(BASE_DIR, 'hmdb')
# FEATURE_DIR = os.path.join(BASE_DIR, 'hmdb_feature_old')

HMDB_DIR = os.path.join(BASE_DIR, 'HMDB')
FEATURE_DIR = os.path.join(BASE_DIR, 'hmdb_feature_cluster_check/newBreastPt')

OUTPUT_DIR = os.path.join(BASE_DIR, 'hmdb_feature_cluster_check/combined_gm_tab20')

# Model and data paths for generating embeddings
MODEL_CONFIG_PATH = '/home/wyf/coding_test/storm/STORM_code/large_ddp_rpb_lowres_benchmark.yaml'
MODEL_CHECKPOINT_PATH = '/data200T/STORM/ckpt_HD/ckpt-epoch-300.pth'
GENE_TOKEN_PATH = '/home/wyf/coding_test/storm/gene_token_homologs.csv'
SAMPLE_LIST_PATH = '/home/wyf/coding_test/storm/STORM_code/functionDebuger/texture_mapping/histMol.xlsx'

# Offset file paths
OFFSET_CSV = '/home/wyf/coding_test/storm/STORM_code/functionDebuger/texture_mapping/histMol_final_updated.csv'
# OFFSET_JSON = '/home/wyf/coding_test/storm/STORM_code/functionDebuger/texture_mapping/offset.json'
OFFSET_JSON = '/home/wyf/coding_test/storm/STORM_code/functionDebuger/texture_mapping/hmidFiles/offset_updated2.json'

# Processing parameters
PATCH_SIZE = 224
STRIDE = 16
BATCH_SIZE = 16
TARGET_SIZE = (14, 14)

# Analysis parameters
N_CLUSTERS = 10
N_PCA_COMPONENTS = 10
N_NEIGHBORS = 6
N_POINTS = 10000
RANDOM_SEED = 0

# Clustering parameters
CLUSTERING_PARAMS = {
    'gm': {
        'n_components': N_CLUSTERS,
        'covariance_type': 'diag',
        'max_iter': 100,
        'n_init': 1,
        'reg_covar': 1e-4,
        'random_state': RANDOM_SEED
    }
}