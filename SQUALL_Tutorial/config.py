"""Portable settings for the SQUALL tutorial.

All bundled-resource paths are resolved relative to this file.  Users only need
to supply their own checkpoint location when running the embedding script.
"""

from pathlib import Path


TUTORIAL_DIR = Path(__file__).resolve().parent

# Bundled tutorial inputs
DATA_ROOT = TUTORIAL_DIR / "HMDB_sample"
MODEL_CONFIG_PATH = TUTORIAL_DIR / "config.yaml"
GENE_TOKEN_PATH = TUTORIAL_DIR / "gene_token_homologs.csv"
OFFSET_JSON = TUTORIAL_DIR / "offset.json"

# Generated files.  Keeping them in a dedicated directory avoids modifying the
# downloaded example data.
OUTPUT_DIR = TUTORIAL_DIR / "outputs"

# The repository does not include the trained checkpoint.  Put it here or pass
# another location with --checkpoint when running the embedding script.
MODEL_CHECKPOINT_PATH = TUTORIAL_DIR / "weights" / "SQUALL_lowres.pth"

# Embedding settings
PATCH_SIZE = 224
STRIDE = 16
BATCH_SIZE = 2
TARGET_SIZE = (14, 14)
RESOLUTION = 4.0
NUM_WORKERS = 0

# Clustering settings used by feature_clustering_squall_tutorial.py
N_POINTS = 10000
RANDOM_SEED = 0
N_PCA_COMPONENTS = 10
N_NEIGHBORS = 6
N_CLUSTERS = 10
