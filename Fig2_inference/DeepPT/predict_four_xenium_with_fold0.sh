#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT"
INPUT_DIR="${BASE_DIR}/EGNv1_tile_input_allgenes_224"
OUT_ROOT="${BASE_DIR}/outputs"
RESNET_WEIGHT="/lustre1/zxzeng/bwqin/STORM_main/clustering/EGN-main/v2/ResNet50_IMAGENET1K_V2.pt"

cd "${BASE_DIR}"

# ============================================================
# Basic check
# ============================================================

echo "============================================================"
echo "[CHECK] BASE_DIR     = ${BASE_DIR}"
echo "[CHECK] INPUT_DIR    = ${INPUT_DIR}"
echo "[CHECK] OUT_ROOT     = ${OUT_ROOT}"
echo "[CHECK] RESNET       = ${RESNET_WEIGHT}"
echo "============================================================"

if [ ! -f "main.py" ]; then
  echo "[ERROR] main.py not found in ${BASE_DIR}"
  exit 1
fi

if [ ! -d "${INPUT_DIR}/samples" ]; then
  echo "[ERROR] input samples dir not found: ${INPUT_DIR}/samples"
  exit 1
fi

ls "${INPUT_DIR}/samples"

# Macenko import check
if [ ! -f "${BASE_DIR}/utils_color_norm.py" ]; then
  if [ -f "${BASE_DIR}/11slide_processing/utils_color_norm.py" ]; then
    cp "${BASE_DIR}/11slide_processing/utils_color_norm.py" "${BASE_DIR}/utils_color_norm.py"
    echo "[INFO] copied utils_color_norm.py from 11slide_processing"
  elif [ -f "${BASE_DIR}/../11slide_processing/utils_color_norm.py" ]; then
    cp "${BASE_DIR}/../11slide_processing/utils_color_norm.py" "${BASE_DIR}/utils_color_norm.py"
    echo "[INFO] copied utils_color_norm.py from ../11slide_processing"
  else
    echo "[ERROR] utils_color_norm.py not found"
    exit 1
  fi
fi

python - <<'PY'
import spams
import utils_color_norm
m = utils_color_norm.macenko_normalizer()
print("[CHECK] Macenko OK")
PY

# ============================================================
# Shared predict settings
# ============================================================

GPU=0
TILE_SIZE=224
NORM="log1p_cpm"
PRED_BATCH_SIZE=4096
FEATURE_BATCH_SIZE=128
NUM_WORKERS=8

# ============================================================
# Function: predict one sample using a group of fold0 models
# ============================================================

predict_with_fold0_models () {
  local MODEL_PREFIX="$1"      # OV_to_OVXenium or HCC_to_HCCXenium
  local TEST_SAMPLE="$2"       # OV_Xenium_all_new / HCC_Xenium_all_new / OC_all_new / CC_all_new
  local OUT_PREFIX="$3"        # output dir prefix

  for il in 0 1 2 3 4
  do
    TRAIN_DIR="${OUT_ROOT}/DeepPT_nested_macenko_${MODEL_PREFIX}_fold0_${il}"
    OUT_DIR="${OUT_ROOT}/${OUT_PREFIX}_fold0_${il}"

    mkdir -p "${OUT_DIR}"

    echo "============================================================"
    echo "[PREDICT]"
    echo "  MODEL_PREFIX = ${MODEL_PREFIX}"
    echo "  fold         = 0_${il}"
    echo "  TEST_SAMPLE  = ${TEST_SAMPLE}"
    echo "  TRAIN_DIR    = ${TRAIN_DIR}"
    echo "  OUT_DIR      = ${OUT_DIR}"
    echo "============================================================"

    if [ ! -f "${TRAIN_DIR}/deeppt_model_AE.pth" ]; then
      echo "[ERROR] missing AE ckpt: ${TRAIN_DIR}/deeppt_model_AE.pth"
      exit 1
    fi

    if [ ! -f "${TRAIN_DIR}/deeppt_model_MLP.pth" ]; then
      echo "[ERROR] missing MLP ckpt: ${TRAIN_DIR}/deeppt_model_MLP.pth"
      exit 1
    fi

    if [ ! -f "${TRAIN_DIR}/gene_list.tsv" ]; then
      echo "[ERROR] missing gene_list: ${TRAIN_DIR}/gene_list.tsv"
      exit 1
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      --input_dir "${INPUT_DIR}" \
      --test_sample "${TEST_SAMPLE}" \
      --out_dir "${OUT_DIR}" \
      --predict_only \
      --ae_ckpt "${TRAIN_DIR}/deeppt_model_AE.pth" \
      --mlp_ckpt "${TRAIN_DIR}/deeppt_model_MLP.pth" \
      --gene_list "${TRAIN_DIR}/gene_list.tsv" \
      --resnet_weight "${RESNET_WEIGHT}" \
      --tile_size ${TILE_SIZE} \
      --norm ${NORM} \
      --pred_batch_size ${PRED_BATCH_SIZE} \
      --feature_batch_size ${FEATURE_BATCH_SIZE} \
      --num_workers ${NUM_WORKERS} \
      --gpu 0 \
      2>&1 | tee "${OUT_DIR}/run.log"
  done
}

# ============================================================
# 1. OV fold0 models -> OV Xenium
# ============================================================

predict_with_fold0_models \
  "OV_to_OVXenium" \
  "OV_Xenium_all_new" \
  "DeepPT_fold0_OVmodel_predict_OVXenium"

# ============================================================
# 2. HCC fold0 models -> HCC Xenium
# ============================================================

predict_with_fold0_models \
  "HCC_to_HCCXenium" \
  "HCC_Xenium_all_new" \
  "DeepPT_fold0_HCCmodel_predict_HCCXenium"

# ============================================================
# 3. OV fold0 models -> OC
# ============================================================

predict_with_fold0_models \
  "OV_to_OVXenium" \
  "OC_all_new" \
  "DeepPT_fold0_OVmodel_predict_OC"

# ============================================================
# 4. OV fold0 models -> CC
# ============================================================

predict_with_fold0_models \
  "OV_to_OVXenium" \
  "CC_all_new" \
  "DeepPT_fold0_OVmodel_predict_CC"

# ============================================================
# 5. Ensemble fold0_0 ~ fold0_4 predictions for each test sample
# ============================================================

python - <<'PY'
import glob
import h5py
import numpy as np
from pathlib import Path

OUT_ROOT = Path("/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT/outputs")

jobs = {
    "OV_Xenium_all_new": {
        "pattern": "DeepPT_fold0_OVmodel_predict_OVXenium_fold0_*",
        "h5_name": "OV_Xenium_all_new_predicted_expression.h5",
        "out_dir": "DeepPT_fold0_OVmodel_predict_OVXenium_ensemble5",
    },
    "HCC_Xenium_all_new": {
        "pattern": "DeepPT_fold0_HCCmodel_predict_HCCXenium_fold0_*",
        "h5_name": "HCC_Xenium_all_new_predicted_expression.h5",
        "out_dir": "DeepPT_fold0_HCCmodel_predict_HCCXenium_ensemble5",
    },
    "OC_all_new": {
        "pattern": "DeepPT_fold0_OVmodel_predict_OC_fold0_*",
        "h5_name": "OC_all_new_predicted_expression.h5",
        "out_dir": "DeepPT_fold0_OVmodel_predict_OC_ensemble5",
    },
    "CC_all_new": {
        "pattern": "DeepPT_fold0_OVmodel_predict_CC_fold0_*",
        "h5_name": "CC_all_new_predicted_expression.h5",
        "out_dir": "DeepPT_fold0_OVmodel_predict_CC_ensemble5",
    },
}

def read_genes(ds):
    out = []
    for x in ds[:]:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out

for sample, cfg in jobs.items():
    print("\n" + "=" * 100)
    print("[ENSEMBLE]", sample)

    dirs = sorted(glob.glob(str(OUT_ROOT / cfg["pattern"])))
    files = [Path(d) / cfg["h5_name"] for d in dirs]
    files = [f for f in files if f.exists()]

    print("n_files:", len(files))
    for f in files:
        print(" ", f)

    if len(files) == 0:
        print("[WARN] no prediction files found, skip")
        continue

    if len(files) != 5:
        print(f"[WARN] expected 5 fold0 files, found {len(files)}")

    pred_sum = None
    true_expression = None
    tile_coords = None
    genes = None
    attrs = {}

    for fp in files:
        with h5py.File(fp, "r") as f:
            pred = f["predicted_expression"][:].astype(np.float32)

            if pred_sum is None:
                pred_sum = np.zeros_like(pred, dtype=np.float64)

                if "true_expression" in f:
                    true_expression = f["true_expression"][:].astype(np.float32)

                tile_coords = f["tile_coords"][:].astype(np.float32)
                genes = read_genes(f["genes"])
                attrs = dict(f.attrs)
            else:
                if pred.shape != pred_sum.shape:
                    raise ValueError(f"Shape mismatch: {fp}, {pred.shape} vs {pred_sum.shape}")

            pred_sum += pred.astype(np.float64)

    pred_mean = (pred_sum / len(files)).astype(np.float32)

    out_dir = OUT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = out_dir / f"{sample}_predicted_expression_ensemble5_fold0.h5"

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "predicted_expression",
            data=pred_mean,
            compression="gzip",
            chunks=(min(512, pred_mean.shape[0]), min(512, pred_mean.shape[1])),
        )

        if true_expression is not None:
            f.create_dataset(
                "true_expression",
                data=true_expression,
                compression="gzip",
                chunks=(min(512, true_expression.shape[0]), min(512, true_expression.shape[1])),
            )

        f.create_dataset("tile_coords", data=tile_coords, compression="gzip")
        f.create_dataset("genes", data=np.asarray(genes, dtype=object), dtype=str_dtype)

        f.attrs["ensemble_n"] = len(files)
        f.attrs["source"] = "DeepPT fold0 internal models ensemble average"

        for k, v in attrs.items():
            try:
                f.attrs[k] = v
            except Exception:
                pass

    print("[SAVED]", out_h5)

print("\n[DONE] fold0 prediction + ensemble")
PY

echo "[ALL DONE]"
