#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT"
INPUT_DIR="${BASE_DIR}/EGNv1_tile_input_allgenes_224"
RESNET_WEIGHT="/lustre1/zxzeng/bwqin/STORM_main/clustering/EGN-main/v2/ResNet50_IMAGENET1K_V2.pt"
OUT_ROOT="${BASE_DIR}/outputs"

cd "${BASE_DIR}"
mkdir -p "${OUT_ROOT}"

# Make Macenko importable
if [ ! -f "${BASE_DIR}/utils_color_norm.py" ]; then
  if [ -f "${BASE_DIR}/11slide_processing/utils_color_norm.py" ]; then
    cp "${BASE_DIR}/11slide_processing/utils_color_norm.py" "${BASE_DIR}/utils_color_norm.py"
  elif [ -f "${BASE_DIR}/../11slide_processing/utils_color_norm.py" ]; then
    cp "${BASE_DIR}/../11slide_processing/utils_color_norm.py" "${BASE_DIR}/utils_color_norm.py"
  else
    echo "[ERROR] Cannot find utils_color_norm.py"
    exit 1
  fi
fi

python - <<'PY'
import spams
import utils_color_norm
m = utils_color_norm.macenko_normalizer()
print("[CHECK] Macenko OK")
PY

GPU=7

for ik in 0 1 2 3 4
do
  for il in 0 1 2 3 4
  do
    OUT_DIR="${OUT_ROOT}/DeepPT_nested_macenko_HCC_to_HCCXenium_fold${ik}_${il}"
    mkdir -p "${OUT_DIR}"

    echo "===================================================================="
    echo "[RUN] HCC VisiumHD -> HCC Xenium fold ${ik}_${il}"
    echo "===================================================================="

    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      --input_dir "${INPUT_DIR}" \
      --train_sample HCC_VisiumHD_all_new \
      --test_sample HCC_Xenium_all_new \
      --out_dir "${OUT_DIR}" \
      --resnet_weight "${RESNET_WEIGHT}" \
      --tile_size 224 \
      --norm log1p_cpm \
      --n_outer_folds 5 \
      --n_inner_folds 5 \
      --ik_fold ${ik} \
      --il_fold ${il} \
      --ae_epochs 500 \
      --ae_batch_size 2048 \
      --ae_lr 1e-4 \
      --epochs 500 \
      --patience 50 \
      --batch_size 4096 \
      --pred_batch_size 2048 \
      --feature_batch_size 2048 \
      --dropout 0.2 \
      --lr 1e-4 \
      --num_workers 4 \
      --gpu 0 \
      2>&1 | tee "${OUT_DIR}/run.log"
  done
done

echo "[DONE] HCC -> HCC Xenium all 25 folds"
