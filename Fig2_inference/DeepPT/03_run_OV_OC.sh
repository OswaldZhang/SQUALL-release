#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/lustre1/zxzeng/bwqin/STORM_main/clustering/DeepPT"
INPUT_DIR="${BASE_DIR}/EGNv1_tile_input_allgenes"
RESNET_WEIGHT="/lustre1/zxzeng/bwqin/STORM_main/clustering/EGN-main/v2/ResNet50_IMAGENET1K_V2.pt"
OUT_ROOT="${BASE_DIR}/outputs"

cd "${BASE_DIR}"
mkdir -p "${OUT_ROOT}"

GPU=0

for ik in 0 1 2 3 4
do
  for il in 0 1 2 3 4
  do
    TRAIN_DIR="${OUT_ROOT}/DeepPT_nested_macenko_OV_to_OVXenium_fold${ik}_${il}"
    OUT_DIR="${OUT_ROOT}/DeepPT_nested_macenko_OV_fold${ik}_${il}_predict_OC"
    mkdir -p "${OUT_DIR}"

    if [ ! -f "${TRAIN_DIR}/deeppt_model_AE.pth" ]; then
      echo "[ERROR] Missing AE ckpt: ${TRAIN_DIR}/deeppt_model_AE.pth"
      exit 1
    fi

    if [ ! -f "${TRAIN_DIR}/deeppt_model_MLP.pth" ]; then
      echo "[ERROR] Missing MLP ckpt: ${TRAIN_DIR}/deeppt_model_MLP.pth"
      exit 1
    fi

    echo "===================================================================="
    echo "[PREDICT] OV model fold ${ik}_${il} -> OC"
    echo "===================================================================="

    CUDA_VISIBLE_DEVICES=${GPU} python main.py \
      --input_dir "${INPUT_DIR}" \
      --test_sample OC_all_new \
      --out_dir "${OUT_DIR}" \
      --predict_only \
      --ae_ckpt "${TRAIN_DIR}/deeppt_model_AE.pth" \
      --mlp_ckpt "${TRAIN_DIR}/deeppt_model_MLP.pth" \
      --gene_list "${TRAIN_DIR}/gene_list.tsv" \
      --resnet_weight "${RESNET_WEIGHT}" \
      --tile_size 256 \
      --norm log1p_cpm \
      --pred_batch_size 4096 \
      --feature_batch_size 64 \
      --num_workers 4 \
      --gpu 0 \
      2>&1 | tee "${OUT_DIR}/run.log"
  done
done

echo "[DONE] OV 25 fold models -> OC"
