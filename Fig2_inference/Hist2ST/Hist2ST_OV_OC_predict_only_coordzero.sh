#!/usr/bin/env bash
set -euo pipefail

TILE_ROOT=./EGNv1_tile_input_allgenes_224
TRAIN_OUT=outputs/Hist2ST_HD_native_OV_to_OVXenium_coordzero

CUDA_VISIBLE_DEVICES=0 python main_hist2st_hdgenes.py \
  --mode predict_only \
  --tile_root ${TILE_ROOT} \
  --train_sample OV_VisiumHD_all_new \
  --test_sample OC_all_new \
  --out_dir outputs/Hist2ST_HD_native_OVmodel_to_OC_coordzero \
  --train_out_dir ${TRAIN_OUT} \
  --ckpt ${TRAIN_OUT}/best_model.pt \
  --gpu 0 \
  --crop_size 224 \
  --fig_size 112 \
  --max_pred_nodes 0 \
  2>&1 | tee outputs/Hist2ST_HD_native_OVmodel_to_OC_coordzero.log
