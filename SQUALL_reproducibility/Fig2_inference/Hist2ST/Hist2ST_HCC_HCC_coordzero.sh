#!/usr/bin/env bash
set -euo pipefail

TILE_ROOT=./EGNv1_tile_input_allgenes_224

CUDA_VISIBLE_DEVICES=0 python main_hist2st_hdgenes.py \
  --mode train_predict \
  --tile_root ${TILE_ROOT} \
  --train_sample HCC_VisiumHD_all_new \
  --test_sample HCC_Xenium_all_new \
  --out_dir outputs/Hist2ST_HD_native_HCC_to_HCCXenium_coordzero \
  --gpu 0 \
  --epochs 350 \
  --lr 1e-5 \
  --dropout 0.2 \
  --tag 5-7-2-8-4-16-32 \
  --bake 5 \
  --lamb 0.5 \
  --zinb 0.25 \
  --nb F \
  --prune NA \
  --policy mean \
  --neighbor 4 \
  --crop_size 224 \
  --fig_size 112 \
  --coord_mode zero \
  --n_pos 1 \
  --gene_source common \
  --val_fraction 0.1 \
  --max_train_nodes 0 \
  --max_pred_nodes 0 \
  2>&1 | tee outputs/Hist2ST_HD_native_HCC_to_HCCXenium_coordzero.log
