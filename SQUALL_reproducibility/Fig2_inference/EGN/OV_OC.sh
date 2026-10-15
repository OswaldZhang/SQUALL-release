TILE_ROOT=./EGNv1_tile_input_allgenes_224
RESNET=/lustre1/zxzeng/bwqin/STORM_main/clustering/EGN-main/v2/ResNet50_IMAGENET1K_V2.pt

CUDA_VISIBLE_DEVICES=2 python main_hdgenes.py \
  --mode predict_only \
  --tile_root ${TILE_ROOT} \
  --train_sample OV_VisiumHD_all_new \
  --test_sample OC_all_new \
  --out_dir outputs/EGNv1_HD_original_OVmodel_to_OC \
  --train_out_dir outputs/EGNv1_HD_original_OV_to_OVXenium \
  --ckpt outputs/EGNv1_HD_original_OV_to_OVXenium/best_model.pt \
  --resnet_weight ${RESNET} \
  --batch 4 \
  --embed_batch_size 64 \
  --workers 8 \
  --size 224 \
  --patch_size 32 \
  --dim 1024 \
  --depth 8 \
  --heads 16 \
  --mlp_dim 4096 \
  --bhead 8 \
  --bdim 64 \
  --bfre 2 \
  --mdim 2048 \
  --player 1 \
  --linear_projection True \
  --numk 16 \
  --knn_batch 512 \
  2>&1 | tee outputs/EGNv1_HD_original_OVmodel_to_OC.log
