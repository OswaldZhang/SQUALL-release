python -m stnet.prepare.spatial --root /lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/OV_VisiumHD/OV1 --dest /lustre1/zxzeng/bwqin/STORM_main/clustering/ST-NET/data/spatial-processed
python -m stnet.run_spatial \
  --task gene \
  --model resnet18 \
  --root results/OV1/ \
  --trainpatients OV1 \
  --testpatients OV1 \
  --epochs 10 \
  --batch 32 \
  --gpu \
  --checkpoint results/OV1/checkpoints/epoch_ \
  --pred_root results/OV1/ \
  --logfile results/OV1/train.log
