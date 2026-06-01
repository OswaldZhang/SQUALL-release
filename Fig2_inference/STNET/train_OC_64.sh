#!/bin/bash

ngenes=64
model=densenet121
window=224
train_patient=OVVisiumHD
test_patient=OCXenium
outdir=output_all_OC_backup/${model}_${window}/top_${ngenes}/${train_patient}_${test_patient}
echo "Start time: $(date)"
# 训练模型
python3 -m stnet run_spatial \
  --gene \
  --trainpatients OVVisiumHD \
  --testpatients OCXenium \
  --logfile output_all_OC_backup/densenet121_224/top_250/OC1_OCXenium/gene.log \
  --epochs 50 \
  --checkpoint output_all_OC_backup/densenet121_224/top_250/OC1_OCXenium/checkpoints/epoch_ \
  --pred_root output_all_OC_backup/densenet121_224/top_250/OC1_OCXenium/ \
  --lr 1e-6 \
  --window 224 \
  --model densenet121 \
  --pretrain \
  --average \
  --batch 32 \
  --workers 8 \
  --gene_n 64 \
  --norm
echo "End time: $(date)"
