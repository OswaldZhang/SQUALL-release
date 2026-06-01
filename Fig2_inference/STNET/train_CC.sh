#!/bin/bash

ngenes=5000
model=densenet121
window=224
train_patient=OVVisiumHD
test_patient=CCXenium
outdir=output_all_CC/${model}_${window}/top_${ngenes}/${train_patient}_${test_patient}
echo "Start time: $(date)"
# 训练模型
python3 -m stnet run_spatial \
  --gene \
  --trainpatients OVVisiumHD \
  --testpatients CCXenium \
  --logfile output_all_CC/densenet121_224/top_250/CC1_CCXenium/gene.log \
  --epochs 50 \
  --checkpoint output_all_CC/densenet121_224/top_250/CC1_CCXenium/checkpoints/epoch_ \
  --pred_root output_all_CC/densenet121_224/top_250/CC1_CCXenium/ \
  --lr 1e-6 \
  --window 224 \
  --model densenet121 \
  --pretrain \
  --average \
  --batch 32 \
  --workers 8 \
  --gene_n 5000 \
  --norm
echo "End time: $(date)"
