#!/bin/bash

ngenes=5000
model=densenet121
window=224
train_patient=HCCVisiumHD
test_patient=HCCXenium
outdir=output_all_HCC/${model}_${window}/top_${ngenes}/${train_patient}_${test_patient}
echo "Start time: $(date)"
# 训练模型
python3 -m stnet run_spatial \
  --gene \
  --trainpatients HCCVisiumHD \
  --testpatients HCCXenium \
  --logfile output_all_HCC/densenet121_224/top_250/HCC1_HCCXenium/gene.log \
  --epochs 50 \
  --checkpoint output_all_HCC/densenet121_224/top_250/HCC1_HCCXenium/checkpoints/epoch_ \
  --pred_root output_all_HCC/densenet121_224/top_250/HCC1_HCCXenium/ \
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
