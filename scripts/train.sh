CUDA_VISIBLE_DEVICES=0 python train.py \
--pretrained \
--vision-backbone densenet121 \
--save-dir checkpoints \
--batch-size 32 \
--epochs 40 \
--lr 0.0001 \
--beta-rank 1 \
--beta-map 0.01 \
--beta-con 0.01 \
--neg-penalty 0.20 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--data-root data/CXR8
