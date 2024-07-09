CUDA_VISIBLE_DEVICES=0 python test_cheXpert.py \
--vision-backbone densenet121 \
--textual-embeddings embeddings/cheXpert_biobert_12.npy \
--load-from checkpoints/best_auroc_checkpoint.pth156.tar \
--data-root data/CXR8 \
--dataset CheXpert_12_5 \
--test-file data/CheXpert-v1.0-small/CheXpert-v1.0-small/test.csv

