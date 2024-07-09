CUDA_VISIBLE_DEVICES=0 python test_chestxray14.py \
--vision-backbone densenet121 \
--textual-embeddings embeddings/nih_chest_xray_biobert.npy \
--load-from checkpoints/best_auroc_checkpoint.pth156.tar \
--data-root data/CXR8 \
--dataset ChestX-ray14 \
--test-file data/CXR8/preprocess/transformed_official_test.csv

