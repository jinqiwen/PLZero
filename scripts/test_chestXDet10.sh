CUDA_VISIBLE_DEVICES=0 python test_chestXDet10.py \
--vision-backbone densenet121 \
--textual-embeddings embeddings/chexdet10_biobert.npy \
--load-from checkpoints/best_auroc_checkpoint.pth156.tar \
--dataset chestXDet_10 \
--test-file data/ChestX-Det10/preprocess/test.csv

