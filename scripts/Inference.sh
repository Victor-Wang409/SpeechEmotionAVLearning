python inference.py \
    --checkpoint_path ./models/fold30/epoch1/pytorch_model.bin  \
    --emotion2vec_dir ./data/emo2vec_features \
    --hubert_dir ./data/hubert_features \
    --csv_path ./csv_files/IEMOCAP.csv \
    --output_path ./data/embeddings.pickle \
    --device cuda