python wavLM.py --data ./data/audio_partial5_train_dataset.pickle \
                --mode inference \
                --num-labels 5 \
                --load-path ./dump/train/model_best.pth \
                --save-path ./dump \
                --batch-size 16 