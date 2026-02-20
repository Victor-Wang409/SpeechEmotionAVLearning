python wavLM.py --data ./data/audio_partial5_train_dataset.pickle \
                --mode inference \
                --num-labels 5 \
                --load-path ./dump/train/model_last_epoch.pth \
                --save-path ./dump/inference \
                --batch-size 16 