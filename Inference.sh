python wavLM.py --mode inference \
                --num-labels 5 \
                --load-path /home/victor/Github/SpeechEmotionAVLearning/dump/train/model_last_epoch.pth \
                --save-path ./dump \
                --batch-size 16 \
                --data /home/victor/Github/SpeechEmotionAVLearning/data/audio_partial5_train_dataset.pickle \
                --write