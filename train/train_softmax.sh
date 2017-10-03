#!/bin/bash bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"

python src/train_softmax.py --logs_base_dir  ~/facenet/logs/facenet/ --models_base_dir ~/facenet/models/facenet/ --data_dir /home/lipreading/LFW/lfw-deepfunneled --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir /home/lipreading/LFW/lfw_mtcnpy_160 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file ~/facenet/data/learning_rate_schedule_classifier_casia.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9  --gpu_memory_fraction 0.48 &