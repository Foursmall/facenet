#!/bin/bash bash
for N in {1..4};
  do python ~/facenet/src/align/align_dataset_mtcnn.py ~/LFW/lfw  ~/LFW/lfw_mtcnpy_160  --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done