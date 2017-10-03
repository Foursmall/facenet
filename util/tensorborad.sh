#!/usr/bin/env bash

tensorboard --logdir=~/Desktop/facenet  --port 6006
#lsof -i tcp:6006
#kill PID