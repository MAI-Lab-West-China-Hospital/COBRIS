#!/bin/bash

torchrun --nproc_per_node=8 train_classic_DDP.py \
  --train-csv ./DATA/Dataset_train.csv \
  --val-csv ./DATA/Dataset_val.csv \
  --root-dir ./DATA/ \
  --cache-dir /Cache/cache \
  --output-dir ./runs_regression \
  --batch-size 16 \
  --epochs 200 \
  --lr 1e-4 \
  --backbone seresnet101 \
  --val-every 1 \
  --amp
