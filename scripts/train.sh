#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cd "$SCRIPT_DIR"/..

python ./train.py \
    --data_dir path/to/your/dataset \
    --batch_size 4 \
    --valid_batch_size 2 \
    --num_workers 8 \
    --epochs 70 \
    --device cuda \
    --lr 0.0001 \
    --T_max 50 \
    --eta_min 0.000001 \
    --seed 3407 \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs \
    --save_freq 10