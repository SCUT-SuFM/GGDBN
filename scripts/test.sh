#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cd "$SCRIPT_DIR"/../

python ./test.py \
    --data_dir path/to/data \
    --ckpt_path path/to/ckpt \
    --output_dir path/to/output \
    --device cuda