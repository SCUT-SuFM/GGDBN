#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

cd "$SCRIPT_DIR"/../

python ./eval.py \
    --gt_dir path/to/gt \
    --eval_dir path/to/eval \
    --device cuda