#!/bin/bash

# Set environment variables
export DATA_DIR="./data/CIFAR10"
export SAVE_DIR="./save"
export BACKBONE="ViT-B/32"
export BATCH_SIZE=128
export NUM_WORKERS=4
export TEMPERATURE=0.01
export EPSILON=0.7
export NUM_TEMPLATES=8
export METHOD="clipot"

# Run the script
CUDA_VISIBLE_DEVICES=4 python main.py \
    --dataset cifar10 \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --epsilon "$EPSILON" \
    --num_templates "$NUM_TEMPLATES"\
    --corruptions_list original\
    --disable_wandb
