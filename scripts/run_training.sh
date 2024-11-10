#!/bin/bash

# Shell script to run the MM-SCS model training with GAT

# Define arguments
DATA_PATH="./dataset/train.csv"
SAVE_PATH="./mmscs_model.pth"
BATCH_SIZE=8
EPOCHS=10
LEARNING_RATE=3e-5
MAX_LENGTH=128
MARGIN=0.1

# Execute the training script
python train_with_gat.py \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --margin $MARGIN
