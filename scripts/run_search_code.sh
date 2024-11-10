#!/bin/bash

# Define arguments
MODEL_PATH="./mmscs_model.pth"
DATA_PATH="./dataset/train.csv"
DEVICE="cuda"
BATCH_SIZE=4
MAX_LENGTH=128

# Execute the search script
python searchCode.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH
