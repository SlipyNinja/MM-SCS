#!/bin/bash

# Define arguments
TEST_DATA_PATH="./dataset/test.csv"
MODEL_PATH="./mmscs_model.pth"
BATCH_SIZE=4
MAX_LENGTH=128

# Execute the evaluation script
python evaluate_model.py \
    --test_data_path $TEST_DATA_PATH \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH
