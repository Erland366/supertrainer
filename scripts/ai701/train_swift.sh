#!/bin/bash

# Check if a model type is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <model_type>"
  exit 1
fi

# Assign the first argument to the model_type variable
MODEL_TYPE=$1

# Set CUDA device to GPU 0
export CUDA_VISIBLE_DEVICES=0

# Run the swift sft command with the specified model type
swift sft \
    --model_type "$MODEL_TYPE" \
    --dataset train.jsonl \
    --max_length 144 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --lora_rank 64 \
    --lora_alpha 128
