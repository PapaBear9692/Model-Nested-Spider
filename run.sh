#!/bin/bash
# Model-Nested-Spider Training Runner for Linux/MacOS

# Get the script directory (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded from .env"
else
    echo "⚠ Warning: .env file not found. Using defaults."
fi

# Default arguments
GPU="${GPU:-0}"
SEED="${SEED:-1}"
TRAIN_DATASET="${TRAIN_DATASET:-CIFAR10}"
TEST_DATASET="${TEST_DATASET:-CIFAR10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_EPOCH="${MAX_EPOCH:-50}"
LR="${LR:-0.01}"
OPTIMIZER="${OPTIMIZER:-Adam}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
DATA_SUB_URL="${DATA_SUB_URL:-swin_base_7_checkpoint}"
PRETRAINED_URL="${PRETRAINED_URL:-}"

# Print configuration
echo "========================================"
echo "Model-Nested-Spider Configuration"
echo "========================================"
echo "GPU: $GPU"
echo "Seed: $SEED"
echo "Train Dataset: $TRAIN_DATASET"
echo "Test Dataset: $TEST_DATASET"
echo "Batch Size: $BATCH_SIZE"
echo "Max Epoch: $MAX_EPOCH"
echo "Learning Rate: $LR"
echo "Optimizer: $OPTIMIZER"
echo "LR Scheduler: $LR_SCHEDULER"
echo "Data SubURL: $DATA_SUB_URL"
if [ ! -z "$PRETRAINED_URL" ]; then
    echo "Pretrained URL: $PRETRAINED_URL"
fi
echo "========================================"
echo ""

# Build command
CMD="python trainer.py \
    --gpu $GPU \
    --seed $SEED \
    --train_dataset $TRAIN_DATASET \
    --test_dataset $TEST_DATASET \
    --batch_size $BATCH_SIZE \
    --max_epoch $MAX_EPOCH \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --lr_scheduler $LR_SCHEDULER \
    --data_sub_url $DATA_SUB_URL"

# Add pretrained URL if provided
if [ ! -z "$PRETRAINED_URL" ]; then
    CMD="$CMD --pretrained_url $PRETRAINED_URL"
fi

# Add any additional arguments passed to this script
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

# Run the training
echo "Starting training..."
echo "Command: $CMD"
echo ""
eval $CMD
