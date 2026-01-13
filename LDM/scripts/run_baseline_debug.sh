#!/bin/bash
# Quick debug run for local testing on Mac M4
# Shorter training to verify everything works
# 128px input -> 32x32 latent (f=4)

set -e

# Configuration - debug settings
TIMESTEPS=1000
SCHEDULE="cosine"
TOTAL_STEPS=1000
BATCH_SIZE=8   # Very small for M4 Mac with 128px
DEVICE="mps"   # Use Metal for Mac
IMAGE_SIZE=128
DATASET="celeba_hq_256"
DOWNSAMPLE_FACTOR=4

# Directories
OUTPUT_DIR="./outputs"
DATA_DIR="./data"

# Run training with uv and debug flag
uv run python train.py \
    --timesteps $TIMESTEPS \
    --beta-schedule $SCHEDULE \
    --total-steps $TOTAL_STEPS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --output-dir $OUTPUT_DIR \
    --data-dir $DATA_DIR \
    --image-size $IMAGE_SIZE \
    --dataset $DATASET \
    --downsample-factor $DOWNSAMPLE_FACTOR \
    --exp-name "ldm_debug" \
    --debug

echo "Debug run complete!"
