#!/bin/bash
# Quick debug run for local testing on Mac M4
# Shorter training to verify everything works

set -e

# Configuration - debug settings
TIMESTEPS=1000
SCHEDULE="linear"
TOTAL_STEPS=1000
BATCH_SIZE=32  # Smaller for M4 Mac
DEVICE="mps"   # Use Metal for Mac

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
    --exp-name "ddpm_debug" \
    --debug

echo "Debug run complete!"
