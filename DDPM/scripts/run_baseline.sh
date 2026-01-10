#!/bin/bash
# Run baseline DDPM training (T=1000, linear schedule)
# Expected FID: ~3.17

set -e

# Configuration
TIMESTEPS=1000
SCHEDULE="linear"
TOTAL_STEPS=800000
BATCH_SIZE=128
DEVICE="cuda"  # Change to "mps" for Mac or "cpu" for CPU

# Directories
OUTPUT_DIR="./outputs"
DATA_DIR="./data"

# Run training with uv
uv run python train.py \
    --timesteps $TIMESTEPS \
    --beta-schedule $SCHEDULE \
    --total-steps $TOTAL_STEPS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --output-dir $OUTPUT_DIR \
    --data-dir $DATA_DIR \
    --exp-name "ddpm_baseline_T${TIMESTEPS}_${SCHEDULE}"

echo "Baseline training complete!"
