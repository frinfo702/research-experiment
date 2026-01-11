#!/bin/bash
# Run baseline LDM training with CelebA-HQ
# 128px input -> 16x16 latent (f=8)

set -e

# Configuration
TIMESTEPS=1000 # denoising process step
SCHEDULE="cosine"
TOTAL_STEPS=100000
BATCH_SIZE=64  # If you hit OOM, try 32
DEVICE="mps"   # Change to "cuda" for GPU
IMAGE_SIZE=128
DATASET="celeba_hq_256"
# sd-vae-ft-mse expects f=8; keep 8 for safety
DOWNSAMPLE_FACTOR=8
# Higher-capacity U-Net for better quality (matches notebook capacity)
MODEL_CHANNELS=128
CHANNEL_MULT="1,2,3,4"

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
    --image-size $IMAGE_SIZE \
    --dataset $DATASET \
    --downsample-factor $DOWNSAMPLE_FACTOR \
    --model-channels $MODEL_CHANNELS \
    --channel-mult $CHANNEL_MULT \
    --exp-name "ldm_${DATASET}_T${TIMESTEPS}_${SCHEDULE}"

echo "Baseline training complete!"
