#!/bin/bash
# Run baseline LDM training with CelebA-HQ
# 128px input -> 32x32 latent (f=4)

set -e

# Configuration
TIMESTEPS=1000 # denoising process step
SCHEDULE="cosine"
TOTAL_STEPS=100000
DEVICE="mps"   # Change to "cuda" for GPU
IMAGE_SIZE=128
DATASET="celeba_hq_256"
# f=4 for 32x32 latent (Colab setting for better quality)
DOWNSAMPLE_FACTOR=4

# MPS tuning profile: turbo / fast / default
PROFILE="${PROFILE:-balanced}"
case "$PROFILE" in
  turbo)
    BATCH_SIZE=32
    MODEL_CHANNELS=64
    CHANNEL_MULT="1,2,2,2"
    NUM_RES_BLOCKS=1
    NUM_HEADS=4
    ATTENTION_RES="16"
    ;;
  fast)
    BATCH_SIZE=32
    MODEL_CHANNELS=64
    CHANNEL_MULT="1,2,2,3"
    NUM_RES_BLOCKS=2
    NUM_HEADS=4
    ATTENTION_RES="16,8"
    ;;
  *)
    # default
    BATCH_SIZE=32
    MODEL_CHANNELS=128
    CHANNEL_MULT="1,2,3,4"
    NUM_RES_BLOCKS=2
    NUM_HEADS=8
    ATTENTION_RES="16,8"
    ;;
esac

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
    --num-res-blocks $NUM_RES_BLOCKS \
    --num-heads $NUM_HEADS \
    --attention-resolutions $ATTENTION_RES \
    --exp-name "ldm_${DATASET}_T${TIMESTEPS}_${SCHEDULE}"

echo "Baseline training complete!"
