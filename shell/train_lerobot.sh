#!/bin/bash

set -x
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="70959bcf4562cbb6a910d3842b8ae3dd7d1e35e0"

# === Basic Configuration ===
POLICY_PATH=${POLICY_PATH:-"/mnt/pfs/scalelab/smolvla_ckpt/ti_2254/checkpoints/010000/pretrained_model"}
DATASET_ROOT=${DATASET_ROOT:-"/mnt/pfs/scalelab/dataset/huggingface/lerobot/messy_gs_tasks"}
DATASET_REPO_ID=${DATASET_REPO_ID:-"messy_gs_tasks"}
BATCH_SIZE=${BATCH_SIZE:-192}
STEPS=${STEPS:-20000}
POLICY_DEVICE=${POLICY_DEVICE:-"cuda"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/pfs/scalelab/smolvla_ckpt/ti_8161829_messy_gs_tasks"}
WANDB_ENABLE=${WANDB_ENABLE:-true}
WANDB_PROJECT=${WANDB_PROJECT:-"messy_gs_tasks_smolvla"}
WANDB_ENTITY=${WANDB_ENTITY:-"violetevar-shanghai-jiao-tong-university"}
WANDB_RUN_ID=${WANDB_RUN_ID:-"run_001"}
WANDB_MODE=${WANDB_MODE:-"online"}
POLICY_PUSH_TO_HUB=${POLICY_PUSH_TO_HUB:-false}

echo "============================================================================"
echo "🚀 LeRobot Training Configuration"
echo "============================================================================"

# === Launch Training ===
python src/lerobot/scripts/train.py \
  --policy.path="${POLICY_PATH}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --batch_size=${BATCH_SIZE} \
  --steps=${STEPS} \
  --policy.device="${POLICY_DEVICE}" \
  --output_dir="${OUTPUT_DIR}" \
  --wandb.enable=${WANDB_ENABLE} \
  --wandb.project="${WANDB_PROJECT}" \
  --wandb.entity="${WANDB_ENTITY}" \
  --wandb.run_id="${WANDB_RUN_ID}" \
  --wandb.mode="${WANDB_MODE}" \
  --policy.push_to_hub=${POLICY_PUSH_TO_HUB} \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

echo "============================================================================"
echo "🎉 Training completed!"
echo "📋 Logs: ${OUTPUT_DIR}/training_log.txt"
echo "💾 Checkpoints: $OUTPUT_DIR"
echo "🔗 Monitor training: https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
echo "============================================================================"