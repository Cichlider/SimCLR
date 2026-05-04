#!/bin/zsh
set -e

cd "$(dirname "$0")"

run_one() {
  local augmentation="$1"
  local experiment_name="$2"
  local checkpoint_path="$3"

  echo "========================================"
  echo "Resuming ${augmentation} to epoch 100"
  echo "Experiment: ${experiment_name}"
  echo "Checkpoint: ${checkpoint_path}"
  echo "========================================"

  python3 run.py \
    -data ./datasets_local \
    -dataset-name cifar10 \
    -a resnet18 \
    --epochs 100 \
    -b 256 \
    --temperature 0.07 \
    --seed 0 \
    --n-views 2 \
    --lr 0.0003 \
    --wd 1e-4 \
    -j 0 \
    --disable-cuda \
    --max-samples 5000 \
    --augmentation "${augmentation}" \
    --experiment-name "${experiment_name}" \
    --output-dir outputs/augmentation_suite \
    --resume-from-checkpoint "${checkpoint_path}"
}

run_one "baseline" \
  "cifar10_resnet18_baseline_e100_subset5000_ckpt10" \
  "outputs/augmentation_suite/cifar10_resnet18_baseline_e100_subset5000_ckpt10/checkpoint_0050.pth.tar"

run_one "no_blur" \
  "cifar10_resnet18_no_blur_e100_subset5000_ckpt10" \
  "outputs/augmentation_suite/cifar10_resnet18_no_blur_e100_subset5000_ckpt10/checkpoint_0050.pth.tar"

run_one "no_color_jitter" \
  "cifar10_resnet18_no_color_jitter_e100_subset5000_ckpt10" \
  "outputs/augmentation_suite/cifar10_resnet18_no_color_jitter_e100_subset5000_ckpt10/checkpoint_0050.pth.tar"

run_one "no_grayscale" \
  "cifar10_resnet18_no_grayscale_e100_subset5000_ckpt10" \
  "outputs/augmentation_suite/cifar10_resnet18_no_grayscale_e100_subset5000_ckpt10/checkpoint_0050.pth.tar"
