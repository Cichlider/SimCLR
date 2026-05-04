# Augmentation Experiment Handoff

This file is the teammate-facing summary for augmentation ablations.

## How To Read These Results

- `final_loss`, `final_top1`, and `final_top5` are **contrastive training metrics**, not downstream classification accuracy.
- For `frozen encoder` or `few-label` evaluation, the most important artifact is the **checkpoint path** for each augmentation setting.
- Recommendation: use the `e10_subset5000` checkpoints first, because they are more meaningful than the quick `e2_subset512` smoke-test runs.

## Shared Training Setup

| Item | Value |
|---|---|
| dataset | `cifar10` |
| backbone | `resnet18` |
| out_dim | `128` |
| batch_size | `256` |
| temperature | `0.07` |
| seed | `0` |
| n_views | `2` |
| lr | `0.0003` |
| weight_decay | `1e-4` |
| workers | `0` |
| device | `cpu` |

## Augmentation Presets

| Augmentation | Transform Difference |
|---|---|
| `baseline` | `RandomResizedCrop + HorizontalFlip + ColorJitter + Grayscale + GaussianBlur + ToTensor` |
| `no_blur` | baseline without `GaussianBlur` |
| `no_color_jitter` | baseline without `ColorJitter` |
| `no_grayscale` | baseline without `RandomGrayscale` |
| `crop_only` | `RandomResizedCrop + ToTensor` |

## Main Checkpoints For Downstream Evaluation

Use these first for frozen-encoder / few-label evaluation.

| Augmentation | Epochs | Max Samples | final_loss | contrastive top1 | contrastive top5 | Checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 10 | 5000 | 5.485037 | 5.2529 | 14.5559 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_baseline/checkpoint_0010.pth.tar` |
| `no_blur` | 10 | 5000 | 5.450078 | 5.1295 | 14.8849 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_blur/checkpoint_0010.pth.tar` |
| `no_color_jitter` | 10 | 5000 | 3.305897 | 37.6645 | 56.8154 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_color_jitter/checkpoint_0010.pth.tar` |
| `no_grayscale` | 10 | 5000 | 5.258235 | 7.0724 | 18.4519 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_grayscale/checkpoint_0010.pth.tar` |

## Run Directories And Metadata

| Augmentation | Run Dir | Config | Metrics |
|---|---|---|---|
| `baseline` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_baseline` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_baseline/config.yml` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_baseline/metrics.csv` |
| `no_blur` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_blur` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_blur/config.yml` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_blur/metrics.csv` |
| `no_color_jitter` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_color_jitter` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_color_jitter/config.yml` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_color_jitter/metrics.csv` |
| `no_grayscale` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_grayscale` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_grayscale/config.yml` | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_grayscale/metrics.csv` |

## Quick Sweep Results

These were the fast smoke-test runs used to compare more variants quickly.

| Augmentation | Epochs | Max Samples | final_loss | contrastive top1 | contrastive top5 | Checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 2 | 512 | 6.274026 | 1.4648 | 3.4180 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_baseline/checkpoint_0002.pth.tar` |
| `no_blur` | 2 | 512 | 6.235386 | 1.0742 | 3.3203 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_blur/checkpoint_0002.pth.tar` |
| `no_color_jitter` | 2 | 512 | 5.594493 | 6.4453 | 14.6484 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_color_jitter/checkpoint_0002.pth.tar` |
| `no_grayscale` | 2 | 512 | 6.238386 | 1.5625 | 3.3203 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_grayscale/checkpoint_0002.pth.tar` |
| `crop_only` | 2 | 512 | 5.516730 | 7.5195 | 15.6250 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_crop_only/checkpoint_0002.pth.tar` |

## Suggested Message To Teammates

You can send this:

`I fixed the augmentation ablation pipeline and exported checkpoints for each setting. Please use the e10_subset5000 checkpoints first for frozen-encoder / few-label evaluation. The contrastive metrics are summarized above, but the final conclusion should be based on downstream evaluation accuracy.`
