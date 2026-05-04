# cifar10_resnet18_linear_eval_e100_subset5000_train5000_fast

## Frozen Encoder + Linear Classifier Results

| augmentation | train_accuracy | test_accuracy | test_top5_accuracy | train_samples | test_samples |
|---|---:|---:|---:|---:|---:|
| baseline | 71.5000 | 43.5700 | 89.6300 | 5000 | 10000 |
| no_blur | 71.9000 | 44.5600 | 89.7800 | 5000 | 10000 |
| no_color_jitter | 71.4800 | 42.0700 | 88.4700 | 5000 | 10000 |
| no_grayscale | 72.9400 | 44.9200 | 89.6200 | 5000 | 10000 |

## Checkpoints

- `baseline`: `outputs/augmentation_suite/cifar10_resnet18_baseline_e100_subset5000_ckpt10/checkpoint_0100.pth.tar`
- `no_blur`: `outputs/augmentation_suite/cifar10_resnet18_no_blur_e100_subset5000_ckpt10/checkpoint_0100.pth.tar`
- `no_color_jitter`: `outputs/augmentation_suite/cifar10_resnet18_no_color_jitter_e100_subset5000_ckpt10/checkpoint_0100.pth.tar`
- `no_grayscale`: `outputs/augmentation_suite/cifar10_resnet18_no_grayscale_e100_subset5000_ckpt10/checkpoint_0100.pth.tar`
