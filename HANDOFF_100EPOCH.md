# SimCLR Handoff for 100-Epoch Run

This repo already contains the code needed to run the CIFAR-10 SimCLR augmentation experiments.

## What to share

Share these source files and folders:

- `run.py`
- `simclr.py`
- `utils.py`
- `run_augmentation_suite.py`
- `linear_eval.py`
- `run_linear_eval_suite.py`
- `data_aug/`
- `models/`
- `exceptions/`
- `env.yml`
- `README.md`
- `data.txt`

Do not include:

- `outputs/`
- `datasets_local/`
- any `__pycache__/` folders

## Recommended command

For the main 100-epoch CIFAR-10 augmentation comparison:

```bash
python run_augmentation_suite.py \
  --data ./datasets_local \
  --dataset-name cifar10 \
  -a resnet18 \
  --epochs 100 \
  -b 256 \
  --temperature 0.07 \
  --seed 0 \
  --n-views 2 \
  --lr 0.0003 \
  --wd 1e-4 \
  -j 4 \
  --augmentations baseline no_color_jitter no_grayscale no_blur \
  --suite-name cifar10_resnet18_aug_ablation_e100
```

If GPU is unavailable, add:

```bash
--disable-cuda
```

## Notes

- The dataset will download automatically into `./datasets_local`.
- Results will be written to `outputs/augmentation_suite/`.
- Each run also writes `training_curves.png`, which contains `top1/top5`, `learning rate`, and `loss` against epoch.
