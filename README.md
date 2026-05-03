# SimCLR 增强实验交接说明

这个仓库当前用于运行 `CIFAR-10` 上的 `SimCLR` 自监督对比学习实验，重点是比较不同数据增强方案对训练结果的影响。

## 这个项目在训练什么

这不是普通的有监督图像分类训练。

这里训练的是 `SimCLR` 对比学习模型，核心思想是：

- 主干网络使用 `ResNet-18`
- 数据集使用 `CIFAR-10`
- 同一张图片经过两次不同增强后，得到两个 view
- 训练目标是让同一张图片的两个 view 在特征空间里更接近，让不同图片更远

当前主要比较以下增强方案：

- `baseline`
- `no_color_jitter`
- `no_grayscale`
- `no_blur`

## 当前主实验配置

当前统一主实验参数如下：

- 数据集：`cifar10`
- 网络结构：`resnet18`
- 投影头维度：`128`
- 训练轮数：`100`
- batch size：`256`
- temperature：`0.07`
- learning rate：`0.0003`
- weight decay：`1e-4`
- view 数量：`2`
- 随机种子：`0`

## 环境配置

推荐使用 `conda`：

```bash
conda env create --name simclr --file env.yml
conda activate simclr
```

仓库里也有 `requirements.txt`，但本项目建议优先使用 `env.yml`。

## 推荐运行命令

运行当前主要的 `100 epoch` augmentation ablation：

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

如果只能用 CPU，额外加上：

```bash
--disable-cuda
```

## 数据说明

- 数据目录：`./datasets_local`
- 如果本地没有数据，代码会自动下载 `CIFAR-10`

以下目录或文件不要提交，也不用共享给别人：

- `outputs/`
- `datasets_local/`
- `__pycache__/`

## 输出结果说明

每个增强方案都会在 `outputs/augmentation_suite/` 下生成一个单独的实验目录。

每个实验最重要的文件有：

- `config.yml`：保存本次运行配置
- `metrics.csv`：按 epoch 记录的指标
- `training.log`：训练日志
- `summary.json`：本次运行的汇总结果
- `checkpoint_0100.pth.tar`：100 epoch 最终 checkpoint
- `training_curves.png`：训练曲线图

批量实验还会额外生成整组增强方案的汇总文件：

- `*_summary.csv`
- `*_summary.md`
- `*_final_top1.png`

## 训练曲线说明

每个实验结束后都会自动生成 `training_curves.png`，横轴都是 `epoch`，包含三部分：

1. `top1` 和 `top5` 对照图
2. `learning rate` 曲线
3. `loss` 曲线

需要注意：

这里的 `top1/top5` 不是 CIFAR-10 分类准确率，而是对比学习训练过程中 batch 内的排序指标，用来反映正样本匹配效果。

如果要看更接近分类任务的效果，需要做线性评估。

## 线性评估

仓库里已经包含线性评估脚本：

- `linear_eval.py`
- `run_linear_eval_suite.py`

线性评估的做法是：

- 先把训练好的 SimCLR 编码器冻结
- 再在提取出的特征上训练一个线性分类器

如果需要更标准的分类准确率，建议在预训练完成后再跑这一部分。

## 关键文件

- `run.py`：单次 SimCLR 训练入口
- `run_augmentation_suite.py`：批量运行增强对比实验
- `simclr.py`：训练主循环、checkpoint、指标保存
- `data_aug/contrastive_learning_dataset.py`：数据增强定义
- `linear_eval.py`：冻结编码器后的线性评估
- `data.txt`：当前主实验参数简要记录

## 交接备注

- 这次交接的目标实验是 `100 epoch` 的 CIFAR-10 augmentation ablation
- 如果机器有 GPU，优先使用 GPU
- 如果没有 GPU，可以加 `--disable-cuda`，但训练会明显更慢
- 最有价值的结果通常是：汇总表、checkpoint、`training_curves.png`
