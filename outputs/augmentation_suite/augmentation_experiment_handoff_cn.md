# 数据增强实验中文说明

这份文档是给组内汇报和队友交接用的中文版本，重点回答两个问题：

1. 我们用什么来衡量数据增强的影响？
2. 目前实验结果说明了什么？

## 一、我们用什么衡量“增强方案的影响”

这次实验里，我们主要看两类东西。

### 1. 训练阶段指标

这是目前已经直接跑出来的指标，用来衡量不同增强方案对 `SimCLR` 自监督训练过程的影响：

| 指标 | 含义 | 怎么理解 |
|---|---|---|
| `final_loss` | 最终对比学习损失 | 越低通常越好，表示模型更容易把同一张图的两种视图拉近、把不同图分开 |
| `contrastive top1` | 对比学习任务中的 top-1 指标 | 越高越好，表示模型更容易找到正确的正样本对 |
| `contrastive top5` | 对比学习任务中的 top-5 指标 | 越高越好，表示模型在候选中更容易把正确匹配排进前 5 |

注意：

- 这些指标反映的是 **自监督训练本身学得好不好**
- 它们 **不是最终分类准确率**
- 所以它们可以用来比较“哪种增强更有利于 SimCLR 训练”，但不能直接当作最终项目结论

### 2. 下游评估指标

队友后续会做：

- `frozen encoder evaluation`
- `少标签 / few-label evaluation`

这部分才是最终判断“哪种增强方案更好”的关键。

也就是说：

- 你这边负责的是：先比较不同增强对 `SimCLR` 训练的影响
- 队友负责的是：拿你产出的 checkpoint 做下游分类评估

## 二、这次实验固定了什么

为了公平比较增强方案，我们尽量固定其余条件不变。

| 项目 | 数值 |
|---|---|
| 数据集 | `cifar10` |
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

这张表的意义是：

除了增强策略之外，其他主要训练条件基本一致，所以不同实验之间可以直接对比。

## 三、每种增强方案改了什么

| 增强方案 | 具体含义 |
|---|---|
| `baseline` | 标准 SimCLR 增强：`RandomResizedCrop + HorizontalFlip + ColorJitter + Grayscale + GaussianBlur + ToTensor` |
| `no_blur` | 在 baseline 基础上去掉 `GaussianBlur` |
| `no_color_jitter` | 在 baseline 基础上去掉 `ColorJitter` |
| `no_grayscale` | 在 baseline 基础上去掉 `RandomGrayscale` |
| `crop_only` | 只保留 `RandomResizedCrop + ToTensor` |

这张表的意义是：

明确每一组实验到底“改了哪个增强”，这样后面看到结果时才知道影响来自哪里。

## 四、建议队友优先使用的 checkpoint

下面这张表最重要，因为它是给后续 `frozen encoder / 少标签评估` 直接用的。

优先建议用 `e10_subset5000` 这组结果，因为它比快速试跑版更有参考价值。

| 增强方案 | epochs | 样本数上限 | final_loss | contrastive top1 | contrastive top5 | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 10 | 5000 | 5.485037 | 5.2529 | 14.5559 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_baseline/checkpoint_0010.pth.tar` |
| `no_blur` | 10 | 5000 | 5.450078 | 5.1295 | 14.8849 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_blur/checkpoint_0010.pth.tar` |
| `no_color_jitter` | 10 | 5000 | 3.305897 | 37.6645 | 56.8154 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_color_jitter/checkpoint_0010.pth.tar` |
| `no_grayscale` | 10 | 5000 | 5.258235 | 7.0724 | 18.4519 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e10_subset5000_no_grayscale/checkpoint_0010.pth.tar` |

这张表怎么读：

- `final_loss` 越低，说明训练目标优化得通常更好
- `contrastive top1/top5` 越高，说明模型更容易把正样本匹配出来
- `checkpoint` 是最关键的交付物，队友后面要拿这个做冻结编码器和少标签评估

## 五、快速实验结果

这部分是我先做的快速试跑，用来快速看趋势，不是最终正式结论。

| 增强方案 | epochs | 样本数上限 | final_loss | contrastive top1 | contrastive top5 | checkpoint |
|---|---:|---:|---:|---:|---:|---|
| `baseline` | 2 | 512 | 6.274026 | 1.4648 | 3.4180 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_baseline/checkpoint_0002.pth.tar` |
| `no_blur` | 2 | 512 | 6.235386 | 1.0742 | 3.3203 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_blur/checkpoint_0002.pth.tar` |
| `no_color_jitter` | 2 | 512 | 5.594493 | 6.4453 | 14.6484 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_color_jitter/checkpoint_0002.pth.tar` |
| `no_grayscale` | 2 | 512 | 6.238386 | 1.5625 | 3.3203 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_no_grayscale/checkpoint_0002.pth.tar` |
| `crop_only` | 2 | 512 | 5.516730 | 7.5195 | 15.6250 | `outputs/augmentation_suite/cifar10_resnet18_aug_ablation_e2_subset512_crop_only/checkpoint_0002.pth.tar` |

这张表的作用是：

- 先快速观察趋势
- 看哪些增强值得继续深入
- 不建议直接把这组结果当成最终结论写进报告

## 六、目前结果说明了什么

根据目前已经跑出来的结果，可以先得到这些阶段性结论：

1. 不同数据增强方案，确实会明显影响 `SimCLR` 的训练效果。
2. `no_color_jitter` 的训练指标明显最好，说明在当前设置下，`ColorJitter` 可能过强，反而增加了学习难度。
3. `no_blur` 和 `baseline` 很接近，说明 `GaussianBlur` 在当前设置下影响不算大。
4. `no_grayscale` 比 `baseline` 略好，说明灰度增强可能也会略微增加训练难度。
5. 这些结论目前还只是“训练阶段结论”，最终还是要看队友做的 `frozen encoder / 少标签评估`。

更稳妥的汇报说法是：

> 在固定数据集、backbone 和训练超参数的条件下，我们比较了不同数据增强策略对 SimCLR 训练表现的影响。结果表明，增强策略会显著影响对比学习指标，其中去掉 `ColorJitter` 的方案表现最好，说明当前设置下颜色扰动可能过强；而去掉 `GaussianBlur` 的影响较小，去掉 `Grayscale` 有小幅提升。不过这些还只是自监督训练阶段的结果，最终仍需要结合 frozen encoder 和少标签评估来判断哪种增强方案最好。

## 七、建议你怎么对队友说

你可以直接把这段发给队友：

> 我这边已经把 augmentation ablation 跑出来了，统一配置下比较了 baseline、no_blur、no_color_jitter、no_grayscale 等方案。当前的 `loss / contrastive top1 / top5` 主要反映 SimCLR 训练阶段的效果，不是最终分类准确率。你们后面做 frozen encoder 或少标签评估时，建议优先使用 `e10_subset5000` 这几组 checkpoint。
