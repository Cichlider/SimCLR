import os
import shutil
import csv
import json

import torch
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_json(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def save_metrics_csv(rows, filename):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filename, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_csv(filename):
    if not os.path.exists(filename):
        return []

    with open(filename, newline='') as infile:
        reader = csv.DictReader(infile)
        rows = []
        for row in reader:
            rows.append({
                'epoch': int(row['epoch']),
                'loss': float(row['loss']),
                'top1': float(row['top1']),
                'top5': float(row['top5']),
                'learning_rate': float(row['learning_rate']),
            })
        return rows


def save_training_curves(rows, filename):
    if plt is None or not rows:
        return None

    epochs = [row['epoch'] for row in rows]
    top1 = [row['top1'] for row in rows]
    top5 = [row['top5'] for row in rows]
    learning_rate = [row['learning_rate'] for row in rows]
    loss = [row['loss'] for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(epochs, top1, label='top1', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, top5, label='top5', color='#ff7f0e', linewidth=2)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Top-1 and Top-5 vs Epoch')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, learning_rate, color='#2ca02c', linewidth=2)
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate vs Epoch')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss, color='#d62728', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Loss vs Epoch')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    return filename
