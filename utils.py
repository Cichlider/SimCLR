import os
import shutil
import csv
import json

import torch
import yaml


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
