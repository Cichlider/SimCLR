import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Frozen encoder linear evaluation for SimCLR checkpoints.')
    parser.add_argument('--checkpoint', required=True, help='path to SimCLR checkpoint')
    parser.add_argument('--dataset-name', default='cifar10', choices=['cifar10', 'stl10'])
    parser.add_argument('--data', default='./datasets_local', help='dataset root')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--max-train-samples', default=None, type=int)
    parser.add_argument('--max-test-samples', default=None, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output-dir', required=True, help='directory for evaluation outputs')
    parser.add_argument('--classifier-max-iter', default=200, type=int)
    parser.add_argument('--classifier-solver', default='liblinear',
                        choices=['liblinear', 'lbfgs', 'saga', 'newton-cg', 'newton-cholesky', 'sag'])
    return parser.parse_args()


def maybe_subset(dataset, max_samples, seed):
    if max_samples is None or max_samples >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


def get_datasets(name, root, max_train_samples, max_test_samples, seed):
    transform = transforms.ToTensor()

    if name == 'cifar10':
        train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        train_dataset = datasets.STL10(root, split='train', download=True, transform=transform)
        test_dataset = datasets.STL10(root, split='test', download=True, transform=transform)
        num_classes = 10

    train_dataset = maybe_subset(train_dataset, max_train_samples, seed)
    test_dataset = maybe_subset(test_dataset, max_test_samples, seed)
    return train_dataset, test_dataset, num_classes


def build_encoder(arch):
    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        feature_dim = 512
    else:
        model = models.resnet50(weights=None)
        feature_dim = 2048

    model.fc = torch.nn.Identity()
    return model, feature_dim


def load_simclr_encoder(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    converted_state = {}

    for key, value in state_dict.items():
        if not key.startswith('backbone.'):
            continue
        if key.startswith('backbone.fc'):
            continue
        converted_state[key[len('backbone.'):]] = value

    log = model.load_state_dict(converted_state, strict=False)
    missing_keys = set(log.missing_keys)
    if missing_keys:
        raise RuntimeError(f'Unexpected missing keys: {log.missing_keys}')
    if log.unexpected_keys:
        raise RuntimeError(f'Unexpected keys when loading encoder: {log.unexpected_keys}')


def extract_features(model, loader, device):
    features = []
    labels = []
    model.eval()

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, num_classes = get_datasets(
        args.dataset_name, args.data, args.max_train_samples, args.max_test_samples, args.seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False
    )

    model, feature_dim = build_encoder(args.arch)
    model = model.to(device)
    load_simclr_encoder(model, args.checkpoint, device)

    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    classifier = LogisticRegression(
        random_state=args.seed,
        max_iter=args.classifier_max_iter,
        solver=args.classifier_solver,
        multi_class='auto',
        n_jobs=-1,
    )
    classifier.fit(train_features, train_labels)

    train_accuracy = float(classifier.score(train_features, train_labels) * 100.0)
    test_accuracy = float(classifier.score(test_features, test_labels) * 100.0)
    test_probabilities = classifier.predict_proba(test_features)
    labels = list(range(num_classes))
    test_top5 = float(top_k_accuracy_score(test_labels, test_probabilities, k=min(5, num_classes), labels=labels) * 100.0)

    summary = {
        'checkpoint': args.checkpoint,
        'dataset_name': args.dataset_name,
        'arch': args.arch,
        'feature_dim': feature_dim,
        'train_samples': int(len(train_dataset)),
        'test_samples': int(len(test_dataset)),
        'train_accuracy': round(train_accuracy, 4),
        'test_accuracy': round(test_accuracy, 4),
        'test_top5_accuracy': round(test_top5, 4),
        'duration_seconds': round(time.time() - start_time, 2),
    }

    summary_path = os.path.join(args.output_dir, 'linear_eval_summary.json')
    with open(summary_path, 'w') as outfile:
        json.dump(summary, outfile, indent=2)

    print(f'Linear evaluation summary saved to {summary_path}')
    print(
        f"Train acc: {summary['train_accuracy']:.4f}, "
        f"Test acc: {summary['test_accuracy']:.4f}, "
        f"Test top5: {summary['test_top5_accuracy']:.4f}"
    )


if __name__ == '__main__':
    main()
