import argparse
import os

from utils import load_metrics_csv

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


DEFAULT_RUN_DIRS = [
    ('baseline', 'outputs/augmentation_suite/cifar10_resnet18_baseline_e100_subset5000_ckpt10'),
    ('no_blur', 'outputs/augmentation_suite/cifar10_resnet18_no_blur_e100_subset5000_ckpt10'),
    ('no_color_jitter', 'outputs/augmentation_suite/cifar10_resnet18_no_color_jitter_e100_subset5000_ckpt10'),
    ('no_grayscale', 'outputs/augmentation_suite/cifar10_resnet18_no_grayscale_e100_subset5000_ckpt10'),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training comparisons for augmentation experiments.')
    parser.add_argument('--output-dir', default='outputs/augmentation_suite')
    parser.add_argument('--suite-name', default='e100_subset5000')
    return parser.parse_args()


def plot_metric(run_metrics, metric_name, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (run_name, rows), color in zip(run_metrics, colors):
        epochs = [row['epoch'] for row in rows]
        values = [row[metric_name] for row in rows]
        ax.plot(epochs, values, label=run_name, color=color, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    if plt is None:
        raise RuntimeError('matplotlib is required to generate comparison plots.')

    run_metrics = []
    for run_name, run_dir in DEFAULT_RUN_DIRS:
        metrics_path = os.path.join(run_dir, 'metrics.csv')
        rows = load_metrics_csv(metrics_path)
        if not rows:
            raise FileNotFoundError(f'No metrics found at {metrics_path}')
        run_metrics.append((run_name, rows))

    accuracy_path = os.path.join(args.output_dir, f'{args.suite_name}_accuracy_vs_epoch_comparison.png')
    loss_path = os.path.join(args.output_dir, f'{args.suite_name}_loss_vs_epoch_comparison.png')

    plot_metric(
        run_metrics,
        metric_name='top1',
        ylabel='Contrastive Top-1',
        title=f'{args.suite_name} top-1 comparison',
        output_path=accuracy_path,
    )
    plot_metric(
        run_metrics,
        metric_name='loss',
        ylabel='Loss',
        title=f'{args.suite_name} loss comparison',
        output_path=loss_path,
    )

    print(f'Accuracy comparison plot: {accuracy_path}')
    print(f'Loss comparison plot: {loss_path}')


if __name__ == '__main__':
    main()
