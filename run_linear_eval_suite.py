import argparse
import csv
import json
import os
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


LINEAR_EVAL_EPOCH = 100

DEFAULT_RUN_DIRS = [
    ('baseline', 'cifar10_resnet18_baseline_e100_subset5000_ckpt10'),
    ('no_blur', 'cifar10_resnet18_no_blur_e100_subset5000_ckpt10'),
    ('no_color_jitter', 'cifar10_resnet18_no_color_jitter_e100_subset5000_ckpt10'),
    ('no_grayscale', 'cifar10_resnet18_no_grayscale_e100_subset5000_ckpt10'),
]

DEFAULT_RUNS = [
    (
        run_name,
        os.path.join(
            'outputs',
            'augmentation_suite',
            experiment_name,
            f'checkpoint_{LINEAR_EVAL_EPOCH:04d}.pth.tar',
        ),
    )
    for run_name, experiment_name in DEFAULT_RUN_DIRS
]


def parse_args():
    parser = argparse.ArgumentParser(description='Batch linear evaluation for SimCLR checkpoints.')
    parser.add_argument('--data', default='./datasets_local')
    parser.add_argument('--dataset-name', default='cifar10', choices=['cifar10', 'stl10'])
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--max-train-samples', default=None, type=int)
    parser.add_argument('--max-test-samples', default=None, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output-dir', default='outputs/linear_eval')
    parser.add_argument(
        '--suite-name',
        default=f'cifar10_resnet18_linear_eval_e{LINEAR_EVAL_EPOCH}_subset5000',
    )
    return parser.parse_args()


def run_single(args, run_name, checkpoint):
    output_dir = os.path.join(args.output_dir, f'{args.suite_name}_{run_name}')
    cmd = [
        sys.executable, 'linear_eval.py',
        '--checkpoint', checkpoint,
        '--dataset-name', args.dataset_name,
        '--data', args.data,
        '--arch', args.arch,
        '--batch-size', str(args.batch_size),
        '--workers', str(args.workers),
        '--seed', str(args.seed),
        '--output-dir', output_dir,
    ]
    if args.max_train_samples is not None:
        cmd.extend(['--max-train-samples', str(args.max_train_samples)])
    if args.max_test_samples is not None:
        cmd.extend(['--max-test-samples', str(args.max_test_samples)])

    subprocess.run(cmd, check=True)
    summary_path = os.path.join(output_dir, 'linear_eval_summary.json')
    with open(summary_path) as infile:
        summary = json.load(infile)
    summary['run_name'] = run_name
    summary['summary_path'] = summary_path
    return summary


def write_outputs(args, summaries):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f'{args.suite_name}_summary.csv')
    md_path = os.path.join(args.output_dir, f'{args.suite_name}_summary.md')
    plot_path = os.path.join(args.output_dir, f'{args.suite_name}_test_accuracy.png')
    visualization_path = os.path.join(args.output_dir, f'{args.suite_name}_visualization.png')

    rows = []
    for summary in summaries:
        rows.append({
            'run_name': summary['run_name'],
            'checkpoint': summary['checkpoint'],
            'train_samples': summary['train_samples'],
            'test_samples': summary['test_samples'],
            'train_accuracy': summary['train_accuracy'],
            'test_accuracy': summary['test_accuracy'],
            'test_top5_accuracy': summary['test_top5_accuracy'],
            'duration_seconds': summary['duration_seconds'],
            'summary_path': summary['summary_path'],
        })

    with open(csv_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f'# {args.suite_name}',
        '',
        '## Frozen Encoder + Linear Classifier Results',
        '',
        '| augmentation | train_accuracy | test_accuracy | test_top5_accuracy | train_samples | test_samples |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['train_accuracy']:.4f} | {row['test_accuracy']:.4f} | {row['test_top5_accuracy']:.4f} | {row['train_samples']} | {row['test_samples']} |"
        )

    lines.extend(['', '## Checkpoints', ''])
    for row in rows:
        lines.append(f"- `{row['run_name']}`: `{row['checkpoint']}`")

    with open(md_path, 'w') as outfile:
        outfile.write('\n'.join(lines) + '\n')

    if plt is not None:
        run_names = [row['run_name'] for row in rows]
        test_accuracies = [row['test_accuracy'] for row in rows]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(run_names, test_accuracies, color='#2A9D8F')
        plt.ylabel('Linear eval test accuracy')
        plt.xlabel('Augmentation preset')
        plt.title(args.suite_name)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        for bar, score in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.2f}',
                     ha='center', va='bottom', fontsize=9)
        plt.savefig(plot_path, dpi=200)
        plt.close()

        top5_scores = [row['test_top5_accuracy'] for row in rows]
        durations = [row['duration_seconds'] for row in rows]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

        bar_groups = [
            ('Test Accuracy', test_accuracies, '#2A9D8F'),
            ('Top-5 Accuracy', top5_scores, '#E9C46A'),
            ('Duration (s)', durations, '#264653'),
        ]
        for axis, (title, values, color) in zip(axes, bar_groups):
            bars = axis.bar(run_names, values, color=color)
            axis.set_title(title)
            axis.tick_params(axis='x', rotation=30)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment('right')
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

        fig.suptitle(args.suite_name)
        fig.tight_layout()
        fig.savefig(visualization_path, dpi=200)
        plt.close(fig)
    else:
        plot_path = None
        visualization_path = None

    return csv_path, md_path, plot_path, visualization_path


def main():
    args = parse_args()
    summaries = []
    for run_name, checkpoint in DEFAULT_RUNS:
        print(f'Running linear evaluation for: {run_name}', flush=True)
        summaries.append(run_single(args, run_name, checkpoint))

    csv_path, md_path, plot_path, visualization_path = write_outputs(args, summaries)
    print(f'Linear eval summary CSV: {csv_path}')
    print(f'Linear eval summary Markdown: {md_path}')
    if plot_path is not None:
        print(f'Linear eval plot: {plot_path}')
    if visualization_path is not None:
        print(f'Linear eval visualization: {visualization_path}')


if __name__ == '__main__':
    main()
