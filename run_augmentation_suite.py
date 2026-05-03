import argparse
import csv
import json
import os
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


DEFAULT_AUGMENTATIONS = [
    'baseline',
    'no_blur',
    'no_color_jitter',
    'no_grayscale',
    'no_flip',
    'crop_only',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run a SimCLR augmentation ablation suite.')
    parser.add_argument('--augmentations', nargs='+', default=DEFAULT_AUGMENTATIONS,
                        help='augmentation presets to run')
    parser.add_argument('--data', default='./datasets_local', help='dataset root')
    parser.add_argument('--dataset-name', default='cifar10', choices=['cifar10', 'stl10'])
    parser.add_argument('-a', '--arch', default='resnet18')
    parser.add_argument('--out_dim', default=128, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n-views', default=2, type=int)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--output-dir', default='outputs/augmentation_suite')
    parser.add_argument('--suite-name', default='cifar10_resnet18_aug_ablation')
    parser.add_argument('--max-samples', default=None, type=int,
                        help='optional cap on the number of training samples per experiment')
    return parser.parse_args()


def run_experiment(args, augmentation):
    experiment_name = f"{args.suite_name}_{augmentation}"
    cmd = [
        sys.executable, 'run.py',
        '-data', args.data,
        '-dataset-name', args.dataset_name,
        '-a', args.arch,
        '--out_dim', str(args.out_dim),
        '--epochs', str(args.epochs),
        '-b', str(args.batch_size),
        '--temperature', str(args.temperature),
        '--seed', str(args.seed),
        '--n-views', str(args.n_views),
        '--lr', str(args.lr),
        '--wd', str(args.wd),
        '-j', str(args.workers),
        '--augmentation', augmentation,
        '--experiment-name', experiment_name,
        '--output-dir', args.output_dir,
    ]
    if args.max_samples is not None:
        cmd.extend(['--max-samples', str(args.max_samples)])
    if args.disable_cuda:
        cmd.append('--disable-cuda')

    subprocess.run(cmd, check=True)
    summary_path = os.path.join(args.output_dir, experiment_name, 'summary.json')
    with open(summary_path) as infile:
        return json.load(infile)


def write_summary_files(output_dir, suite_name, summaries):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{suite_name}_summary.csv')
    md_path = os.path.join(output_dir, f'{suite_name}_summary.md')

    rows = []
    for summary in summaries:
        rows.append({
            'augmentation': summary['augmentation'],
            'final_loss': summary['final_loss'],
            'final_top1': summary['final_top1'],
            'final_top5': summary['final_top5'],
            'duration_seconds': summary['duration_seconds'],
            'run_dir': summary['run_dir'],
            'checkpoint_path': summary['checkpoint_path'],
        })

    with open(csv_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f'# {suite_name}',
        '',
        '| augmentation | final_loss | final_top1 | final_top5 | duration_seconds |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['augmentation']} | {row['final_loss']:.6f} | {row['final_top1']:.4f} | {row['final_top5']:.4f} | {row['duration_seconds']:.2f} |"
        )

    lines.extend([
        '',
        '## Run Directories',
        '',
    ])
    for row in rows:
        lines.append(f"- `{row['augmentation']}`: `{row['run_dir']}`")

    with open(md_path, 'w') as outfile:
        outfile.write('\n'.join(lines) + '\n')

    plot_path = None
    if plt is not None:
        plot_path = os.path.join(output_dir, f'{suite_name}_final_top1.png')
        augmentations = [row['augmentation'] for row in rows]
        top1_scores = [row['final_top1'] for row in rows]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(augmentations, top1_scores, color='#4C72B0')
        plt.ylabel('Final contrastive top-1')
        plt.xlabel('Augmentation preset')
        plt.title(f'{suite_name} augmentation comparison')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()

        for bar, score in zip(bars, top1_scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{score:.2f}',
                     ha='center', va='bottom', fontsize=9)

        plt.savefig(plot_path, dpi=200)
        plt.close()

    return csv_path, md_path, plot_path


def main():
    args = parse_args()
    summaries = []

    for augmentation in args.augmentations:
        print(f'Running augmentation: {augmentation}', flush=True)
        summaries.append(run_experiment(args, augmentation))

    csv_path, md_path, plot_path = write_summary_files(args.output_dir, args.suite_name, summaries)
    print(f'Suite summary CSV: {csv_path}')
    print(f'Suite summary Markdown: {md_path}')
    if plot_path is not None:
        print(f'Suite summary plot: {plot_path}')


if __name__ == '__main__':
    main()
