#!/usr/bin/env python
"""归档一个 MMSegmentation 实验。

此脚本将单次训练运行中的重要产物保存到一个紧凑的实验文件夹中。

它会保存：
1. source_config.py：你用于启动训练的配置文件。
2. config.py：从 work_dir/*/vis_data/config.py 复制过来的已解析配置。
3. train.log：work_dir 下最新的训练日志。
4. scalars.json 和 run.json：MMEngine 可视化记录（如果存在）。
5. best_mIoU*.pth：最佳检查点，除非使用了 --no-weights。
6. iter_*.pth 或 last checkpoint：最终检查点，除非使用了 --no-weights。
7. last_checkpoint：work_dir 中的检查点标记文件。
8. metrics.txt：每个验证间隔解析出的验证指标。
9. per_class_iou.txt：解析出的最佳/最终逐类 IoU 和 Acc 表格。
10. summary.txt 和 metadata.json：一份可读的摘要和一份机器可读的元数据，
    用于后续对比。

每次归档前，请修改以下命令行参数：
1. --work-dir：要归档的训练输出目录。
2. --config：本次运行使用的配置文件。
3. --out-dir：你想要创建/更新的实验归档目录。
4. --name：一个简短可读的实验名称。
5. --note：可选，但建议填写，用于记录本次运行的目的。

可选参数：
1. --val-interval：仅在脚本无法从配置中解析出 val_interval 时需要指定。
2. --no-weights：当你只需要日志/指标/配置文件，而不想复制较大的 .pth
   检查点文件时使用。

示例：
python tools/my_tools/archive_experiment.py \
  --work-dir work_dirs/quick40k/segformer_b0_suim_baseline \
  --config configs/segformer/segformer_mit-b0_suim_quick_40k.py \
  --out-dir experiments/quick40k/segformer_b0_suim_baseline \
  --name "Quick40k Baseline" \
  --note "Reference baseline for quick 40k FreqFusion experiments."

python tools/my_tools/archive_experiment.py \
  --work-dir work_dirs/quick40k/segformer_b0_suim_segformer_freqfusion_c4c3 \
  --config configs/segformer/segformer_mit-b0_suim_segformer_freqfusion_c4c3_quick_40k.py \
  --out-dir experiments/quick40k/segformer_b0_suim_segformer_freqfusion_c4c3 \
  --name "Quick40k FreqFusion c4->c3"
"""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


METRIC_RE = re.compile(
    r'(aAcc|mIoU|mAcc|data_time|time):\s*([0-9]+(?:\.[0-9]+)?)')
BEST_RE = re.compile(
    r'The best checkpoint with\s+([0-9]+(?:\.[0-9]+)?)\s+(\w+)'
    r'\s+at\s+(\d+)\s+iter')
CLASS_ROW_RE = re.compile(
    r'\|\s*([^|]+?)\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|'
    r'\s*([0-9]+(?:\.[0-9]+)?)\s*\|')
VAL_INTERVAL_RE = re.compile(r'val_interval\s*=\s*(\d+)')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Archive checkpoints, logs, config, and parsed metrics.')
    parser.add_argument('--work-dir', required=True, help='MMSeg work_dir.')
    parser.add_argument('--out-dir', required=True, help='Archive directory.')
    parser.add_argument('--config', default=None, help='Source config path.')
    parser.add_argument('--name', default=None, help='Experiment name.')
    parser.add_argument('--note', default=None, help='Short note for summary.')
    parser.add_argument(
        '--val-interval',
        type=int,
        default=None,
        help='Validation interval. If omitted, the script tries to parse it.')
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Do not copy checkpoint weights.')
    return parser.parse_args()


def latest_file(paths):
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def copy_file(src, dst_dir, dst_name=None):
    if src is None or not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (dst_name or src.name)
    shutil.copy2(src, dst)
    return dst


def read_text(path):
    if path is None or not path.exists():
        return ''
    return path.read_text(encoding='utf-8', errors='ignore')


def parse_val_interval(config_path, copied_config):
    for path in (config_path, copied_config):
        text = read_text(path)
        match = VAL_INTERVAL_RE.search(text)
        if match:
            return int(match.group(1))
    return None


def parse_log(log_path, val_interval=None):
    lines = read_text(log_path).splitlines()
    validations = []
    best_events = []
    pending_table = None
    table_rows = None

    for line in lines:
        if '|     Class' in line and '|  IoU' in line and '|  Acc' in line:
            table_rows = []
            continue

        if table_rows is not None:
            stripped = line.strip()
            if stripped.startswith('+'):
                if table_rows:
                    pending_table = table_rows
                    table_rows = None
                continue
            row_match = CLASS_ROW_RE.search(line)
            if row_match:
                table_rows.append({
                    'class': row_match.group(1).strip(),
                    'iou': float(row_match.group(2)),
                    'acc': float(row_match.group(3)),
                })
                continue
            if table_rows:
                pending_table = table_rows
            table_rows = None

        if 'Iter(val)' in line and 'mIoU:' in line:
            metrics = {
                key: float(value)
                for key, value in METRIC_RE.findall(line)
            }
            val_index = len(validations) + 1
            if val_interval is not None:
                metrics['iter'] = val_index * val_interval
            metrics['per_class'] = pending_table or []
            validations.append(metrics)
            pending_table = None

        best_match = BEST_RE.search(line)
        if best_match:
            best_events.append({
                'score': float(best_match.group(1)),
                'metric': best_match.group(2),
                'iter': int(best_match.group(3)),
            })

    best = None
    if best_events:
        best = best_events[-1]
        for item in validations:
            if item.get('iter') == best['iter']:
                best = {**item, **best}
                break
    elif validations:
        best = max(validations, key=lambda item: item.get('mIoU', -1.0))

    final = validations[-1] if validations else None
    return {'validations': validations, 'best': best, 'final': final}


def find_last_checkpoint(work_dir):
    marker = work_dir / 'last_checkpoint'
    if marker.exists():
        text = marker.read_text(encoding='utf-8', errors='ignore').strip()
        if text:
            path = Path(text)
            if not path.is_absolute():
                path = work_dir / path
            if path.exists():
                return path

    iter_ckpts = sorted(
        work_dir.glob('iter_*.pth'),
        key=lambda p: int(re.search(r'iter_(\d+)\.pth', p.name).group(1))
    )
    return iter_ckpts[-1] if iter_ckpts else None


def format_metrics(item):
    if not item:
        return 'N/A'
    parts = []
    for key in ('iter', 'aAcc', 'mIoU', 'mAcc', 'data_time', 'time'):
        if key in item:
            parts.append(f'{key}={item[key]}')
    return ', '.join(parts)


def write_metrics(path, validations):
    with path.open('w', encoding='utf-8') as f:
        f.write('iter,aAcc,mIoU,mAcc,data_time,time\n')
        for item in validations:
            f.write(','.join([
                str(item.get('iter', '')),
                str(item.get('aAcc', '')),
                str(item.get('mIoU', '')),
                str(item.get('mAcc', '')),
                str(item.get('data_time', '')),
                str(item.get('time', '')),
            ]))
            f.write('\n')


def write_per_class(path, best, final):
    with path.open('w', encoding='utf-8') as f:
        for title, item in (('best', best), ('final', final)):
            f.write(f'[{title}]\n')
            f.write(f'{format_metrics(item)}\n')
            f.write('Class,IoU,Acc\n')
            for row in (item or {}).get('per_class', []):
                f.write(f"{row['class']},{row['iou']},{row['acc']}\n")
            f.write('\n')


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    config_path = Path(args.config).resolve() if args.config else None

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = latest_file(work_dir.rglob('*.log'))
    vis_config = latest_file(work_dir.rglob('vis_data/config.py'))
    scalars = latest_file(work_dir.rglob('vis_data/scalars.json'))
    run_json = latest_file(work_dir.rglob('vis_data/*.json'))
    work_config = latest_file(work_dir.glob('*.py'))

    copied = {}
    copied['source_config'] = copy_file(config_path, out_dir, 'source_config.py')
    copied['config'] = copy_file(vis_config or work_config or config_path,
                                 out_dir, 'config.py')
    copied['log'] = copy_file(log_path, out_dir, 'train.log')
    copied['scalars'] = copy_file(scalars, out_dir, 'scalars.json')
    copied['run_json'] = copy_file(run_json, out_dir, 'run.json')
    copied['last_checkpoint_marker'] = copy_file(work_dir / 'last_checkpoint',
                                                out_dir, 'last_checkpoint')

    best_ckpt = latest_file(work_dir.glob('best_mIoU*.pth'))
    last_ckpt = find_last_checkpoint(work_dir)
    if not args.no_weights:
        copied['best_checkpoint'] = copy_file(best_ckpt, out_dir)
        copied['last_checkpoint'] = copy_file(last_ckpt, out_dir)

    val_interval = args.val_interval or parse_val_interval(config_path,
                                                           copied['config'])
    parsed = parse_log(log_path, val_interval=val_interval)

    write_metrics(out_dir / 'metrics.txt', parsed['validations'])
    write_per_class(out_dir / 'per_class_iou.txt', parsed['best'],
                    parsed['final'])

    metadata = {
        'name': args.name,
        'note': args.note,
        'archived_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'work_dir': str(work_dir),
        'out_dir': str(out_dir),
        'source_config': str(config_path) if config_path else None,
        'log': str(log_path) if log_path else None,
        'best_checkpoint': str(best_ckpt) if best_ckpt else None,
        'last_checkpoint': str(last_ckpt) if last_ckpt else None,
        'val_interval': val_interval,
        'best': parsed['best'],
        'final': parsed['final'],
        'copied_files': {
            key: str(value) for key, value in copied.items() if value
        },
    }
    (out_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding='utf-8')

    summary_lines = [
        f"Experiment: {args.name or out_dir.name}",
        f"Archived at: {metadata['archived_at']}",
        f"Work dir: {work_dir}",
        f"Source config: {config_path or 'N/A'}",
        f"Validation interval: {val_interval or 'N/A'}",
        '',
        f"Best validation: {format_metrics(parsed['best'])}",
        f"Final validation: {format_metrics(parsed['final'])}",
        '',
        f"Best checkpoint: {best_ckpt or 'N/A'}",
        f"Last checkpoint: {last_ckpt or 'N/A'}",
    ]
    if args.note:
        summary_lines.extend(['', f'Note: {args.note}'])
    summary_lines.extend(['', 'Copied files:'])
    summary_lines.extend([
        f"- {key}: {value.name}"
        for key, value in copied.items() if value
    ])
    (out_dir / 'summary.txt').write_text(
        '\n'.join(summary_lines) + '\n',
        encoding='utf-8')

    print(f'Archived experiment to: {out_dir}')
    print(f"Best: {format_metrics(parsed['best'])}")
    print(f"Final: {format_metrics(parsed['final'])}")


if __name__ == '__main__':
    main()
