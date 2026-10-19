"""Validate the fixed experiment splits and optionally create boundary masks.

This script expects already converted single-channel semantic class masks.
It never downloads data, regenerates splits, or overwrites existing boundaries.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def foreground_boundary(mask):
    binary = Image.fromarray(((mask > 0) * 255).astype(np.uint8))
    dilated = np.asarray(binary.filter(ImageFilter.MaxFilter(3)), dtype=np.int16)
    eroded = np.asarray(binary.filter(ImageFilter.MinFilter(3)), dtype=np.int16)
    return (dilated - eroded).astype(np.uint8)


def validate(root, dataset, write_boundaries=False, limit=None):
    root = Path(root)
    split_dir = root / ('splits' if dataset == 'SUIM' else 'ImageSets/Segmentation')
    nclasses = 8 if dataset == 'SUIM' else 5
    splits = {}
    for split in ('train', 'val'):
        ids = (split_dir / f'{split}.txt').read_text(encoding='utf-8-sig').split()
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {split}')
        if any('/' in item or '\\' in item or item in ('.', '..') for item in ids):
            raise ValueError(f'Invalid sample ID in {split}')
        splits[split] = ids
    if set(splits['train']) & set(splits['val']):
        raise ValueError('The training and validation lists overlap')
    ids = splits['train'] + splits['val']
    if limit is not None:
        ids = ids[:limit]
    written = 0
    missing_boundaries = 0
    for sample_id in ids:
        image = root / 'JPEGImages' / f'{sample_id}.jpg'
        mask_file = root / 'Segmentation' / f'{sample_id}.png'
        if not image.is_file() or not mask_file.is_file():
            raise FileNotFoundError(f'Missing image or semantic mask for {sample_id}')
        with Image.open(mask_file) as source:
            mask = np.asarray(source)
        if mask.ndim != 2:
            raise ValueError(f'{mask_file} is not a single-channel class-index mask')
        invalid = set(np.unique(mask).tolist()) - set(range(nclasses))
        if invalid:
            raise ValueError(f'{mask_file} has unexpected class IDs: {sorted(invalid)}')
        with Image.open(image) as source:
            if source.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f'Image/mask dimensions differ for {sample_id}')
        boundary_file = root / 'boundary' / f'{sample_id}.png'
        expected = foreground_boundary(mask)
        if boundary_file.exists():
            with Image.open(boundary_file) as source:
                actual = np.asarray(source)
            if not np.array_equal(actual, expected):
                raise ValueError(f'Boundary differs from the foreground-gradient rule: {boundary_file}')
        elif write_boundaries:
            boundary_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(expected).save(boundary_file)
            written += 1
        else:
            missing_boundaries += 1
    return dict(dataset=dataset, train=len(splits['train']), val=len(splits['val']),
                samples_checked=len(ids), boundaries_written=written,
                missing_boundaries=missing_boundaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--dataset', required=True, choices=['SUIM', 'USEG'])
    parser.add_argument('--write-boundaries', action='store_true')
    parser.add_argument('--limit', type=int, help='Check only the first N samples')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    print(json.dumps(validate(args.data_root, args.dataset, args.write_boundaries, args.limit), indent=2))
