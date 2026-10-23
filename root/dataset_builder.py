"""Prepare paired datasets for amp/FX training and Conformer training."""
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from .logging_setup import setup_logging
from .paths import data_dir

log = setup_logging()

def pair_files(dry_dir: str, wet_dir: str,
               dry_pattern: str = "*.wav",
               wet_pattern: str = "*.wav") -> List[Tuple[Path, Path]]:
    """Match dry/wet files by stem name."""
    dry_root = Path(dry_dir)
    wet_root = Path(wet_dir)
    pairs = []
    for dry in sorted(dry_root.rglob(dry_pattern)):
        wet = wet_root / f"{dry.stem}{Path(wet_pattern).suffix}"
        if not wet.exists():
            # try recursively
            candidates = list(wet_root.rglob(f"{dry.stem}*"))
            if candidates:
                wet = candidates[0]
        if wet.exists():
            pairs.append((dry, wet))
    return pairs

def validate_pair(dry_path: Path, wet_path: Path,
                  target_sr: int = 48000) -> dict:
    dry, sr_d = sf.read(str(dry_path), dtype='float32')
    wet, sr_w = sf.read(str(wet_path), dtype='float32')
    if dry.ndim > 1: dry = dry.mean(axis=-1)
    if wet.ndim > 1: wet = wet.mean(axis=-1)
    return {
        'dry': str(dry_path),
        'wet': str(wet_path),
        'dry_sr': sr_d, 'wet_sr': sr_w,
        'dry_len': len(dry), 'wet_len': len(wet),
        'sr_match': sr_d == sr_w == target_sr,
        'len_match': abs(len(dry) - len(wet)) < target_sr * 0.1,
    }

def build_paired_manifest(dry_dir: str, wet_dir: str,
                          out_path: str = None) -> str:
    pairs = pair_files(dry_dir, wet_dir)
    log.info(f"Found {len(pairs)} candidate pairs")

    manifest = []
    for dry, wet in pairs:
        info = validate_pair(dry, wet)
        status = 'ok' if (info['sr_match'] and info['len_match']) else 'warn'
        info['status'] = status
        manifest.append(info)

    if out_path is None:
        out_path = data_dir("manifests") / "paired_manifest.json"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))

    n_ok = sum(1 for m in manifest if m['status'] == 'ok')
    log.info(f"Manifest: {out_path} ({n_ok}/{len(manifest)} valid)")
    return str(out_path)

def build_guitarset_manifest(root_dir: str, out_path: str = None) -> str:
    """Build a Conformer manifest from a GuitarSet-style directory tree."""
    root = Path(root_dir)
    clips = []
    for audio in sorted(root.rglob("audio.wav")):
        notes = audio.with_name("notes.txt")
        if not notes.exists():
            continue
        audio_data, sr = sf.read(str(audio), dtype='float32')
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=-1)
        n_notes = sum(1 for _ in notes.read_text().splitlines()
                      if _.strip())
        clips.append({
            'audio': str(audio),
            'notes': str(notes),
            'sr': sr,
            'duration': len(audio_data) / sr,
            'n_notes': n_notes,
        })

    if out_path is None:
        out_path = data_dir("manifests") / "guitarset_manifest.json"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clips, indent=2))

    log.info(f"Conformer manifest: {out_path} ({len(clips)} clips, "
             f"{sum(c['n_notes'] for c in clips)} notes)")
    return str(out_path)

def split_manifest(manifest_path: str, val_ratio: float = 0.1,
                   seed: int = 42) -> Tuple[str, str]:
    path = Path(manifest_path)
    items = json.loads(path.read_text())
    random.Random(seed).shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    val, train = items[:n_val], items[n_val:]

    train_path = path.with_name(path.stem + "_train.json")
    val_path = path.with_name(path.stem + "_val.json")
    train_path.write_text(json.dumps(train, indent=2))
    val_path.write_text(json.dumps(val, indent=2))

    log.info(f"Split: train={len(train)} ({train_path}), "
             f"val={len(val)} ({val_path})")
    return str(train_path), str(val_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    pair = sub.add_parser('pair')
    pair.add_argument('--dry', required=True)
    pair.add_argument('--wet', required=True)
    pair.add_argument('--output', default=None)

    gset = sub.add_parser('guitarset')
    gset.add_argument('--root', required=True)
    gset.add_argument('--output', default=None)

    split = sub.add_parser('split')
    split.add_argument('--manifest', required=True)
    split.add_argument('--val_ratio', type=float, default=0.1)

    a = p.parse_args()
    if a.cmd == 'pair':
        build_paired_manifest(a.dry, a.wet, a.output)
    elif a.cmd == 'guitarset':
        build_guitarset_manifest(a.root, a.output)
    elif a.cmd == 'split':
        split_manifest(a.manifest, a.val_ratio)
