"""Measure transcription accuracy against ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.front_end import yin_pitch
from aura_g.transcription import hz_to_midi

def evaluate_file(audio_path: str, notes_path: str, sr: int = 48000):
    audio, _ = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Ground truth notes
    gt = []
    for line in Path(notes_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            gt.append({'onset': float(parts[0]),
                       'offset': float(parts[1]),
                       'midi': int(parts[2])})

    # Predicted notes
    pred = []
    prev = None
    for i in range(0, len(audio) - 2048, 512):
        f0, conf = yin_pitch(audio[i:i + 2048], sr)
        if conf > 0.6 and f0 > 50:
            m = hz_to_midi(f0)
            if m != prev:
                pred.append({'onset': i / sr, 'midi': m})
                prev = m

    # Simple matching
    tp = fp = fn = 0
    used = set()
    for p in pred:
        matched = False
        for i, g in enumerate(gt):
            if i in used:
                continue
            if abs(p['onset'] - g['onset']) < 0.1 and p['midi'] == g['midi']:
                tp += 1
                used.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(gt) - len(used)

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return {'file': audio_path, 'gt': len(gt), 'pred': len(pred),
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1}

def evaluate_manifest(manifest_path: str):
    items = json.loads(Path(manifest_path).read_text())
    results = []
    for item in items:
        r = evaluate_file(item['audio'], item['notes'])
        results.append(r)
        print(f"{Path(item['audio']).name}: "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    if results:
        mean_p = np.mean([r['precision'] for r in results])
        mean_r = np.mean([r['recall'] for r in results])
        mean_f = np.mean([r['f1'] for r in results])
        print(f"\nMean: P={mean_p:.3f} R={mean_r:.3f} F1={mean_f:.3f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    a = p.parse_args()
    evaluate_manifest(a.manifest)
