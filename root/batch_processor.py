"""Batch-process a directory of audio files through AURA-G."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .config import get_config
from .logging_setup import setup_logging
from .paths import output_dir

log = setup_logging()

def _process_one(args):
    input_path, output_dir_path, midi_out = args
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent))

    from aura_g.pipeline import AURAG

    input_path = Path(input_path)
    output_dir_path = Path(output_dir_path)

    try:
        audio, sr = sf.read(str(input_path), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        ag = AURAG(sr=sr)
        out, events, bus = ag.process(audio)

        out_path = output_dir_path / f"{input_path.stem}_processed.wav"
        sf.write(str(out_path), out, sr)

        if midi_out and events:
            from extensions.midi_export import events_to_midi
            midi_events = [
                {'start': e['start'], 'end': e['start'] + 0.5,
                 'midi': e['midi'],
                 'velocity': e.get('velocity', 0.8)}
                for e in events
            ]
            midi_path = output_dir_path / f"{input_path.stem}.mid"
            events_to_midi(midi_events, str(midi_path))

        return {
            'input': str(input_path),
            'output': str(out_path),
            'events': len(events),
            'status': 'ok',
        }
    except Exception as e:
        return {
            'input': str(input_path),
            'status': 'error',
            'error': str(e),
        }

def batch_process(input_dir: str, out_dir: Optional[str] = None,
                  pattern: str = "*.wav", workers: int = 4,
                  midi_out: bool = True) -> List[dict]:
    cfg = get_config()
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    out_path = Path(out_dir) if out_dir else (output_dir() / "batch")
    out_path.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.rglob(pattern))
    log.info(f"Found {len(files)} files matching {pattern}")

    tasks = [(str(f), str(out_path), midi_out) for f in files]
    results = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r['status'] == 'ok':
                log.info(f"[{i}/{len(tasks)}] OK  {Path(r['input']).name} "
                         f"({r['events']} events)")
            else:
                log.error(f"[{i}/{len(tasks)}] ERR {Path(r['input']).name}: "
                          f"{r['error']}")

    summary_path = out_path / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    log.info(f"Wrote summary: {summary_path}")

    ok = sum(1 for r in results if r['status'] == 'ok')
    log.info(f"Done: {ok}/{len(results)} succeeded")
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', default=None)
    p.add_argument('--pattern', default='*.wav')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--no-midi', action='store_true')
    a = p.parse_args()
    batch_process(a.input, a.output, a.pattern, a.workers,
                  midi_out=not a.no_midi)
