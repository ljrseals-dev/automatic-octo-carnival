"""Measure peak memory for each AURA-G component."""
import argparse
import gc
import tracemalloc

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def measure(fn, label: str):
    gc.collect()
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label}: peak={peak / 1e6:.2f} MB")
    if torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"{label} GPU: peak={gpu_peak:.2f} MB")

def run_all(sr=48000, duration=5.0):
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1

    def load_pipeline():
        from aura_g.pipeline import AURAG
        ag = AURAG(sr=sr)
        ag.process(audio)

    def load_amp():
        from aura_g.ddsp_amp import DDSPAmp
        m = DDSPAmp(sr=sr)
        m(torch.randn(1, 8192))

    def load_conformer():
        from transcription.conformer import GuitarConformer
        m = GuitarConformer()
        m(torch.randn(1, 200, 229))

    measure(load_amp,       "Amp")
    measure(load_pipeline,  "Pipeline")
    measure(load_conformer, "Conformer")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=5.0)
    a = p.parse_args()
    run_all(duration=a.duration)
