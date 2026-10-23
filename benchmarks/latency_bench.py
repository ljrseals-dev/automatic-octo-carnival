"""Measure end-to-end latency of the AURA-G pipeline."""
import argparse
import time
from statistics import mean, stdev

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.pipeline import AURAG
from aura_g.ddsp_amp import DDSPAmp
from aura_g.rnn_fx import SampleRateIndependentFX

def bench_pipeline(sr=48000, duration=2.0, n_iter=10):
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
    ag = AURAG(sr=sr)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ag.process(audio)
        times.append(time.perf_counter() - t0)

    rtf = mean(times) / duration
    print(f"Pipeline: mean={mean(times)*1000:.1f} ms "
          f"std={stdev(times)*1000:.1f} ms "
          f"RTF={rtf:.3f} ({'real-time' if rtf < 1 else 'SLOWER'})")

def bench_amp(sr=48000, block=4096, n_iter=50):
    amp = DDSPAmp(sr=sr).eval()
    x = torch.randn(1, block)
    latent = torch.rand(1, 32)

    with torch.no_grad():
        for _ in range(5):
            amp(x, latent)

        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            amp(x, latent)
            times.append(time.perf_counter() - t0)

    dt = mean(times)
    print(f"Amp: mean={dt*1000:.3f} ms per {block} samples "
          f"({block / sr * 1000:.1f} ms audio) "
          f"RTF={dt / (block / sr):.4f}")

def bench_fx(sr=48000, block=4096, n_iter=50):
    fx = SampleRateIndependentFX(sr_in=sr, sr_internal=sr).eval()
    x = torch.randn(1, block)
    with torch.no_grad():
        for _ in range(5):
            fx(x, torch.zeros(1, 32))
        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            fx(x, torch.zeros(1, 32))
            times.append(time.perf_counter() - t0)
    dt = mean(times)
    print(f"FX:  mean={dt*1000:.3f} ms per {block} samples "
          f"RTF={dt / (block / sr):.4f}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sr', type=int, default=48000)
    p.add_argument('--duration', type=float, default=2.0)
    p.add_argument('--iters', type=int, default=10)
    a = p.parse_args()

    print(f"=== Latency benchmarks (sr={a.sr}) ===")
    bench_pipeline(a.sr, a.duration, a.iters)
    print()
    bench_amp(a.sr)
    bench_fx(a.sr)
