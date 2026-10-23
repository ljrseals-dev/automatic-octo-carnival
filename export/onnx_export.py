"""Export AURA-G models to ONNX."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.ddsp_amp import DDSPAmp
from aura_g.rnn_fx import SampleRateIndependentFX
from transcription.conformer import GuitarConformer

def export_amp(output: str = "exports/amp.onnx", block: int = 8192,
               sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = DDSPAmp(sr=sr).eval()
    dummy_audio = torch.randn(1, block)
    dummy_latent = torch.rand(1, 32)

    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x, latent):
            return self.m(x, latent)

    w = Wrapper(model).eval()
    torch.onnx.export(
        w, (dummy_audio, dummy_latent), output,
        input_names=['audio', 'latent'],
        output_names=['output'],
        dynamic_axes={
            'audio': {1: 'time'},
            'output': {1: 'time'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported: {output}")

def export_fx(output: str = "exports/fx.onnx", block: int = 8192,
              sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = SampleRateIndependentFX(
        sr_in=sr, sr_internal=sr, hidden=32).eval()

    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            return self.m(x, torch.zeros(1, 32))

    w = Wrapper(model).eval()
    dummy = torch.randn(1, block)
    torch.onnx.export(
        w, (dummy,), output,
        input_names=['audio'], output_names=['output'],
        dynamic_axes={'audio': {1: 'time'}, 'output': {1: 'time'}},
        opset_version=17,
    )
    print(f"Exported: {output}")

def export_conformer(output: str = "exports/conformer.onnx",
                     frames: int = 200, n_mels: int = 229):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = GuitarConformer(n_mels=n_mels).eval()
    dummy = torch.randn(1, frames, n_mels)

    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, mel):
            pitch, onset, offset = self.m(mel)
            return pitch, onset, offset

    w = Wrapper(model).eval()
    torch.onnx.export(
        w, (dummy,), output,
        input_names=['mel'], output_names=['pitch', 'onset', 'offset'],
        dynamic_axes={
            'mel': {1: 'frames'},
            'pitch': {1: 'frames'},
            'onset': {1: 'frames'},
            'offset': {1: 'frames'},
        },
        opset_version=17,
    )
    print(f"Exported: {output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['amp', 'fx', 'conformer', 'all'],
                   default='all')
    p.add_argument('--out_dir', default='exports')
    a = p.parse_args()

    out = Path(a.out_dir)
    if a.model in ('amp', 'all'):
        export_amp(str(out / 'amp.onnx'))
    if a.model in ('fx', 'all'):
        export_fx(str(out / 'fx.onnx'))
    if a.model in ('conformer', 'all'):
        export_conformer(str(out / 'conformer.onnx'))
