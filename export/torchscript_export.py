"""TorchScript export for C++ deployment."""
import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.ddsp_amp import DDSPAmp
from aura_g.rnn_fx import SampleRateIndependentFX
from transcription.conformer import GuitarConformer

def export_amp(output: str = "exports/amp.pt", sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = DDSPAmp(sr=sr).eval()
    example = (torch.randn(1, 4096), torch.rand(1, 32))
    scripted = torch.jit.trace(
        lambda x, l: model(x, l), example, strict=False)
    scripted.save(output)
    print(f"Exported: {output}")

def export_fx(output: str = "exports/fx.pt", sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = SampleRateIndependentFX(sr_in=sr, sr_internal=sr).eval()
    example = torch.randn(1, 4096)
    scripted = torch.jit.trace(
        lambda x: model(x, torch.zeros(1, 32)), example, strict=False)
    scripted.save(output)
    print(f"Exported: {output}")

def export_conformer(output: str = "exports/conformer.pt"):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = GuitarConformer().eval()
    example = torch.randn(1, 200, 229)
    scripted = torch.jit.trace(model, example, strict=False)
    scripted.save(output)
    print(f"Exported: {output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['amp', 'fx', 'conformer', 'all'],
                   default='all')
    p.add_argument('--out_dir', default='exports')
    a = p.parse_args()

    out = Path(a.out_dir)
    if a.model in ('amp', 'all'):
        export_amp(str(out / 'amp.pt'))
    if a.model in ('fx', 'all'):
        export_fx(str(out / 'fx.pt'))
    if a.model in ('conformer', 'all'):
        export_conformer(str(out / 'conformer.pt'))
