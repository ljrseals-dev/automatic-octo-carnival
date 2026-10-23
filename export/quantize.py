"""Post-training quantization (dynamic + static INT8)."""
import argparse
from pathlib import Path

import torch
import torch.quantization as tq

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.ddsp_amp import DDSPAmp
from aura_g.rnn_fx import SampleRateIndependentFX

def dynamic_quantize_fx(output: str = "exports/fx_int8.pt", sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = SampleRateIndependentFX(sr_in=sr, sr_internal=sr).eval()
    q = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8)
    torch.save(q.state_dict(), output)
    print(f"Exported: {output}")

def static_quantize_amp(output: str = "exports/amp_int8.pt", sr: int = 48000):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model = DDSPAmp(sr=sr).eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prep = torch.quantization.prepare(model, inplace=False)

    # Calibrate with random inputs (swap for real calibration data)
    with torch.no_grad():
        for _ in range(20):
            prep(torch.randn(1, 4096), torch.rand(1, 32))

    q = torch.quantization.convert(prep, inplace=False)
    torch.save(q.state_dict(), output)
    print(f"Exported: {output}")

def compare_sizes(orig_path: str, quant_path: str):
    a = Path(orig_path).stat().st_size if Path(orig_path).exists() else 0
    b = Path(quant_path).stat().st_size if Path(quant_path).exists() else 0
    print(f"Original: {a / 1e6:.2f} MB")
    print(f"Quantized: {b / 1e6:.2f} MB")
    if a:
        print(f"Reduction: {100 * (1 - b / a):.1f}%")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['amp', 'fx', 'all'], default='all')
    p.add_argument('--out_dir', default='exports')
    a = p.parse_args()
    out = Path(a.out_dir)
    if a.model in ('fx', 'all'):
        dynamic_quantize_fx(str(out / 'fx_int8.pt'))
    if a.model in ('amp', 'all'):
        static_quantize_amp(str(out / 'amp_int8.pt'))
