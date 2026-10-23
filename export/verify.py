"""Verify exported models produce numerically equivalent output."""
import argparse
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_onnx_vs_torch(onnx_path: str, torch_model_fn,
                         input_names: list, example_inputs: list,
                         rtol: float = 1e-3, atol: float = 1e-4):
    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        return False

    session = ort.InferenceSession(onnx_path,
                                   providers=['CPUExecutionProvider'])
    np_inputs = [x.detach().numpy() for x in example_inputs]

    onnx_out = session.run(None, dict(zip(input_names, np_inputs)))

    torch_model = torch_model_fn().eval()
    with torch.no_grad():
        torch_out = torch_model(*example_inputs)

    if not isinstance(torch_out, (tuple, list)):
        torch_out = (torch_out,)

    all_ok = True
    for i, (o, t) in enumerate(zip(onnx_out, torch_out)):
        diff = np.abs(o - t.detach().numpy())
        rel = diff / (np.abs(t.detach().numpy()) + 1e-8)
        ok = np.allclose(o, t.detach().numpy(), rtol=rtol, atol=atol)
        print(f"  output[{i}]: max_abs={diff.max():.6e} "
              f"max_rel={rel.max():.6e} ok={ok}")
        all_ok = all_ok and ok
    return all_ok

def verify_amp(onnx_path: str = "exports/amp.onnx"):
    from aura_g.ddsp_amp import DDSPAmp
    model = DDSPAmp()

    class Wrapper(torch.nn.Module):
        def __init__(self): super().__init__(); self.m = model
        def forward(self, x, latent): return self.m(x, latent)

    inputs = [torch.randn(1, 8192), torch.rand(1, 32)]
    print("Verifying amp.onnx")
    return verify_onnx_vs_torch(onnx_path, Wrapper,
                                ['audio', 'latent'], inputs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--kind', choices=['amp'], default='amp')
    a = p.parse_args()
    if a.kind == 'amp':
        ok = verify_amp(a.onnx)
        print("PASS" if ok else "FAIL")
