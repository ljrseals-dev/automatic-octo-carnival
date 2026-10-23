"""Export to TensorFlow Lite for Android/iOS deployment."""
import argparse
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def torch_to_tflite(onnx_path: str, tflite_path: str):
    """ONNX -> TF SavedModel -> TFLite."""
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf
    except ImportError:
        raise SystemExit("pip install onnx onnx-tf tensorflow")

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    saved_dir = Path(tflite_path).parent / "saved_model"
    tf_rep.export_graph(str(saved_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    Path(tflite_path).write_bytes(tflite_model)
    print(f"Exported: {tflite_path} ({len(tflite_model) / 1e6:.2f} MB)")

def benchmark_tflite(tflite_path: str, n_iter: int = 100):
    try:
        import tensorflow as tf
    except ImportError:
        return
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    shape = [d if d > 0 else 1 for d in inp['shape']]
    dummy = np.random.randn(*shape).astype(inp['dtype'])

    import time
    t0 = time.perf_counter()
    for _ in range(n_iter):
        interp.set_tensor(inp['index'], dummy)
        interp.invoke()
        _ = interp.get_tensor(out['index'])
    dt = (time.perf_counter() - t0) / n_iter
    print(f"TFLite inference: {dt*1000:.2f} ms")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--benchmark', action='store_true')
    a = p.parse_args()
    torch_to_tflite(a.onnx, a.output)
    if a.benchmark:
        benchmark_tflite(a.output)
