"""End-to-end demo: synthesize guitar-ish input, run AURA-G, print results."""
import numpy as np
import soundfile as sf
from aura_g.pipeline import AURAG
from aura_g.physical_model import PhysicalGuitarModel

def make_test_input(sr=48000, duration=3.0):
    model = PhysicalGuitarModel(sr=sr, f0=196.0)
    buf = np.zeros(int(sr * duration), dtype=np.float32)
    for f0, t in [(196.0, 0.0), (98.0, 0.75), (123.5, 1.5), (164.8, 2.25)]:
        model.string.f0 = f0
        model.string.delay = int(sr / f0)
        model.string.buffer = np.random.randn(model.string.delay) * 0.01
        seg = model.synthesize(int(sr * 0.7), fret=2, pluck_pos=0.3, dynamics=0.9)
        s = int(t * sr)
        e = min(s + len(seg), len(buf))
        buf[s:e] += seg[:e - s]
    buf /= (np.max(np.abs(buf)) + 1e-9)
    return buf

def main():
    sr = 48000
    print("Rendering test input...")
    dry = make_test_input(sr=sr, duration=3.0)
    sf.write("input.wav", dry, sr)
    print("Running AURA-G pipeline...")
    ag = AURAG(sr=sr, hop=480)
    out, events, bus = ag.process(dry)
    sf.write("output.wav", out, sr)
    print(f"\nLatent bus snapshots: {len(bus)}")
    print(f"Sample bus vector (first 16 dims):")
    print(np.round(bus[0].read()[:16], 3))
    print(f"\nTranscription events ({len(events)}):")
    for e in events:
        print(f"  t={e['start']:.2f}s  {e['name']:>4s}  "
              f"string={e['string']}  fret={e['fret']}  vel={e['velocity']:.2f}")
    print("\nDone. Wrote input.wav and output.wav")

if __name__ == "__main__":
    main()
