"""AURA-G unified CLI."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def run_demo():
    import numpy as np
    import soundfile as sf
    from aura_g.pipeline import AURAG
    from aura_g.physical_model import PhysicalGuitarModel

    print("=" * 60)
    print("AURA-G demo mode")
    print("=" * 60)
    sr = 48000
    model = PhysicalGuitarModel(sr=sr, f0=196.0)
    buf = np.zeros(sr * 3, dtype=np.float32)
    for f0, t in [(196.0, 0.0), (98.0, 0.75), (123.5, 1.5), (164.8, 2.25)]:
        model.string.f0 = f0
        model.string.delay = int(sr / f0)
        model.string.buffer = np.random.randn(model.string.delay) * 0.01
        seg = model.synthesize(int(sr * 0.7), fret=2, pluck_pos=0.3, dynamics=0.9)
        s = int(t * sr)
        e = min(s + len(seg), len(buf))
        buf[s:e] += seg[:e - s]
    buf /= (np.max(np.abs(buf)) + 1e-9)
    sf.write("demo_input.wav", buf, sr)
    ag = AURAG(sr=sr)
    out, events, bus = ag.process(buf)
    sf.write("demo_output.wav", out, sr)
    print(f"Input:  demo_input.wav")
    print(f"Output: demo_output.wav")
    print(f"Bus snapshots: {len(bus)}")
    print(f"Transcription events: {len(events)}")
    for e in events:
        print(f"  t={e['start']:.2f}s  {e['name']}  "
              f"string={e['string']}  fret={e['fret']}")

def run_live():
    from realtime.live_engine import RealtimeAURAG
    RealtimeAURAG(sr=48000, blocksize=256).run()

def run_train_amp(dry, wet):
    from training.train_amp import train_amp
    train_amp(dry, wet, epochs=50, output_dir='checkpoints/amp')

def run_train_fx(dry, wet):
    from training.train_fx import train_fx
    train_fx(dry, wet, epochs=30, output_dir='checkpoints/fx')

def run_transcribe(audio_path):
    import soundfile as sf
    from aura_g.front_end import yin_pitch
    from aura_g.transcription import hz_to_midi, midi_to_name

    print(f"Transcribing: {audio_path}")
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    events, prev = [], None
    for i in range(0, len(audio) - 2048, 512):
        f0, conf = yin_pitch(audio[i:i + 2048], sr)
        if conf > 0.6 and f0 > 50:
            name = midi_to_name(hz_to_midi(f0))
            if name != prev:
                events.append({'start': i / sr, 'name': name,
                               'confidence': conf})
                prev = name
    print(f"\nDetected {len(events)} note events:")
    for e in events:
        print(f"  t={e['start']:.3f}s  {e['name']}  (conf={e['confidence']:.2f})")

def main():
    p = argparse.ArgumentParser(description='AURA-G console')
    p.add_argument('--mode', required=True,
                   choices=['demo', 'live', 'train_amp', 'train_fx', 'transcribe'])
    p.add_argument('--dry')
    p.add_argument('--wet')
    p.add_argument('--input')
    a = p.parse_args()

    if a.mode == 'demo':
        run_demo()
    elif a.mode == 'live':
        run_live()
    elif a.mode == 'train_amp':
        if not a.dry or not a.wet:
            print("Error: --dry and --wet required"); sys.exit(1)
        run_train_amp(a.dry, a.wet)
    elif a.mode == 'train_fx':
        if not a.dry or not a.wet:
            print("Error: --dry and --wet required"); sys.exit(1)
        run_train_fx(a.dry, a.wet)
    elif a.mode == 'transcribe':
        if not a.input:
            print("Error: --input required"); sys.exit(1)
        run_transcribe(a.input)

if __name__ == '__main__':
    main()
