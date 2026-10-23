"""End-to-end integration tests."""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

def test_full_workflow(tmp_path):
    sr = 48000
    t = np.arange(sr) / sr
    # Synthesize a short guitar-like tone
    from aura_g.physical_model import PhysicalGuitarModel
    m = PhysicalGuitarModel(sr=sr, f0=196.0)
    audio = m.synthesize(int(sr * 1.0), fret=2,
                         pluck_pos=0.3, dynamics=0.9).astype(np.float32)

    in_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    midi_path = tmp_path / "out.mid"
    sf.write(str(in_path), audio, sr)

    # Process through pipeline
    from aura_g.pipeline import AURAG
    ag = AURAG(sr=sr)
    out, events, bus = ag.process(audio)
    sf.write(str(out_path), out, sr)

    # Write MIDI
    if events:
        from extensions.midi_export import events_to_midi
        midi_events = [
            {'start': e['start'], 'end': e['start'] + 0.5,
             'midi': e['midi'], 'velocity': e.get('velocity', 0.8)}
            for e in events
        ]
        events_to_midi(midi_events, str(midi_path))

    assert in_path.exists()
    assert out_path.exists()
    assert len(out) == len(audio)
    assert np.isfinite(out).all()
