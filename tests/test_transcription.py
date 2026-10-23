import numpy as np
import pytest

from aura_g.transcription import (hz_to_midi, midi_to_name,
                                  fingering_from_midi,
                                  TranscriptionEngine)
from aura_g.latent_bus import LatentBus

def test_hz_to_midi():
    assert hz_to_midi(440.0) == 69
    assert hz_to_midi(220.0) == 57

def test_midi_to_name():
    assert midi_to_name(69) == "A4"
    assert midi_to_name(40) == "E2"

def test_fingering():
    # E2 = MIDI 40, string 0, fret 0
    assert fingering_from_midi(40) == (0, 0)
    # A2 = MIDI 45, string 1, fret 0
    assert fingering_from_midi(45) == (1, 0)

def test_transcription_engine():
    bus = LatentBus()
    bus.write_pitch_gesture(f0_log_hz=np.log2(196 / 440) / 6,
                            onset=1.0, bend=0.0, vib_rate=0.5,
                            vib_depth=0.0, pluck_pos=0.5,
                            dynamics=0.8, palm_mute=0.0)
    eng = TranscriptionEngine()
    events = eng.process([bus])
    assert len(events) >= 1
    assert events[0]['midi'] > 0
