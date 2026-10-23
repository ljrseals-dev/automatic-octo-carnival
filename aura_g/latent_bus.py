"""Shared Latent Control Bus — the core abstraction of AURA-G."""
from dataclasses import dataclass, field
import numpy as np

BUS_DIM = 32
PITCH_GESTURE = slice(0, 8)
TIMBRE        = slice(8, 16)
EXPRESSION    = slice(16, 24)
TRANSCRIPTION = slice(24, 28)
RESERVED      = slice(28, 32)

@dataclass
class LatentBus:
    vector: np.ndarray = field(default_factory=lambda: np.zeros(BUS_DIM, dtype=np.float32))
    sample_rate: int = 48000
    hop: int = 480

    def write_pitch_gesture(self, f0_log_hz, onset, bend, vib_rate, vib_depth,
                            pluck_pos, dynamics, palm_mute):
        self.vector[PITCH_GESTURE] = np.array(
            [f0_log_hz, onset, bend, vib_rate, vib_depth,
             pluck_pos, dynamics, palm_mute], dtype=np.float32)

    def write_timbre(self, centroid, rolloff, flatness, zcr,
                     harmonic_ratio, even_odd, attack_slope, sustain):
        self.vector[TIMBRE] = np.array(
            [centroid, rolloff, flatness, zcr,
             harmonic_ratio, even_odd, attack_slope, sustain], dtype=np.float32)

    def write_expression(self, macros: np.ndarray):
        assert macros.shape == (8,)
        self.vector[EXPRESSION] = macros.astype(np.float32)

    def write_transcription(self, note_id, string, fret, confidence):
        self.vector[TRANSCRIPTION] = np.array(
            [note_id, string, fret, confidence], dtype=np.float32)

    def read(self):
        return self.vector.copy()

    def get_pitch_gesture(self): return self.vector[PITCH_GESTURE]
    def get_timbre(self):        return self.vector[TIMBRE]
    def get_expression(self):    return self.vector[EXPRESSION]
    def get_transcription(self): return self.vector[TRANSCRIPTION]
