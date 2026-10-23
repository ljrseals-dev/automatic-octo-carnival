"""Joint note + fingering from latent bus."""
import numpy as np

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4

def hz_to_midi(f0):
    if f0 <= 0: return 0
    return int(round(69 + 12 * np.log2(f0 / 440.0)))

def midi_to_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def fingering_from_midi(midi, max_fret=15):
    candidates = [(s, midi - o) for s, o in enumerate(OPEN_STRINGS)
                  if 0 <= midi - o <= max_fret]
    if not candidates:
        return None
    scored = [(s, f, f * 0.5 + s * 0.1) for s, f in candidates]
    scored.sort(key=lambda x: x[2])
    return scored[0][0], scored[0][1]

class TranscriptionEngine:
    def __init__(self, sr=48000, hop=480):
        self.sr, self.hop = sr, hop
        self.active_note = None

    def process(self, bus_history):
        events = []
        for i, bus in enumerate(bus_history):
            pg = bus.get_pitch_gesture()
            f0 = 440.0 * (2 ** (pg[0] * 6.0)) if pg[0] != 0 else 0.0
            if pg[1] > 0.5 and f0 > 0:
                if self.active_note is not None:
                    events.append(self.active_note)
                midi = hz_to_midi(f0)
                fing = fingering_from_midi(midi)
                self.active_note = {
                    'start': i * self.hop / self.sr,
                    'midi': midi,
                    'name': midi_to_name(midi),
                    'string': fing[0] if fing else -1,
                    'fret': fing[1] if fing else -1,
                    'confidence': 1.0,
                    'velocity': float(pg[6]),
                }
                bus.write_transcription(midi / 127.0,
                                        (fing[0] if fing else -1) / 5.0,
                                        (fing[1] if fing else -1) / 24.0, 1.0)
        if self.active_note is not None:
            events.append(self.active_note)
        return events
