"""Convert AURA-G transcription events to Standard MIDI File (.mid)."""
import numpy as np
from pathlib import Path

class MIDIWriter:
    """Minimal, dependency-free Standard MIDI File writer (format 0)."""

    def __init__(self, ticks_per_beat=480, tempo_bpm=120):
        self.ticks_per_beat = ticks_per_beat
        self.tempo_us = int(60_000_000 / tempo_bpm)
        self.events = []  # (abs_tick, status_byte, data1, data2)

    def _secs_to_ticks(self, secs):
        beats = secs * (self.tempo_us / 60_000_000) * (120 / 120)
        return int(round(beats * self.ticks_per_beat))

    def add_note(self, start_sec, end_sec, midi, velocity=100, channel=0):
        s = self._secs_to_ticks(start_sec)
        e = self._secs_to_ticks(end_sec)
        v = max(1, min(127, int(velocity * 127)))
        self.events.append((s, 0x90 | channel, midi & 0x7F, v))
        self.events.append((e, 0x80 | channel, midi & 0x7F, 0))

    @staticmethod
    def _vlq(value):
        """Variable-length quantity encoding."""
        if value < 0: value = 0
        out = [value & 0x7F]
        value >>= 7
        while value:
            out.append(0x80 | (value & 0x7F))
            value >>= 7
        return bytes(reversed(out))

    def _track_chunk(self):
        self.events.sort(key=lambda x: (x[0], x[1] & 0xF0 == 0x90))  # note-offs first at same tick
        track = bytearray()
        # Tempo meta
        track += self._vlq(0) + b'\xFF\x51\x03' + self.tempo_us.to_bytes(3, 'big')
        # Time signature 4/4
        track += self._vlq(0) + b'\xFF\x58\x04\x04\x02\x18\x08'
        # Track name
        name = b'AURA-G'
        track += self._vlq(0) + b'\xFF\x03' + self._vlq(len(name)) + name

        last_tick = 0
        for tick, status, d1, d2 in self.events:
            delta = tick - last_tick
            track += self._vlq(delta) + bytes([status, d1, d2])
            last_tick = tick

        # End of track
        track += self._vlq(0) + b'\xFF\x2F\x00'
        return b'MTrk' + len(track).to_bytes(4, 'big') + bytes(track)

    def write(self, path):
        header = b'MThd' + (6).to_bytes(4, 'big') + \
                 (0).to_bytes(2, 'big') + \
                 (1).to_bytes(2, 'big') + \
                 self.ticks_per_beat.to_bytes(2, 'big')
        track = self._track_chunk()
        Path(path).write_bytes(header + track)

def events_to_midi(events, output_path, tempo_bpm=120,
                   default_dur=0.5, velocity_scale=127):
    """Convert AURA-G event list to a .mid file.

    Each event should have 'start', 'midi', and optionally 'end'/'velocity'.
    """
    writer = MIDIWriter(tempo_bpm=tempo_bpm)
    for ev in events:
        start = ev['start']
        end = ev.get('end', start + default_dur)
        midi = ev['midi']
        vel = int(ev.get('velocity', 0.8) * velocity_scale)
        writer.add_note(start, end, midi, velocity=vel)
    writer.write(output_path)
    return output_path

def tabs_to_midi_text(events):
    """Render a simple ASCII tab from events for terminal display."""
    lines = ['e|', 'B|', 'G|', 'D|', 'A|', 'E|']
    open_strings = [64, 59, 55, 50, 45, 40]
    for ev in events:
        midi = ev['midi']
        placed = False
        for s_idx, op in enumerate(open_strings):
            fret = midi - op
            if 0 <= fret <= 22:
                lines[s_idx] += f'{fret}-'
                placed = True
                break
        if not placed:
            for i in range(6):
                lines[i] += '--'
    return '\n'.join(lines)

if __name__ == '__main__':
    # Quick self-test
    demo_events = [
        {'start': 0.0, 'end': 0.4, 'midi': 40, 'velocity': 0.9},
        {'start': 0.5, 'end': 0.9, 'midi': 43, 'velocity': 0.85},
        {'start': 1.0, 'end': 1.4, 'midi': 47, 'velocity': 0.88},
        {'start': 1.5, 'end': 1.9, 'midi': 52, 'velocity': 0.92},
    ]
    events_to_midi(demo_events, 'demo_output.mid')
    print("Wrote demo_output.mid")
    print(tabs_to_midi_text(demo_events))
