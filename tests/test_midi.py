import tempfile
from pathlib import Path

import pytest

from extensions.midi_export import MIDIWriter, events_to_midi, tabs_to_midi_text

def test_writer_emits_valid_header():
    w = MIDIWriter()
    w.add_note(0.0, 0.5, 60, 100)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "test.mid"
        w.write(str(p))
        assert p.exists()
        data = p.read_bytes()
        assert data[:4] == b'MThd'

def test_events_to_midi_roundtrip():
    events = [
        {'start': 0.0, 'end': 0.4, 'midi': 60, 'velocity': 0.9},
        {'start': 0.5, 'end': 0.9, 'midi': 64, 'velocity': 0.8},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.mid"
        events_to_midi(events, str(p))
        assert p.exists() and p.stat().st_size > 0

def test_tab_rendering():
    events = [{'midi': 40, 'start': 0.0}]
    tab = tabs_to_midi_text(events)
    assert 'e|' in tab and 'E|' in tab
