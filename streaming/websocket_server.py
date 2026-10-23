"""WebSocket server for real-time streaming inference."""
import asyncio
import base64
import io
import json
import struct

import numpy as np
import soundfile as sf

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aura_g.pipeline import AURAG

class StreamSession:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.ag = AURAG(sr=sr)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window = sr  # 1 s chunks

    def push(self, samples: np.ndarray) -> np.ndarray:
        self.buffer = np.concatenate([self.buffer, samples])
        if len(self.buffer) < self.window:
            return np.zeros(0, dtype=np.float32)

        chunk = self.buffer[:self.window]
        self.buffer = self.buffer[self.window:]

        out, events, _ = self.ag.process(chunk)
        self.last_events = events
        return out.astype(np.float32)

    def drain_events(self):
        evs = getattr(self, 'last_events', [])
        self.last_events = []
        return evs

async def handler(websocket, path=None):
    session = StreamSession()
    await websocket.send(json.dumps({'type': 'hello', 'sr': session.sr}))

    try:
        async for message in websocket:
            if isinstance(message, str):
                # Control message
                msg = json.loads(message)
                if msg.get('type') == 'reset':
                    session = StreamSession()
                    await websocket.send(json.dumps({'type': 'reset_ok'}))
                continue

            # Binary audio: int16 PCM
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32)
            samples /= 32768.0

            out = session.push(samples)
            if len(out) > 0:
                pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
                await websocket.send(pcm.tobytes())
                events = session.drain_events()
                if events:
                    await websocket.send(json.dumps({
                        'type': 'events',
                        'events': [
                            {'start': e['start'], 'name': e['name'],
                             'midi': e['midi']}
                            for e in events
                        ]
                    }))
    except websockets.exceptions.ConnectionClosed:
        pass

async def main(host: str = '0.0.0.0', port: int = 8765):
    print(f"WebSocket server on ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        await asyncio.Future()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8765)
    a = p.parse_args()
    asyncio.run(main(a.host, a.port))
