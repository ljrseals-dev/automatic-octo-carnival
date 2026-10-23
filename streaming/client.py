"""WebSocket client: stream mic audio to server, play back."""
import asyncio
import json
import numpy as np

try:
    import sounddevice as sd
    import websockets
except ImportError:
    raise SystemExit("pip install sounddevice websockets")

SR = 48000
BLOCK = 1024

class StreamClient:
    def __init__(self, url: str = "ws://localhost:8765"):
        self.url = url
        self.in_q = asyncio.Queue(maxsize=32)
        self.out_q = asyncio.Queue(maxsize=32)

    def mic_callback(self, indata, frames, time_info, status):
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16)
        try:
            self.in_q.put_nowait(pcm.tobytes())
        except asyncio.QueueFull:
            pass

    async def receive_loop(self, ws):
        async for message in ws:
            if isinstance(message, bytes):
                pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32)
                pcm /= 32768.0
                try:
                    self.out_q.put_nowait(pcm)
                except asyncio.QueueFull:
                    pass
            else:
                msg = json.loads(message)
                if msg.get('type') == 'events':
                    for e in msg['events']:
                        print(f"  t={e['start']:.2f}s {e['name']}")

    async def send_loop(self, ws):
        while True:
            data = await self.in_q.get()
            await ws.send(data)

    def playback_callback(self, outdata, frames, time_info, status):
        try:
            chunk = self.out_q.get_nowait()
        except asyncio.QueueEmpty:
            outdata.fill(0)
            return
        n = min(frames, len(chunk))
        outdata[:n, 0] = chunk[:n]
        if n < frames:
            outdata[n:, 0] = 0

    async def run(self):
        async with websockets.connect(self.url) as ws:
            hello = await ws.recv()
            print(f"Connected: {hello}")

            recv_task = asyncio.create_task(self.receive_loop(ws))
            send_task = asyncio.create_task(self.send_loop(ws))

            with sd.InputStream(samplerate=SR, blocksize=BLOCK, channels=1,
                                callback=self.mic_callback, dtype='float32'):
                with sd.OutputStream(samplerate=SR, blocksize=BLOCK, channels=1,
                                     callback=self.playback_callback,
                                     dtype='float32'):
                    print("Streaming. Ctrl+C to stop.")
                    try:
                        await asyncio.gather(recv_task, send_task)
                    except asyncio.CancelledError:
                        pass

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='ws://localhost:8765')
    a = p.parse_args()
    try:
        asyncio.run(StreamClient(a.url).run())
    except KeyboardInterrupt:
        print("\nStopped.")
