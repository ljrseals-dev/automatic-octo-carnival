"""Async job worker with a simple file-based queue."""
import json
import time
import traceback
from pathlib import Path

from .config import get_config
from .logging_setup import setup_logging
from .paths import output_dir

log = setup_logging()
cfg = get_config()

class JobQueue:
    def __init__(self, queue_dir: str = None):
        self.root = Path(queue_dir) if queue_dir else (output_dir() / "queue")
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = self.root / "pending"
        self.running = self.root / "running"
        self.done = self.root / "done"
        self.failed = self.root / "failed"
        for d in (self.pending, self.running, self.done, self.failed):
            d.mkdir(exist_ok=True)

    def enqueue(self, payload: dict) -> str:
        import uuid
        jid = uuid.uuid4().hex[:12]
        payload['job_id'] = jid
        payload['created_at'] = time.time()
        (self.pending / f"{jid}.json").write_text(json.dumps(payload))
        return jid

    def claim(self):
        for f in sorted(self.pending.glob("*.json")):
            target = self.running / f.name
            try:
                f.rename(target)
                return target
            except FileNotFoundError:
                continue
        return None

    def finish(self, path: Path, success: bool, result: dict = None):
        dest = self.done if success else self.failed
        payload = json.loads(path.read_text())
        payload['finished_at'] = time.time()
        if result:
            payload['result'] = result
        (dest / path.name).write_text(json.dumps(payload))
        path.unlink(missing_ok=True)

def _process(job: dict) -> dict:
    from aura_g.pipeline import AURAG
    import soundfile as sf

    audio_path = job['audio_path']
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    ag = AURAG(sr=sr)
    out, events, _ = ag.process(audio)

    out_path = output_dir() / f"{job['job_id']}_out.wav"
    sf.write(str(out_path), out, sr)

    return {'output': str(out_path), 'events': len(events), 'sr': sr}

def run_worker(poll_interval: float = 1.0, once: bool = False):
    q = JobQueue()
    log.info(f"Worker started (queue={q.root})")
    while True:
        job_file = q.claim()
        if job_file is None:
            if once:
                break
            time.sleep(poll_interval)
            continue

        job = json.loads(job_file.read_text())
        log.info(f"Processing job {job['job_id']}: {job['audio_path']}")
        try:
            result = _process(job)
            q.finish(job_file, True, result)
            log.info(f"Job {job['job_id']} done: {result['events']} events")
        except Exception as e:
            log.error(f"Job {job['job_id']} failed: {e}")
            log.error(traceback.format_exc())
            q.finish(job_file, False, {'error': str(e)})

        if once:
            break
    log.info("Worker stopped")

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--poll', type=float, default=1.0)
    p.add_argument('--enqueue')
    a = p.parse_args()

    if a.enqueue:
        q = JobQueue()
        jid = q.enqueue({'audio_path': a.enqueue})
        print(f"Enqueued job: {jid}")
    else:
        run_worker(a.poll, a.once)
