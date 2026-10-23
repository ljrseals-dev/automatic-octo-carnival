"""FastAPI HTTP server for AURA-G inference."""
import io
import uuid
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
    from fastapi.responses import Response, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise SystemExit("Install API dependencies: pip install fastapi uvicorn "
                     "python-multipart")

from .config import get_config
from .logging_setup import setup_logging
from .paths import output_dir
from .telemetry import session

log = setup_logging()
app = FastAPI(title="AURA-G API", version="0.1.0")
cfg = get_config()

# ------------------------------------------------------------------
# In-memory job store
# ------------------------------------------------------------------
JOBS: dict = {}

class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    finished_at: Optional[float] = None
    events: int = 0
    error: Optional[str] = None

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": cfg.device,
            "version": "0.1.0", "sr": cfg.audio.sample_rate}

@app.get("/metrics")
def metrics():
    return session().summary()

# ------------------------------------------------------------------
# Synchronous processing
# ------------------------------------------------------------------
@app.post("/process")
async def process_audio(file: UploadFile = File(...),
                        midi: bool = False):
    t0 = time.perf_counter()
    raw = await file.read()
    if len(raw) > cfg.api.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")

    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype='float32')
    except Exception as e:
        raise HTTPException(400, f"Invalid audio: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    from aura_g.pipeline import AURAG
    ag = AURAG(sr=sr)
    out, events, _ = ag.process(audio)
    session().record("process_sec", time.perf_counter() - t0)
    session().count("files_processed")

    buf = io.BytesIO()
    sf.write(buf, out, sr, format='WAV')
    buf.seek(0)

    headers = {
        "X-Event-Count": str(len(events)),
        "X-Sample-Rate": str(sr),
    }
    if midi and events:
        from extensions.midi_export import events_to_midi
        midi_events = [
            {'start': e['start'], 'end': e['start'] + 0.5,
             'midi': e['midi'], 'velocity': e.get('velocity', 0.8)}
            for e in events
        ]
        midi_path = output_dir() / f"{uuid.uuid4().hex}.mid"
        events_to_midi(midi_events, str(midi_path))
        headers["X-MIDI-Path"] = str(midi_path)

    return Response(content=buf.read(), media_type="audio/wav",
                    headers=headers)

# ------------------------------------------------------------------
# Async job queue
# ------------------------------------------------------------------
@app.post("/jobs", response_model=JobStatus)
async def create_job(background: BackgroundTasks,
                     file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    raw = await file.read()

    JOBS[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'created_at': time.time(),
        'events': 0,
        'error': None,
    }

    async def run():
        JOBS[job_id]['status'] = 'running'
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            from aura_g.pipeline import AURAG
            ag = AURAG(sr=sr)
            out, events, _ = ag.process(audio)
            out_path = output_dir() / f"{job_id}.wav"
            sf.write(str(out_path), out, sr)
            JOBS[job_id]['status'] = 'done'
            JOBS[job_id]['events'] = len(events)
            JOBS[job_id]['output'] = str(out_path)
        except Exception as e:
            JOBS[job_id]['status'] = 'error'
            JOBS[job_id]['error'] = str(e)
        finally:
            JOBS[job_id]['finished_at'] = time.time()

    background.add_task(run)
    return JobStatus(**{k: JOBS[job_id][k] for k in
                        ('job_id', 'status', 'created_at',
                         'events', 'error')})

@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JobStatus(**{k: JOBS[job_id].get(k) for k in
                        ('job_id', 'status', 'created_at',
                         'finished_at', 'events', 'error')})

@app.get("/jobs")
def list_jobs():
    return list(JOBS.values())

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run():
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port,
                workers=1, log_level="info")

if __name__ == '__main__':
    run()
