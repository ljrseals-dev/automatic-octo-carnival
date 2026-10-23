"""Google Cloud Function for AURA-G inference."""
import base64
import io

import numpy as np
import soundfile as sf

_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from aura_g.pipeline import AURAG
        _MODEL = AURAG(sr=48000)
    return _MODEL

def aura_g(request):
    """HTTP entry point. Request body: base64 WAV."""
    try:
        data = request.get_json(silent=True) or {}
        raw = base64.b64decode(data.get('audio_b64', ''))
        audio, sr = sf.read(io.BytesIO(raw), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        ag = _load_model()
        out, events, _ = ag.process(audio)

        buf = io.BytesIO()
        sf.write(buf, out, sr, format='WAV')
        buf.seek(0)

        return {
            'audio_b64': base64.b64encode(buf.read()).decode('ascii'),
            'events': len(events),
            'sr': sr,
        }
    except Exception as e:
        return {'error': str(e)}, 500
