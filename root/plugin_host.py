"""Headless renderer for the AURA-G VST3/AU plugin.

Runs the plugin without a DAW: loads a plugin binary, feeds it audio,
captures output. Requires pedalboard (Spotify) or JUCE host binary.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

from .logging_setup import setup_logging

log = setup_logging()

def render_through_plugin(input_path: str, output_path: str,
                          plugin_path: str,
                          params: dict = None,
                          block_size: int = 512):
    try:
        from pedalboard import load_plugin
    except ImportError:
        raise SystemExit("pip install pedalboard")

    plugin = load_plugin(plugin_path)
    log.info(f"Loaded plugin: {plugin.name} ({plugin_path})")

    audio, sr = sf.read(input_path, dtype='float32')
    if audio.ndim == 1:
        audio = audio[:, None]

    if params:
        for k, v in params.items():
            if hasattr(plugin, k):
                setattr(plugin, k, v)
            else:
                log.warning(f"Plugin has no parameter '{k}'")

    out = plugin.process(audio, sr, buffer_size=block_size)
    sf.write(output_path, out.T if out.ndim > 1 else out, sr)
    log.info(f"Rendered: {output_path} ({out.shape})")
    return output_path

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--plugin', required=True,
                   help='Path to .vst3/.component/.dll')
    p.add_argument('--param', action='append', default=[],
                   help='name=value (repeatable)')
    a = p.parse_args()

    params = {}
    for kv in a.param:
        k, v = kv.split('=', 1)
        params[k] = float(v)

    render_through_plugin(a.input, a.output, a.plugin, params)
