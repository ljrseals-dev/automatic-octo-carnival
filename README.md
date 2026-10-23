# AURA-G

**Adaptive Unified Real-time Architecture for Guitar.**

A hybrid guitar processing platform that unifies four paradigms under
a shared 32-dimensional **Latent Control Bus**:

1. **DDSP amplifier** — interpretable preamp/tone-stack/power-amp model
2. **Sample-rate-independent RNN FX** — interpolation-filter frontend + GRU
3. **Physical waveguide model** — digital string with fret-aware damping
4. **Blind effect extraction** — STFT Wiener inversion from dry/wet pairs

Plus: MIDI export, JUCE VST3/AU plugin, real-time Conformer transcription,
full FPGA HLS pipeline, Streamlit + Gradio UIs, HTTP API, and batch processing.

## Quick Start

```bash
git clone <repo>
cd aura-g
pip install -e '.[all]'
python run_all.py --mode demo
