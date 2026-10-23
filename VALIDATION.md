# AURA-G Validation Record

Validation performed on the reconciled source tree after extraction from the supplied AURA-G documents.

## Passed

- Python syntax parse: 58 Python files, 0 syntax errors.
- Core tests: 12 passed (`test_bus`, `test_front_end`, `test_transcription`, `test_midi`).
- DSP/pipeline tests: 12 passed (`test_ddsp_amp`, `test_rnn_fx`, `test_physical`, `test_pipeline`).
- End-to-end integration: 1 passed (`test_integration`).
- Total executed tests: 25 passed.

## Corrective change

The supplied transcription implementation scored fingering around the fifth fret, which caused the included test for MIDI 45 (open A string) to fail. The reconciled implementation now prefers lower fret positions, with a small string penalty; the supplied test then passes.

## Not claimed as locally verified

- JUCE VST3/AU build: requires a native C++ toolchain and JUCE dependency resolution.
- FPGA kernels: require the vendor HLS toolchain/target hardware.
- Full Conformer training: requires an appropriate labeled guitar dataset.
- CUDA performance or real-time hardware latency targets: not benchmarked here.
