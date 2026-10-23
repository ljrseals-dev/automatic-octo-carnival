# AURA-G Reconciliation

This package consolidates the four supplied AURA-G source documents into one filesystem tree.

## Source basis
- Part 1: architecture, latent bus, core Python proof-of-concept.
- Part 2: complete implementation package, MIDI, JUCE plugin, UI, FPGA extensions.
- Part 3: root configuration, API/worker/batch tooling, tests, deployment/export/mobile/cloud/monitoring extensions.
- Part 4: repeated core material plus expanded plugin/preset/API/integration material.

## Reconciliation rule
Where the same path was supplied more than once, the last complete code block encountered across the four supplied documents was selected. No outside implementation was substituted.

## Important validation note
The source documents use the word "complete/runnable", but they also explicitly identify some components as scaffolds or requiring external toolchains/datasets. This package therefore distinguishes source completeness from verified buildability.

## Expected external toolchains
- Python runtime and dependencies for the reference implementation.
- CMake/compiler/JUCE for the native plugin.
- AMD/Xilinx Vitis HLS and compatible FPGA hardware for FPGA kernels.
- Labeled guitar data for full Conformer training.
