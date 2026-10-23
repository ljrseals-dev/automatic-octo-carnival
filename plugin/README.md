# AURA-G JUCE Plugin

VST3 / AU / Standalone build of the AURA-G guitar processing engine.

## Build

### Prerequisites

- CMake ≥ 3.22
- A C++17 compiler (Clang, GCC, MSVC 2019+)
- Git (JUCE is fetched automatically)
- macOS: Xcode Command Line Tools
- Windows: Visual Studio 2019+ with "Desktop C++" workload
- Linux: `sudo apt install build-essential libasound2-dev libfreetype6-dev libx11-dev libxext-dev libxrandr-dev libxinerama-dev libxcursor-dev libcurl4-openssl-dev`

### Configure & compile

```bash
cd plugin
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
