from setuptools import setup, find_packages

setup(
    name="aura-g",
    version="0.1.0",
    description="AURA-G: Adaptive Unified Real-time Architecture for Guitar",
    author="AURA-G Contributors",
    packages=find_packages(exclude=("tests", "plugin", "fpga", "ui")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "soundfile>=0.12",
        "librosa>=0.10",
        "torch>=2.0",
        "torchaudio>=2.0",
    ],
    extras_require={
        "realtime": ["sounddevice>=0.4.6"],
        "ui": ["streamlit>=1.30", "gradio>=4.20"],
        "api": ["fastapi", "uvicorn", "python-multipart"],
        "plugins": ["pedalboard>=0.8"],
        "midi": ["mido>=1.3"],
        "all": [
            "sounddevice>=0.4.6",
            "streamlit>=1.30",
            "gradio>=4.20",
            "fastapi", "uvicorn", "python-multipart",
            "pedalboard>=0.8",
            "mido>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "aura-g=run_all:main",
            "aura-g-batch=root.batch_processor:batch_process",
            "aura-g-server=root.server:run",
            "aura-g-worker=root.worker:run_worker",
        ],
    },
)
