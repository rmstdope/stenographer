# stenographer

A Python application that transcribes WAV or MP3 recordings to text.

## Tech stack

- **Audio processing**: `ffmpeg-python`
- **Inference**: `faster-whisper`
- **Default Swedish model**: `KBLab/kb-whisper-small`

## Setup

```bash
python --version  # Requires Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need `ffmpeg` installed on your system.

## Usage

```bash
python stenographer.py path/to/audio.wav
python stenographer.py path/to/audio.mp3 --language sv --output transcript.txt
```

Options:

- `--model` (default: `KBLab/kb-whisper-small-ct2`)
- `--compute-type` (default: `auto`)
- `--beam-size` (default: `5`)
- `--language` (optional)
