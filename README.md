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
python stenographer.py path/to/recording.mp4 --language sv --output transcript.txt
```

Options:

- `--language` — ISO 639-1 language code, e.g. `sv` for Swedish, `en` for English. Omit to let the model detect the language automatically.
- `--output` / `-o` — Path to write the transcript text. If omitted, the transcript is printed to stdout.
- `--model` — HuggingFace model ID (default: `KBLab/kb-whisper-small`). Other KBLab sizes: `kb-whisper-tiny`, `kb-whisper-base`, `kb-whisper-medium`, `kb-whisper-large`.
- `--beam-size` — Number of candidate sequences the decoder considers at each step (default: `5`). Higher values can improve accuracy at the cost of speed. `1` is greedy (fastest); `5` is a good general balance.
- `--compute-type` — Numerical precision used during inference (default: `auto`):
  - `auto` — picks the best type for your hardware automatically.
  - `int8` — fastest and most memory-efficient; recommended for CPU. Slight accuracy trade-off.
  - `float16` — fast on GPU and Apple Silicon with good accuracy.
  - `float32` — full precision; most accurate but slowest and most memory-intensive.
