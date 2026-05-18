# stenographer

A Python application that transcribes WAV, MP3, or MP4 recordings to text.

> **Requires Apple Silicon (M1/M2/M3/M4).** Transcription uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), which leverages the Apple Neural Engine and GPU via the [MLX](https://github.com/ml-explore/mlx) framework.

## Tech stack

- **Inference**: `mlx-whisper` (Apple MLX — GPU + Neural Engine)
- **Default Swedish model**: `KBLab/kb-whisper-small`

## Setup

```bash
python --version  # Requires Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You also need `ffmpeg` installed on your system (`brew install ffmpeg`).

### First-run model conversion

On the first transcription, `stenographer` automatically downloads `KBLab/kb-whisper-small` from HuggingFace and converts it to MLX format. The converted model is cached at `~/.cache/stenographer/mlx_models/` and reused on subsequent runs.

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
- `--compute-type` — Accepted for CLI compatibility but not used by the mlx-whisper backend (MLX manages precision automatically).
