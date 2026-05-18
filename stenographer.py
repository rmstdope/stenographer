from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "KBLab/kb-whisper-small"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".mp4"}


def _validate_audio_path(audio_path: str | Path) -> Path:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio/video format '{path.suffix}'. Supported formats: WAV, MP3, MP4"
        )
    return path


def load_audio(audio_path: str | Path, sample_rate: int = 16000) -> Any:
    path = _validate_audio_path(audio_path)

    import ffmpeg  # type: ignore
    import numpy as np  # type: ignore

    try:
        output, _ = (
            ffmpeg.input(str(path))
            .output(
                "pipe:",
                format="s16le",
                acodec="pcm_s16le",
                ac=1,
                ar=sample_rate,
            )
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
    except ffmpeg.Error as exc:  # type: ignore[attr-defined]
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise RuntimeError(f"Failed to decode audio with ffmpeg: {stderr}") from exc

    return np.frombuffer(output, np.int16).astype(np.float32) / 32768.0


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    beam_size: int = 5,
    compute_type: str = "auto",
) -> dict[str, Any]:
    _validate_audio_path(audio_path)
    audio = load_audio(audio_path)

    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(model_name, compute_type=compute_type)

    segments, info = model.transcribe(audio, language=language, beam_size=beam_size)

    segment_texts = []
    for segment in segments:
        cleaned_text = getattr(segment, "text", "").strip()
        if cleaned_text:
            segment_texts.append(cleaned_text)
    text = " ".join(segment_texts)
    return {
        "text": text,
        "language": getattr(info, "language", language),
        "duration": getattr(info, "duration", None),
        "model": model_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe WAV/MP3/MP4 recordings to text using faster-whisper and KBLab models."
    )
    parser.add_argument("audio", help="Path to input audio/video file (.wav, .mp3 or .mp4)")
    parser.add_argument("-o", "--output", help="Optional path to write transcript text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name")
    parser.add_argument("--language", help="Language code (e.g. sv, en)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "int8", "float16", "float32"],
        help="Compute type for faster-whisper (auto, int8, float16, float32)",
    )
    args = parser.parse_args(argv)

    try:
        result = transcribe_audio(
            args.audio,
            model_name=args.model,
            language=args.language,
            beam_size=args.beam_size,
            compute_type=args.compute_type,
        )
        if args.output:
            Path(args.output).write_text(result["text"], encoding="utf-8")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Keep CLI output user-friendly instead of displaying a raw traceback.
        print(f"Unexpected error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    print(result["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
