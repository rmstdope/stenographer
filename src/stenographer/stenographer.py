from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "KBLab/kb-whisper-small"
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".mp4"}
_CACHE_DIR = Path.home() / ".cache" / "stenographer" / "mlx_models"


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_segments(segments: list) -> str:
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {text}")
    return "\n".join(lines)


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


_SKIP_KEYS = {"proj_out.weight", "encoder.embed_positions.weight"}
_KEY_REPLACEMENTS = [
    (".layers.", ".blocks."),
    (".self_attn.", ".attn."),
    (".self_attn_layer_norm.", ".attn_ln."),
    (".encoder_attn.", ".cross_attn."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    (".final_layer_norm.", ".mlp_ln."),
    (".q_proj.", ".query."),
    (".k_proj.", ".key."),
    (".v_proj.", ".value."),
    (".out_proj.", ".out."),
    (".fc1.", ".mlp1."),
    (".fc2.", ".mlp2."),
    (".mlp.0.", ".mlp1."),
    (".mlp.2.", ".mlp2."),
    ("embed_positions.weight", "positional_embedding"),
    ("embed_tokens.weight", "token_embedding.weight"),
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("decoder.layer_norm.", "decoder.ln."),
]


def _convert_hf_to_mlx(model_name: str, output_path: Path) -> None:
    import json

    import mlx.core as mx  # type: ignore
    from huggingface_hub import snapshot_download  # type: ignore

    model_dir = Path(snapshot_download(model_name))

    with open(model_dir / "config.json", encoding="utf-8") as f:
        hf_config = json.load(f)

    mlx_config = {
        "model_type": "whisper",
        "n_mels": hf_config["num_mel_bins"],
        "n_audio_ctx": hf_config["max_source_positions"],
        "n_audio_state": hf_config["d_model"],
        "n_audio_head": hf_config["encoder_attention_heads"],
        "n_audio_layer": hf_config["encoder_layers"],
        "n_vocab": hf_config["vocab_size"],
        "n_text_ctx": hf_config["max_target_positions"],
        "n_text_state": hf_config["d_model"],
        "n_text_head": hf_config["decoder_attention_heads"],
        "n_text_layer": hf_config["decoder_layers"],
    }

    raw_weights = dict(mx.load(str(model_dir / "model.safetensors")))

    new_weights: dict[str, Any] = {}
    for key, value in raw_weights.items():
        if key.startswith("model."):
            key = key[6:]
        if key in _SKIP_KEYS:
            continue
        for old, new in _KEY_REPLACEMENTS:
            key = key.replace(old, new)
        if "conv" in key and value.ndim == 3:
            value = mx.swapaxes(value, 1, 2)
        new_weights[key] = value.astype(mx.float16)

    output_path.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output_path / "weights.safetensors"), new_weights)
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(mlx_config, f, indent=2)


def _ensure_mlx_model(model_name: str) -> str:
    import hashlib

    local = Path(model_name)
    if local.exists() and (local / "weights.safetensors").exists():
        return model_name

    model_hash = hashlib.md5(model_name.encode()).hexdigest()
    model_cache = _CACHE_DIR / model_hash
    if (model_cache / "weights.safetensors").exists() and (model_cache / "config.json").exists():
        return str(model_cache)

    _convert_hf_to_mlx(model_name, model_cache)
    return str(model_cache)


def transcribe_audio(
    audio_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    beam_size: int = 5,
    compute_type: str = "auto",
) -> dict[str, Any]:
    path = _validate_audio_path(audio_path)
    mlx_model_path = _ensure_mlx_model(model_name)

    import mlx_whisper  # type: ignore

    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=mlx_model_path,
        language=language,
    )

    segments = result.get("segments", [])
    duration = segments[-1]["end"] if segments else 0.0
    return {
        "text": result["text"].strip(),
        "segments": segments,
        "language": result.get("language") or language,
        "duration": duration,
        "model": model_name,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe WAV/MP3/MP4 recordings to text using mlx-whisper and KBLab models."
    )
    parser.add_argument("audio", help="Path to input audio/video file (.wav, .mp3 or .mp4)")
    parser.add_argument("-o", "--output", help="Optional path to write transcript text")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name")
    parser.add_argument("--language", help="Language code (e.g. sv, en)")
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Number of candidate sequences considered at each decoding step (default: 5). "
        "Higher values improve accuracy at the cost of speed; 1 is greedy (fastest).",
    )
    parser.add_argument(
        "--format",
        default="segments",
        choices=["segments", "text"],
        help="Output format: 'segments' (default) includes timestamps per segment; "
        "'text' outputs the flat transcript.",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "int8", "float16", "float32"],
        help="Numerical precision for inference (default: auto). "
        "auto: hardware-optimised automatically. "
        "int8: fastest/smallest memory, best for CPU, slight accuracy trade-off. "
        "float16: fast on GPU/Apple Silicon with good accuracy. "
        "float32: full precision, slowest, most accurate.",
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
        if args.format == "text":
            output = result["text"]
        else:
            output = _format_segments(result.get("segments", [])) or result["text"]

        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Keep CLI output user-friendly instead of displaying a raw traceback.
        print(f"Unexpected error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1

    if not args.output:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
