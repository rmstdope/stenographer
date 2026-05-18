import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from stenographer import stenographer


class StenographerTests(unittest.TestCase):
    def test_default_model_is_valid_kblab_id(self) -> None:
        # KBLab publishes the model as "KBLab/kb-whisper-small" for faster-whisper.
        # The "-ct2" suffix does not exist on HuggingFace and causes a 401 error.
        self.assertEqual(stenographer.DEFAULT_MODEL, "KBLab/kb-whisper-small")

    def test_validate_audio_path_rejects_directory(self) -> None:
        with (
            tempfile.TemporaryDirectory(suffix=".mp3") as tmpdir,
            self.assertRaises(ValueError),
        ):
            stenographer._validate_audio_path(tmpdir)

    def test_transcribe_audio_rejects_unsupported_extension(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".txt") as tmp,
            self.assertRaises(ValueError),
        ):
            stenographer.transcribe_audio(tmp.name)

    def test_transcribe_audio_happy_path(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": " Hej världen ",
                    "segments": [{"start": 0.0, "end": 12.5, "text": " Hej världen "}],
                    "language": "sv",
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["text"], "Hej världen")
        self.assertEqual(result["language"], "sv")
        self.assertEqual(result["model"], stenographer.DEFAULT_MODEL)

    def test_main_writes_output_file(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            output_file = Path(tmpdir) / "out.txt"
            with (
                patch(
                    "stenographer.stenographer.transcribe_audio",
                    return_value={"text": "hej"},
                ),
                patch("sys.stdout", new=io.StringIO()) as stdout,
            ):
                exit_code = stenographer.main(
                    [wav.name, "--output", str(output_file)]
                )
            output_text = output_file.read_text(encoding="utf-8")
            cli_output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertEqual(output_text, "hej")
        self.assertEqual(cli_output, "")  # no stdout when -o is given

    def test_main_returns_error_when_output_write_fails(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch(
                "stenographer.stenographer.transcribe_audio",
                return_value={"text": "hej"},
            ),
            patch(
                "stenographer.stenographer.Path.write_text",
                side_effect=OSError("cannot write"),
            ),
            patch("sys.stderr", new=io.StringIO()) as stderr,
        ):
            exit_code = stenographer.main(
                [wav.name, "--output", "/tmp/nowhere/out.txt"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: cannot write", stderr.getvalue())

    # --- _validate_audio_path ---

    def test_validate_audio_path_rejects_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            stenographer._validate_audio_path(
                "/nonexistent/certainly/not/here/audio.wav"
            )

    def test_validate_audio_path_is_case_insensitive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".WAV") as tmp:
            path = stenographer._validate_audio_path(tmp.name)
            self.assertIsInstance(path, Path)

    def test_validate_audio_path_accepts_mp4(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            path = stenographer._validate_audio_path(tmp.name)
            self.assertEqual(path.suffix.lower(), ".mp4")

    # --- transcribe_audio ---

    def test_transcribe_audio_returns_duration_from_last_segment(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": "hello",
                    "segments": [{"start": 0.0, "end": 42.0, "text": "hello"}],
                    "language": "sv",
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["duration"], 42.0)

    def test_transcribe_audio_duration_zero_with_no_segments(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": "",
                    "segments": [],
                    "language": "sv",
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["duration"], 0.0)
        self.assertEqual(result["text"], "")

    def test_transcribe_audio_language_fallback_to_parameter(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": "",
                    "segments": [],
                    "language": None,
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name, language="fr")

        self.assertEqual(result["language"], "fr")

    def test_transcribe_audio_passes_model_name_to_ensure_mlx_model(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={"text": "", "segments": [], "language": "sv"}
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ) as mock_ensure,
        ):
            stenographer.transcribe_audio(tmp.name, model_name="custom/model")

        mock_ensure.assert_called_once_with("custom/model")

    def test_transcribe_audio_passes_language_to_mlx_not_beam_size(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={"text": "", "segments": [], "language": "en"}
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            stenographer.transcribe_audio(tmp.name, language="en", beam_size=3)

        call_kwargs = fake_mlx.transcribe.call_args
        # language must be forwarded
        self.assertEqual(call_kwargs.kwargs.get("language"), "en")
        # beam_size must NOT be forwarded
        # (mlx-whisper raises NotImplementedError for it)
        self.assertNotIn("beam_size", call_kwargs.kwargs)

    def test_transcribe_audio_accepts_mp4(self) -> None:
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": "Hello Teams",
                    "segments": [{"start": 0.0, "end": 5.0, "text": "Hello Teams"}],
                    "language": "en",
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".mp4") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["text"], "Hello Teams")
        self.assertEqual(result["language"], "en")

    # --- _ensure_mlx_model ---

    def test_ensure_mlx_model_returns_local_path_when_already_mlx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mlx_dir = Path(tmpdir)
            (mlx_dir / "weights.safetensors").write_bytes(b"fake")
            (mlx_dir / "config.json").write_text("{}")

            result = stenographer._ensure_mlx_model(str(mlx_dir))

        self.assertEqual(result, str(mlx_dir))

    def test_ensure_mlx_model_returns_cache_path_when_cache_exists(self) -> None:
        import hashlib

        model_name = "KBLab/kb-whisper-small"
        model_hash = hashlib.md5(model_name.encode()).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_base = Path(tmpdir)
            model_cache = cache_base / model_hash
            model_cache.mkdir(parents=True)
            (model_cache / "weights.safetensors").write_bytes(b"fake")
            (model_cache / "config.json").write_text("{}")

            with patch("stenographer.stenographer._CACHE_DIR", cache_base):
                result = stenographer._ensure_mlx_model(model_name)

        self.assertEqual(result, str(model_cache))

    def test_ensure_mlx_model_converts_when_cache_missing(self) -> None:
        model_name = "KBLab/kb-whisper-small"

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_base = Path(tmpdir)
            with (
                patch("stenographer.stenographer._CACHE_DIR", cache_base),
                patch(
                    "stenographer.stenographer._convert_hf_to_mlx"
                ) as mock_convert,
            ):
                stenographer._ensure_mlx_model(model_name)

        mock_convert.assert_called_once()

    # --- _convert_hf_to_mlx ---

    def _make_fake_mx(self, raw_weights: dict) -> tuple[types.SimpleNamespace, dict]:
        """Return (fake_mx_module, saved_weights_dict) for patching mlx.core."""
        saved: dict = {}

        class FakeArray:
            def __init__(self, ndim: int = 1) -> None:
                self.ndim = ndim

            def astype(self, dtype: object) -> "FakeArray":
                return self

        fake_mx = types.SimpleNamespace(
            float16="float16",
            load=MagicMock(
                return_value={k: FakeArray(v) for k, v in raw_weights.items()}
            ),
            save_safetensors=MagicMock(
                side_effect=lambda path, w, **kw: saved.update(w)
            ),
            swapaxes=MagicMock(side_effect=lambda v, a, b: v),
        )
        return fake_mx, saved

    def _hf_config(self) -> dict:
        return {
            "num_mel_bins": 80,
            "max_source_positions": 1500,
            "d_model": 512,
            "encoder_attention_heads": 8,
            "encoder_layers": 6,
            "vocab_size": 51865,
            "max_target_positions": 448,
            "decoder_attention_heads": 8,
            "decoder_layers": 6,
        }

    def _run_convert(self, fake_mx: types.SimpleNamespace, hf_config: dict) -> None:
        import json

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, "config.json").write_text(
                json.dumps(hf_config), encoding="utf-8"
            )
            with tempfile.TemporaryDirectory() as out_dir:
                output_path = Path(out_dir) / "converted"
                with patch.dict(
                    sys.modules,
                    {
                        "mlx": types.SimpleNamespace(core=fake_mx),
                        "mlx.core": fake_mx,
                        "huggingface_hub": types.SimpleNamespace(
                            snapshot_download=MagicMock(return_value=model_dir)
                        ),
                    },
                ):
                    stenographer._convert_hf_to_mlx("some/model", output_path)

    def test_convert_hf_to_mlx_skips_encoder_embed_positions(self) -> None:
        # AudioEncoder stores positional embedding as _positional_embedding (private,
        # not a loadable parameter). Sending encoder.positional_embedding to
        # model.update() raises "Module does not have parameter
        # named 'positional_embedding'".
        fake_mx, saved = self._make_fake_mx(
            {
                "model.encoder.embed_positions.weight": 1,  # must be skipped
                # must be remapped to decoder.positional_embedding
                "model.decoder.embed_positions.weight": 1,
            }
        )
        self._run_convert(fake_mx, self._hf_config())

        self.assertFalse(
            any(
                "encoder.positional_embedding" in k or k == "positional_embedding"
                for k in saved
            ),
            f"encoder positional_embedding should be skipped"
            f" but found: {list(saved.keys())}",
        )
        self.assertIn("decoder.positional_embedding", saved)

    def test_convert_hf_to_mlx_skips_proj_out_weight(self) -> None:
        # Bug: proj_out.weight (with or without "model." prefix) was not skipped.
        # mlx-whisper rejects the converted weights with "Module does not have
        # parameter named 'proj_out'".
        fake_mx, saved = self._make_fake_mx(
            {
                "model.proj_out.weight": 1,  # must be skipped (with prefix)
                "proj_out.weight": 1,  # must be skipped (no prefix)
                "model.encoder.conv1.weight": 3,  # must be kept
            }
        )
        self._run_convert(fake_mx, self._hf_config())

        self.assertFalse(
            any("proj_out" in k for k in saved),
            f"proj_out should be skipped"
            f" but found: {[k for k in saved if 'proj_out' in k]}",
        )
        self.assertIn("encoder.conv1.weight", saved)

    def test_convert_hf_to_mlx_remaps_embed_tokens_to_token_embedding(self) -> None:
        fake_mx, saved = self._make_fake_mx(
            {
                "model.decoder.embed_tokens.weight": 1,
            }
        )
        self._run_convert(fake_mx, self._hf_config())

        self.assertIn("decoder.token_embedding.weight", saved)
        self.assertNotIn("model.decoder.embed_tokens.weight", saved)

    def test_convert_hf_to_mlx_swaps_conv_weight_axes(self) -> None:
        fake_mx, _ = self._make_fake_mx(
            {
                "model.encoder.conv1.weight": 3,  # ndim=3 → swapaxes must be called
            }
        )
        self._run_convert(fake_mx, self._hf_config())

        fake_mx.swapaxes.assert_called_once()

    def test_main_prints_transcript_without_output_flag(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch(
                "stenographer.stenographer.transcribe_audio",
                return_value={"text": "hello"},
            ),
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            exit_code = stenographer.main([wav.name])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "hello\n")

    # --- _format_timestamp ---

    def test_format_timestamp_seconds_only(self) -> None:
        self.assertEqual(stenographer._format_timestamp(5.9), "00:00:05")

    def test_format_timestamp_minutes_and_seconds(self) -> None:
        self.assertEqual(stenographer._format_timestamp(75.0), "00:01:15")

    def test_format_timestamp_hours(self) -> None:
        self.assertEqual(stenographer._format_timestamp(3665.0), "01:01:05")

    # --- _format_segments ---

    def test_format_segments_single_segment(self) -> None:
        segments = [{"start": 4.0, "end": 9.0, "text": " Hej världen"}]
        result = stenographer._format_segments(segments)
        self.assertEqual(result, "[00:00:04 --> 00:00:09] Hej världen")

    def test_format_segments_omits_empty_text(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 5.0, "text": "Hello"},
        ]
        result = stenographer._format_segments(segments)
        self.assertEqual(result, "[00:00:02 --> 00:00:05] Hello")

    def test_format_segments_multiple_segments(self) -> None:
        segments = [
            {"start": 0.0, "end": 4.0, "text": " First."},
            {"start": 4.0, "end": 9.0, "text": " Second."},
        ]
        result = stenographer._format_segments(segments)
        self.assertEqual(
            result,
            "[00:00:00 --> 00:00:04] First.\n[00:00:04 --> 00:00:09] Second.",
        )

    def test_format_segments_returns_empty_string_for_no_segments(self) -> None:
        self.assertEqual(stenographer._format_segments([]), "")

    # --- transcribe_audio returns segments ---

    def test_transcribe_audio_result_includes_segments(self) -> None:
        seg = {"start": 0.0, "end": 3.0, "text": " Hej"}
        fake_mlx = types.SimpleNamespace(
            transcribe=MagicMock(
                return_value={
                    "text": " Hej",
                    "segments": [seg],
                    "language": "sv",
                }
            )
        )
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as tmp,
            patch.dict(sys.modules, {"mlx_whisper": fake_mlx}),
            patch(
                "stenographer.stenographer._ensure_mlx_model",
                return_value="/fake/model",
            ),
        ):
            result = stenographer.transcribe_audio(tmp.name)

        self.assertIn("segments", result)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["start"], 0.0)
        self.assertEqual(result["segments"][0]["end"], 3.0)

    # --- main() formatted output ---

    def test_main_prints_formatted_segments_by_default(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": " Hej världen"}]
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch(
                "stenographer.stenographer.transcribe_audio",
                return_value={"text": "Hej världen", "segments": segments},
            ),
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            exit_code = stenographer.main([wav.name])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "[00:00:00 --> 00:00:04] Hej världen\n")

    def test_main_prints_flat_text_with_format_text_flag(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": " Hej världen"}]
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch(
                "stenographer.stenographer.transcribe_audio",
                return_value={"text": "Hej världen", "segments": segments},
            ),
            patch("sys.stdout", new=io.StringIO()) as stdout,
        ):
            exit_code = stenographer.main([wav.name, "--format", "text"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "Hej världen\n")

    def test_main_writes_formatted_segments_to_file_by_default(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "text": " Hej"}]
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            output_file = Path(tmpdir) / "out.txt"
            with patch(
                "stenographer.stenographer.transcribe_audio",
                return_value={"text": "Hej", "segments": segments},
            ):
                exit_code = stenographer.main([wav.name, "--output", str(output_file)])
            output_text = output_file.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(output_text, "[00:00:00 --> 00:00:04] Hej")

    def test_main_returns_error_for_nonexistent_file(self) -> None:
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            exit_code = stenographer.main(["/nonexistent/audio.wav"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error:", stderr.getvalue())

    def test_main_handles_unexpected_exception(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch(
                "stenographer.stenographer.transcribe_audio",
                side_effect=MemoryError("out of memory"),
            ),
            patch("sys.stderr", new=io.StringIO()) as stderr,
        ):
            exit_code = stenographer.main([wav.name])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unexpected error (MemoryError):", stderr.getvalue())

    def test_main_passes_model_arg_to_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            audio_path = wav.name
            with (
                patch(
                    "stenographer.stenographer.transcribe_audio",
                    return_value={"text": ""},
                ) as mock_transcribe,
                patch("sys.stdout", new=io.StringIO()),
            ):
                stenographer.main([audio_path, "--model", "custom/model"])

        mock_transcribe.assert_called_once_with(
            audio_path,
            model_name="custom/model",
            language=None,
            beam_size=5,
            compute_type="auto",
        )

    def test_main_passes_language_arg_to_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            audio_path = wav.name
            with (
                patch(
                    "stenographer.stenographer.transcribe_audio",
                    return_value={"text": ""},
                ) as mock_transcribe,
                patch("sys.stdout", new=io.StringIO()),
            ):
                stenographer.main([audio_path, "--language", "sv"])

        mock_transcribe.assert_called_once_with(
            audio_path,
            model_name=stenographer.DEFAULT_MODEL,
            language="sv",
            beam_size=5,
            compute_type="auto",
        )

    def test_main_passes_beam_size_arg_to_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            audio_path = wav.name
            with (
                patch(
                    "stenographer.stenographer.transcribe_audio",
                    return_value={"text": ""},
                ) as mock_transcribe,
                patch("sys.stdout", new=io.StringIO()),
            ):
                stenographer.main([audio_path, "--beam-size", "3"])

        mock_transcribe.assert_called_once_with(
            audio_path,
            model_name=stenographer.DEFAULT_MODEL,
            language=None,
            beam_size=3,
            compute_type="auto",
        )

    def test_main_passes_compute_type_arg_to_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            audio_path = wav.name
            with (
                patch(
                    "stenographer.stenographer.transcribe_audio",
                    return_value={"text": ""},
                ) as mock_transcribe,
                patch("sys.stdout", new=io.StringIO()),
            ):
                stenographer.main([audio_path, "--compute-type", "int8"])

        mock_transcribe.assert_called_once_with(
            audio_path,
            model_name=stenographer.DEFAULT_MODEL,
            language=None,
            beam_size=5,
            compute_type="int8",
        )


if __name__ == "__main__":
    unittest.main()
