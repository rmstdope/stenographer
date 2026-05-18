import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import stenographer


class StenographerTests(unittest.TestCase):
    def test_validate_audio_path_rejects_directory(self) -> None:
        with tempfile.TemporaryDirectory(suffix=".mp3") as tmpdir:
            with self.assertRaises(ValueError):
                stenographer._validate_audio_path(tmpdir)

    def test_transcribe_audio_rejects_unsupported_extension(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            with self.assertRaises(ValueError):
                stenographer.transcribe_audio(tmp.name)

    def test_transcribe_audio_happy_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [
                    types.SimpleNamespace(text="Hej"),
                    types.SimpleNamespace(text=" världen "),
                    types.SimpleNamespace(text="   "),
                    types.SimpleNamespace(text=""),
                ],
                types.SimpleNamespace(language="sv", duration=12.5),
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )

            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["text"], "Hej världen")
        self.assertEqual(result["language"], "sv")
        self.assertEqual(result["model"], stenographer.DEFAULT_MODEL)

    def test_load_audio_decodes_with_ffmpeg_pipeline(self) -> None:
        class FakeArray:
            def __init__(self, values: list[int]) -> None:
                self.values = values

            def astype(self, _dtype: object) -> "FakeArray":
                return self

            def __truediv__(self, divisor: float) -> list[float]:
                return [value / divisor for value in self.values]

        class FakeFfmpegError(Exception):
            def __init__(self, stderr: bytes = b"") -> None:
                super().__init__("ffmpeg failed")
                self.stderr = stderr

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_stream = MagicMock()
            fake_stream.output.return_value = fake_stream
            fake_stream.run.return_value = (b"\x00\x00\x00@\x00\x80", b"")

            fake_ffmpeg = types.SimpleNamespace(
                input=MagicMock(return_value=fake_stream),
                Error=FakeFfmpegError,
            )
            fake_numpy = types.SimpleNamespace(
                int16="int16",
                float32="float32",
                frombuffer=MagicMock(return_value=FakeArray([0, 16384, -32768])),
            )

            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg, "numpy": fake_numpy}):
                result = stenographer.load_audio(tmp.name)

        self.assertEqual(result, [0.0, 0.5, -1.0])
        fake_ffmpeg.input.assert_called_once_with(tmp.name)
        fake_stream.output.assert_called_once_with(
            "pipe:",
            format="s16le",
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
        )
        fake_numpy.frombuffer.assert_called_once_with(b"\x00\x00\x00@\x00\x80", "int16")

    def test_load_audio_wraps_ffmpeg_error(self) -> None:
        class FakeFfmpegError(Exception):
            def __init__(self, stderr: bytes = b"") -> None:
                super().__init__("ffmpeg failed")
                self.stderr = stderr

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_stream = MagicMock()
            fake_stream.output.return_value = fake_stream
            fake_stream.run.side_effect = FakeFfmpegError(b"decoder broke")
            fake_ffmpeg = types.SimpleNamespace(
                input=MagicMock(return_value=fake_stream),
                Error=FakeFfmpegError,
            )
            fake_numpy = types.SimpleNamespace(int16="int16", float32="float32")

            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg, "numpy": fake_numpy}):
                with self.assertRaises(RuntimeError) as exc_info:
                    stenographer.load_audio(tmp.name)

        self.assertIn("Failed to decode audio with ffmpeg: decoder broke", str(exc_info.exception))

    def test_main_writes_output_file(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            output_file = Path(tmpdir) / "out.txt"
            with patch("stenographer.transcribe_audio", return_value={"text": "hej"}):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    exit_code = stenographer.main([wav.name, "--output", str(output_file)])
            output_text = output_file.read_text(encoding="utf-8")
            cli_output = stdout.getvalue()

        self.assertEqual(exit_code, 0)
        self.assertEqual(output_text, "hej")
        self.assertEqual(cli_output, "hej\n")

    def test_main_returns_error_when_output_write_fails(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            patch("stenographer.transcribe_audio", return_value={"text": "hej"}),
            patch("stenographer.Path.write_text", side_effect=OSError("cannot write")),
            patch("sys.stderr", new=io.StringIO()) as stderr,
        ):
            exit_code = stenographer.main([wav.name, "--output", "/tmp/nowhere/out.txt"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: cannot write", stderr.getvalue())

    # --- _validate_audio_path ---

    def test_validate_audio_path_rejects_nonexistent_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            stenographer._validate_audio_path("/nonexistent/certainly/not/here/audio.wav")

    def test_validate_audio_path_is_case_insensitive(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".WAV") as tmp:
            path = stenographer._validate_audio_path(tmp.name)
            self.assertIsInstance(path, Path)

    def test_validate_audio_path_accepts_mp4(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            path = stenographer._validate_audio_path(tmp.name)
            self.assertEqual(path.suffix.lower(), ".mp4")

    # --- load_audio ---

    def test_load_audio_uses_custom_sample_rate(self) -> None:
        class FakeArray:
            def astype(self, _dtype: object) -> "FakeArray":
                return self

            def __truediv__(self, _divisor: float) -> "FakeArray":
                return self

        class FakeFfmpegError(Exception):
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_stream = MagicMock()
            fake_stream.output.return_value = fake_stream
            fake_stream.run.return_value = (b"", b"")
            fake_ffmpeg = types.SimpleNamespace(
                input=MagicMock(return_value=fake_stream),
                Error=FakeFfmpegError,
            )
            fake_numpy = types.SimpleNamespace(
                int16="int16",
                float32="float32",
                frombuffer=MagicMock(return_value=FakeArray()),
            )
            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg, "numpy": fake_numpy}):
                stenographer.load_audio(tmp.name, sample_rate=8000)

        fake_stream.output.assert_called_once_with(
            "pipe:",
            format="s16le",
            acodec="pcm_s16le",
            ac=1,
            ar=8000,
        )

    def test_load_audio_wraps_ffmpeg_error_with_no_stderr(self) -> None:
        class FakeFfmpegError(Exception):
            def __init__(self) -> None:
                super().__init__("ffmpeg failed")
                self.stderr = None

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_stream = MagicMock()
            fake_stream.output.return_value = fake_stream
            fake_stream.run.side_effect = FakeFfmpegError()
            fake_ffmpeg = types.SimpleNamespace(
                input=MagicMock(return_value=fake_stream),
                Error=FakeFfmpegError,
            )
            fake_numpy = types.SimpleNamespace(int16="int16", float32="float32")
            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg, "numpy": fake_numpy}):
                with self.assertRaises(RuntimeError) as exc_info:
                    stenographer.load_audio(tmp.name)

        self.assertIn("Failed to decode audio with ffmpeg:", str(exc_info.exception))

    # --- transcribe_audio ---

    def test_transcribe_audio_returns_duration_in_result(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(language="sv", duration=42.0),
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["duration"], 42.0)

    def test_transcribe_audio_language_fallback_to_parameter(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(duration=1.0),  # no language attr
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    result = stenographer.transcribe_audio(tmp.name, language="fr")

        self.assertEqual(result["language"], "fr")

    def test_transcribe_audio_returns_empty_text_when_no_segments(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(language="sv", duration=0.0),
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["text"], "")

    def test_transcribe_audio_passes_model_name_to_whisper(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(language="sv", duration=0.0),
            )
            fake_whisper_cls = MagicMock(return_value=fake_model)
            fake_whisper_module = types.SimpleNamespace(WhisperModel=fake_whisper_cls)
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    stenographer.transcribe_audio(tmp.name, model_name="custom/model")

        fake_whisper_cls.assert_called_once_with("custom/model", compute_type="auto")

    def test_transcribe_audio_passes_params_to_model_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(language="en", duration=0.0),
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    stenographer.transcribe_audio(tmp.name, language="en", beam_size=3)

        fake_model.transcribe.assert_called_once_with("AUDIO", language="en", beam_size=3)

    def test_transcribe_audio_passes_compute_type_to_whisper_model(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [],
                types.SimpleNamespace(language="sv", duration=0.0),
            )
            fake_whisper_cls = MagicMock(return_value=fake_model)
            fake_whisper_module = types.SimpleNamespace(WhisperModel=fake_whisper_cls)
            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    stenographer.transcribe_audio(tmp.name, compute_type="int8")

        fake_whisper_cls.assert_called_once_with(
            stenographer.DEFAULT_MODEL, compute_type="int8"
        )

    # --- main ---

    def test_main_prints_transcript_without_output_flag(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            with patch("stenographer.transcribe_audio", return_value={"text": "hello"}):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    exit_code = stenographer.main([wav.name])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "hello\n")

    def test_main_returns_error_for_nonexistent_file(self) -> None:
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            exit_code = stenographer.main(["/nonexistent/audio.wav"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error:", stderr.getvalue())

    def test_main_handles_unexpected_exception(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            with patch(
                "stenographer.transcribe_audio",
                side_effect=MemoryError("out of memory"),
            ):
                with patch("sys.stderr", new=io.StringIO()) as stderr:
                    exit_code = stenographer.main([wav.name])

        self.assertEqual(exit_code, 1)
        self.assertIn("Unexpected error (MemoryError):", stderr.getvalue())

    def test_main_passes_model_arg_to_transcribe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            audio_path = wav.name
            with patch(
                "stenographer.transcribe_audio", return_value={"text": ""}
            ) as mock_transcribe:
                with patch("sys.stdout", new=io.StringIO()):
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
            with patch(
                "stenographer.transcribe_audio", return_value={"text": ""}
            ) as mock_transcribe:
                with patch("sys.stdout", new=io.StringIO()):
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
            with patch(
                "stenographer.transcribe_audio", return_value={"text": ""}
            ) as mock_transcribe:
                with patch("sys.stdout", new=io.StringIO()):
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
            with patch(
                "stenographer.transcribe_audio", return_value={"text": ""}
            ) as mock_transcribe:
                with patch("sys.stdout", new=io.StringIO()):
                    stenographer.main([audio_path, "--compute-type", "int8"])

        mock_transcribe.assert_called_once_with(
            audio_path,
            model_name=stenographer.DEFAULT_MODEL,
            language=None,
            beam_size=5,
            compute_type="int8",
        )

    def test_transcribe_audio_accepts_mp4(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            fake_model = MagicMock()
            fake_model.transcribe.return_value = (
                [types.SimpleNamespace(text="Hello Teams")],
                types.SimpleNamespace(language="en", duration=5.0),
            )
            fake_whisper_module = types.SimpleNamespace(
                WhisperModel=MagicMock(return_value=fake_model)
            )

            with patch.dict(sys.modules, {"faster_whisper": fake_whisper_module}):
                with patch("stenographer.load_audio", return_value="AUDIO"):
                    result = stenographer.transcribe_audio(tmp.name)

        self.assertEqual(result["text"], "Hello Teams")
        self.assertEqual(result["language"], "en")


if __name__ == "__main__":
    unittest.main()
