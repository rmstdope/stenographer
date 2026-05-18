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

    def test_validate_audio_path_accepts_mp4(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            path = stenographer._validate_audio_path(tmp.name)
            self.assertEqual(path.suffix.lower(), ".mp4")

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
