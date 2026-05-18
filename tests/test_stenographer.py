import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import stenographer


class StenographerTests(unittest.TestCase):
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
        self.assertEqual(result["model"], "KBLab/kb-whisper-small")

    def test_main_writes_output_file(self) -> None:
        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as wav,
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            output_file = Path(tmpdir) / "out.txt"
            with patch("stenographer.transcribe_audio", return_value={"text": "hej"}):
                exit_code = stenographer.main([wav.name, "--output", str(output_file)])
            output_text = output_file.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(output_text, "hej")


if __name__ == "__main__":
    unittest.main()
