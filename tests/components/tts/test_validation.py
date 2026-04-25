from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from trillim.components.tts._limits import (
    MAX_HTTP_TEXT_BYTES,
    MAX_INPUT_TEXT_CHARS,
    MAX_VOICE_UPLOAD_BYTES,
)
from trillim.components.tts._validation import (
    PayloadTooLargeError,
    dump_voice_state_safetensors_bytes,
    load_safe_voice_state_safetensors_bytes,
    normalize_optional_name,
    normalize_required_name,
    open_validated_source_audio_file,
    validate_http_speech_body,
    validate_http_speech_request,
    validate_http_voice_upload_request,
    validate_speed,
    validate_text,
    validate_voice_bytes,
)
from trillim.errors import InvalidRequestError

from tests.components.tts.support import fake_voice_state


class TTSValidationTests(unittest.TestCase):
    def test_text_name_speed_and_body_validation(self):
        self.assertEqual(validate_text(" hello "), " hello ")
        self.assertEqual(normalize_required_name(" custom ", field_name="voice"), "custom")
        self.assertIsNone(normalize_optional_name(None, field_name="voice"))
        self.assertEqual(validate_speed("1.5"), 1.5)
        self.assertEqual(validate_http_speech_body("hi".encode()), "hi")

        invalid_cases = (
            (validate_text, (123,), "text must be a string"),
            (validate_text, (" ",), "must not be empty"),
            (validate_text, ("x" * (MAX_INPUT_TEXT_CHARS + 1),), "character limit"),
            (normalize_required_name, (None,), "header is required"),
            (normalize_optional_name, ("bad-name",), "letters and digits"),
            (validate_speed, ("fast",), "number"),
            (validate_speed, (99,), "between"),
            (validate_http_speech_body, (b"",), "must not be empty"),
            (validate_http_speech_body, (b"\xff",), "valid UTF-8"),
        )
        for func, args, message in invalid_cases:
            with self.subTest(func=func.__name__, message=message):
                kwargs = (
                    {"field_name": "voice"}
                    if func in {normalize_required_name, normalize_optional_name}
                    else {}
                )
                with self.assertRaisesRegex(InvalidRequestError, message):
                    func(*args, **kwargs)

    def test_http_metadata_and_size_validation(self):
        speech = validate_http_speech_request(
            content_length="2",
            voice="alba",
            speed=None,
        )
        self.assertEqual(speech.content_length, 2)
        self.assertEqual(speech.voice, "alba")
        self.assertEqual(speech.speed, 1.0)
        upload = validate_http_voice_upload_request(
            content_length="3",
            name="custom",
        )
        self.assertEqual(upload.content_length, 3)
        self.assertEqual(upload.name, "custom")
        self.assertEqual(validate_voice_bytes(b"voice"), b"voice")

        with self.assertRaisesRegex(InvalidRequestError, "invalid content-length"):
            validate_http_speech_request(content_length="bad", voice=None, speed=None)
        with self.assertRaisesRegex(PayloadTooLargeError, "speech input exceeds"):
            validate_http_speech_request(
                content_length=str(MAX_HTTP_TEXT_BYTES + 1),
                voice=None,
                speed=None,
            )
        with self.assertRaisesRegex(PayloadTooLargeError, "voice upload exceeds"):
            validate_http_voice_upload_request(
                content_length=str(MAX_VOICE_UPLOAD_BYTES + 1),
                name="custom",
            )
        with self.assertRaisesRegex(InvalidRequestError, "audio must be bytes"):
            validate_voice_bytes("voice")
        with self.assertRaisesRegex(InvalidRequestError, "must not be empty"):
            validate_voice_bytes(b"")

    def test_safetensors_voice_state_roundtrip_and_malformed_payload(self):
        payload = dump_voice_state_safetensors_bytes(fake_voice_state())
        state = load_safe_voice_state_safetensors_bytes(payload)
        self.assertEqual(state["module"]["cache"].tolist(), [1.0])

        with self.assertRaisesRegex(InvalidRequestError, "malformed"):
            load_safe_voice_state_safetensors_bytes(b"not safetensors")
        with self.assertRaisesRegex(InvalidRequestError, "malformed"):
            dump_voice_state_safetensors_bytes({"module": {"bad/key": object()}})

    def test_open_validated_source_audio_file_accepts_regular_nonempty_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "voice.wav"
            source.write_bytes(b"voice")
            fd = open_validated_source_audio_file(source)
            try:
                self.assertGreater(os.fstat(fd).st_size, 0)
            finally:
                os.close(fd)

    def test_open_validated_source_audio_file_rejects_unsafe_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = root / "missing.wav"
            empty = root / "empty.wav"
            directory = root / "directory"
            empty.write_bytes(b"")
            directory.mkdir()

            cases = (
                (missing, "does not exist"),
                (empty, "must not be empty"),
                (directory, "not a regular file"),
            )
            for path, message in cases:
                with self.subTest(path=path):
                    with self.assertRaisesRegex(InvalidRequestError, message):
                        open_validated_source_audio_file(path)

            target = root / "target.wav"
            target.write_bytes(b"voice")
            symlink = root / "link.wav"
            symlink.symlink_to(target)
            with self.assertRaisesRegex(InvalidRequestError, "symlinks"):
                open_validated_source_audio_file(symlink)
