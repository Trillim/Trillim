from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from trillim.components._stt import STT
from trillim.server import Server

EXPECTED_TEXT = (
    "on Danny Koviyat was known as the torpedo, which I actually think is unfair, "
    "because like I said, this incident wasn't really his fault. He did torpedo "
    "veil out of the race at the Russian Grand Prix, and that was a proper torpedo. "
    "And he did a similar thing to Fernando Alonso at the Austrian Grand Prix. He "
    "just drove straight to the back of him, which pushed Fernando into Max "
    "Verstappen and took both of them out of the race. And then at the British "
    "Grand Prix, he tried to go side by side with his teammate, Carlos through "
    "maggots and beckons before torpedoing him out with rest. Okay, maybe calling "
    "him the torpedo"
)


class STTRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_path = Path(__file__).with_name("test.wav")
        self.fixture_bytes = self.fixture_path.read_bytes()

    def _make_server(self) -> Server:
        return Server(STT())

    def test_audio_transcriptions_accepts_wav_and_octet_stream(self):
        with TestClient(self._make_server().app) as client:
            for content_type in ("audio/wav", "audio/x-wav", "application/octet-stream"):
                with self.subTest(content_type=content_type):
                    response = client.post(
                        "/v1/audio/transcriptions",
                        content=self.fixture_bytes,
                        headers={"content-type": content_type},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {"text": EXPECTED_TEXT})

    def test_audio_transcriptions_rejects_invalid_requests(self):
        cases = (
            (
                {"content-type": "text/plain"},
                {},
                self.fixture_bytes,
                400,
                "content-type",
            ),
            (
                {"content-type": "audio/wav"},
                {"language": "en_us"},
                self.fixture_bytes,
                400,
                "letters and hyphens",
            ),
            (
                {"content-type": "audio/wav", "content-length": "bad"},
                {},
                self.fixture_bytes,
                400,
                "content-length",
            ),
            (
                {"content-type": "audio/wav"},
                {},
                b"",
                400,
                "audio_bytes must not be empty",
            ),
            (
                {"content-type": "audio/wav", "content-length": "999999"},
                {},
                self.fixture_bytes,
                400,
                "content-length",
            ),
        )
        with TestClient(self._make_server().app) as client:
            for headers, params, body, status_code, message in cases:
                with self.subTest(headers=headers, params=params, body_len=len(body)):
                    response = client.post(
                        "/v1/audio/transcriptions",
                        params=params,
                        content=body,
                        headers=headers,
                    )
                    self.assertEqual(response.status_code, status_code)
                    self.assertIn(message, response.json()["detail"])

    def test_audio_transcriptions_rejects_shorter_than_claimed_body(self):
        with TestClient(self._make_server().app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                content=self.fixture_bytes,
                headers={
                    "content-type": "audio/wav",
                    "content-length": str(len(self.fixture_bytes) + 1),
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("content-length", response.json()["detail"])

    def test_concurrent_router_requests_return_success_and_busy(self):
        with TestClient(self._make_server().app) as client:
            def send_request():
                return client.post(
                    "/v1/audio/transcriptions",
                    content=self.fixture_bytes,
                    headers={"content-type": "audio/wav"},
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                responses = list(executor.map(lambda _: send_request(), range(2)))

        statuses = sorted(response.status_code for response in responses)
        self.assertEqual(statuses, [200, 429])
        successful = next(response for response in responses if response.status_code == 200)
        rejected = next(response for response in responses if response.status_code == 429)
        self.assertEqual(successful.json(), {"text": EXPECTED_TEXT})
        self.assertIn("busy", rejected.json()["detail"].lower())
