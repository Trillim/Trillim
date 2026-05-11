from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

from fastapi.testclient import TestClient

from trillim.components import Component
from trillim.server import Server

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class E2EServerTests(unittest.TestCase):
    def test_server_serves_health_endpoint_through_real_fastapi_app(self):
        server = Server(Component())
        pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

        with TestClient(server.app) as client:
            response = client.get("/healthz")
            openapi_response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(openapi_response.status_code, 200)
        self.assertEqual(
            openapi_response.json()["info"]["version"],
            pyproject["project"]["version"],
        )
