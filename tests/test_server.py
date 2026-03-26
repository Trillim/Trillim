"""Tests for server/app composition."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import APIRouter
from fastapi.testclient import TestClient

from trillim.components import Component
from trillim.errors import ComponentLifecycleError
from trillim.server import Server


class _RouteComponent(Component):
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls = calls
        self._component_name = name

    @property
    def component_name(self) -> str:
        return self._component_name

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.get(f"/{self.component_name}")
        async def route():
            return {"name": self.component_name}

        return router

    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")

    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")


class _BrokenStartComponent(_RouteComponent):
    async def start(self) -> None:
        self.calls.append(f"{self.component_name}.start")
        raise RuntimeError("broken start")


class _BrokenStopComponent(_RouteComponent):
    async def stop(self) -> None:
        self.calls.append(f"{self.component_name}.stop")
        raise RuntimeError("broken stop")


class _HotSwapAwareComponent(_RouteComponent):
    def __init__(self, calls: list[str], name: str) -> None:
        super().__init__(calls, name)
        self.enabled: bool | None = None

    def _set_hot_swap_routes_enabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _ModelInfoComponent(_RouteComponent):
    def __init__(self, calls: list[str], name: str, *, state: str) -> None:
        super().__init__(calls, name)
        self._state = state

    def model_info(self):
        return SimpleNamespace(state=self._state)


class _EnumStateComponent(_RouteComponent):
    def __init__(self, calls: list[str], name: str, *, state: str) -> None:
        super().__init__(calls, name)
        self._state = state

    def model_info(self):
        return SimpleNamespace(state=SimpleNamespace(value=self._state))


class ServerTests(unittest.TestCase):
    def test_server_requires_components(self):
        with self.assertRaisesRegex(ValueError, "at least one component"):
            Server()

    def test_server_rejects_duplicate_component_names(self):
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, "Duplicate component name"):
            Server(_RouteComponent(calls, "dup"), _RouteComponent(calls, "dup"))

    def test_server_builds_app_and_runs_lifecycle(self):
        calls: list[str] = []
        server = Server(
            _RouteComponent(calls, "one"),
            _RouteComponent(calls, "two"),
            allow_hot_swap=True,
        )
        self.assertTrue(server.allow_hot_swap)
        with TestClient(server.app) as client:
            self.assertEqual(client.get("/healthz").json(), {"status": "ok"})
            self.assertEqual(client.get("/one").json(), {"name": "one"})
            self.assertEqual(client.get("/two").json(), {"name": "two"})
        self.assertEqual(
            calls,
            ["one.start", "two.start", "two.stop", "one.stop"],
        )

    def test_server_healthz_returns_503_for_unready_component_states(self):
        for state in ("server_error", "draining", "swapping", "unavailable"):
            with self.subTest(state=state):
                calls: list[str] = []
                server = Server(_ModelInfoComponent(calls, "llm", state=state))
                with TestClient(server.app) as client:
                    response = client.get("/healthz")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(
                    response.json(),
                    {"status": "degraded", "components": {"llm": {"state": state}}},
                )

    def test_server_healthz_keeps_running_model_info_components_ready(self):
        calls: list[str] = []
        server = Server(_ModelInfoComponent(calls, "llm", state="running"))
        with TestClient(server.app) as client:
            response = client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_server_stops_started_components_when_startup_fails(self):
        calls: list[str] = []
        server = Server(_RouteComponent(calls, "ok"), _BrokenStartComponent(calls, "bad"))
        with self.assertRaisesRegex(RuntimeError, "Component startup failed"):
            with TestClient(server.app):
                pass
        self.assertEqual(calls, ["ok.start", "bad.start", "ok.stop"])

    def test_server_configures_hot_swap_routes_only_for_llm_components(self):
        calls: list[str] = []
        llm_component = _HotSwapAwareComponent(calls, "llm")
        other_component = _HotSwapAwareComponent(calls, "other")

        server = Server(llm_component, other_component, allow_hot_swap=True)
        _ = server.app

        self.assertTrue(llm_component.enabled)
        self.assertFalse(other_component.enabled)

    def test_server_healthz_uses_state_value_objects(self):
        calls: list[str] = []
        server = Server(_EnumStateComponent(calls, "llm", state="swapping"))

        with TestClient(server.app) as client:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json(),
            {"status": "degraded", "components": {"llm": {"state": "swapping"}}},
        )

    def test_server_exposes_components_and_run_delegates_to_uvicorn(self):
        calls: list[str] = []
        component = _RouteComponent(calls, "one")
        server = Server(component)

        self.assertEqual(server.components, (component,))
        with patch("trillim.server.uvicorn.run") as run:
            server.run(host="0.0.0.0", port=9000, log_level="debug")

        run.assert_called_once_with(server.app, host="0.0.0.0", port=9000, log_level="debug")

    def test_server_raises_component_lifecycle_error_when_shutdown_fails(self):
        calls: list[str] = []
        server = Server(_RouteComponent(calls, "ok"), _BrokenStopComponent(calls, "bad"))

        with self.assertRaisesRegex(ComponentLifecycleError, "Component shutdown failed"):
            with TestClient(server.app):
                pass

        self.assertEqual(calls, ["ok.start", "bad.start", "bad.stop", "ok.stop"])

    def test_server_startup_cleanup_swallows_shutdown_errors(self):
        calls: list[str] = []
        server = Server(_BrokenStopComponent(calls, "ok"), _BrokenStartComponent(calls, "bad"))

        with self.assertRaisesRegex(ComponentLifecycleError, "Component startup failed"):
            with TestClient(server.app):
                pass

        self.assertEqual(calls, ["ok.start", "bad.start", "ok.stop"])
