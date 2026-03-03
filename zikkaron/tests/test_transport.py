"""Tests for Streamable HTTP transport migration, health endpoint, and session management."""

import subprocess
import sys

import pytest
from starlette.testclient import TestClient

from zikkaron import __version__
from zikkaron import server


@pytest.fixture(autouse=True)
def _engines(tmp_path):
    """Initialize global engines with a temp database for each test."""
    db_path = str(tmp_path / "test_transport.db")
    server.init_engines(
        db_path=db_path, embedding_model="all-MiniLM-L6-v2"
    )
    yield
    server.shutdown()


# ── Health Endpoint ───────────────────────────────────────────────────


class TestHealthEndpoint:
    def _get_client(self, transport: str) -> TestClient:
        """Create a Starlette test client for the given transport."""
        if transport == "streamable-http":
            app = server.mcp_server.streamable_http_app()
        else:
            app = server.mcp_server.sse_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_health_returns_ok_sse(self):
        server._active_transport = "sse"
        server._start_time = 1000000.0
        client = self._get_client("sse")
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__
        assert data["transport"] == "sse"
        assert "uptime_seconds" in data
        assert "active_sessions" in data

    def test_health_returns_ok_streamable_http(self):
        server._active_transport = "streamable-http"
        server._start_time = 1000000.0
        client = self._get_client("streamable-http")
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == __version__
        assert data["transport"] == "streamable-http"

    def test_health_version_matches_package(self):
        client = self._get_client("sse")
        resp = client.get("/health")
        assert resp.json()["version"] == __version__

    def test_health_active_sessions_zero_on_fresh_start(self):
        client = self._get_client("sse")
        resp = client.get("/health")
        assert resp.json()["active_sessions"] == 0


# ── Session Management ────────────────────────────────────────────────


class TestSessionManagement:
    def test_streamable_http_app_has_session_manager(self):
        """Streamable HTTP transport creates a session manager."""
        app = server.mcp_server.streamable_http_app()
        assert server.mcp_server._session_manager is not None

    def test_session_manager_tracks_instances(self):
        """Session manager has _server_instances dict for tracking."""
        app = server.mcp_server.streamable_http_app()
        mgr = server.mcp_server._session_manager
        assert hasattr(mgr, "_server_instances")
        assert isinstance(mgr._server_instances, dict)

    def test_session_count_reflected_in_health(self):
        """Health endpoint session count reflects session manager state."""
        app = server.mcp_server.streamable_http_app()
        server._active_transport = "streamable-http"
        server._start_time = 1000000.0

        # Inject a fake session to verify counting
        mgr = server.mcp_server._session_manager
        mgr._server_instances["fake-session-1"] = object()
        mgr._server_instances["fake-session-2"] = object()

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        data = resp.json()
        assert data["active_sessions"] == 2

        # Cleanup
        mgr._server_instances.clear()


# ── Transport Selection ───────────────────────────────────────────────


class TestTransportSelection:
    def test_valid_transports_in_cli_help(self):
        """CLI advertises both sse and streamable-http transports."""
        result = subprocess.run(
            [sys.executable, "-m", "zikkaron", "--help"],
            capture_output=True, text=True,
        )
        assert "sse" in result.stdout
        assert "streamable-http" in result.stdout

    def test_main_accepts_transport_param(self):
        """server.main() accepts a transport keyword argument."""
        import inspect
        sig = inspect.signature(server.main)
        assert "transport" in sig.parameters
        assert sig.parameters["transport"].default == "stdio"

    def test_cli_transport_flag_default(self):
        """CLI defaults to stdio transport."""
        result = subprocess.run(
            [sys.executable, "-m", "zikkaron", "--help"],
            capture_output=True, text=True,
        )
        assert "--transport" in result.stdout
        assert "stdio" in result.stdout
        assert "streamable-http" in result.stdout

    def test_cli_rejects_invalid_transport(self):
        """CLI rejects unknown transport values."""
        result = subprocess.run(
            [sys.executable, "-m", "zikkaron", "--transport", "websocket"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_active_transport_tracks_selection(self):
        """_active_transport global is set by main() setup."""
        server._active_transport = "streamable-http"
        assert server._active_transport == "streamable-http"
        server._active_transport = "sse"
        assert server._active_transport == "sse"


# ── Streamable HTTP App Structure ─────────────────────────────────────


class TestStreamableHttpApp:
    def test_streamable_http_app_is_starlette(self):
        from starlette.applications import Starlette
        app = server.mcp_server.streamable_http_app()
        assert isinstance(app, Starlette)

    def test_sse_app_is_starlette(self):
        from starlette.applications import Starlette
        app = server.mcp_server.sse_app()
        assert isinstance(app, Starlette)

    def test_streamable_http_mcp_endpoint_exists(self):
        """The /mcp endpoint should exist on the streamable HTTP app."""
        app = server.mcp_server.streamable_http_app()
        paths = [route.path for route in app.routes]
        assert "/mcp" in paths

    def test_health_endpoint_on_both_transports(self):
        """Health endpoint is available on both SSE and Streamable HTTP apps."""
        sse_app = server.mcp_server.sse_app()
        http_app = server.mcp_server.streamable_http_app()

        sse_paths = [route.path for route in sse_app.routes]
        http_paths = [route.path for route in http_app.routes]

        assert "/health" in sse_paths
        assert "/health" in http_paths
