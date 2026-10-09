"""Exercise server readiness and shutdown without launching Paddle or a GPU."""

import io
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from workers import run as worker


@pytest.fixture
def server(monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(
        worker, "paddle_server_command", lambda: (["server"], "http://localhost/health")
    )
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(worker.time, "sleep", MagicMock())
    monkeypatch.setattr(worker, "stop_paddle_server", MagicMock())
    return process


@pytest.mark.parametrize(
    "first_response",
    [
        pytest.param(OSError("starting"), id="connection-unavailable"),
        pytest.param(503, id="not-ready"),
    ],
)
def test_server_waits_until_ready(server, monkeypatch, first_response):
    not_ready, ready = MagicMock(), MagicMock()
    not_ready.__enter__.return_value.status = first_response
    ready.__enter__.return_value.status = 200
    first = first_response if isinstance(first_response, OSError) else not_ready
    health = MagicMock(side_effect=[first, ready])
    monkeypatch.setattr(worker.urllib.request, "urlopen", health)

    assert worker.start_paddle_server(io.BytesIO()) is server
    assert health.call_count == 2
    worker.stop_paddle_server.assert_not_called()


def test_readiness_timeout_stops_the_server(server, monkeypatch):
    monkeypatch.setattr(worker.time, "monotonic", MagicMock(side_effect=[0, 0, 5]))
    monkeypatch.setattr(
        worker.urllib.request, "urlopen", MagicMock(side_effect=OSError("starting"))
    )

    with pytest.raises(TimeoutError, match="did not become ready"):
        worker.start_paddle_server(io.BytesIO(), timeout=4)
    worker.stop_paddle_server.assert_called_once_with(server)


def test_interrupt_during_startup_stops_the_server(server, monkeypatch):
    monkeypatch.setattr(
        worker.urllib.request, "urlopen", MagicMock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        worker.start_paddle_server(io.BytesIO())
    worker.stop_paddle_server.assert_called_once_with(server)


def test_missing_server_executable_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("PADDLE_SERVER_PYTHON", str(tmp_path / "missing-python"))
    monkeypatch.setenv("PADDLE_SERVER_CLI", str(tmp_path / "missing-cli"))
    launch = MagicMock()
    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    with pytest.raises(FileNotFoundError, match="Base image server executable missing"):
        worker.start_paddle_server(io.BytesIO())
    launch.assert_not_called()


@pytest.mark.parametrize("parent_times_out", [False, True])
def test_shutdown_cleans_up_children_even_when_parent_has_exited(
    monkeypatch, parent_times_out
):
    # The worker runs on Linux. Substitute its Unix APIs so this test also runs on Windows.
    killpg = MagicMock()
    monkeypatch.setattr(worker.os, "killpg", killpg, raising=False)
    monkeypatch.setattr(worker, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    process = MagicMock(pid=1234)
    if parent_times_out:
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 15), None]

    worker.stop_paddle_server(process)

    assert killpg.call_args_list == [call(1234, 15), call(1234, 9)]
    assert process.wait.call_args_list == [call(timeout=15), call()]


def test_shutdown_handles_an_already_gone_process(monkeypatch):
    monkeypatch.setattr(
        worker.os, "killpg", MagicMock(side_effect=ProcessLookupError), raising=False
    )
    process = MagicMock(pid=1234)
    worker.stop_paddle_server(process)
    process.wait.assert_not_called()
