"""Shared paths and local fixtures; ordinary tests cannot access the network."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--azure-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.json",
        help="Workspace configuration for integration tests (default: config.json)",
    )


@pytest.fixture(autouse=True)
def offline_by_default(request):
    if request.node.get_closest_marker("integration") is None:
        request.getfixturevalue("socket_disabled")


@pytest.fixture(scope="session")
def repo_root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def run_command(tmp_path):
    """Run a CLI in a temporary directory using the installed package."""

    def run(*args, cwd=tmp_path):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [str(arg) for arg in args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    return run


@pytest.fixture
def pdf_folder(tmp_path):
    folder = tmp_path / "inputs"
    folder.mkdir()
    return folder


@pytest.fixture
def make_pdf(pdf_folder):
    """Create a placeholder input for tests that simulate the parser."""

    def create(relative):
        path = pdf_folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test input")
        return path

    return create


@pytest.fixture
def output_folder(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def read_report(output_folder):
    def read():
        return json.loads((output_folder / "report.json").read_text())

    return read


@pytest.fixture
def worker_environment():
    """Restore environment variables changed by the in-process worker."""
    with patch.dict(os.environ):
        yield
