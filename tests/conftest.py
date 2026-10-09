"""Keep ordinary tests offline; Azure tests require an explicit integration selection."""

from pathlib import Path

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
