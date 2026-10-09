"""Run the installed Paddle and MinerU clients from outside the repository."""

import json
import sys
import sysconfig
from pathlib import Path

import pytest


@pytest.fixture
def command(run_command):
    def run(name, *args):
        suffix = ".exe" if sys.platform == "win32" else ""
        executable = Path(sysconfig.get_path("scripts")) / (name + suffix)
        return run_command(executable, *args)

    return run


@pytest.mark.parametrize("name", ["run-mineru", "run-paddle"])
def test_help_works_outside_the_repository(command, tmp_path, name):
    result = command(name, "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: {name}" in result.stdout
    options = ["--config", "--output", "--resume", "--no-wait", "--browser-login"]
    assert all(option in result.stdout for option in options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["run-mineru", "run-paddle"])
def test_missing_input_returns_cli_usage_error(command, name):
    result = command(name)
    assert result.returncode == 2
    assert "required" in result.stderr


@pytest.mark.parametrize(
    "command_parser, receipt_parser", [("paddle", "mineru"), ("mineru", "paddle")]
)
def test_wrong_parser_receipt_recommends_the_other_command(
    command, tmp_path, command_parser, receipt_parser
):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"parser": receipt_parser}))
    result = command(f"run-{command_parser}", "--resume", str(receipt))
    assert result.returncode == 1
    assert f"Use run-{receipt_parser} for this receipt" in result.stderr


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_parser_module_help_works_outside_the_repository(run_command, tmp_path, parser):
    result = run_command(sys.executable, "-m", f"pdf_parsers.{parser}", "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: run-{parser}" in result.stdout
    options = ["--config", "--output", "--resume", "--no-wait", "--browser-login"]
    assert all(option in result.stdout for option in options)
    assert not list(tmp_path.iterdir())
