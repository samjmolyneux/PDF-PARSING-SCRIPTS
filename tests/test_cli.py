"""Exercise installed commands and script entry points without Azure calls."""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def command(tmp_path):
    def run(name, *args):
        suffix = ".exe" if sys.platform == "win32" else ""
        executable = Path(sysconfig.get_path("scripts")) / (name + suffix)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [str(executable), *args],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    return run


@pytest.fixture
def python_command(tmp_path):
    def run(*args, cwd=tmp_path):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        # Fixed interpreter and project entry points; no shell is involved.
        return subprocess.run(  # noqa: S603
            [sys.executable, *args],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

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
def test_parser_module_help_works_outside_the_repository(
    python_command, tmp_path, parser
):
    result = python_command("-m", f"pdf_parsers.{parser}", "--help")
    assert result.returncode == 0, result.stderr
    assert f"usage: run-{parser}" in result.stdout
    options = ["--config", "--output", "--resume", "--no-wait", "--browser-login"]
    assert all(option in result.stdout for option in options)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("module", "options"),
    [
        ("admin.deploy_parsers", ["--parser", "--config", "--apply"]),
        ("admin.setup_compute", ["--config", "--apply"]),
    ],
    ids=["deploy-parsers", "setup-compute"],
)
def test_admin_module_help_lists_supported_options(python_command, module, options):
    result = python_command("-m", module, "--help", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert all(option in result.stdout for option in options)


def test_worker_script_help_lists_required_options(python_command, tmp_path):
    result = python_command(str(ROOT / "workers/run.py"), "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: run.py" in result.stdout
    assert all(
        option in result.stdout for option in ("--parser", "--input", "--output")
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_worker_script_empty_input_reports_failure(python_command, tmp_path, parser):
    inputs = tmp_path / "empty inputs"
    inputs.mkdir()
    output = tmp_path / "worker results"
    result = python_command(
        str(ROOT / "workers/run.py"),
        "--parser",
        parser,
        "--input",
        str(inputs),
        "--output",
        str(output),
    )
    assert result.returncode == 1, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["parser"] == parser
    assert report["total"] == 0
    assert report["documents"] == []
    assert report["finished"] is False
    assert "No PDFs" in report["fatal_error"]
    assert "No PDFs" in result.stdout
    assert "No PDFs" in (output / "parser.log").read_text()
    assert not (output / "documents").exists()
    assert not (output / "server.log").exists()
