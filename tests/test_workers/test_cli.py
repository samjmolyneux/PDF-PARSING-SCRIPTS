"""Run the worker script's command-line entry point without parser dependencies."""

import json
import sys

import pytest


def test_worker_script_help_lists_required_options(repo_root, run_command, tmp_path):
    result = run_command(sys.executable, str(repo_root / "workers/run.py"), "--help")
    assert result.returncode == 0, result.stderr
    assert "usage: run.py" in result.stdout
    assert all(
        option in result.stdout for option in ("--parser", "--input", "--output")
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_worker_script_empty_input_reports_failure(
    repo_root, run_command, tmp_path, parser
):
    inputs = tmp_path / "empty inputs"
    inputs.mkdir()
    output = tmp_path / "worker results"
    result = run_command(
        sys.executable,
        str(repo_root / "workers/run.py"),
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
