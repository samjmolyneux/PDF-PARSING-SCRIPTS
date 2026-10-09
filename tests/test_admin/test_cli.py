"""Check the command-line options of the administration scripts."""

import sys

import pytest


@pytest.mark.parametrize(
    ("module", "options"),
    [
        ("admin.deploy_parsers", ["--parser", "--config", "--apply"]),
        ("admin.setup_compute", ["--config", "--apply"]),
    ],
    ids=["deploy-parsers", "setup-compute"],
)
def test_admin_module_help_lists_supported_options(
    repo_root, run_command, module, options
):
    result = run_command(sys.executable, "-m", module, "--help", cwd=repo_root)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert all(option in result.stdout for option in options)
