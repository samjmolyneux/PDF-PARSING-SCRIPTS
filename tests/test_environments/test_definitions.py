"""Validate the build context of each parser environment."""

from pathlib import Path

import pytest
from azure.ai.ml import load_environment


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_environment_build_context_resolves(repo_root, parser):
    environment = load_environment(repo_root / f"environments/{parser}/environment.yml")
    environment.validate()
    # Azure resolves the upload folder relative to environment.yml.
    context = Path(environment.path).resolve()
    assert context == repo_root / f"environments/{parser}"
    assert (context / environment.build.dockerfile_path).is_file()
    if parser == "paddle":
        assert (context / "conda.yml").is_file()
    assert environment.image is None
    assert environment.conda_file is None
