"""Both admin scripts must preview without authentication or Azure access."""

import sys
from unittest.mock import MagicMock

import pytest

from admin import deploy_parsers, setup_compute


@pytest.mark.parametrize(
    "module", [deploy_parsers, setup_compute], ids=["deploy-parsers", "setup-compute"]
)
def test_preview_does_not_authenticate_or_connect_even_without_config(
    module, monkeypatch
):
    credential, client = MagicMock(), MagicMock()
    monkeypatch.setattr(
        sys, "argv", [module.__name__, "--config", "/missing/config.json"]
    )
    monkeypatch.setattr(module, "AzureCliCredential", credential)
    monkeypatch.setattr(module, "MLClient", client)
    module.main()
    credential.assert_not_called()
    client.assert_not_called()
