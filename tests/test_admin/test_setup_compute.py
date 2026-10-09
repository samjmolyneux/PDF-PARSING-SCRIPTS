"""Check compute setup against a simulated Azure workspace."""

import json
import sys
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import HttpResponseError

from admin import setup_compute


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(None, id="succeeded"),
        pytest.param(HttpResponseError("Compute provisioning failed"), id="failed"),
    ],
)
def test_compute_targets_configured_workspace_and_waits_for_completion(
    tmp_path, monkeypatch, capsys, error
):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "tenant_id": "tenant",
                "subscription_id": "sub",
                "resource_group": "rg",
                "workspace": "ws",
            }
        )
    )
    client, credential = MagicMock(), MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(
        sys, "argv", ["setup_compute", "--apply", "--config", str(config)]
    )
    monkeypatch.setattr(setup_compute, "AzureCliCredential", credential)
    monkeypatch.setattr(setup_compute, "MLClient", factory)
    poller = client.compute.begin_create_or_update.return_value
    poller.result.side_effect = error

    with (
        pytest.raises(HttpResponseError, match="Compute provisioning failed")
        if error
        else nullcontext()
    ):
        setup_compute.main()

    credential.assert_called_once_with(tenant_id="tenant")
    factory.assert_called_once_with(credential.return_value, "sub", "rg", "ws")
    client.compute.begin_create_or_update.assert_called_once()
    compute = client.compute.begin_create_or_update.call_args.args[0]
    assert compute.name == "pdf-parsers-a100"
    assert compute.size == "Standard_NC24ads_A100_v4"
    assert (compute.min_instances, compute.max_instances) == (0, 2)
    assert compute.identity.type == "system_assigned"
    poller.result.assert_called_once_with()
    assert ("Compute configured:" in capsys.readouterr().out) is (error is None)
