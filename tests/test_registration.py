"""Registration previews stay offline; applying uses the normal Azure SDK."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from azure.ai.ml.entities import Model
from azure.core.exceptions import HttpResponseError
from admin import deploy_parsers, register_models, setup_compute


class RegistrationTests(unittest.TestCase):
    def test_previews_do_not_authenticate_or_connect_even_without_config(self):
        for module in (deploy_parsers, register_models, setup_compute):
            with self.subTest(module=module.__name__), \
                 patch.object(sys, "argv", [module.__name__, "--config", "/missing/config.json"]), \
                 patch.object(module, "AzureCliCredential") as credential, \
                 patch.object(module, "MLClient") as client, \
                 contextlib.redirect_stdout(io.StringIO()):
                module.main()
                credential.assert_not_called()
                client.assert_not_called()

    def test_registration_needs_no_manifest_and_propagates_azure_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"weights")
            config = root / "config.json"
            config.write_text(json.dumps({"tenant_id": "tenant", "subscription_id": "sub",
                                          "resource_group": "rg", "workspace": "ws"}))
            model = Model(name="test-model", version="2", type="custom_model", path=str(root))
            client = MagicMock()
            with patch.object(sys, "argv", ["register_models", "--parser", "mineru", "--apply", "--config", str(config)]), \
                 patch.object(register_models, "load_model", return_value=model), \
                 patch.object(register_models, "AzureCliCredential"), \
                 patch.object(register_models, "MLClient", return_value=client), \
                 contextlib.redirect_stdout(io.StringIO()):
                register_models.main()
                client.models.create_or_update.assert_called_once_with(model)
                client.models.get.assert_not_called()
                client.models.create_or_update.side_effect = HttpResponseError("403 forbidden")
                with self.assertRaises(HttpResponseError):
                    register_models.main()

    def test_compute_targets_configured_workspace_and_waits_for_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"tenant_id": "tenant", "subscription_id": "sub",
                                          "resource_group": "rg", "workspace": "ws"}))
            client = MagicMock()
            output = io.StringIO()
            with patch.object(sys, "argv", ["setup_compute", "--apply", "--config", str(config)]), \
                 patch.object(setup_compute, "AzureCliCredential") as credential, \
                 patch.object(setup_compute, "MLClient", return_value=client) as factory, \
                 contextlib.redirect_stdout(output):
                setup_compute.main()
                credential.assert_called_once_with(tenant_id="tenant")
                factory.assert_called_once_with(credential.return_value, "sub", "rg", "ws")
                client.compute.begin_create_or_update.assert_called_once()
                compute = client.compute.begin_create_or_update.call_args.args[0]
                self.assertEqual(compute.name, "pdf-parsers-a100")
                self.assertEqual(compute.size, "Standard_NC24ads_A100_v4")
                self.assertEqual((compute.min_instances, compute.max_instances), (0, 1))
                self.assertEqual(compute.identity.type, "system_assigned")
                poller = client.compute.begin_create_or_update.return_value
                poller.result.assert_called_once_with()

                output.seek(0)
                output.truncate()
                poller.result.side_effect = HttpResponseError("Compute provisioning failed")
                with self.assertRaises(HttpResponseError):
                    setup_compute.main()
                self.assertNotIn("Compute configured:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
