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
from admin import register_environments, register_models


class RegistrationTests(unittest.TestCase):
    def test_previews_do_not_authenticate_or_connect_even_without_config(self):
        for module in (register_environments, register_models):
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


if __name__ == "__main__":
    unittest.main()
