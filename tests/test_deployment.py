"""Parser deployment selects the right assets and stops when Azure reports failure."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from azure.core.exceptions import HttpResponseError
from admin import deploy_parsers


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config = Path(temporary.name) / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "tenant_id": "tenant",
                    "subscription_id": "subscription",
                    "resource_group": "group",
                    "workspace": "workspace",
                    "endpoint": "custom-endpoint",
                }
            )
        )

    def test_parser_selection_and_configured_endpoint(self):
        for selection, parsers in [
            ("paddle", ["paddle"]),
            ("mineru", ["mineru"]),
            ("both", ["mineru", "paddle"]),
            (None, ["mineru", "paddle"]),
        ]:
            client = MagicMock()
            arguments = ["deploy_parsers", "--apply", "--config", str(self.config)]
            if selection:
                arguments += ["--parser", selection]
            with (
                self.subTest(selection=selection),
                patch.object(sys, "argv", arguments),
                patch.object(deploy_parsers, "AzureCliCredential") as credential,
                patch.object(
                    deploy_parsers, "MLClient", return_value=client
                ) as factory,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                deploy_parsers.main()

                credential.assert_called_once_with(tenant_id="tenant")
                factory.assert_called_once_with(
                    credential.return_value, "subscription", "group", "workspace"
                )
                environments = [
                    call.args[0]
                    for call in client.environments.create_or_update.call_args_list
                ]
                pipelines = [
                    call.args[0]
                    for call in client.components.create_or_update.call_args_list
                ]
                deployments = [
                    call.args[0]
                    for call in client.batch_deployments.begin_create_or_update.call_args_list
                ]
                self.assertEqual(
                    [env.name for env in environments],
                    [f"pdf-{parser}" for parser in parsers],
                )
                pipeline_names = {
                    "mineru": "pdf_mineru_pipeline",
                    "paddle": "paddle_vl_pipeline",
                }
                self.assertEqual(
                    [pipeline.name for pipeline in pipelines],
                    [pipeline_names[parser] for parser in parsers],
                )
                deployment_names = {"mineru": "mineru", "paddle": "paddle-vl"}
                self.assertEqual(
                    [deployment.name for deployment in deployments],
                    [deployment_names[parser] for parser in parsers],
                )
                endpoint = client.batch_endpoints.begin_create_or_update.call_args.args[
                    0
                ]
                self.assertEqual(endpoint.name, "custom-endpoint")
                self.assertEqual(endpoint.auth_mode, "aad_token")
                for pipeline, deployment in zip(pipelines, deployments):
                    self.assertEqual(deployment.endpoint_name, "custom-endpoint")
                    self.assertEqual(
                        deployment.component.removeprefix("azureml:"),
                        f"{pipeline.name}:{pipeline.version}",
                    )
                    self.assertEqual(
                        deployment.settings["default_compute"], "pdf-parsers-a100"
                    )

                # Only the selected assets are registered; no models, compute or jobs.
                expected = ["environments.create_or_update"] * len(parsers)
                expected += ["components.create_or_update"] * len(parsers)
                expected += [
                    "batch_endpoints.begin_create_or_update",
                    "batch_endpoints.begin_create_or_update().result",
                ]
                expected += [
                    "batch_deployments.begin_create_or_update",
                    "batch_deployments.begin_create_or_update().result",
                ] * len(parsers)
                self.assertEqual([call[0] for call in client.mock_calls], expected)

    def test_failed_registration_or_provisioning_stops_deployment(self):
        for stage, expected_calls in enumerate((1, 2, 4, 6)):
            client = MagicMock()
            operations = [
                client.environments.create_or_update,
                client.components.create_or_update,
                client.batch_endpoints.begin_create_or_update.return_value.result,
                client.batch_deployments.begin_create_or_update.return_value.result,
            ]
            operations[stage].side_effect = HttpResponseError("Azure operation failed")
            output = io.StringIO()
            with (
                self.subTest(stage=stage),
                patch.object(
                    sys,
                    "argv",
                    [
                        "deploy_parsers",
                        "--parser",
                        "paddle",
                        "--apply",
                        "--config",
                        str(self.config),
                    ],
                ),
                patch.object(deploy_parsers, "AzureCliCredential"),
                patch.object(deploy_parsers, "MLClient", return_value=client),
                contextlib.redirect_stdout(output),
            ):
                with self.assertRaises(HttpResponseError):
                    deploy_parsers.main()
                self.assertEqual(len(client.mock_calls), expected_calls)
                self.assertNotIn("Parser deployments configured.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
