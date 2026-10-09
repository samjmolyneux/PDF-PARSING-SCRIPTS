"""Parser deployment selects the right assets and stops when Azure reports failure."""

import contextlib
import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import HttpResponseError

from admin import deploy_parsers


class TestDeployment:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.config = tmp_path / "config.json"
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

    @pytest.mark.parametrize(
        "selection, parsers",
        [
            pytest.param("paddle", ["paddle"], id="paddle"),
            pytest.param("mineru", ["mineru"], id="mineru"),
            pytest.param("both", ["mineru", "paddle"], id="both"),
            pytest.param(None, ["mineru", "paddle"], id="default"),
        ],
    )
    def test_parser_selection_and_configured_endpoint(self, selection, parsers):
        client = MagicMock()
        arguments = ["deploy_parsers", "--apply", "--config", str(self.config)]
        if selection:
            arguments += ["--parser", selection]
        with (
            patch.object(sys, "argv", arguments),
            patch.object(deploy_parsers, "AzureCliCredential") as credential,
            patch.object(deploy_parsers, "MLClient", return_value=client) as factory,
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
            assert [env.name for env in environments] == [
                f"pdf-{parser}" for parser in parsers
            ]
            pipeline_names = {
                "mineru": "pdf_mineru_pipeline",
                "paddle": "paddle_vl_pipeline",
            }
            assert [pipeline.name for pipeline in pipelines] == [
                pipeline_names[parser] for parser in parsers
            ]
            deployment_names = {"mineru": "mineru", "paddle": "paddle-vl"}
            assert [deployment.name for deployment in deployments] == [
                deployment_names[parser] for parser in parsers
            ]
            endpoint = client.batch_endpoints.begin_create_or_update.call_args.args[0]
            assert endpoint.name == "custom-endpoint"
            assert endpoint.auth_mode == "aad_token"
            for pipeline, deployment in zip(pipelines, deployments):
                assert deployment.endpoint_name == "custom-endpoint"
                assert (
                    deployment.component.removeprefix("azureml:")
                    == f"{pipeline.name}:{pipeline.version}"
                )
                assert deployment.settings["default_compute"] == "pdf-parsers-a100"

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
            assert [call[0] for call in client.mock_calls] == expected

    @pytest.mark.parametrize(
        "stage, expected_calls",
        [
            pytest.param(0, 1, id="environment"),
            pytest.param(1, 2, id="pipeline"),
            pytest.param(2, 4, id="endpoint"),
            pytest.param(3, 6, id="deployment"),
        ],
    )
    def test_failed_registration_or_provisioning_stops_deployment(
        self, stage, expected_calls
    ):
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
            pytest.raises(HttpResponseError, match="Azure operation failed"),
        ):
            deploy_parsers.main()
        assert len(client.mock_calls) == expected_calls
        assert "Parser deployments configured." not in output.getvalue()
