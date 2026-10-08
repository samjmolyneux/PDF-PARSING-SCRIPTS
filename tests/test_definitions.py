"""Load definitions through Azure ML's schemas without contacting Azure."""

from pathlib import Path
import unittest

from azure.ai.ml import (
    load_batch_endpoint,
    load_component,
    load_compute,
    load_environment,
)
from azure.ai.ml.entities._load_functions import (
    load_pipeline_component_batch_deployment,
)
from pdf_parsers.client import DEPLOYMENT_NAMES

ROOT = Path(__file__).resolve().parents[1]


class DefinitionTests(unittest.TestCase):
    def test_compute_endpoint_and_environments_load(self):
        compute = load_compute(ROOT / "azure/compute.yml")
        self.assertEqual(compute.min_instances, 0)
        self.assertEqual(compute.max_instances, 1)
        self.assertEqual(
            load_batch_endpoint(ROOT / "azure/endpoint.yml").auth_mode, "aad_token"
        )
        for parser in ("mineru", "paddle"):
            with self.subTest(parser=parser):
                environment = load_environment(
                    ROOT / f"environments/{parser}/environment.yml"
                )
                environment.validate()
                # Azure resolves the upload folder relative to environment.yml.
                context = Path(environment.path).resolve()
                self.assertEqual(context, ROOT / f"environments/{parser}")
                self.assertTrue((context / environment.build.dockerfile_path).is_file())
                if parser == "paddle":
                    self.assertTrue((context / "conda.yml").is_file())
                self.assertIsNone(environment.image)
                self.assertIsNone(environment.conda_file)

    def test_components_and_pipeline_deployments_validate(self):
        for parser in ("mineru", "paddle"):
            with self.subTest(parser=parser):
                for kind in ("command", "pipeline"):
                    component = load_component(ROOT / f"azure/{parser}-{kind}.yml")
                    self.assertTrue(component._validate().passed)
                    self.assertEqual(component.outputs["results"].mode, "rw_mount")
                deployment = load_pipeline_component_batch_deployment(
                    ROOT / f"azure/{parser}-deployment.yml"
                )
                self.assertEqual(deployment.type, "pipeline")
                self.assertEqual(
                    deployment.settings["default_compute"], "pdf-parsers-a100"
                )
                self.assertTrue(deployment.settings["force_rerun"])

    def test_runtime_variables_survive_pipeline_serialization(self):
        parser_variables = {
            "mineru": {
                "MINERU_MODEL_SOURCE": "local",
                "MINERU_TASK_RESULT_TIMEOUT_SECONDS": "86400",
            },
            "paddle": {
                "PADDLE_SERVER_PYTHON": "/usr/local/bin/python",
                "PADDLE_SERVER_CLI": "/usr/local/bin/paddleocr",
            },
        }
        # Schema validation alone didn't catch variables being lost when they
        # were placed on the reusable component instead of the pipeline job.
        for parser, variables in parser_variables.items():
            with self.subTest(parser=parser):
                pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
                expected = {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", **variables}
                self.assertEqual(pipeline.jobs["parse"].environment_variables, expected)
                registered = pipeline._to_rest_object().properties.component_spec
                self.assertEqual(
                    registered["jobs"]["parse"]["environment_variables"], expected
                )

    def test_version_references_agree_and_pdfs_are_the_only_job_input(self):
        for parser in ("mineru", "paddle"):
            environment = load_environment(
                ROOT / f"environments/{parser}/environment.yml"
            )
            command = load_component(ROOT / f"azure/{parser}-command.yml")
            pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
            deployment = load_pipeline_component_batch_deployment(
                ROOT / f"azure/{parser}-deployment.yml"
            )
            self.assertEqual(
                command.environment.removeprefix("azureml:"),
                f"{environment.name}:{environment.version}",
            )
            self.assertEqual(set(command.inputs), {"pdfs"})
            self.assertEqual(set(pipeline.jobs["parse"].inputs), {"pdfs"})
            self.assertEqual(set(pipeline.inputs), {"pdfs"})
            self.assertEqual(
                deployment.component.removeprefix("azureml:"),
                f"{pipeline.name}:{pipeline.version}",
            )
            self.assertEqual(deployment.name, DEPLOYMENT_NAMES[parser])


if __name__ == "__main__":
    unittest.main()
