"""Load definitions through Azure ML's schemas without contacting Azure."""
from pathlib import Path
import unittest

from azure.ai.ml import load_batch_endpoint, load_component, load_compute, load_environment, load_model
from azure.ai.ml.entities._load_functions import load_pipeline_component_batch_deployment

ROOT = Path(__file__).resolve().parents[1]


class DefinitionTests(unittest.TestCase):
    def test_compute_endpoint_and_environments_load(self):
        compute = load_compute(ROOT / "azure/compute.yml")
        self.assertEqual(compute.min_instances, 0)
        self.assertEqual(compute.max_instances, 1)
        self.assertEqual(load_batch_endpoint(ROOT / "azure/endpoint.yml").auth_mode, "aad_token")
        for parser in ("mineru", "paddle"):
            with self.subTest(parser=parser):
                environment = load_environment(ROOT / f"environments/{parser}/environment.yml")
                self.assertIsNone(environment.build)
                self.assertTrue(environment.image)
                self.assertTrue(environment.conda_file["dependencies"])

    def test_components_and_pipeline_deployments_validate(self):
        for parser in ("mineru", "paddle"):
            with self.subTest(parser=parser):
                for kind in ("command", "pipeline"):
                    component = load_component(ROOT / f"azure/{parser}-{kind}.yml")
                    self.assertTrue(component._validate().passed)
                    self.assertEqual(component.outputs["results"].mode, "rw_mount")
                deployment = load_pipeline_component_batch_deployment(ROOT / f"azure/{parser}-deployment.yml")
                self.assertEqual(deployment.type, "pipeline")
                self.assertEqual(deployment.settings["default_compute"], "pdf-parsers-a100")
                self.assertTrue(deployment.settings["force_rerun"])

    def test_version_references_agree_and_models_stay_internal_to_pipeline(self):
        for parser in ("mineru", "paddle"):
            environment = load_environment(ROOT / f"environments/{parser}/environment.yml")
            model = load_model(ROOT / f"models/{parser}.yml")
            command = load_component(ROOT / f"azure/{parser}-command.yml")
            pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
            deployment = load_pipeline_component_batch_deployment(ROOT / f"azure/{parser}-deployment.yml")
            self.assertEqual(command.environment.removeprefix("azureml:"), f"{environment.name}:{environment.version}")
            self.assertEqual(command.inputs["models"].type, "custom_model")
            self.assertEqual(pipeline.jobs["parse"].inputs["models"].path.removeprefix("azureml:"), f"{model.name}:{model.version}")
            self.assertEqual(pipeline.jobs["parse"].inputs["models"].mode, "download")
            self.assertEqual(set(pipeline.inputs), {"pdfs"})
            self.assertEqual(deployment.component.removeprefix("azureml:"), f"{pipeline.name}:{pipeline.version}")


if __name__ == "__main__":
    unittest.main()
