"""Load definitions through Azure ML's schemas without contacting Azure."""

from pathlib import Path

import pytest
from azure.ai.ml import (
    load_batch_endpoint,
    load_component,
    load_compute,
    load_environment,
)
from azure.ai.ml.entities._load_functions import (
    load_pipeline_component_batch_deployment,
)

from pdf_parsers.client import DEPLOYMENT_NAMES, MAX_ACTIVE_JOBS

ROOT = Path(__file__).resolve().parents[1]


def test_compute_loads_with_scale_to_zero():
    compute = load_compute(ROOT / "azure/compute.yml")
    assert (compute.min_instances, compute.max_instances) == (0, 2)
    assert MAX_ACTIVE_JOBS == compute.max_instances


def test_endpoint_loads_with_aad_authentication():
    assert load_batch_endpoint(ROOT / "azure/endpoint.yml").auth_mode == "aad_token"


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_environment_build_context_resolves(parser):
    environment = load_environment(ROOT / f"environments/{parser}/environment.yml")
    environment.validate()
    # Azure resolves the upload folder relative to environment.yml.
    context = Path(environment.path).resolve()
    assert context == ROOT / f"environments/{parser}"
    assert (context / environment.build.dockerfile_path).is_file()
    if parser == "paddle":
        assert (context / "conda.yml").is_file()
    assert environment.image is None
    assert environment.conda_file is None


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
@pytest.mark.parametrize("kind", ["command", "pipeline"])
def test_component_validates(parser, kind):
    component = load_component(ROOT / f"azure/{parser}-{kind}.yml")
    assert component._validate().passed
    assert component.outputs["results"].mode == "rw_mount"


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_pipeline_deployment_validates(parser):
    deployment = load_pipeline_component_batch_deployment(
        ROOT / f"azure/{parser}-deployment.yml"
    )
    assert deployment.type == "pipeline"
    assert deployment.settings["default_compute"] == "pdf-parsers-a100"
    assert deployment.settings["force_rerun"]


@pytest.mark.parametrize(
    "parser, variables",
    [
        (
            "mineru",
            {
                "MINERU_MODEL_SOURCE": "local",
                "MINERU_TASK_RESULT_TIMEOUT_SECONDS": "86400",
            },
        ),
        (
            "paddle",
            {
                "PADDLE_SERVER_PYTHON": "/usr/local/bin/python",
                "PADDLE_SERVER_CLI": "/usr/local/bin/paddleocr",
            },
        ),
    ],
)
def test_runtime_variables_survive_pipeline_serialization(parser, variables):
    # Schema validation alone didn't catch variables being lost when they
    # were placed on the reusable component instead of the pipeline job.
    pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
    expected = {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", **variables}
    assert pipeline.jobs["parse"].environment_variables == expected
    registered = pipeline._to_rest_object().properties.component_spec
    assert registered["jobs"]["parse"]["environment_variables"] == expected


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_version_references_agree_and_pdfs_are_the_only_job_input(parser):
    environment = load_environment(ROOT / f"environments/{parser}/environment.yml")
    command = load_component(ROOT / f"azure/{parser}-command.yml")
    pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
    deployment = load_pipeline_component_batch_deployment(
        ROOT / f"azure/{parser}-deployment.yml"
    )
    assert (
        command.environment.removeprefix("azureml:")
        == f"{environment.name}:{environment.version}"
    )
    assert set(command.inputs) == {"pdfs"}
    assert set(pipeline.jobs["parse"].inputs) == {"pdfs"}
    assert set(pipeline.inputs) == {"pdfs"}
    assert (
        deployment.component.removeprefix("azureml:")
        == f"{pipeline.name}:{pipeline.version}"
    )
    assert deployment.name == DEPLOYMENT_NAMES[parser]
