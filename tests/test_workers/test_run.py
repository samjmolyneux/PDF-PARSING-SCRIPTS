"""Check batch-worker behavior shared by Paddle and MinerU."""

import contextlib
import io
import os
from unittest.mock import patch

import pytest

from workers import run as worker

pytestmark = pytest.mark.usefixtures("worker_environment")


@pytest.mark.parametrize("parser", ["mineru", "paddle"])
def test_empty_batch_is_failure(pdf_folder, output_folder, read_report, parser):
    with (
        patch.object(worker, "start_paddle_server") as start,
        patch.object(worker.subprocess, "run") as run,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert worker.process_batch(parser, pdf_folder, output_folder) == 1
    start.assert_not_called()
    run.assert_not_called()
    assert "No PDFs" in read_report()["fatal_error"]


@pytest.mark.parametrize(
    "parser, operation",
    [("mineru", "subprocess.run"), ("paddle", "start_paddle_server")],
)
def test_worker_preserves_model_locations_from_the_image(
    tmp_path, pdf_folder, make_pdf, output_folder, parser, operation
):
    make_pdf("a.pdf")
    config = tmp_path / "image-mineru.json"
    contents = '{"models-dir": {"pipeline": "/image/pipeline", "vlm": "/image/vlm"}}'
    config.write_text(contents)
    variables = {
        "MINERU_TOOLS_CONFIG_JSON": str(config),
        "MINERU_MODEL_SOURCE": "local",
        "PADDLE_PDX_CACHE_HOME": "/image/paddle",
    }
    observed = []

    def record_environment(*args, **kwargs):
        observed.append(
            (
                {key: os.environ.get(key) for key in variables},
                config.read_text(),
                os.environ.get("PYTHONNOUSERSITE"),
            )
        )

    with (
        patch.dict(os.environ, variables),
        patch(f"workers.run.{operation}", side_effect=record_environment) as launch,
        patch.object(worker, "parse_paddle_pdfs", return_value=iter([])),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        worker.process_batch(parser, pdf_folder, output_folder)
    launch.assert_called_once()
    # Assert outside the worker, which deliberately catches parser exceptions.
    assert observed == [(variables, contents, "1")]
    assert not (output_folder / "mineru-runtime.json").exists()
