"""Check MinerU folder invocation, export confirmation and batch errors."""

import contextlib
import io
import subprocess
from unittest.mock import patch

import pytest

from workers import run as worker

pytestmark = pytest.mark.usefixtures("worker_environment")


@pytest.mark.parametrize(
    "exported_names, exit_code, statuses",
    [
        pytest.param(["a", "b"], 0, ["succeeded", "succeeded"], id="complete"),
        pytest.param(["a"], 0, ["succeeded", "unconfirmed"], id="skipped-pdf"),
        pytest.param(["a"], 1, ["unconfirmed", "unconfirmed"], id="native-error"),
    ],
)
def test_mineru_folder_batch_retains_exports_and_confirms_only_success(
    pdf_folder,
    make_pdf,
    output_folder,
    read_report,
    exported_names,
    exit_code,
    statuses,
):
    make_pdf("a.pdf")
    make_pdf("b.PDF")

    def export(command, **kwargs):
        for name in exported_names:
            target = output_folder / "documents" / name
            target.mkdir()
            (target / "result.md").write_text("export")
            (target / "result.json").write_text("{}")
        if exit_code:
            kwargs["stdout"].write("native error: b.pdf could not be parsed\n")
            raise subprocess.CalledProcessError(exit_code, command)

    with (
        patch.object(worker, "start_paddle_server") as start,
        patch.object(worker.subprocess, "run", side_effect=export) as run,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        code = worker.process_batch("mineru", pdf_folder, output_folder)
    assert code == int("unconfirmed" in statuses)
    start.assert_not_called()
    run.assert_called_once()
    command = run.call_args.args[0]
    assert command[command.index("-p") + 1] == str(pdf_folder.resolve())
    assert "--api-url" not in command
    assert "timeout" not in run.call_args.kwargs
    assert run.call_args.kwargs["check"]
    assert [d["status"] for d in read_report()["documents"]] == statuses
    assert read_report()["finished"] is (exit_code == 0)
    assert all(
        (output_folder / "documents" / name / "result.md").is_file()
        for name in exported_names
    )
    if exit_code:
        assert "native error: b.pdf" in (output_folder / "parser.log").read_text()
    elif "unconfirmed" in statuses:
        assert "Expected exports not found" in read_report()["documents"][1]["error"]
