"""Local fault-injection tests. No Azure calls, model downloads or GPU required."""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf_parsers import client
from workers import run as worker


class TestWorkflow:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.root = tmp_path
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "results"
        self.models = self.root / "models"
        with patch.dict(os.environ):
            yield

    def pdf(self, relative):
        file = self.inputs / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test input")
        return file

    def report(self):
        return json.loads((self.output / "report.json").read_text())

    @pytest.mark.parametrize(
        "exported_names, exit_code, statuses",
        [
            pytest.param(["a", "b"], 0, ["succeeded", "succeeded"], id="complete"),
            pytest.param(["a"], 0, ["succeeded", "unconfirmed"], id="skipped-pdf"),
            pytest.param(["a"], 1, ["unconfirmed", "unconfirmed"], id="native-error"),
        ],
    )
    def test_mineru_folder_batch_retains_exports_and_confirms_only_success(
        self, exported_names, exit_code, statuses
    ):
        self.pdf("a.pdf")
        self.pdf("b.PDF")

        def export(command, **kwargs):
            for name in exported_names:
                target = self.output / "documents" / name
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
            code = worker.process_batch("mineru", self.inputs, self.output)
        assert code == int("unconfirmed" in statuses)
        start.assert_not_called()
        run.assert_called_once()
        command = run.call_args.args[0]
        assert command[command.index("-p") + 1] == str(self.inputs.resolve())
        assert "--api-url" not in command
        assert "timeout" not in run.call_args.kwargs
        assert run.call_args.kwargs["check"]
        assert [d["status"] for d in self.report()["documents"]] == statuses
        assert self.report()["finished"] is (exit_code == 0)
        assert all(
            (self.output / "documents" / name / "result.md").is_file()
            for name in exported_names
        )
        if exit_code:
            assert "native error: b.pdf" in (self.output / "parser.log").read_text()
        elif "unconfirmed" in statuses:
            assert (
                "Expected exports not found" in self.report()["documents"][1]["error"]
            )

    def test_paddle_batch_failure_preserves_confirmed_documents_without_retry(self):
        a, b = self.pdf("a.pdf"), self.pdf("b.pdf")

        def parse(pdfs, output):
            assert pdfs == [a.resolve(), b.resolve()]
            yield a
            assert self.report()["documents"][0]["status"] == "succeeded"
            raise RuntimeError("native pipeline failed")

        with (
            patch.object(worker, "start_paddle_server", return_value=object()) as start,
            patch.object(worker, "stop_paddle_server") as stop,
            patch.object(worker, "parse_paddle_pdfs", side_effect=parse) as parse_call,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = worker.process_batch("paddle", self.inputs, self.output)
        assert code == 1
        start.assert_called_once()
        stop.assert_called_once()
        parse_call.assert_called_once()
        assert [d["status"] for d in self.report()["documents"]] == [
            "succeeded",
            "unconfirmed",
        ]
        assert "native pipeline failed" in (self.output / "parser.log").read_text()

    def test_paddle_returning_no_results_cannot_report_success(self):
        self.pdf("a.pdf")
        with (
            patch.object(worker, "start_paddle_server", return_value=object()),
            patch.object(worker, "stop_paddle_server"),
            patch.object(worker, "parse_paddle_pdfs", return_value=iter([])),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert worker.process_batch("paddle", self.inputs, self.output) == 1
        assert "did not confirm" in self.report()["fatal_error"]

    def test_server_start_failure_leaves_every_pdf_unconfirmed(self):
        self.pdf("a.pdf")
        self.pdf("b.pdf")
        with (
            patch.object(
                worker, "start_paddle_server", side_effect=RuntimeError("missing model")
            ) as start,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = worker.process_batch("paddle", self.inputs, self.output)
        assert code == 1
        assert start.call_count == 1
        assert not self.report()["finished"]
        assert "missing model" in self.report()["fatal_error"]
        assert len(self.report()["documents"]) == 2

    @pytest.mark.parametrize("parser", ["mineru", "paddle"])
    def test_empty_batch_is_failure(self, parser):
        with (
            patch.object(worker, "start_paddle_server") as start,
            patch.object(worker.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert worker.process_batch(parser, self.inputs, self.output) == 1
        start.assert_not_called()
        run.assert_not_called()
        assert "No PDFs" in self.report()["fatal_error"]

    @pytest.mark.parametrize(
        "parser, operation",
        [("mineru", "subprocess.run"), ("paddle", "start_paddle_server")],
    )
    def test_worker_preserves_model_locations_from_the_image(self, parser, operation):
        self.pdf("a.pdf")
        config = self.root / "image-mineru.json"
        contents = (
            '{"models-dir": {"pipeline": "/image/pipeline", "vlm": "/image/vlm"}}'
        )
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
            worker.process_batch(parser, self.inputs, self.output)
        launch.assert_called_once()
        # Assert outside the worker, which deliberately catches parser exceptions.
        assert observed == [(variables, contents, "1")]
        assert not (self.output / "mineru-runtime.json").exists()

    def test_paddle_server_launch_preserves_logs_and_process_group_until_ready(self):
        cli = self.root / "server cli.py"
        cli.touch()
        os.environ.update(
            PADDLE_SERVER_PYTHON=sys.executable,
            PADDLE_SERVER_CLI=str(cli),
            PADDLE_PDX_CACHE_HOME=str(self.models),
        )
        process = MagicMock()
        process.poll.return_value = None
        log = io.BytesIO()
        with (
            patch.object(worker.subprocess, "Popen", return_value=process) as launch,
            patch.object(worker.urllib.request, "urlopen") as health,
        ):
            health.return_value.__enter__.return_value.status = 200
            assert worker.start_paddle_server(log) is process
        command = launch.call_args.args[0]
        assert command[0] == "bash"
        assert Path(command[1]) == Path(worker.__file__).with_name(
            "start_paddle_server.sh"
        )
        assert Path(command[1]).is_file()
        assert command[2:5] == [sys.executable, str(cli), "genai_server"]
        # Inherit Conda's activation state so the shell can undo it correctly.
        assert "env" not in launch.call_args.kwargs
        assert launch.call_args.kwargs["stdout"] is log
        assert launch.call_args.kwargs["stderr"] == subprocess.STDOUT
        assert launch.call_args.kwargs["start_new_session"]
        health.assert_called_once_with("http://127.0.0.1:8118/health", timeout=5)

    def test_server_launcher_failure_stops_immediately_and_cleans_up(self):
        process = MagicMock()
        process.poll.return_value = 17
        process.returncode = 17
        with (
            patch.object(
                worker,
                "paddle_server_command",
                return_value=(["bash", "launcher.sh"], "health"),
            ),
            patch.object(worker.subprocess, "Popen", return_value=process),
            patch.object(worker, "stop_paddle_server") as stop,
            patch.object(worker.urllib.request, "urlopen") as health,
        ):
            with pytest.raises(RuntimeError, match="exited \\(17\\).*server.log"):
                worker.start_paddle_server(io.BytesIO())
        health.assert_not_called()
        stop.assert_called_once_with(process)

    def test_flat_pdf_folder_accepts_spaces_and_uppercase_extension(self):
        files = [self.pdf("a.pdf"), self.pdf("space name.PDF")]
        folder, found = client.pdf_files(self.inputs)
        assert found == [p.resolve() for p in files]
        assert folder == self.inputs.resolve()

    @pytest.mark.parametrize("extra", ["nested/b.pdf", "picture.png", "notes.txt"])
    def test_client_ignores_other_entries_but_worker_requires_a_pdf_only_batch(
        self, extra
    ):
        pdf = self.pdf("a.pdf")
        self.pdf(extra)
        assert client.pdf_files(self.inputs)[1] == [pdf.resolve()]
        with (
            patch.object(worker.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            assert worker.process_batch("mineru", self.inputs, self.output) == 1
        run.assert_not_called()

    @pytest.mark.parametrize("target", ["client", "mineru", "paddle"])
    def test_case_insensitive_name_collisions_are_rejected(self, target):
        entries = [self.pdf("same.pdf"), self.pdf("SAME.pdf")]
        # Simulate a Linux directory listing even on a case-insensitive Mac.
        # Each scan needs a fresh iterator: the worker reads the folder twice.
        with patch.object(Path, "iterdir", side_effect=lambda: iter(entries)):
            if target == "client":
                with pytest.raises(ValueError, match="distinct"):
                    client.pdf_files(self.inputs)
                return
            with (
                patch.object(worker.subprocess, "run") as run,
                patch.object(worker, "start_paddle_server") as start,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = worker.process_batch(target, self.inputs, self.output)

        assert code == 1
        run.assert_not_called()
        start.assert_not_called()
        report = self.report()
        assert report["parser"] == target
        assert report["total"] == 2
        assert report["finished"] is False
        assert "PDF filenames must be distinct" in report["fatal_error"]
        assert {item["input"] for item in report["documents"]} == {
            pdf.name for pdf in entries
        }
        assert [item["status"] for item in report["documents"]] == [
            "unconfirmed",
            "unconfirmed",
        ]
        assert (
            "PDF filenames must be distinct" in (self.output / "parser.log").read_text()
        )

    def test_linked_pdf_is_rejected(self):
        outside = self.root / "outside.pdf"
        outside.write_text("external")
        try:
            (self.inputs / "linked.pdf").symlink_to(outside)
        except OSError:
            pytest.skip("Symlinks unavailable")
        with pytest.raises(ValueError, match="linked PDFs"):
            client.pdf_files(self.inputs)

    @pytest.mark.parametrize(
        "status, changes, succeeds",
        [
            pytest.param("Completed", {}, True, id="complete"),
            pytest.param("Failed", {}, False, id="azure-failed"),
            pytest.param("Canceled", {}, False, id="azure-canceled"),
            pytest.param("NotResponding", {}, False, id="azure-not-responding"),
            pytest.param("Completed", {"finished": False}, False, id="unfinished"),
            pytest.param("Completed", {"parser": "paddle"}, False, id="wrong-parser"),
            pytest.param("Completed", {"documents": []}, False, id="no-documents"),
            pytest.param(
                "Completed", {"fatal_error": "disk full"}, False, id="fatal-error"
            ),
            pytest.param("Completed", {"total": 2}, False, id="wrong-total"),
            pytest.param(
                "Completed",
                {"documents": [{"input": "different.pdf", "status": "succeeded"}]},
                False,
                id="wrong-document",
            ),
            pytest.param(
                "Completed",
                {"documents": [{"input": "a.pdf", "status": "failed"}]},
                False,
                id="failed-document",
            ),
        ],
    )
    def test_success_requires_azure_completion_and_every_submitted_pdf(
        self, status, changes, succeeds
    ):
        receipt = {"parser": "mineru", "pdf_count": 1, "pdfs": ["a.pdf"]}
        report = {
            "parser": "mineru",
            "total": 1,
            "finished": True,
            "documents": [{"input": "a.pdf", "status": "succeeded"}],
            **changes,
        }
        assert client.report_success(report, receipt, status) is succeeds
