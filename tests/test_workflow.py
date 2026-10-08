"""Local fault-injection tests. No Azure calls, model downloads or GPU required."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pdf_parsers import client
from workers import run as worker


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "results"
        self.models = self.root / "models"
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)

    def pdf(self, relative):
        file = self.inputs / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"test input")
        return file

    def report(self):
        return json.loads((self.output / "report.json").read_text())

    def test_mineru_uses_one_native_folder_call_and_no_external_server(self):
        self.pdf("a.pdf")
        self.pdf("b.PDF")
        def export(command, **kwargs):
            for name in ("a", "b"):
                target = self.output / "documents" / name
                target.mkdir()
                (target / "result.md").write_text("export")
                (target / "result.json").write_text("{}")
        with patch.object(worker, "start_server") as start, \
             patch.object(worker.subprocess, "run", side_effect=export) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            code = worker.process_batch("mineru", self.inputs, self.output)
        self.assertEqual(code, 0)
        start.assert_not_called()
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-p") + 1], str(self.inputs.resolve()))
        self.assertNotIn("--api-url", command)
        self.assertNotIn("timeout", run.call_args.kwargs)
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertEqual([d["status"] for d in self.report()["documents"]], ["succeeded", "succeeded"])

    def test_mineru_zero_exit_cannot_hide_a_silently_skipped_pdf(self):
        self.pdf("a.pdf")
        self.pdf("skipped.pdf")
        def export(command, **kwargs):
            target = self.output / "documents/a"
            target.mkdir()
            (target / "result.md").write_text("export")
            (target / "result.json").write_text("{}")
        with patch.object(worker.subprocess, "run", side_effect=export), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.process_batch("mineru", self.inputs, self.output), 1)
        self.assertEqual([d["status"] for d in self.report()["documents"]], ["succeeded", "unconfirmed"])
        self.assertIn("Expected exports not found", self.report()["documents"][1]["error"])

    def test_mineru_failed_batch_retains_exports_but_never_confirms_partial_files(self):
        self.pdf("a.pdf")
        self.pdf("b.pdf")
        def fail(command, **kwargs):
            exported = self.output / "documents/a"
            exported.mkdir()
            (exported / "a.md").write_text("possibly partial")
            kwargs["stdout"].write("native error: b.pdf could not be parsed\n")
            raise subprocess.CalledProcessError(1, command)
        with patch.object(worker.subprocess, "run", side_effect=fail) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            code = worker.process_batch("mineru", self.inputs, self.output)
        self.assertEqual(code, 1)
        run.assert_called_once()
        self.assertEqual([d["status"] for d in self.report()["documents"]], ["unconfirmed", "unconfirmed"])
        self.assertFalse(self.report()["finished"])
        self.assertTrue((self.output / "documents/a/a.md").is_file())
        self.assertIn("native error: b.pdf", (self.output / "parser.log").read_text())

    def test_paddle_batch_failure_preserves_confirmed_documents_without_retry(self):
        a, b = self.pdf("a.pdf"), self.pdf("b.pdf")
        def parse(pdfs, output):
            self.assertEqual(pdfs, [a.resolve(), b.resolve()])
            yield a
            self.assertEqual(self.report()["documents"][0]["status"], "succeeded")
            raise RuntimeError("native pipeline failed")
        with patch.object(worker, "start_server", return_value=object()) as start, \
             patch.object(worker, "stop_process") as stop, \
             patch.object(worker, "parse_pdfs", side_effect=parse) as parse_call, \
             contextlib.redirect_stdout(io.StringIO()):
            code = worker.process_batch("paddle", self.inputs, self.output)
        self.assertEqual(code, 1)
        start.assert_called_once()
        stop.assert_called_once()
        parse_call.assert_called_once()
        self.assertEqual([d["status"] for d in self.report()["documents"]], ["succeeded", "unconfirmed"])
        self.assertIn("native pipeline failed", (self.output / "parser.log").read_text())

    def test_paddle_returning_no_results_cannot_report_success(self):
        self.pdf("a.pdf")
        with patch.object(worker, "start_server", return_value=object()), \
             patch.object(worker, "stop_process"), \
             patch.object(worker, "parse_pdfs", return_value=iter([])), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.process_batch("paddle", self.inputs, self.output), 1)
        self.assertIn("did not confirm", self.report()["fatal_error"])

    def test_server_start_failure_leaves_every_pdf_unconfirmed(self):
        self.pdf("a.pdf")
        self.pdf("b.pdf")
        with patch.object(worker, "start_server", side_effect=RuntimeError("missing model")) as start, \
             contextlib.redirect_stdout(io.StringIO()):
            code = worker.process_batch("paddle", self.inputs, self.output)
        self.assertEqual(code, 1)
        self.assertEqual(start.call_count, 1)
        self.assertFalse(self.report()["finished"])
        self.assertIn("missing model", self.report()["fatal_error"])
        self.assertEqual(len(self.report()["documents"]), 2)

    def test_empty_batch_is_failure(self):
        with patch.object(worker, "start_server") as start, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(worker.process_batch("mineru", self.inputs, self.output), 1)
        start.assert_not_called()
        self.assertIn("No PDFs", self.report()["fatal_error"])

    def test_worker_preserves_model_locations_from_the_image(self):
        self.pdf("a.pdf")
        config = self.root / "image-mineru.json"
        contents = '{"models-dir": {"pipeline": "/image/pipeline", "vlm": "/image/vlm"}}'
        config.write_text(contents)
        variables = {
            "MINERU_TOOLS_CONFIG_JSON": str(config),
            "MINERU_MODEL_SOURCE": "local",
            "PADDLE_PDX_CACHE_HOME": "/image/paddle",
        }

        observed = []

        def record_environment(*args, **kwargs):
            observed.append(({key: os.environ.get(key) for key in variables},
                             config.read_text(), os.environ.get("PYTHONNOUSERSITE")))

        with patch.dict(os.environ, variables), \
             patch.object(worker.subprocess, "run", side_effect=record_environment) as mineru, \
             patch.object(worker, "start_server", side_effect=record_environment) as paddle, \
             patch.object(worker, "parse_pdfs", return_value=iter([])), \
             contextlib.redirect_stdout(io.StringIO()):
            for parser in ("mineru", "paddle"):
                worker.process_batch(parser, self.inputs, self.output)
        mineru.assert_called_once()
        paddle.assert_called_once()
        # Assert outside the worker, which deliberately catches parser exceptions.
        self.assertEqual(observed, [(variables, contents, "1"), (variables, contents, "1")])
        self.assertFalse((self.output / "mineru-runtime.json").exists())

    def test_paddle_server_launch_preserves_logs_and_process_group_until_ready(self):
        cli = self.root / "server cli.py"
        cli.touch()
        os.environ.update(PADDLE_SERVER_PYTHON=sys.executable, PADDLE_SERVER_CLI=str(cli),
                          PADDLE_PDX_CACHE_HOME=str(self.models))
        process = MagicMock()
        process.poll.return_value = None
        log = io.BytesIO()
        with patch.object(worker.subprocess, "Popen", return_value=process) as launch, \
             patch.object(worker.urllib.request, "urlopen") as health:
            health.return_value.__enter__.return_value.status = 200
            self.assertIs(worker.start_server(log), process)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], "bash")
        self.assertEqual(Path(command[1]), Path(worker.__file__).with_name("start_paddle_server.sh"))
        self.assertTrue(Path(command[1]).is_file())
        self.assertEqual(command[2:5], [sys.executable, str(cli), "genai_server"])
        # Inherit Conda's activation state so the shell can undo it correctly.
        self.assertNotIn("env", launch.call_args.kwargs)
        self.assertIs(launch.call_args.kwargs["stdout"], log)
        self.assertEqual(launch.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        health.assert_called_once_with("http://127.0.0.1:8118/health", timeout=5)

    def test_server_launcher_failure_stops_immediately_and_cleans_up(self):
        process = MagicMock()
        process.poll.return_value = 17
        process.returncode = 17
        with patch.object(worker, "server_command", return_value=(["bash", "launcher.sh"], "health")), \
             patch.object(worker.subprocess, "Popen", return_value=process), \
             patch.object(worker, "stop_process") as stop, \
             patch.object(worker.urllib.request, "urlopen") as health:
            with self.assertRaisesRegex(RuntimeError, r"exited \(17\).*server.log"):
                worker.start_server(io.BytesIO())
        health.assert_not_called()
        stop.assert_called_once_with(process)

    def test_flat_pdf_folder_accepts_spaces_and_uppercase_extension(self):
        files = [self.pdf("a.pdf"), self.pdf("space name.PDF")]
        folder, found = client.pdf_files(self.inputs)
        self.assertEqual(found, [p.resolve() for p in files])
        self.assertEqual(folder, self.inputs.resolve())

    def test_client_ignores_other_entries_but_worker_requires_a_pdf_only_batch(self):
        pdf = self.pdf("a.pdf")
        for extra in ("nested/b.pdf", "picture.png", "notes.txt"):
            with self.subTest(extra=extra):
                path = self.pdf(extra)
                try:
                    self.assertEqual(client.pdf_files(self.inputs)[1], [pdf.resolve()])
                    with patch.object(worker.subprocess, "run") as run, \
                         contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(worker.process_batch("mineru", self.inputs, self.output), 1)
                    run.assert_not_called()
                finally:
                    path.unlink()
                    if path.parent != self.inputs:
                        path.parent.rmdir()

    def test_case_insensitive_name_collisions_are_rejected(self):
        self.pdf("same.pdf")
        self.pdf("same.PDF")
        # Simulate a Linux directory listing even on a case-insensitive Mac.
        entries = [self.inputs / "same.pdf", self.inputs / "same.PDF"]
        with patch.object(Path, "iterdir", return_value=iter(entries)), \
             self.assertRaisesRegex(ValueError, "distinct"):
            client.pdf_files(self.inputs)

    def test_linked_pdf_is_rejected(self):
        outside = self.root / "outside.pdf"
        outside.write_text("external")
        try:
            (self.inputs / "linked.pdf").symlink_to(outside)
        except OSError:
            self.skipTest("Symlinks unavailable")
        with self.assertRaisesRegex(ValueError, "linked PDFs"):
            client.pdf_files(self.inputs)

    def test_success_requires_azure_completion_and_every_submitted_pdf(self):
        receipt = {"parser": "mineru", "pdf_count": 1, "pdfs": ["a.pdf"]}
        report = {"parser": "mineru", "total": 1, "finished": True,
                  "documents": [{"input": "a.pdf", "status": "succeeded"}]}
        self.assertTrue(client.report_success(report, receipt, "Completed"))
        for status in ("Failed", "Canceled", "NotResponding"):
            self.assertFalse(client.report_success(report, receipt, status))
        for changes in ({"finished": False}, {"parser": "paddle"}, {"documents": []},
                        {"fatal_error": "disk full"}, {"total": 2},
                        {"documents": [{"input": "different.pdf", "status": "succeeded"}]},
                        {"documents": [{"input": "a.pdf", "status": "failed"}]}):
            with self.subTest(changes=changes):
                self.assertFalse(client.report_success({**report, **changes}, receipt, "Completed"))


if __name__ == "__main__":
    unittest.main()
