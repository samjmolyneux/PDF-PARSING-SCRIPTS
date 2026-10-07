"""Exercise submission/resumption with the real SDK types and mocked services."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from azure.ai.ml import MLClient
from pdf_parsers import mineru, paddle
from pdf_parsers.client import download_results


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        previous_directory = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous_directory)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        (self.inputs / "a.pdf").write_bytes(b"test PDF")
        config = {"subscription_id": "sub", "resource_group": "rg", "workspace": "ws",
                  "tenant_id": "tenant", "endpoint": "endpoint"}
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        self.ml = MagicMock()
        self.ml.batch_endpoints.list_jobs.return_value = []
        self.outputs = {}
        self.ml.jobs.list.return_value = []
        self.events = []
        self.expected_upload = {"a.pdf": b"test PDF"}
        self.upload_directories = []

        def invoke(**kwargs):
            # A receipt must be durable before the HTTP request might succeed.
            receipt = json.loads(self.receipt().read_text())
            self.assertEqual(receipt["state"], "submitting")
            self.assertEqual(kwargs["job_name"], receipt["job_name"])
            expected_deployment = "paddle-vl" if receipt["parser"] == "paddle" else "mineru"
            self.assertEqual(kwargs["deployment_name"], expected_deployment)
            staged = Path(kwargs["inputs"]["pdfs"].path)
            self.upload_directories.append(staged)
            self.assertNotEqual(staged, self.inputs)
            self.assertEqual({p.name: p.read_bytes() for p in staged.iterdir()}, self.expected_upload)
            self.assertEqual(kwargs["inputs"]["pdfs"].type, "uri_folder")
            self.assertNotIn("outputs", kwargs)
            self.events.append("invoked")
            return SimpleNamespace(name=kwargs["job_name"])

        self.ml.batch_endpoints.invoke.side_effect = invoke
        self.ml.jobs.get.return_value = SimpleNamespace(status="Completed")
        self.ml.jobs.download.side_effect = self.download

    def download(self, job_name, *, download_path, output_name):
        self.assertEqual(output_name, "results")
        for name, content in self.outputs.items():
            file = Path(download_path) / "named-outputs/results" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)

    def receipt(self):
        return next((self.root / "runs").glob("*.json"))

    def call(self, *args, parser="mineru"):
        with patch.object(sys, "argv", [f"run-{parser}", *map(str, args)]), \
             patch("azure.ai.ml.MLClient", return_value=self.ml), \
             patch("azure.identity.InteractiveBrowserCredential"), \
             contextlib.redirect_stdout(io.StringIO()) as output, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            code = {"mineru": mineru, "paddle": paddle}[parser].main()
        return code, output.getvalue() + errors.getvalue()

    def submit(self):
        code, message = self.call(self.inputs, "--config", self.config, "--no-wait")
        self.assertEqual(code, 0, message)
        self.assertIn("Submitted:", message)
        self.assertIn("Later: run-mineru --resume", message)
        receipt = json.loads(self.receipt().read_text())
        self.assertEqual(receipt["state"], "submitted")
        return receipt

    def cloud_report(self, receipt, **overrides):
        report = {"parser": "mineru", "total": 1, "finished": True, "fatal_error": None,
                  "documents": [{"input": "a.pdf", "status": "succeeded"}]}
        report.update(overrides)
        self.outputs["report.json"] = json.dumps(report).encode()

    def test_submit_then_resume_downloads_without_uploading_or_invoking_again(self):
        receipt = self.submit()
        self.assertEqual(receipt["pdfs"], ["a.pdf"])
        self.cloud_report(receipt)
        self.outputs["documents/a/result.md"] = b"all exports"
        # Resuming must work even while the endpoint has an unfinished job.
        self.ml.batch_endpoints.list_jobs.side_effect = AssertionError("Resume must skip the busy check")
        destination = self.root / "downloaded"
        code, message = self.call("--resume", self.receipt(), "--output", destination)
        self.assertEqual(code, 0, message)
        self.assertEqual(self.events, ["invoked"])
        self.assertEqual((destination / "documents/a/result.md").read_bytes(), b"all exports")
        self.assertIn("1/1 PDFs succeeded", message)
        self.ml.batch_endpoints.list_jobs.assert_called_once_with(endpoint_name="endpoint")
        self.assertEqual(len(self.upload_directories), 1)
        self.assertFalse(self.upload_directories[0].exists())

    def test_mixed_folders_upload_only_direct_pdfs_for_both_parsers(self):
        (self.inputs / "Second PDF.PDF").write_bytes(b"second PDF")
        self.expected_upload["Second PDF.PDF"] = b"second PDF"
        for name in ("notes.txt", "photo.png", ".DS_Store", ".amlignore", ".gitignore"):
            (self.inputs / name).write_text("*.pdf")
        nested = self.inputs / "nested.pdf"
        nested.mkdir()
        (nested / "nested.pdf").write_bytes(b"not selected")
        for parser in ("mineru", "paddle"):
            with self.subTest(parser=parser):
                code, message = self.call(self.inputs, "--no-wait", parser=parser)
                self.assertEqual(code, 0, message)
                expected_deployment = "paddle-vl" if parser == "paddle" else "mineru"
                self.assertEqual(self.ml.batch_endpoints.invoke.call_args.kwargs["deployment_name"], expected_deployment)
                self.assertFalse(self.upload_directories[-1].exists())
                receipt_path = self.receipt()
                receipt = json.loads(receipt_path.read_text())
                self.assertEqual(set(receipt["pdfs"]), set(self.expected_upload))
                self.assertEqual(receipt["pdf_count"], 2)
                receipt_path.unlink()
        self.assertEqual((self.inputs / "a.pdf").read_bytes(), b"test PDF")
        self.assertEqual((self.inputs / "notes.txt").read_text(), "*.pdf")
        self.assertTrue((nested / "nested.pdf").is_file())

    def test_current_directory_excludes_config_receipts_and_nested_pdfs(self):
        (self.root / "a.pdf").write_bytes(b"test PDF")
        code, message = self.call(".", "--no-wait")
        self.assertEqual(code, 0, message)
        self.assertEqual(json.loads(self.receipt().read_text())["pdfs"], ["a.pdf"])
        self.assertFalse(self.upload_directories[0].exists())

    def test_no_direct_pdfs_does_not_contact_azure_or_submit(self):
        (self.inputs / "a.pdf").unlink()
        nested = self.inputs / "nested"
        nested.mkdir()
        (nested / "a.pdf").write_bytes(b"not selected")
        (self.inputs / "notes.txt").write_text("not a PDF")
        code, message = self.call(self.inputs, "--no-wait")
        self.assertEqual(code, 1)
        self.assertIn("No PDFs found", message)
        self.ml.batch_endpoints.list_jobs.assert_not_called()
        self.ml.batch_endpoints.invoke.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_copy_failure_cleans_staging_and_never_uploads(self):
        destinations = []
        def fail_copy(source, destination):
            destinations.append(destination)
            destination.write_bytes(b"partial copy")
            raise OSError("disk full")
        with patch("pdf_parsers.client.shutil.copyfile", side_effect=fail_copy):
            code, message = self.call(self.inputs, "--no-wait")
        self.assertEqual(code, 1)
        self.assertIn("disk full", message)
        self.assertFalse(destinations[0].parent.exists())
        self.ml.batch_endpoints.invoke.assert_not_called()
        self.assertEqual((self.inputs / "a.pdf").read_bytes(), b"test PDF")

    def test_busy_endpoint_blocks_both_parsers_before_upload_or_receipt(self):
        for parser, other in (("mineru", "paddle"), ("paddle", "mineru")):
            for status in ("NotStarted", "Queued", "Preparing", "Provisioning", "Starting", "Running",
                           "Finalizing", "CancelRequested", "Paused", None):
                with self.subTest(parser=parser, status=status):
                    self.ml.batch_endpoints.list_jobs.return_value = [
                        SimpleNamespace(name="old-job", status="Completed"),
                        SimpleNamespace(name=f"{other}-colleague-job", status=status),
                    ]
                    code, message = self.call(self.inputs, "--no-wait", parser=parser)
                    self.assertEqual(code, 1)
                    self.assertIn("Parser endpoint is busy", message)
                    self.assertIn(f"{other}-colleague-job", message)
                    self.assertIn("No PDFs were uploaded", message)
                    self.assertFalse((self.root / "runs").exists())
                    self.ml.batch_endpoints.invoke.assert_not_called()

    def test_finished_jobs_do_not_block_submission(self):
        self.ml.batch_endpoints.list_jobs.return_value = [
            SimpleNamespace(name=f"old-{status}", status=status)
            for status in ("Completed", "Failed", "Canceled", "Cancelled", "NotResponding")
        ]
        self.submit()
        self.ml.batch_endpoints.list_jobs.assert_called_once_with(endpoint_name="endpoint")

    def test_job_listing_error_does_not_upload_or_submit(self):
        self.ml.batch_endpoints.list_jobs.side_effect = RuntimeError("Cannot read endpoint jobs")
        code, message = self.call(self.inputs, "--no-wait")
        self.assertEqual(code, 1)
        self.assertIn("Cannot read endpoint jobs", message)
        self.assertFalse((self.root / "runs").exists())
        self.ml.batch_endpoints.invoke.assert_not_called()

    def test_failed_job_downloads_available_exports_and_reports_failure(self):
        receipt = self.submit()
        self.ml.jobs.get.return_value.status = "Failed"
        self.cloud_report(receipt, documents=[{"input": "a.pdf", "status": "failed", "error": "bad PDF"}])
        self.outputs["server.log"] = b"diagnostic"
        self.outputs["documents/good/result.md"] = b"successful export"
        destination = self.root / "downloaded"
        code, message = self.call("--resume", self.receipt(), "--output", destination)
        self.assertEqual(code, 1)
        self.assertIn("FAILED: a.pdf: bad PDF", message)
        self.assertTrue((destination / "server.log").is_file())
        self.assertEqual((destination / "documents/good/result.md").read_bytes(), b"successful export")

    def test_unconfirmed_results_are_not_mislabeled_as_known_pdf_failures(self):
        receipt = self.submit()
        self.ml.jobs.get.return_value.status = "Failed"
        self.cloud_report(receipt, documents=[{"input": "a.pdf", "status": "unconfirmed"}])
        self.outputs["parser.log"] = b"native parser diagnostics"
        destination = self.root / "downloaded"
        code, message = self.call("--resume", self.receipt(), "--output", destination)
        self.assertEqual(code, 1)
        self.assertIn("UNCONFIRMED: a.pdf", message)
        self.assertNotIn("FAILED: a.pdf", message)
        self.assertEqual((destination / "parser.log").read_bytes(), b"native parser diagnostics")

    def test_missing_parent_output_downloads_single_worker_output(self):
        receipt = self.submit()
        self.ml.jobs.get.return_value.status = "Failed"
        self.ml.jobs.list.return_value = [SimpleNamespace(name="worker-job")]
        self.cloud_report(receipt, fatal_error="parser startup failed", finished=False)

        def download(job_name, **kwargs):
            if job_name == "worker-job":
                self.download(job_name, **kwargs)

        self.ml.jobs.download.side_effect = download
        code, message = self.call("--resume", self.receipt())
        self.assertEqual(code, 1)
        self.assertIn("BATCH FAILED: parser startup failed", message)
        self.assertEqual([call.args[0] for call in self.ml.jobs.download.call_args_list],
                         [receipt["job_name"], "worker-job"])

    def test_stale_local_report_cannot_hide_missing_cloud_outputs(self):
        receipt = self.submit()
        destination = self.root / "downloaded"
        destination.mkdir()
        self.cloud_report(receipt)
        content = self.outputs.pop("report.json")
        (destination / "report.json").write_bytes(content)
        code, message = self.call("--resume", self.receipt(), "--output", destination)
        self.assertEqual(code, 1)
        self.assertIn("no report was downloaded", message)

    def test_ambiguous_submission_can_be_recovered_without_a_second_job(self):
        def uncertain(**kwargs):
            raise ConnectionError("Response lost after server accepted job")
        self.ml.batch_endpoints.invoke.side_effect = uncertain
        code, _ = self.call(self.inputs, "--config", self.config, "--no-wait")
        self.assertEqual(code, 1)
        receipt = json.loads(self.receipt().read_text())
        self.assertEqual(receipt["state"], "submitting")
        self.cloud_report(receipt)
        code, message = self.call("--resume", self.receipt())
        self.assertEqual(code, 0, message)
        self.ml.batch_endpoints.invoke.assert_called_once()

    def test_interrupted_sdk_upload_leaves_receipt_and_resume_never_resubmits(self):
        staged_directories = []
        def interrupted(**kwargs):
            staged = Path(kwargs["inputs"]["pdfs"].path)
            self.assertEqual((staged / "a.pdf").read_bytes(), b"test PDF")
            staged_directories.append(staged)
            raise OSError("upload interrupted")
        self.ml.batch_endpoints.invoke.side_effect = interrupted
        code, _ = self.call(self.inputs, "--config", self.config, "--no-wait")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(self.receipt().read_text())["state"], "submitting")
        self.assertFalse(staged_directories[0].exists())
        self.assertEqual((self.inputs / "a.pdf").read_bytes(), b"test PDF")
        self.ml.jobs.get.side_effect = RuntimeError("Job not found")
        code, message = self.call("--resume", self.receipt())
        self.assertEqual(code, 1)
        self.assertIn("Job not found", message)
        self.ml.batch_endpoints.invoke.assert_called_once()

    def test_default_config_receipts_and_results_use_working_directory(self):
        code, message = self.call(self.inputs, "--no-wait")
        self.assertEqual(code, 0, message)
        receipt = json.loads(self.receipt().read_text())
        self.cloud_report(receipt)
        code, message = self.call("--resume", self.receipt())
        self.assertEqual(code, 0, message)
        self.assertTrue((self.root / "results" / receipt["run_id"] / "report.json").is_file())

    def test_real_sdk_download_layout_for_completed_and_failed_pipelines(self):
        ml = MLClient(MagicMock(), "sub", "rg", "ws")
        for status in ("Completed", "Failed"):
            job = SimpleNamespace(name="job", status=status, properties={},
                                  tags={"azureml.batchrun": "true", "azureml.jobtype": "azureml.pipelinejob"})

            def transfer(*, uri, destination, datastore_operation):
                self.assertEqual(Path(destination).parts[-2:], ("named-outputs", "results"))
                Path(destination).mkdir(parents=True)
                (Path(destination) / "report.json").write_text('{"finished": true}')
                (Path(destination) / "export.md").write_text("export")

            with self.subTest(status=status), \
                 patch("azure.core.pipeline.transport.RequestsTransport.send", side_effect=AssertionError("No network in tests")), \
                 patch.object(ml.jobs, "get", return_value=job), \
                 patch.object(ml.jobs, "_get_named_output_uri", return_value={"results": "azureml://test"}), \
                 patch("azure.ai.ml.operations._job_operations.download_artifact_from_aml_uri", side_effect=transfer):
                destination = self.root / status
                self.assertEqual(download_results(ml, "job", destination), {"finished": True})
                self.assertEqual((destination / "export.md").read_text(), "export")


if __name__ == "__main__":
    unittest.main()
