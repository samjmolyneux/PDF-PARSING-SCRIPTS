"""Exercise submission/resumption with the real SDK types and mocked services."""

import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from azure.ai.ml import MLClient
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from azure.identity import CredentialUnavailableError

from pdf_parsers import mineru, paddle
from pdf_parsers.client import download_results


class TestClient:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.root = tmp_path.resolve()
        monkeypatch.chdir(self.root)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        (self.inputs / "a.pdf").write_bytes(b"test PDF")
        config = {
            "subscription_id": "sub",
            "resource_group": "rg",
            "workspace": "ws",
            "tenant_id": "tenant",
            "endpoint": "endpoint",
        }
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        self.ml = MagicMock()
        self.ml_factory = MagicMock(return_value=self.ml)
        self.azure_cli = MagicMock()
        self.browser = MagicMock()
        self.device_code = MagicMock()
        self.ml.batch_endpoints.list_jobs.return_value = []
        self.outputs = {}
        self.ml.jobs.list.return_value = []
        self.events = []
        self.expected_upload = {"a.pdf": b"test PDF"}
        self.upload_directories = []

        def invoke(**kwargs):
            # A receipt must be durable before the HTTP request might succeed.
            receipt = json.loads(self.receipt().read_text())
            assert receipt["state"] == "submitting"
            assert kwargs["job_name"] == receipt["job_name"]
            expected_deployment = (
                "paddle-vl" if receipt["parser"] == "paddle" else "mineru"
            )
            assert kwargs["deployment_name"] == expected_deployment
            staged = Path(kwargs["inputs"]["pdfs"].path)
            self.upload_directories.append(staged)
            assert staged != self.inputs
            assert {
                p.name: p.read_bytes() for p in staged.iterdir()
            } == self.expected_upload
            assert kwargs["inputs"]["pdfs"].type == "uri_folder"
            assert "outputs" not in kwargs
            self.events.append("invoked")
            return SimpleNamespace(name=kwargs["job_name"])

        self.ml.batch_endpoints.invoke.side_effect = invoke
        self.ml.jobs.get.return_value = SimpleNamespace(status="Completed")
        self.ml.jobs.download.side_effect = self.download

    def download(self, job_name, *, download_path, output_name):
        assert output_name == "results"
        for name, content in self.outputs.items():
            file = Path(download_path) / "named-outputs/results" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(content)

    def receipt(self):
        return next((self.root / "runs").glob("*.json"))

    def call(self, *args, parser="mineru"):
        with (
            patch.object(sys, "argv", [f"run-{parser}", *map(str, args)]),
            patch("azure.ai.ml.MLClient", self.ml_factory),
            patch("azure.identity.AzureCliCredential", self.azure_cli),
            patch("azure.identity.InteractiveBrowserCredential", self.browser),
            patch("azure.identity.DeviceCodeCredential", self.device_code),
            contextlib.redirect_stdout(io.StringIO()) as output,
            contextlib.redirect_stderr(io.StringIO()) as errors,
        ):
            code = {"mineru": mineru, "paddle": paddle}[parser].main()
        return code, output.getvalue() + errors.getvalue()

    def submit(self, use_defaults=False):
        args = [self.inputs, "--no-wait"]
        if not use_defaults:
            args += ["--config", self.config]
        code, message = self.call(*args)
        assert code == 0, message
        assert "Submitted:" in message
        assert "Later: run-mineru --resume" in message
        receipt = json.loads(self.receipt().read_text())
        assert receipt["state"] == "submitted"
        return receipt

    def cloud_report(self, receipt, **overrides):
        report = {
            "parser": "mineru",
            "total": 1,
            "finished": True,
            "fatal_error": None,
            "documents": [{"input": "a.pdf", "status": "succeeded"}],
        }
        report.update(overrides)
        self.outputs["report.json"] = json.dumps(report).encode()

    def test_default_reuses_working_cli_login(self):
        self.submit()
        self.azure_cli.assert_called_once_with(tenant_id="tenant")
        self.azure_cli.return_value.get_token.assert_called_once_with(
            "https://management.azure.com/.default"
        )
        assert self.ml_factory.call_args.args[0] is self.azure_cli.return_value
        self.browser.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                CredentialUnavailableError("Please run az login"), id="unavailable"
            ),
            pytest.param(ClientAuthenticationError("Login has expired"), id="expired"),
        ],
    )
    def test_default_falls_back_to_browser_when_cli_is_unavailable_or_expired(
        self, error
    ):
        self.azure_cli.return_value.get_token.side_effect = error
        code, message = self.call(self.inputs, "--no-wait")
        assert code == 0, message
        self.browser.assert_called_once_with(tenant_id="tenant")
        assert self.ml_factory.call_args.args[0] is self.browser.return_value
        assert "opening browser sign-in" in message

    @pytest.mark.parametrize(
        "login_args, error, expected_message",
        [
            pytest.param(
                ["--az-login"],
                ClientAuthenticationError("Please run az login"),
                "Please run az login",
                id="explicit-cli-login-failed",
            ),
            pytest.param(
                [],
                HttpResponseError("Forbidden: workspace access denied"),
                "workspace access denied",
                id="workspace-forbidden",
            ),
        ],
    )
    def test_azure_access_errors_do_not_trigger_browser_fallback(
        self, login_args, error, expected_message
    ):
        self.ml.batch_endpoints.list_jobs.side_effect = error
        code, message = self.call(self.inputs, "--no-wait", *login_args)
        assert code == 1
        assert expected_message in message
        assert self.ml_factory.call_args.args[0] is self.azure_cli.return_value
        self.browser.assert_not_called()
        self.ml.batch_endpoints.invoke.assert_not_called()

    @pytest.mark.parametrize(
        "flag, selected, unused",
        [
            ("--browser-login", "browser", "device_code"),
            ("--device-code", "device_code", "browser"),
        ],
    )
    def test_explicit_interactive_login_skips_other_credentials(
        self, flag, selected, unused
    ):
        code, message = self.call(self.inputs, "--no-wait", flag)
        assert code == 0, message
        assert self.ml_factory.call_args.args[0] is getattr(self, selected).return_value
        self.azure_cli.assert_not_called()
        getattr(self, unused).assert_not_called()

    @pytest.mark.parametrize(
        "use_defaults", [True, False], ids=["default-paths", "explicit-paths"]
    )
    def test_submit_then_resume_downloads_without_uploading_or_invoking_again(
        self, use_defaults
    ):
        receipt = self.submit(use_defaults=use_defaults)
        assert receipt["pdfs"] == ["a.pdf"]
        self.cloud_report(receipt)
        self.outputs["documents/a/result.md"] = b"all exports"
        self.ml.jobs.get.side_effect = [
            SimpleNamespace(status="Running"),
            SimpleNamespace(status="Running"),
            SimpleNamespace(status="Completed"),
        ]
        # Resuming must work even while the endpoint has an unfinished job.
        self.ml.batch_endpoints.list_jobs.side_effect = AssertionError(
            "Resume must skip the busy check"
        )
        destination = (
            self.root / "results" / receipt["run_id"]
            if use_defaults
            else self.root / "downloaded"
        )
        args = ["--resume", self.receipt()]
        if not use_defaults:
            args += ["--output", destination]
        with patch("pdf_parsers.client.time.sleep") as sleep:
            code, message = self.call(*args)
        assert code == 0, message
        assert self.ml.jobs.get.call_args_list == [call(receipt["job_name"])] * 3
        assert sleep.call_args_list == [call(30), call(30)]
        assert message.count("Azure job: Running") == 1
        assert message.count("Azure job: Completed") == 1
        assert self.events == ["invoked"]
        assert (destination / "report.json").is_file()
        assert (destination / "documents/a/result.md").read_bytes() == b"all exports"
        assert "1/1 PDFs succeeded" in message
        self.ml.batch_endpoints.list_jobs.assert_called_once_with(
            endpoint_name="endpoint"
        )
        assert len(self.upload_directories) == 1
        assert not self.upload_directories[0].exists()

    @pytest.mark.parametrize("parser", ["mineru", "paddle"])
    def test_keyboard_interrupt_preserves_receipt_and_resume_never_resubmits(
        self, parser
    ):
        self.ml.jobs.get.return_value.status = "Running"
        with patch(
            "pdf_parsers.client.time.sleep", side_effect=KeyboardInterrupt
        ) as sleep:
            code, message = self.call(
                self.inputs, "--config", self.config, parser=parser
            )
        assert code == 1
        assert "Client stopped. Any accepted Azure job continues." in message
        sleep.assert_called_once_with(30)
        receipt_path = self.receipt()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["parser"] == parser
        assert receipt["state"] == "submitted"
        assert f"Receipt: {receipt_path}." in message
        self.ml.jobs.get.assert_called_once_with(receipt["job_name"])
        self.ml.jobs.download.assert_not_called()
        self.ml.jobs.cancel.assert_not_called()
        assert len(self.upload_directories) == 1
        assert not self.upload_directories[0].exists()
        assert (self.inputs / "a.pdf").read_bytes() == b"test PDF"

        self.cloud_report(receipt, parser=parser)
        self.outputs["documents/a/result.md"] = b"all exports"
        self.ml.jobs.get.return_value.status = "Completed"
        code, message = self.call("--resume", receipt_path, parser=parser)
        assert code == 0, message
        assert "1/1 PDFs succeeded" in message
        destination = self.root / "results" / receipt["run_id"]
        assert (destination / "documents/a/result.md").read_bytes() == b"all exports"
        assert json.loads(receipt_path.read_text()) == receipt
        self.ml.batch_endpoints.invoke.assert_called_once()
        self.ml.batch_endpoints.list_jobs.assert_called_once_with(
            endpoint_name="endpoint"
        )
        self.ml.jobs.cancel.assert_not_called()
        assert len(self.upload_directories) == 1

    @pytest.mark.parametrize(
        "parser, deployment", [("mineru", "mineru"), ("paddle", "paddle-vl")]
    )
    def test_mixed_folders_upload_only_direct_pdfs(self, parser, deployment):
        (self.inputs / "Second PDF.PDF").write_bytes(b"second PDF")
        self.expected_upload["Second PDF.PDF"] = b"second PDF"
        for name in ("notes.txt", "photo.png", ".DS_Store", ".amlignore", ".gitignore"):
            (self.inputs / name).write_text("*.pdf")
        nested = self.inputs / "nested.pdf"
        nested.mkdir()
        (nested / "nested.pdf").write_bytes(b"not selected")

        code, message = self.call(self.inputs, "--no-wait", parser=parser)
        assert code == 0, message
        assert (
            self.ml.batch_endpoints.invoke.call_args.kwargs["deployment_name"]
            == deployment
        )
        assert not self.upload_directories[0].exists()
        receipt = json.loads(self.receipt().read_text())
        assert set(receipt["pdfs"]) == set(self.expected_upload)
        assert receipt["pdf_count"] == 2
        assert (self.inputs / "a.pdf").read_bytes() == b"test PDF"
        assert (self.inputs / "notes.txt").read_text() == "*.pdf"
        assert (nested / "nested.pdf").is_file()

    def test_current_directory_excludes_config_receipts_and_nested_pdfs(self):
        (self.root / "a.pdf").write_bytes(b"test PDF")
        code, message = self.call(".", "--no-wait")
        assert code == 0, message
        assert json.loads(self.receipt().read_text())["pdfs"] == ["a.pdf"]
        assert not self.upload_directories[0].exists()

    @pytest.mark.parametrize("parser", ["mineru", "paddle"])
    def test_input_file_is_rejected_before_contacting_azure(self, parser):
        code, message = self.call(self.inputs / "a.pdf", "--no-wait", parser=parser)
        assert code == 1
        assert "Input must be a directory of PDFs." in message
        self.azure_cli.assert_not_called()
        self.browser.assert_not_called()
        self.device_code.assert_not_called()
        self.ml_factory.assert_not_called()
        self.ml.batch_endpoints.invoke.assert_not_called()
        assert not (self.root / "runs").exists()

    def test_no_direct_pdfs_does_not_contact_azure_or_submit(self):
        (self.inputs / "a.pdf").unlink()
        nested = self.inputs / "nested"
        nested.mkdir()
        (nested / "a.pdf").write_bytes(b"not selected")
        (self.inputs / "notes.txt").write_text("not a PDF")
        code, message = self.call(self.inputs, "--no-wait")
        assert code == 1
        assert "No PDFs found" in message
        self.ml.batch_endpoints.list_jobs.assert_not_called()
        self.ml.batch_endpoints.invoke.assert_not_called()
        assert not (self.root / "runs").exists()

    def test_copy_failure_cleans_staging_and_never_uploads(self):
        destinations = []

        def fail_copy(source, destination):
            destinations.append(destination)
            destination.write_bytes(b"partial copy")
            raise OSError("disk full")

        with patch("pdf_parsers.client.shutil.copyfile", side_effect=fail_copy):
            code, message = self.call(self.inputs, "--no-wait")
        assert code == 1
        assert "disk full" in message
        assert not destinations[0].parent.exists()
        self.ml.batch_endpoints.invoke.assert_not_called()
        assert (self.inputs / "a.pdf").read_bytes() == b"test PDF"

    @pytest.mark.parametrize(
        "parser, other", [("mineru", "paddle"), ("paddle", "mineru")]
    )
    @pytest.mark.parametrize(
        "status",
        [
            "NotStarted",
            "Queued",
            "Preparing",
            "Provisioning",
            "Starting",
            "Running",
            "Finalizing",
            "CancelRequested",
            "Paused",
            None,
        ],
    )
    @pytest.mark.parametrize("active_jobs, expected_code", [(1, 0), (2, 1)])
    def test_endpoint_allows_a_second_job_but_blocks_a_third_before_upload(
        self, parser, other, status, active_jobs, expected_code
    ):
        self.ml.batch_endpoints.list_jobs.return_value = [
            SimpleNamespace(name="old-job", status="Completed"),
            *[
                SimpleNamespace(name=f"{other}-colleague-job-{index}", status=status)
                for index in range(active_jobs)
            ],
        ]
        code, message = self.call(self.inputs, "--no-wait", parser=parser)
        assert code == expected_code, message
        if expected_code:
            assert "Parser endpoint is busy: 2 jobs are unfinished" in message
            assert f"{other}-colleague-job" in message
            assert "No PDFs were uploaded" in message
            assert not (self.root / "runs").exists()
            self.ml.batch_endpoints.invoke.assert_not_called()
        else:
            self.ml.batch_endpoints.invoke.assert_called_once()
            assert json.loads(self.receipt().read_text())["state"] == "submitted"

    @pytest.mark.parametrize(
        "status", ["Completed", "Failed", "Canceled", "Cancelled", "NotResponding"]
    )
    def test_finished_jobs_do_not_block_submission(self, status):
        # Counting these finished jobs as active would hit the two-job limit.
        self.ml.batch_endpoints.list_jobs.return_value = [
            SimpleNamespace(name=f"old-{status}-{index}", status=status)
            for index in range(2)
        ]
        self.submit()
        self.ml.batch_endpoints.list_jobs.assert_called_once_with(
            endpoint_name="endpoint"
        )

    def test_job_listing_error_does_not_upload_or_submit(self):
        self.ml.batch_endpoints.list_jobs.side_effect = RuntimeError(
            "Cannot read endpoint jobs"
        )
        code, message = self.call(self.inputs, "--no-wait")
        assert code == 1
        assert "Cannot read endpoint jobs" in message
        assert not (self.root / "runs").exists()
        self.ml.batch_endpoints.invoke.assert_not_called()

    @pytest.mark.parametrize(
        "document, expected_label, unwanted_label",
        [
            pytest.param(
                {"input": "a.pdf", "status": "failed", "error": "bad PDF"},
                "FAILED: a.pdf: bad PDF",
                "UNCONFIRMED: a.pdf",
                id="failed",
            ),
            pytest.param(
                {"input": "a.pdf", "status": "unconfirmed"},
                "UNCONFIRMED: a.pdf",
                "FAILED: a.pdf",
                id="unconfirmed",
            ),
        ],
    )
    def test_incomplete_job_preserves_outputs_and_reports_document_status(
        self, document, expected_label, unwanted_label
    ):
        receipt = self.submit()
        self.ml.jobs.get.return_value.status = "Failed"
        self.cloud_report(receipt, documents=[document])
        self.outputs["parser.log"] = b"native parser diagnostics"
        self.outputs["server.log"] = b"diagnostic"
        self.outputs["documents/good/result.md"] = b"successful export"
        destination = self.root / "downloaded"

        code, message = self.call("--resume", self.receipt(), "--output", destination)
        assert code == 1
        assert expected_label in message
        assert unwanted_label not in message
        assert (destination / "server.log").is_file()
        assert (destination / "parser.log").read_bytes() == b"native parser diagnostics"
        assert (
            destination / "documents/good/result.md"
        ).read_bytes() == b"successful export"

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
        assert code == 1
        assert "BATCH FAILED: parser startup failed" in message
        assert [call.args[0] for call in self.ml.jobs.download.call_args_list] == [
            receipt["job_name"],
            "worker-job",
        ]

    def test_stale_local_report_cannot_hide_missing_cloud_outputs(self):
        receipt = self.submit()
        destination = self.root / "downloaded"
        destination.mkdir()
        self.cloud_report(receipt)
        content = self.outputs.pop("report.json")
        (destination / "report.json").write_bytes(content)
        code, message = self.call("--resume", self.receipt(), "--output", destination)
        assert code == 1
        assert "no report was downloaded" in message

    @pytest.mark.parametrize(
        "submission_error, job_found",
        [
            pytest.param(
                ConnectionError("Response lost after server accepted job"),
                True,
                id="accepted-response-lost",
            ),
            pytest.param(OSError("upload interrupted"), False, id="upload-interrupted"),
        ],
    )
    def test_failed_submission_leaves_receipt_and_resume_never_resubmits(
        self, submission_error, job_found
    ):
        staged_directories = []

        def interrupted(**kwargs):
            staged = Path(kwargs["inputs"]["pdfs"].path)
            assert (staged / "a.pdf").read_bytes() == b"test PDF"
            staged_directories.append(staged)
            raise submission_error

        self.ml.batch_endpoints.invoke.side_effect = interrupted
        code, _ = self.call(self.inputs, "--config", self.config, "--no-wait")
        assert code == 1
        receipt = json.loads(self.receipt().read_text())
        assert receipt["state"] == "submitting"
        assert not staged_directories[0].exists()
        assert (self.inputs / "a.pdf").read_bytes() == b"test PDF"
        if job_found:
            self.cloud_report(receipt)
        else:
            self.ml.jobs.get.side_effect = RuntimeError("Job not found")
        code, message = self.call("--resume", self.receipt())
        assert code == (0 if job_found else 1), message
        assert ("1/1 PDFs succeeded" if job_found else "Job not found") in message
        self.ml.batch_endpoints.invoke.assert_called_once()

    @pytest.mark.parametrize("status", ["Completed", "Failed"])
    def test_real_sdk_download_layout(self, status):
        ml = MLClient(MagicMock(), "sub", "rg", "ws")
        job = SimpleNamespace(
            name="job",
            status=status,
            properties={},
            tags={
                "azureml.batchrun": "true",
                "azureml.jobtype": "azureml.pipelinejob",
            },
        )

        def transfer(*, uri, destination, datastore_operation):
            assert Path(destination).parts[-2:] == ("named-outputs", "results")
            Path(destination).mkdir(parents=True)
            (Path(destination) / "report.json").write_text('{"finished": true}')
            (Path(destination) / "export.md").write_text("export")

        with (
            patch(
                "azure.core.pipeline.transport.RequestsTransport.send",
                side_effect=AssertionError("No network in tests"),
            ),
            patch.object(ml.jobs, "get", return_value=job),
            patch.object(
                ml.jobs,
                "_get_named_output_uri",
                return_value={"results": "azureml://test"},
            ),
            patch(
                "azure.ai.ml.operations._job_operations.download_artifact_from_aml_uri",
                side_effect=transfer,
            ),
        ):
            destination = self.root / status
            assert download_results(ml, "job", destination) == {"finished": True}
            assert (destination / "export.md").read_text() == "export"
