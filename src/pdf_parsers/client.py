"""Shared local client. Native parser batches run entirely in Azure."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}
DEPLOYMENT_NAMES = {"mineru": "mineru", "paddle": "paddle-vl"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pdf_files(folder):
    folder = folder.resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Input must be a directory of PDFs.")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not files:
        raise ValueError(f"No PDFs found in {folder}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Copy linked PDFs into the input folder first: {path}")
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("PDF filenames must be distinct, including when compared without letter case.")
    return folder, files


def download_results(ml, job_name, destination):
    """Use Azure's downloader, then flatten its named-outputs/results directory."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A fresh download prevents an old local report from disguising missing outputs.
    with tempfile.TemporaryDirectory(prefix=".pdf-parser-", dir=destination.parent) as temporary:
        ml.jobs.download(job_name, download_path=temporary, output_name="results")
        results = Path(temporary) / "named-outputs" / "results"
        if not (results / "report.json").is_file():
            # A failed pipeline may only expose the output on its single worker job.
            children = list(ml.jobs.list(parent_job_name=job_name))
            if len(children) == 1:
                ml.jobs.download(children[0].name, download_path=temporary, output_name="results")
        if results.is_dir():
            shutil.copytree(results, destination, dirs_exist_ok=True)
        if not (results / "report.json").is_file():
            raise RuntimeError("Incomplete: no report was downloaded. Check Azure job logs and whether output files still exist.")
        return json.loads((results / "report.json").read_text(encoding="utf-8"))


def report_success(report, receipt, status):
    documents = report.get("documents", [])
    return (
        status == "Completed"
        and report.get("finished") is True
        and report.get("parser") == receipt["parser"]
        and report.get("total") == receipt["pdf_count"]
        and len(documents) == receipt["pdf_count"]
        and {d.get("input") for d in documents} == set(receipt["pdfs"])
        and all(d.get("status") == "succeeded" for d in documents)
        and not report.get("fatal_error")
    )


def main(parser_name, argv=None):
    """Run the client; explicit arguments also allow calls from notebooks."""
    working_dir = Path.cwd()
    cli = argparse.ArgumentParser(
        prog=f"run-{parser_name}",
        description=f"Run {parser_name} on Azure ML. Sign-in defaults to Azure CLI, then browser if unavailable.",
    )
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument("input", nargs="?", type=Path, help="Upload only PDFs directly inside this folder; ignore other files and subfolders")
    source.add_argument("--resume", type=Path, help="Receipt JSON from an already submitted job")
    cli.add_argument("--config", type=Path, default=working_dir / "config.json",
                     help="Workspace configuration (default: ./config.json)")
    cli.add_argument("--output", type=Path, help="Local results folder (default: results/RUN_ID)")
    cli.add_argument("--no-wait", action="store_true", help="Return once Azure has accepted the job")
    login = cli.add_mutually_exclusive_group()
    login.add_argument("--device-code", action="store_true", help="Sign in using a device code")
    login.add_argument("--az-login", action="store_true", help="Use only your existing az login session")
    login.add_argument("--browser-login", action="store_true", help="Sign in through the browser directly")
    args = cli.parse_args(argv)
    receipt_path = args.resume
    receipt = None
    try:
        if args.resume:
            receipt = json.loads(args.resume.read_text(encoding="utf-8"))
            if receipt["parser"] != parser_name:
                raise ValueError(f"Use run-{receipt['parser']} for this receipt.")
            config = receipt["config"]
        else:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            folder, files = pdf_files(args.input)

        # Lazy imports make --help and local tests independent of the Azure SDK.
        from azure.ai.ml import MLClient, Input
        from azure.core.exceptions import ClientAuthenticationError
        from azure.identity import AzureCliCredential, DeviceCodeCredential, InteractiveBrowserCredential

        if args.device_code:
            credential = DeviceCodeCredential(tenant_id=config["tenant_id"])
        elif args.browser_login:
            credential = InteractiveBrowserCredential(tenant_id=config["tenant_id"])
        else:
            credential = AzureCliCredential(tenant_id=config["tenant_id"])
            if not args.az_login:
                # Fall back only for sign-in failures, before any workspace operations.
                # This also catches CredentialUnavailableError (e.g. CLI not installed).
                try:
                    credential.get_token("https://management.azure.com/.default")
                except ClientAuthenticationError:
                    print("Azure CLI sign-in unavailable; opening browser sign-in.", flush=True)
                    credential = InteractiveBrowserCredential(tenant_id=config["tenant_id"])
        ml = MLClient(credential, config["subscription_id"], config["resource_group"], config["workspace"])
        if receipt is None:
            # Check both parser deployments before upload. This is not a reservation:
            # simultaneous submissions can still pass before either job is visible.
            for existing in ml.batch_endpoints.list_jobs(endpoint_name=config["endpoint"]):
                if existing.status not in TERMINAL:
                    raise RuntimeError(
                        f"Parser endpoint is busy: job {existing.name} is {existing.status}. "
                        "No PDFs were uploaded. Please try again later."
                    )
            run_id = f"{parser_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:10]}"
            receipt = {
                "parser": parser_name, "run_id": run_id, "job_name": run_id,
                "config": config, "state": "submitting", "pdf_count": len(files),
                "pdfs": [p.relative_to(folder).as_posix() for p in files],
            }
            receipt_path = working_dir / "runs" / f"{run_id}.json"
            save_json(receipt_path, receipt)
        if not args.resume:
            # The SDK uploads a whole folder during invoke. Stage only the selected
            # PDFs so unrelated files and local ignore rules cannot affect the batch.
            print(f"Preparing and uploading {len(files)} PDFs from: {folder}\nReceipt: {receipt_path}", flush=True)
            with tempfile.TemporaryDirectory(prefix="pdf-parser-upload-") as temporary:
                for pdf in files:
                    shutil.copyfile(pdf, Path(temporary) / pdf.name)
                job = ml.batch_endpoints.invoke(
                    endpoint_name=config["endpoint"], deployment_name=DEPLOYMENT_NAMES[parser_name],
                    job_name=receipt["job_name"],
                    inputs={"pdfs": Input(type="uri_folder", mode="download", path=temporary)},
                )
            receipt.update(job_name=job.name, state="submitted")
            save_json(receipt_path, receipt)
            print(f"Submitted: {job.name}\nReceipt: {receipt_path}", flush=True)
            print(f'Later: run-{parser_name} --resume "{receipt_path}"', flush=True)
        if args.no_wait:
            return 0

        status = None
        while True:
            job = ml.jobs.get(receipt["job_name"])
            if job.status != status:
                status = job.status
                print(f"Azure job: {status}", flush=True)
            if status in TERMINAL:
                break
            time.sleep(30)
        destination = args.output or working_dir / "results" / receipt["run_id"]
        report = download_results(ml, receipt["job_name"], destination)
        print(f"Downloaded results to {destination.resolve()}")
        if report.get("fatal_error"):
            print(f"BATCH FAILED: {report['fatal_error']}", file=sys.stderr)
        failed = [d for d in report.get("documents", []) if d.get("status") != "succeeded"]
        for document in failed:
            label = "FAILED" if document.get("status") == "failed" else "UNCONFIRMED"
            print(f"{label}: {document['input']}: {document.get('error') or 'See report.json and parser.log.'}", file=sys.stderr)
        if not report_success(report, receipt, status):
            raise RuntimeError(f"Incomplete batch. Inspect {destination / 'report.json'} and Azure job logs.")
        print(f"Complete: {receipt['pdf_count']}/{receipt['pdf_count']} PDFs succeeded.")
        return 0
    except KeyboardInterrupt:
        print("\nClient stopped. Any accepted Azure job continues.", file=sys.stderr)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
    if receipt_path:
        print(f"Receipt: {receipt_path}. Check it and Azure job status before submitting again.", file=sys.stderr)
    return 1
