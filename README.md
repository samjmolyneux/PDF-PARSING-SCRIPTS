# PDF parsing on Azure ML

Run the team's existing MinerU or PaddleOCR-VL settings from a local PDF folder.
Azure processes the batch even after the submitting laptop disconnects. Two
commands, `run-mineru` and `run-paddle`, have separate entry modules and share the
upload, job tracking and download code.

**Setup is not deployed yet.** See [the administrator guide](docs/DEPLOYMENT.md)
for the one-time setup and GPU smoke test. Local tests cannot verify GPU inference,
Azure permissions, quota, image builds or deployment.

## For colleagues

Install Python 3.10 or newer. Obtain this repository and the administrator's
`config.json`, then install the package from the repository root in a virtual
environment:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install .
```

Installing the package registers both commands through `pyproject.toml`.
Keep the virtual environment activated when using them. Run either parser;
the first run opens the Microsoft sign-in page:

```sh
run-mineru ./pdfs --output ./mineru-results
run-paddle ./pdfs --output ./paddle-results
```

The commands work from any folder. By default, they read `config.json` from the
current folder and save receipts under `./runs/` and results under
`./results/RUN_ID/`. Use `--config /path/to/config.json` and `--output` to choose
different locations. `--resume` uses the configuration saved in the receipt.
Use `run-mineru --help` or `run-paddle --help` to see all options.

### Using a Python notebook

Open [run_parsers.ipynb](run_parsers.ipynb) in VS Code or Jupyter and select the
Python environment where you installed this package. If that environment needs a
notebook kernel, install it with `python -m pip install ipykernel`.

Set the parser, input/output folders and configuration file in the first code
cell, then run the cells in order. Set `WAIT_FOR_RESULTS = False` to return after
submission. To download an existing job later, set `RESUME_RECEIPT` to its receipt
path and rerun the cells. The notebook uses the same Python client as the CLI,
including the busy check, sign-in, failure reporting and downloads.

### What happens during a run

Before uploading, either command checks for unfinished jobs on the configured
endpoint, across both parsers and all users. If it finds one, it exits with the
job name and status: try again later. If the check fails, nothing is submitted.
`--resume` skips this check. This is best effort: simultaneous submissions can
both pass before either job becomes visible, and other clients can submit without
checking. Azure queueing itself remains enabled.

The client selects **PDFs directly inside the folder you choose**, including
`.PDF` extensions, and ignores other files and subfolders. Use `.` to select the
current directory. It copies the selected PDFs into a temporary folder and uploads
only that folder, so configuration files, images and nested PDFs are not sent.
PDF names are preserved and must be distinct even without letter case.

Keep the selected PDFs unchanged until submission finishes. The temporary copy
needs enough local disk space for those PDFs and is removed after submission or
an error; your original files remain in place. Each parser receives the complete
PDF batch in one call, using native batching/concurrency. All generated exports
are retained.

The script normally waits and downloads results. To submit and close your laptop:

```sh
run-paddle ./pdfs --no-wait
```

**Wait until it prints `Submitted` and the receipt filename.** Uploading requires
your laptop to remain connected. After submission, the Azure job is independent
of it. Keep the small JSON receipt printed by the script, then retrieve later:

```sh
run-paddle --resume runs/RECEIPT.json --output ./paddle-results
```

This waits for the existing job and downloads its results; it does not submit a
new batch. Pressing Ctrl+C while waiting detaches the client without cancelling
the Azure job. Use `--device-code` if browser login is unavailable, or `--az-login`
to reuse an existing Azure CLI login. Everyone signs in with their own account.

There are **no automatic PDF retries or custom per-PDF time limits**. The whole
job has a 24-hour execution limit; the parsers' own request timeouts still apply.
A parser error may stop the batch before every PDF is processed. The client
still downloads available exports, `parser.log` and `report.json`, then returns
failure. Paddle also saves `server.log` for its vLLM service.

The report lists each PDF as `succeeded` or `unconfirmed`. Paddle confirms a PDF
after all its pages and exports are saved. MinerU confirms documents when its
native batch succeeds and their expected exports exist; if it exits with an error, all PDFs stay
unconfirmed, even where exports exist. Its log contains the native error details.
Unconfirmed means completion was not established, not necessarily that the PDF
itself is defective. Inspect the logs/results before resubmitting selected PDFs
in another folder. These checks do not assess OCR quality.

## Storage and cost

- Team access is governed by Azure ML and Blob Storage roles.
- Inputs and outputs use the workspace's existing default datastore. This
  project does not automatically delete them; cleanup is manual for now.
  Storage continues to accumulate until files are removed.
- Setup leaves any existing storage-account deletion policies unchanged.
- Model weights, environments and worker code remain centrally managed Azure assets.
- The GPU cluster starts with a zero-node minimum, one-node maximum, and a
  two-minute idle scale-down. More nodes can be allowed later. Each job uses one
  node. To allow concurrent submissions through our CLI, the busy check would
  also need adjusting; increasing the node maximum alone does not change it.
- GPU startup, processing and the idle interval are billable. Storage, registry
  and transfer costs remain when the cluster is at zero.

The implementation uses Azure ML pipeline component batch deployments, one
deployment per parser, backed by one shared A100 compute cluster. Package versions
were identified from the working interactive setup. Model weights come directly
from the publishers' official Hugging Face repositories; setup does not depend
on the existing `sam-a100` compute instance.

## Development and administration

GPU environments are defined in readable YAML pairs:

- `environments/mineru/environment.yml` and `conda.yml`: Azure image/version and MinerU dependencies.
- `environments/paddle/environment.yml` and `conda.yml`: existing Paddle server image and separate client dependencies.
- `models/mineru.yml` and `models/paddle.yml`: separately versioned model weights.
- `azure/`: compute, commands, pipelines and batch deployments that connect them.

Preview the registrations without contacting Azure:

```sh
python admin/register_environments.py
python admin/register_models.py
```

An administrator downloads pinned weights and registers them once, then deploys
the endpoint. From the repository root on the administrator's computer:

```sh
python -m pip install '.[admin]'
python admin/download_models.py --parser mineru
python admin/download_models.py --parser paddle
```

These commands download files locally; they need no GPU and do not upload to
Azure. Follow the [administrator guide](docs/DEPLOYMENT.md) to register the
downloaded folders and deploy.

Model registration is separate from deployment: rerun it only when introducing a
new model version. The repository IDs, revisions and file selection are defined
in `admin/download_models.py`.

The [administrator guide](docs/DEPLOYMENT.md#changing-environments-or-models)
explains how to register one changed environment and update the deployment.

Dependencies, the minimum Python version and console commands are declared in
`pyproject.toml`. The `admin` extra adds the Hugging Face download library;
colleagues only need the normal installation.
For local development, use `python -m pip install -e '.[admin]'` so source edits
are picked up without reinstalling. Run the local checks with
`python -m unittest discover -s tests -v` from the repository root.
