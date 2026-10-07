# Administrator setup

These are instructions for you to run. Nothing has been deployed, uploaded or
submitted to Azure by this implementation. The existing A100 was inspected
read-only to identify its commands and package versions. Setup downloads model
weights from the official Hugging Face repositories and needs no access to that
machine.

The design is one batch endpoint with two pipeline deployments, `mineru` and
`paddle-vl`, on a shared A100 cluster. Each invocation processes one folder on one
GPU. The cluster scales from zero to one node and returns to zero when idle.
The endpoint remains registered while the GPU is off. Azure supports this
[pipeline batch deployment pattern](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-pipeline-deployments?view=azureml-api-2).

## 1. Prepare the configuration and administrator tools

All administrator scripts are Python and work on Windows, macOS and Linux.
Colleagues only need Python and the client package.
Activate your existing Conda environment with Python 3.10+ (Python 3.12 is
recommended). Install the [Azure CLI](https://pypi.org/project/azure-cli/) and the
administrator tools into that environment. From the repository root:

```sh
python -m pip install azure-cli '.[admin]'
az login --tenant f870e5ae-5521-4a94-b9ff-cdde7d36dd35
```

The `admin` extra adds the Hugging Face download library. This installation
supports the administrator Python scripts and creates
the `run-mineru` and `run-paddle` console commands. Administrator helper scripts
still run from the repository because they use its Azure definitions and model
download directories. The client commands can run from any folder; their default
configuration, receipt and result paths are relative to that working folder.

Review `config.json`. The supplied values came from the open Azure ML tab:

| Setting | Value |
| --- | --- |
| Subscription | `56539498-d3d8-4a3b-92f4-f3b098a11d1e` |
| Resource group | `continuous_review_ms_and_ucl` |
| Workspace | `EPPI_DEV` |
| Region | `westeurope` |
| New endpoint | `eppi-pdf-parsers-ccaesjm` |
| Datastore | Workspace's existing default datastore |
| New cluster | `pdf-parsers-a100` |
| VM size | `Standard_NC24ads_A100_v4` |

The endpoint name can be changed in `config.json`; the administrator scripts
override the YAML endpoint name with that value. The cluster name is also used
in the compute and deployment YAML files; change those references together if
you choose a different cluster name.

An administrator needs permission to create compute and register/deploy Azure ML
assets in the existing workspace. The endpoint, deployments and compute belong
to `EPPI_DEV`, within resource group `continuous_review_ms_and_ucl`. Inputs and
outputs use the workspace's existing default datastore. Access is managed
through the workspace's existing administration; these scripts do not create
storage resources, request a team group ID or assign roles.

Check West Europe quota and capacity for the NCads A100 v4 family: this size uses
24 vCPUs and one A100 per node. Existing compute instances also consume applicable
quota. The existing `sam-a100` is an interactive **compute instance**; the batch
deployment uses the new **compute cluster**.

Use the workspace's existing network requirements. Private storage/workspaces
require the submitting machine and compute to have the appropriate network
access. These scripts do not change firewall or private endpoint settings.

## 2. Download the official model weights

Run these commands from this repository on your own computer or another machine
with internet access and enough disk space. No GPU, parser installation, Azure
login or access to the old A100 is needed for this step. Run only the command for
each parser you intend to deploy:

```sh
python admin/download_models.py --parser mineru
python admin/download_models.py --parser paddle
```

The helper downloads these official repositories at fixed commit revisions:

| Repository | Revision |
| --- | --- |
| [opendatalab/PDF-Extract-Kit-1.0](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/tree/ed6b654c018d742e65a17671e379c5e6ecc87ec9) | `ed6b654c018d742e65a17671e379c5e6ecc87ec9` |
| [opendatalab/MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B/tree/bff20d4ae2bf202df9f45284b4d43681555a97ed) | `bff20d4ae2bf202df9f45284b4d43681555a97ed` |
| [PaddlePaddle/PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3/tree/241f8bdfc77a7c7bee915a5057aaee58c235a8d3) | `241f8bdfc77a7c7bee915a5057aaee58c235a8d3` |
| [PaddlePaddle/PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/tree/c5630abae1d940eafe0697512a0325494b02ab42) | `c5630abae1d940eafe0697512a0325494b02ab42` |

`MODELS` in `admin/download_models.py` defines the repository IDs, revisions and
file selection. MinerU's pipeline selection follows its
[3.3.1 downloader](https://github.com/opendatalab/MinerU/blob/mineru-3.3.1-released/mineru/cli/models_download.py),
avoiding unrelated weights in the shared PDF-Extract-Kit repository. The other
three repositories are downloaded in full, including configurations/tokenizers.
The two MinerU revisions preserve those recorded during the original inspection.
The Paddle revisions were verified from the official repositories on 6 October
2026; they have not been compared byte-for-byte with the old machine's caches.

Allow about **4.9 GB for MinerU and 2.1 GB for Paddle**, plus spare disk space for
temporary downloads and uploads. These sizes are based on repository metadata.
The result has the layout the workers already expect:

```text
models/
├── mineru/
│   ├── pipeline/models/...
│   └── vlm/...
└── paddle/official_models/
    ├── PP-DocLayoutV3/...
    └── PaddleOCR-VL-1.6/...
```

The script uses Hugging Face's
[`snapshot_download` with `local_dir`](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder),
which writes real files and keeps its download metadata in `.cache/huggingface/`
inside each destination. If a download is interrupted, rerun the same command
with the same revisions and destination; the library reuses completed files.
Register the folders only after the script reports `Download complete`.
There is no custom retry loop or manifest to maintain.

`--models-dir /path/to/models` changes the parent directory. If you use it, also
update `path` in the corresponding `models/*.yml` before registration. When
changing revisions, use a fresh destination so files removed upstream cannot
remain from an older download. Do not put source PDFs inside the model folders.

The generated `models/mineru/` and `models/paddle/` directories are excluded from
Git. The small `models/*.yml` definitions describe the versioned Azure ML model
assets. Downloading does not upload anything to Azure; the registration step
below uploads these folders. Jobs then use Azure's stored copy of the weights.

## 3. Create the compute cluster

Review `azure/compute.yml` and `admin/setup_compute.py`. Then run:

```sh
python admin/setup_compute.py --apply
```

The helper loads `azure/compute.yml`, reads the workspace details from
`config.json`, and uses the Azure ML Python SDK with your existing `az login`
session to apply the compute definition. It waits for Azure to finish before
reporting success. Without `--apply`, it previews the cluster name, VM size and
node limits locally. Use `--config /path/to/config.json` for another configuration.
The cluster has a zero-node minimum, one-node maximum and a 120-second idle
scale-down. Its system-assigned identity remains part of the compute definition;
the helper assigns no roles to it.

Compute setup is separate from endpoint deployment and submits no parsing job.
Rerunning it reapplies the settings in `azure/compute.yml`.

Azure's SDK uploads PDFs to the existing default datastore, and jobs use that
datastore for their outputs. Keep the workspace's existing access arrangements.
A credential-based datastore can use its stored credentials through authorised
workspace access; an identity-based datastore requires storage access for the
identity being used. If the smoke test reports an access error, have the
workspace administrator resolve the missing access through normal Azure
administration. See [Azure ML data authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-administrate-data-authentication?view=azureml-api-2)
and [batch invocation permissions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-batch-endpoint?view=azureml-api-2).

## 4. Register environments, models and deployments

MinerU uses Azure's image-plus-Conda pattern. Paddle uses a Docker build context
because its official server image does not provide the Conda command required
by Azure's managed Conda build. Paddle's package list remains in `conda.yml`.
Both are supported [Azure environment definitions](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2).

| File | What to edit |
| --- | --- |
| `environments/mineru/environment.yml` | Azure environment name/version and base image |
| `environments/mineru/conda.yml` | Python version and MinerU package versions |
| `environments/paddle/environment.yml` | Azure environment name/version and Docker build folder |
| `environments/paddle/Dockerfile` | Pinned official Paddle server image, Conda installation and client environment creation |
| `environments/paddle/conda.yml` | Python version and Paddle client package versions |
| `workers/start_paddle_client.sh`, `start_paddle_server.sh` | Client Conda activation and server Conda deactivation at job startup |
| `models/mineru.yml`, `models/paddle.yml` | Model asset name/version and local download path |
| `azure/*-command.yml` | Environment version, worker arguments and runtime variables |
| `azure/*-pipeline.yml` | Model version used by that parser's job |

The Paddle Azure asset names are `paddle-vl-models` (model), `paddle_vl_command`
(command component), `paddle_vl_pipeline` (pipeline component), and `paddle-vl`
(deployment). Its environment remains `pdf-paddle`. The local parser selector is
still `--parser paddle`, and colleagues still use `run-paddle`.

Paddle's Dockerfile performs six build steps:

1. `FROM` selects the pinned official Paddle server image.
2. `USER root` selects the container's administrator for installation.
3. `ADD` downloads the fixed Miniforge installer from its official GitHub release.
4. `RUN` installs Conda at `/opt/conda` without interactive prompts.
5. `COPY` puts the local `conda.yml` into the image at `/tmp/client-conda.yml`.
6. `RUN` creates the client environment at `/opt/client` from that YAML, then
   removes Conda package caches. `&&` runs cleanup only after creation succeeds.

These commands run during image construction. Azure jobs use the saved image;
they do not reinstall Conda and the client packages on each invocation. The
Dockerfile creates a separate environment without changing the server's Python
installation. The YAML still explicitly lists the client's dependencies.

Preview the parser deployment or model registration locally; these commands do
not sign in or connect to Azure and do not need the model files:

```sh
python admin/deploy_parsers.py
python admin/register_models.py
```

For initial setup, register both model folders from the repository copy
containing the downloaded files, using your `az login` session:

```sh
python admin/register_models.py --apply
```

Then deploy from any repository copy; local model files are no longer needed:

```sh
python admin/deploy_parsers.py --apply
```

The default is both parsers. For a Paddle-only setup, register only Paddle's
downloaded model folder and select Paddle when deploying:

```sh
python admin/register_models.py --parser paddle --apply
python admin/deploy_parsers.py --parser paddle --apply
```

Use `--parser mineru` for MinerU alone, or `--parser both` for both. A selected
deployment registers only that parser's environment and pipeline and creates its
deployment under the shared endpoint. It does not create or update the other
parser's environment, pipeline or deployment, or remove an existing deployment.
Environment registration is included in `deploy_parsers.py`; it has no separate
helper. The script reads `config.json`, uses your `az login` session, and waits
for the endpoint and deployment operations to finish before reporting success.
Use `--config /path/to/config.json` for another workspace configuration.
Without `--apply`, it previews the selected environments, pipelines, deployments
and compute locally, without connecting to Azure or needing model files.

Deployment does not register models or invoke a job. Model registration is
an explicit administrator action for initial setup or a new weight version.
Jobs reference the already registered version; colleagues never upload weights.
Both clients explicitly select their deployment, so no default deployment is
required. The script uses the Azure ML Python SDK's
[pipeline batch deployment workflow](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-pipeline-deployments?view=azureml-api-2).

Model weights are separate from the environment images. Each command job gets a
pinned model version as a `custom_model` input in `download` mode. This can add
startup time and requires disk space, but changing packages no longer requires
uploading weights. The client submits the PDF folder; model selection is part
of the registered pipeline. The worker configures paths from the downloaded
asset rather than relying on `/opt/models`.

Paddle retains the vendor server's original Python environment. The Dockerfile
creates the layout/export client's Conda environment; activation happens when
the job starts, through two short scripts uploaded with the worker code:

1. `azure/paddle-command.yml` runs `bash start_paddle_client.sh`. This script
   loads Conda's shell commands, activates `/opt/client`, and uses `exec python`
   to replace the launch shell with the Python worker.
2. When the worker needs the server, it starts `start_paddle_server.sh` in a
   separate shell. That script loads Conda's shell commands and deactivates all
   inherited Conda environments, including Conda's own `base` if active. This
   only changes the server launch shell; the client stays in its environment.
3. The server script uses `exec` to replace its shell with the original server
   Python and CLI, supplied through `PADDLE_SERVER_PYTHON` and `PADDLE_SERVER_CLI`
   under `jobs.parse.environment_variables` in `azure/paddle-pipeline.yml`.
   The worker keeps the server's process ID for readiness
   checks and shutdown, including its vLLM child processes.

The client and server run together in one container on the same GPU. The Python
worker no longer rewrites `PATH`, library paths or Conda variables itself;
Conda handles undoing its own activation. Runtime validation must confirm that
the server receives the expected paths and can load its GPU libraries.
The exact image's public registry metadata confirms Python 3.10.16 installed
under `/usr/local`, PaddleOCR 3.6.0 and PaddleX 3.6.1 for the **server**. These are
deliberately separate from the client pins (Python 3.12, PaddleOCR 3.7.0 and
PaddleX 3.7.2). The `/usr/local/bin` paths follow that metadata and the vendor's
[server Dockerfile](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/genai_vllm_server_docker/Dockerfile).
Runtime startup still needs the smoke test; metadata inspection does not run the
image. Missing executables fail with a clear error in `report.json`. The worker
starts the server directly in the Azure job container.

Check **Endpoints → Batch endpoints** for successful provisioning, and
**Environments** for image build status/logs. A registered environment does not
by itself establish that its image has built successfully. Paddle's Docker
build-context definition starts a build when registered; MinerU's image-plus-Conda
definition may wait until first use. The deployment script does not wait for
image build completion. Image builds need access to the base registries, GitHub
installer release and package indexes referenced by the definitions. Check the
build result before submitting PDFs. The existing workspace's registry/build
permissions and compute GPU drivers need verification in the smoke test.

### Apply the Paddle Word-export fix to an existing setup

This revision uses `pdf-paddle:4`, `paddle_vl_command:5` and
`paddle_vl_pipeline:5`. It adds `python-docx==1.2.0` to the client's
`environments/paddle/conda.yml` to fix `ModuleNotFoundError: No module named 'docx'`
when Paddle calls `save_to_word()`. The pinned `paddleocr[doc-parser]` dependencies
do not install this Word-export package automatically.

The Dockerfile also writes a tiny Word document in memory using the client
Python during the build. A missing or broken `python-docx` installation will
fail the build. This check needs no GPU or model weights; full Paddle exports
still need the smoke test.

This change requires a new environment image build. Upload the entire
`environments/paddle/` build folder, including `Dockerfile` and `conda.yml`, and
register it as `pdf-paddle:4`. After its build succeeds, deploy pipeline version 5
so new jobs select the updated image. The model asset, endpoint, deployment name
and compute are unchanged.

The earlier fixes remain included: the Dockerfile installs Conda, and runtime
environment variables are set on the pipeline's `parse` job.

Once the local changes have been reviewed, preview and apply just Paddle:

```sh
python admin/deploy_parsers.py --parser paddle
python admin/deploy_parsers.py --parser paddle --apply
```

If compute and `paddle-vl-models:1` are already registered, do not repeat compute
setup or model download/registration. The latest failed Azure run used
`pdf-paddle:3`: the server handled inference requests and the client saved
Markdown and JSON, then failed on Word export. After building version 4 and
deploying this fix, submit a fresh two-PDF smoke test; the failed job keeps its
old configuration. Confirm that both PDFs have all exports, including `.docx`.

## 5. Smoke-test before handing it to colleagues

This is the first step that intentionally runs paid GPU inference. Use a few
representative PDFs in one flat folder, including several short PDFs and a
longer document. On a colleague-style client installation, run:

```sh
run-mineru ./smoke-pdfs --output ./smoke-mineru
run-paddle ./smoke-pdfs --no-wait
run-paddle --resume runs/RECEIPT_PRINTED_ABOVE.json --output ./smoke-paddle
```

Check that each report covers every submitted filename and that the
exports match the current manual process. MinerU's whole export directory is
retained, including images and visualisation files. Paddle saves JSON, Markdown,
Word and the accompanying assets produced by those export methods.

In a separate small test, include one corrupt PDF and one good PDF. Confirm the
job returns failure, the client downloads any available results and native logs,
and the report does not claim that unconfirmed documents succeeded. The parser
may abort the batch; completing the good PDF is not guaranteed. No document is
automatically retried. Also add a text file and a subfolder with another PDF to
the local input folder; confirm only the PDFs directly inside it are submitted.

Compare elapsed time and GPU memory use with the original manual folder run on
the same PDFs. Native concurrency is enabled, but throughput and memory use have
not been measured for this deployment. Confirm Paddle keeps exports separated by
PDF and still creates merged JSON, Markdown and Word files with their assets.

After `Submitted` prints, stop the local client or close the laptop and retrieve
with `--resume`. Confirm there is only one Azure job. After jobs finish, verify
the cluster returns to zero nodes. Then share the repository and `config.json`.
No model weights, Azure keys or GPU packages are needed on colleagues' machines.

While a batch is starting or running, try submitting through the other parser
command from a colleague's account. Confirm it reports the existing job name and
status without uploading or submitting anything. Confirm `--resume` still works.

## Preserved parser configuration

These direct package versions were read from the working A100 environments:

| Parser | Versions |
| --- | --- |
| MinerU | MinerU 3.3.1; vLLM 0.21.0; PyTorch 2.11.0; Transformers 4.57.6 |
| Paddle | PaddleOCR 3.7.0; PaddleX 3.7.2; PaddlePaddle GPU 3.3.1, CUDA 13.0 wheels |

MinerU keeps the existing flags:

```sh
mineru -p INPUT -o OUTPUT -m ocr -b hybrid-engine --effort high \
  -l en -f true -t true --image-analysis true
```

This command receives the entire input folder. MinerU starts/stops its own API
server and schedules document tasks concurrently (three by default in the pinned
version). We do not override that concurrency or manage a separate MinerU server.
Its behaviour is based on
[MinerU's pinned 3.3.1 CLI](https://github.com/opendatalab/MinerU/blob/mineru-3.3.1-released/docs/en/usage/cli_tools.md).

Paddle keeps `pipeline_version="v1.6"`, `vllm-server`,
`PaddleOCR-VL-1.6-0.9B`, `use_doc_unwarping=False`, chart recognition,
OCR of image blocks, content formatting, and GPU 0. Restructuring merges tables,
relevels titles and concatenates pages before JSON/Markdown/Word export.
`workers/paddle_batch.py` constructs one pipeline and calls `predict_iter()` once
with all PDF paths, preserving native cross-document batching. It gathers and
exports one document at a time as results arrive, rather than accumulating the
whole batch in memory. The output grouping loop does not serialize PDF inference.
The server uses the exact vendor image digest inspected on the existing A100,
recorded in `environments/paddle/environment.yml`, with its existing default memory
settings. Only local model paths and server startup coordination are added.

Direct package pins and fixed model revisions reduce drift; transitive dependencies
and the tagged MinerU base image are not a complete environment lock. After the
smoke test, retain the built image digest and environment/model version references.
Reuse that registered environment for jobs instead of rebuilding on every run.

## Failures, time limits and result layout

Before a new submission, the shared client uses
`ml.batch_endpoints.list_jobs(endpoint_name=...)` to check both deployments on the
configured endpoint. Any unfinished job blocks submission, including jobs still
queued, preparing or cancelling. A failure to list jobs also stops submission.
The check happens before the SDK upload and receipt creation; `--resume` skips it.
This is a best-effort check, not a lock. Jobs can still queue if two submissions
overlap before Azure lists either job, another client skips the check, or unrelated
jobs use the cluster. It does not disable Azure's queue or enforce a spending cap.

The client copies only the PDFs directly inside the selected local folder into a
temporary folder. Azure's SDK uploads that PDF-only folder to the default datastore
during endpoint invocation. Azure chooses the cloud paths; job details expose the input
and output locations. Every invocation gets a new random run ID. Receipts in the
local `runs/` folder record the chosen Azure job name before invocation, allowing
recovery if its HTTP response is interrupted. `--resume` retrieves the same job;
it does not resubmit PDFs. A receipt marked `submitting` can mean either an
failed local copy, interrupted upload or a lost submission response. Check the recorded job name in
Azure before submitting the folder again if `--resume` cannot find it.

The local source folder may contain other files and subfolders: these are ignored,
including `.amlignore` and `.gitignore` files. Selection is not recursive. The clean
temporary upload folder contains only the selected PDFs, with their names preserved;
the Azure worker still expects that flat PDF-only input. Filenames must be distinct
without letter case, so outputs can be downloaded without case collisions. Use
filenames suitable for the computers that will download the results.

Local temporary copies need disk space for the selected PDFs. They are removed
when submission returns or fails, without changing the originals. The receipt
retains the original PDF filenames. Resume uses the existing cloud job and does
not copy or upload local PDFs again. There is no per-document inference subprocess.

`workers/run.py` calls MinerU once for the folder, or starts Paddle's vendor vLLM
server once and runs the streaming Paddle client. The worker does not retry PDFs,
restart the server on individual failures, or impose a per-PDF deadline. Paddle's
server startup still has a 20-minute readiness limit. The Azure command job keeps
its 24-hour execution limit (`86400` seconds) in `azure/*-pipeline.yml`. Native
parser/network timeouts still apply. MinerU's task-result polling timeout is also
set to 24 hours so its shorter default does not cut off long documents; it is not
a per-PDF cancellation mechanism. Execution time does not include a promise
about queue/startup time.

Outputs are written directly to the mounted results directory:

- `parser.log`: native parser messages and the worker traceback on an exception.
  Native libraries may also write to Azure's job logs.
- `server.log`: Paddle's separate vLLM server output (Paddle only).
- `documents/`: every generated export. MinerU uses its native document directory
  names; Paddle writes under `documents/PDF_STEM/` with JSON, Markdown, Word and assets.
- `report.json`: the submitted PDFs, their confirmation status, and any batch error.

Paddle updates the report after all pages and required exports for a document are
saved. If its native iterator raises, already confirmed documents remain
`succeeded`; the remainder stay `unconfirmed`. Some may have partial output.
MinerU's public CLI supplies an overall exit status and human-readable errors,
not a structured per-PDF manifest. After a successful CLI exit, the worker also
checks for each PDF's Markdown and JSON exports, since MinerU can skip inputs it
does not recognise. Missing exports remain `unconfirmed`. MinerU can shorten long
output names; such documents may need manual checking even after a successful run.
On any nonzero exit, all PDFs remain `unconfirmed`. Existing files alone do not
prove completion, so check MinerU's native log before selecting PDFs to rerun.
An unreadable input can abort MinerU's initial scan, and a native Paddle exception
can stop its entire batch. Neither is automatically restarted by this worker.

A cancelled job, lost node or whole-job timeout can leave an incomplete report.
The client does not call that success. It downloads available files even for a
failed Azure job. Resume only retrieves the same job; it never retries inference.
The client uses `ml.jobs.download(..., output_name="results")`, falling back to
the single child worker's output if the parent has no report. It flattens Azure's
`named-outputs/results/` directory into the requested local destination. A fresh
temporary download prevents an old local report from disguising missing results.
Completion checks do not assess OCR quality.

## Stored inputs and results

This project does not configure automatic deletion or delete uploaded PDFs and
cloud results after a run. Completed runs, failed runs and partial uploads remain
in storage for manual cleanup. Storage usage and its associated cost can grow.

Setup does not read or modify the storage account's existing deletion policies.
Any policies already applied outside this project can still affect stored files.
Automatic cleanup can be considered separately later.

## Increase capacity or update the parsers

You can increase the maximum later, subject to quota/capacity:

```sh
az extension add --name ml --upgrade
az ml compute update --name pdf-parsers-a100 --max-instances 2 \
  --resource-group continuous_review_ms_and_ucl --workspace-name EPPI_DEV \
  --subscription 56539498-d3d8-4a3b-92f4-f3b098a11d1e
```

Also update `max_instances` in `azure/compute.yml` so a later setup run preserves
the new maximum. Keep `min_instances: 0`. Two nodes allow two one-GPU jobs at once;
each batch uses its parser's native batching/concurrency within one GPU. The current CLI busy check
allows only one unfinished endpoint job, so it must also be adjusted before
colleagues can use that extra concurrency through the CLI. Scheduling is controlled by
Azure; this does not implement a strict FIFO queue. The
[compute CLI](https://learn.microsoft.com/en-us/cli/azure/ml/compute?view=azure-cli-latest#az-ml-compute-update)
supports this update without replacing the endpoint.

### Changing environments or models

To change only MinerU dependencies:

1. Edit `environments/mineru/conda.yml` and bump `version` in its `environment.yml`
   (currently `2`).
2. Update `environment` in `azure/mineru-command.yml` to that new version and
   bump the command's version. Bump the pipeline version in
   `azure/mineru-pipeline.yml` too, since it includes that command definition.
3. Update `component` in `azure/mineru-deployment.yml` to the new pipeline version.
4. Preview, then apply the parser deployment. This also registers the changed
   environment and pipeline:

```sh
python admin/deploy_parsers.py --parser mineru
python admin/deploy_parsers.py --parser mineru --apply
```

The endpoint name comes from `config.json`. Substitute `paddle` for a Paddle
update (its environment is currently version `4`). Paddle's package list is still
`environments/paddle/conda.yml`; change its Dockerfile only for the server image
or the Conda setup steps. Existing jobs retain their submitted configuration; smoke-test the new
deployment before sharing it with colleagues. Keep the prior versions for rollback.

To change weights, update the repository/revision in `admin/download_models.py`,
then download to a fresh parent directory with
`python admin/download_models.py --parser PARSER --models-dir /path/to/new-models`.
Update `path` in `models/PARSER.yml` to `/path/to/new-models/PARSER` and bump its
`version`, then register with
`python admin/register_models.py --parser PARSER --apply`. Update the model input
reference and bump the version in `azure/PARSER-pipeline.yml`, then update the
deployment's pipeline reference and run
`python admin/deploy_parsers.py --parser PARSER --apply`. The environment can
stay the same. Code or worker-argument changes need new command and pipeline versions; dependency
changes also need a new environment version. Registered versions are not edited
in place. Model registration only uploads weights; `deploy_parsers.py` applies
the environment, pipeline and endpoint deployment. Neither submits a parsing job.

## Local checks and troubleshooting

```sh
python -m pip install '.[admin]'
python -m unittest discover -s tests -v
```

Tests inject parser failures and interrupted submissions and mock Azure services.
Definition checks use the Azure SDK schema loaders. They make no network calls
and do not establish that Azure deployment or GPU parsing works. They were run
with Python 3.12 and `azure-ai-ml` 1.35.1.

| Symptom | Check |
| --- | --- |
| 403 uploading/downloading | Signed-in tenant/account, workspace permissions, default datastore authentication and storage network access; resolve access through the workspace administrator |
| 403 invoking | Workspace role and tenant; endpoint/deployment names in `config.json` |
| Parser endpoint is busy | The named job has not finished; try later or use `--resume` for an existing submission |
| Cannot list endpoint jobs | Workspace job-read permissions, tenant and connectivity; the client will not submit when the check fails |
| Job cannot mount storage | Datastore credentials or the job/compute identity's storage access, plus storage networking; inspect Azure job logs |
| Job stays queued | Cluster provisioning, quota, regional A100 capacity and other active jobs |
| Environment fails to build | Build logs, Conda/pip resolution, registry/package-index access and free build space |
| Paddle build reports `conda: not found` | Version 2 used the server image without installing Conda; environment versions 3 and later install it in the Dockerfile |
| Paddle reports `KeyError: 'PADDLE_SERVER_PYTHON'` | Pipeline versions 4 and later set runtime variables under `jobs.parse.environment_variables` |
| Paddle reports `No module named 'docx'` | Build environment version 4 with `python-docx` in the client, then deploy pipeline version 5 and submit a fresh job |
| Model registration fails | Local weight directory, model version, available disk and workspace storage access |
| Server fails to start | Paddle `server.log` or MinerU `parser.log`, image/driver compatibility and downloaded model paths |
| Parser fails or PDFs are unconfirmed | `report.json` and `parser.log`; inspect partial exports before selecting PDFs to resubmit |
| No report after a terminal job | Azure setup/job logs or missing/deleted output blobs; a stale local report is not accepted |
| Submission response was lost | Use its receipt with `--resume`; inspect the recorded job name in Azure before submitting again |
| Receipt stays `submitting` | Upload and submission share one SDK call; check the recorded job name before submitting again |

Retaining the endpoint does not require an always-running GPU. There is no charge
for the batch endpoint itself; compute used for jobs, storage, registry/builds
and relevant data transfer can still incur charges. Your existing interactive
`sam-a100` has its own billing/lifecycle, independent of this cluster. See
[batch endpoint costs](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-batch?view=azureml-api-2).
