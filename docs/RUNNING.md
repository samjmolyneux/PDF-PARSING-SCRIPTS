# Run parsers

Submit PDFs to an existing Azure ML deployment and download the results. You
need access to the workspace; the supplied `config.json` uses the team's
`EPPI_DEV` deployment.

## Quick start

### 1. Install

**Requires Python 3.10 or later.** Activate an environment using your preferred
tool, such as Conda or `venv`. For example, to create and activate a new Conda
environment:

```bash
conda create --name pdf-parsers python=3.12 -y
conda activate pdf-parsers
```

Download the repository and install the client:

```bash
git clone https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS.git
cd PDF-PARSING-SCRIPTS
python -m pip install .
```

### 2. Run a parser

From the repository directory, choose **one** of the following commands. Replace
`./pdfs` with your PDF folder; put paths containing spaces in quotes.

**PaddleOCR-VL**

```bash
run-paddle "./pdfs" --output "./paddle-results"
```

**MinerU**

```bash
run-mineru "./pdfs" --output "./mineru-results"
```

The command uses your existing Azure CLI sign-in if available; otherwise it
opens your browser to sign in. You do not need to install Azure CLI for this
quick start.

The client uploads PDFs directly inside your chosen folder, waits for Azure to
finish, and downloads the exports, logs and `report.json` to your output folder.
It ignores other files and subfolders, including PDFs in subfolders. Starting
the GPU can take a while.

**That is all you need for a normal run.** Choose a fresh output folder for each
batch so that results from different batches stay separate.

## Default output folder

Without `--output`, results go to `./results/RUN_ID/` relative to your current
directory, with a separate folder for each batch.

## Submit now, download later

PDF parsing can take a long time. If you do not want to leave the command
waiting or keep your computer running, use `--no-wait` to upload the PDFs and
submit the job, then download the results later:

```bash
run-paddle "./pdfs" --no-wait
```

The command returns once the PDFs have been uploaded and Azure has accepted the
job; parsing continues in Azure. **Wait for `Submitted` before closing your
laptop.** The client prints a ready-to-copy `Later:` command for retrieving the
results, along with the path to a small receipt file in `./runs/`. Keep that
file: it identifies your job and its workspace.

To download the results later, use the printed command. For example, replace the
receipt path below with your actual receipt:

```bash
run-paddle --resume "./runs/paddle-YOUR-RUN-ID.json" --output "./paddle-results"
```

`--resume` connects to the existing job. If the job has finished, it downloads
the results; otherwise it waits, then downloads them. It does not upload the
PDFs again or retry parsing. You can download the results again while the files
remain in Azure storage.

Use `run-mineru` for a MinerU receipt. Resume reads the workspace details from
the receipt, so you do not need `--config`. A custom output folder is not saved
in the receipt: supply `--output` again if you want one.

### If your local process stops

You can also retrieve results after an interruption, even if you started the job
**without `--no-wait`**. Every submitted job has a receipt in `./runs/`. Once
Azure has accepted the job, closing the terminal, losing your connection or
pressing **Ctrl+C** does not cancel processing. Use `--resume` with that receipt
to reconnect and download the results.

To cancel processing itself, cancel the job in Azure ML Studio.

## Use a different workspace

Get a configuration file from that workspace's administrator and pass its path:

```bash
run-paddle "./pdfs" --config "./other-workspace.json" --output "./paddle-results"
```

The same option works with `run-mineru`. Without `--config`, the command reads
`config.json` from your current directory. This is why the quick start runs from
the repository directory.

The file specifies the subscription, resource group, workspace, tenant and
endpoint. See the [configuration example](DEPLOYMENT.md#configure-the-workspace)
if you need to prepare one. The chosen parser must already be deployed there.

## Sign-in options

The default tries your existing `az login` session, then falls back to browser
sign-in if the CLI is missing or cannot sign you in. Choose one of the following
flags to select a method explicitly. Each also works with MinerU and `--resume`.

### Open the browser directly

```bash
run-paddle "./pdfs" --browser-login
```

Use this to skip the existing Azure CLI session and sign in through the browser.

### Sign in with a device code

```bash
run-paddle "./pdfs" --device-code
```

The client prints a website and code. Open the website in a browser and enter
the code; this is useful when your terminal cannot open a browser itself.

### Require an existing Azure CLI session

If Azure CLI is installed, first sign in:

```bash
az login
```

Then require that session when submitting:

```bash
run-paddle "./pdfs" --az-login
```

With `--az-login`, a sign-in error stops the command instead of opening a
browser. All methods use your own account and its workspace permissions.

## Results and unsuccessful PDFs

The output directory contains:

| File or directory | What you get                                                |
| ----------------- | ----------------------------------------------------------- |
| `documents/`      | All exports generated by the parser, organised by document. |
| `report.json`     | A status for each input PDF and any batch error.            |
| `parser.log`      | Parser messages and errors.                                 |
| `server.log`      | Paddle's inference server log; Paddle only.                 |

The client prints `Complete: N/N PDFs succeeded.` only when the job and report
confirm the whole batch. If processing fails, the client downloads any available
results and reports the batch as incomplete.

`unconfirmed` means the worker could not establish that a PDF finished; the PDF
may still have partial or complete exports.

## Troubleshooting

| Message or situation                   | What to do                                                                                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Command not found                      | Activate the environment where you installed the package.                                                                                  |
| Cannot find `config.json`              | Run from the repository directory or supply `--config`.                                                                                    |
| No PDFs found                          | Check the folder path. The client does not search subfolders.                                                                              |
| Parser endpoint is busy                | Two jobs are unfinished. Try later; use `--resume` for your own existing job.                                                              |
| Access denied / 403                    | Check the account and tenant, then ask the workspace administrator to check access.                                                        |
| Upload or submission was interrupted   | Keep the receipt and try `--resume`. If Azure cannot find the job, check its recorded name in Studio before resubmitting.                  |
| Job failed or no report was downloaded | Open the job in Azure ML Studio and inspect its logs, including the `parse` child job. Early setup failures may produce no parser outputs. |

The busy check allows two unfinished jobs across both parsers and all users of
the endpoint. It is a best-effort check: simultaneous submissions can still queue
in Azure. This project does not automatically delete uploaded PDFs or cloud results.

For the complete command options:

```bash
run-paddle --help
run-mineru --help
```
