# Testing

Run these commands from the repository in an active Python 3.10, 3.11 or 3.12
environment. Install the package and test tools first:

```bash
python -m pip install --upgrade pip
python -m pip install . --group test
```

## Where tests live

The offline test directories mirror the code and configuration they exercise:

| Directory            | What it tests                                                                         |
| -------------------- | ------------------------------------------------------------------------------------- |
| `test_src/`          | The local client, installed Paddle/MinerU commands, PDF discovery and result reports. |
| `test_admin/`        | Compute setup, parser deployment, previews and the admin CLIs.                        |
| `test_workers/`      | Batch execution, Paddle exports, the Paddle server and the worker CLI.                |
| `test_azure/`        | Azure YAML definitions and consistency between asset references.                      |
| `test_environments/` | Environment build contexts and Paddle model download paths.                           |
| `test_shared/`       | Checks comparing client and worker behavior for the same inputs.                      |
| `integration/`       | Real Azure runs, selected explicitly.                                                 |
| `data/`              | Licensed PDF fixtures and their provenance.                                           |

Within `test_workers/`, `test_run.py` covers behavior shared by both parsers;
`test_mineru_run.py` and `test_paddle_run.py` cover their specific batch behavior.
`test_paddle_server.py` covers server launch, readiness and shutdown.

`conftest.py` contains the shared fixtures: repository paths, CLI execution,
temporary PDF folders, reading worker reports and restoring environment variables.
Mocks specific to a test file stay in that file.

## Offline tests

```bash
python -m pytest
```

These tests exercise the client, workers, admin scripts and Azure YAML definitions
with simulated Azure and parser responses. Network access is disabled during
these tests. They require neither Azure credentials nor a GPU.

Coverage measures statements and branches in `src/pdf_parsers`, `admin` and
`workers`. Coverage below **90%** fails the run. To inspect an HTML report:

```bash
python -m pytest --cov-report=term-missing --cov-report=html
```

Open `htmlcov/index.html`. To test the built wheel in separate environments for
all three Python versions, install those Python interpreters and tox, then run:

```bash
python -m pip install "tox>=4.22,<5"
python -m tox
```

To use just Python 3.12:

```bash
python -m tox -e py312
```

GitHub Actions runs the offline suite on Python 3.10, 3.11 and 3.12 on Linux,
macOS and Windows. Each job uses uv to install a non-editable build of the package
and the test dependencies, then runs pytest directly. It saves each coverage
report as a workflow artifact. The Ubuntu/Python 3.12 default-branch run publishes
the README coverage badge to the `badges` branch, following Flowde's approach.

## Azure integration tests

Integration tests submit the [two sample papers](data/README.md) to your existing
deployments. They use your normal client authentication and the repository's
`config.json`. They consume Azure compute and may take a long time.

```bash
python -m pytest -m integration --no-cov -s
```

Paddle runs first, then MinerU. A shared fixture submits one batch per parser
using the normal blocking client and waits for completion. Parameterised tests
then check each PDF's exports, page count, selected text and table row separately.
They reuse the downloaded batch within that test run, so additional checks do not
submit additional jobs. Results stay in `results/integration/`.

Run just one parser:

```bash
python -m pytest -m integration --no-cov -s -k paddle
```

```bash
python -m pytest -m integration --no-cov -s -k mineru
```

To run both parsers simultaneously, run the two commands above in separate
terminals. Each batch uses one A100 node. Apply the two-node cluster limit once
using your Azure CLI login, and reinstall the local client to update its busy check:

```bash
python admin/setup_compute.py --apply
python -m pip install . --group test
```

Select another workspace configuration:

```bash
python -m pytest -m integration --no-cov -s --azure-config ./other-workspace.json
```

The `integration` marker excludes these tests from ordinary pytest, tox and
GitHub Actions runs. `--no-cov` disables the local coverage requirement for these
tests: the parser workers execute remotely. `-s` displays client progress and
allows interactive authentication.

These tests exercise the currently deployed versions. To validate environment
or worker changes, deploy the changed versions using the admin scripts, confirm
the image build, then run the integration tests. The tests do not deploy anything,
retry jobs, or implement their own timeout or recovery mechanism.

The content checks catch obvious omissions and table errors; they are not a full
OCR quality benchmark. Review representative real-world outputs when changing
parser or model versions.

## Writing tests

Keep tests simple. Use `pytest.mark.parametrize` when the same workflow needs
different inputs, statuses or expected results. Each case should run independently
with fresh fixtures and a useful case name, rather than using loops or `subTest`
to exercise multiple cases inside one test. Loops for constructing a batch or
inspecting the files produced by one workflow are fine.

Place new tests under the directory matching the code they exercise. Keep Paddle
and MinerU cases parameterised together when they test the same workflow; use
`test_shared/` when a test deliberately compares multiple parts of the project.
