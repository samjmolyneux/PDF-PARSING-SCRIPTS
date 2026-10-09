"""One GPU job: run a native parser batch and retain its exports and logs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
import traceback
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any, BinaryIO

if __package__:
    from .paddle_batch import parse_pdfs as parse_paddle_pdfs
else:
    from paddle_batch import parse_pdfs as parse_paddle_pdfs


def write_report(output: Path, report: dict[str, Any]) -> None:
    temporary = output / "report.json.tmp"
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output / "report.json")


def stop_paddle_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    # The Paddle server can leave vLLM children after its parent exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def paddle_server_command() -> tuple[list[str], str]:
    python, cli = os.environ["PADDLE_SERVER_PYTHON"], os.environ["PADDLE_SERVER_CLI"]
    for file in (python, cli):
        if not Path(file).is_file():
            msg = (
                f"Base image server executable missing: {file}; "
                "check azure/paddle-pipeline.yml"
            )
            raise FileNotFoundError(msg)
    return [
        "bash",
        str(Path(__file__).with_name("start_paddle_server.sh")),
        python,
        cli,
        "genai_server",
        "--model_name",
        "PaddleOCR-VL-1.6-0.9B",
        "--model_dir",
        str(
            Path(os.environ["PADDLE_PDX_CACHE_HOME"])
            / "official_models/PaddleOCR-VL-1.6"
        ),
        "--host",
        "127.0.0.1",
        "--port",
        "8118",
        "--backend",
        "vllm",
    ], "http://127.0.0.1:8118/health"


def start_paddle_server(
    log: BinaryIO, timeout: float = 1200
) -> subprocess.Popen[bytes]:
    command, health = paddle_server_command()
    # Fixed launcher and image-configured executables; arguments stay separate.
    process = subprocess.Popen(  # noqa: S603
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                msg = f"Model server exited ({process.returncode}); see server.log"
                raise RuntimeError(msg)
            try:
                # paddle_server_command supplies a fixed loopback HTTP health URL.
                with urllib.request.urlopen(health, timeout=5) as response:  # noqa: S310
                    if response.status == HTTPStatus.OK:
                        return process
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        msg = "Model server did not become ready; see server.log"
        raise TimeoutError(msg)
    except BaseException:
        stop_paddle_server(process)
        raise


def process_batch(parser: str, input_dir: Path, output: Path) -> int:
    input_dir, output = input_dir.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    report = {
        "parser": parser,
        "total": len(pdfs),
        "finished": False,
        "fatal_error": None,
        "documents": [
            {
                "input": p.name,
                "status": "unconfirmed",
                "output": f"documents/{p.stem}" if parser == "paddle" else "documents",
            }
            for p in pdfs
        ],
    }
    paddle_server = None
    write_report(output, report)
    with (output / "parser.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            if not pdfs:
                msg = "No PDFs in the submitted input."
                raise ValueError(msg)
            if any(
                p.is_symlink() or not p.is_file() or p.suffix.lower() != ".pdf"
                for p in input_dir.iterdir()
            ):
                msg = "Use a flat folder containing only PDFs; copy linked files first."
                raise ValueError(msg)
            if len({p.stem.casefold() for p in pdfs}) != len(pdfs):
                msg = "PDF filenames must be distinct without letter case."
                raise ValueError(msg)
            os.environ["PYTHONNOUSERSITE"] = "1"
            exports = output / "documents"
            exports.mkdir(exist_ok=True)
            if parser == "mineru":
                # One folder call. MinerU owns its server, task scheduling and cleanup.
                # Fixed command/options; resolved paths are separate arguments.
                # Resolve the installed MinerU executable through the image's PATH.
                subprocess.run(  # noqa: S603
                    [  # noqa: S607
                        "mineru",
                        "-p",
                        str(input_dir),
                        "-o",
                        str(exports),
                        "-m",
                        "ocr",
                        "-b",
                        "hybrid-engine",
                        "--effort",
                        "high",
                        "-l",
                        "en",
                        "-f",
                        "true",
                        "-t",
                        "true",
                        "--image-analysis",
                        "true",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                # A zero exit can still omit unrecognised inputs. Check exports
                # only after success; partial files from a failed run prove nothing.
                for pdf, item in zip(pdfs, report["documents"], strict=True):
                    found = {
                        p.suffix.lower()
                        for p in (exports / pdf.stem).rglob("*")
                        if p.is_file() and p.stat().st_size
                    }
                    if {".md", ".json"} <= found:
                        item["status"] = "succeeded"
                    else:
                        item["error"] = (
                            "Expected exports not found; inspect parser.log and "
                            "documents/ (MinerU may rename long filenames)."
                        )
            else:
                documents = {item["input"]: item for item in report["documents"]}
                with (output / "server.log").open("ab", buffering=0) as server_log:
                    paddle_server = start_paddle_server(server_log)
                    with (
                        contextlib.redirect_stdout(log),
                        contextlib.redirect_stderr(log),
                    ):
                        for pdf in parse_paddle_pdfs(pdfs, exports):
                            documents[pdf.name]["status"] = "succeeded"
                            write_report(output, report)
            report["finished"] = True
            if any(item["status"] != "succeeded" for item in report["documents"]):
                msg = "Parser did not confirm every PDF; see parser.log."
                raise RuntimeError(msg)
        # Record any parser failure while preserving partial exports and cleanup.
        except Exception as error:  # noqa: BLE001
            report["fatal_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc(file=log)
            print(report["fatal_error"], flush=True)
        finally:
            stop_paddle_server(paddle_server)
            write_report(output, report)
    failures = sum(item["status"] != "succeeded" for item in report["documents"])
    print(
        f"Finished: {len(pdfs) - failures}/{len(pdfs)} PDFs confirmed; "
        f"{failures} unconfirmed. See parser.log.",
        flush=True,
    )
    return 1 if failures or report["fatal_error"] else 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--parser", choices=["mineru", "paddle"], required=True)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    raise SystemExit(process_batch(args.parser, args.input, args.output))
