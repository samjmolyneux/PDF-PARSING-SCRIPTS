"""One GPU job: run a native parser batch and retain its exports and logs."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import urllib.request

if __package__:
    from .paddle_batch import parse_pdfs
else:
    from paddle_batch import parse_pdfs


def configure_models(parser, models, output):
    models = models.resolve()
    required = ("pipeline", "vlm") if parser == "mineru" else (
        "official_models/PaddleOCR-VL-1.6", "official_models/PP-DocLayoutV3")
    for directory in required:
        if not (models / directory).is_dir() or not any((models / directory).rglob("*")):
            raise FileNotFoundError(f"Missing/empty model directory: {models / directory}")
    if parser == "mineru":
        config = output / "mineru-runtime.json"
        config.write_text(json.dumps({"models-dir": {
            name: str(models / name) for name in required}}, indent=2) + "\n", encoding="utf-8")
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config)
        os.environ["MINERU_MODEL_SOURCE"] = "local"
    else:
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(models)
    os.environ["PYTHONNOUSERSITE"] = "1"


def write_report(output, report):
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")


def stop_process(process):
    if process is None:
        return
    # The Paddle server can leave vLLM children after its parent exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def server_command():
    python, cli = os.environ["PADDLE_SERVER_PYTHON"], os.environ["PADDLE_SERVER_CLI"]
    for file in (python, cli):
        if not Path(file).is_file():
            raise FileNotFoundError(f"Base image server executable missing: {file}; check azure/paddle-command.yml")
    return [
        "bash", str(Path(__file__).with_name("start_paddle_server.sh")),
        python, cli, "genai_server", "--model_name", "PaddleOCR-VL-1.6-0.9B",
        "--model_dir", str(Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models/PaddleOCR-VL-1.6"),
        "--host", "127.0.0.1", "--port", "8118", "--backend", "vllm",
    ], "http://127.0.0.1:8118/health"


def start_server(log, timeout=1200):
    command, health = server_command()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Model server exited ({process.returncode}); see server.log")
            try:
                with urllib.request.urlopen(health, timeout=5) as response:
                    if response.status == 200:
                        return process
            except (OSError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError("Model server did not become ready; see server.log")
    except BaseException:
        stop_process(process)
        raise


def process_batch(parser, input_dir, output, models):
    input_dir, output = input_dir.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    report = {
        "parser": parser, "total": len(pdfs), "finished": False, "fatal_error": None,
        "documents": [{"input": p.name, "status": "unconfirmed",
                       "output": f"documents/{p.stem}" if parser == "paddle" else "documents"}
                      for p in pdfs],
    }
    server = None
    write_report(output, report)
    with (output / "parser.log").open("w", encoding="utf-8", buffering=1) as log:
        try:
            if not pdfs:
                raise ValueError("No PDFs in the submitted input.")
            if any(p.is_symlink() or not p.is_file() or p.suffix.lower() != ".pdf"
                   for p in input_dir.iterdir()):
                raise ValueError("Use a flat folder containing only PDFs; copy linked files first.")
            if len({p.stem.casefold() for p in pdfs}) != len(pdfs):
                raise ValueError("PDF filenames must be distinct without letter case.")
            configure_models(parser, models, output)
            exports = output / "documents"
            exports.mkdir(exist_ok=True)
            if parser == "mineru":
                # One folder call. MinerU owns its server, task scheduling and cleanup.
                subprocess.run([
                    "mineru", "-p", str(input_dir), "-o", str(exports),
                    "-m", "ocr", "-b", "hybrid-engine", "--effort", "high", "-l", "en",
                    "-f", "true", "-t", "true", "--image-analysis", "true",
                ], stdout=log, stderr=subprocess.STDOUT, check=True)
                # A zero exit can still omit unrecognised inputs. Check exports
                # only after success; partial files from a failed run prove nothing.
                for pdf, item in zip(pdfs, report["documents"]):
                    found = {p.suffix.lower() for p in (exports / pdf.stem).rglob("*")
                             if p.is_file() and p.stat().st_size}
                    if {".md", ".json"} <= found:
                        item["status"] = "succeeded"
                    else:
                        item["error"] = "Expected exports not found; inspect parser.log and documents/ (MinerU may rename long filenames)."
            else:
                documents = {item["input"]: item for item in report["documents"]}
                with (output / "server.log").open("ab", buffering=0) as server_log:
                    server = start_server(server_log)
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        for pdf in parse_pdfs(pdfs, exports):
                            documents[pdf.name]["status"] = "succeeded"
                            write_report(output, report)
            report["finished"] = True
            if any(item["status"] != "succeeded" for item in report["documents"]):
                raise RuntimeError("Parser did not confirm every PDF; see parser.log.")
        except Exception as error:
            report["fatal_error"] = f"{type(error).__name__}: {error}"
            traceback.print_exc(file=log)
            print(report["fatal_error"], flush=True)
        finally:
            stop_process(server)
            write_report(output, report)
    failures = sum(item["status"] != "succeeded" for item in report["documents"])
    print(f"Finished: {len(pdfs) - failures}/{len(pdfs)} PDFs confirmed; {failures} unconfirmed. See parser.log.", flush=True)
    return 1 if failures or report["fatal_error"] else 0


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--parser", choices=["mineru", "paddle"], required=True)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--models", type=Path, required=True, help="Downloaded Azure ML model asset directory")
    args = cli.parse_args()
    raise SystemExit(process_batch(args.parser, args.input, args.output, args.models))
