"""Check image download destinations against Paddle consumers, without a GPU."""

import os
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from workers import run as worker

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "repository, consumer",
    [
        ("PaddlePaddle/PP-DocLayoutV3", "layout"),
        ("PaddlePaddle/PaddleOCR-VL-1.6", "server"),
    ],
)
def test_pinned_model_download_matches_consumer_path(repository, consumer):
    dockerfile = (ROOT / "environments/paddle/Dockerfile").read_text()
    lines = dockerfile.replace("\\\n", " ").splitlines()
    cache = next(
        line.split("=", 1)[1]
        for line in lines
        if line.startswith("ENV PADDLE_PDX_CACHE_HOME=")
    )
    pipeline = MagicMock()
    pipeline.predict_iter.return_value = iter([])
    factory = MagicMock(return_value=pipeline)
    with (
        patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PaddleOCRVL=factory)}),
        patch.dict(
            os.environ,
            {
                "PADDLE_PDX_CACHE_HOME": cache,
                "PADDLE_SERVER_PYTHON": sys.executable,
                "PADDLE_SERVER_CLI": __file__,
            },
        ),
    ):
        commands = [
            shlex.split(os.path.expandvars(line.removeprefix("RUN ")))
            for line in lines
            if line.startswith("RUN /opt/client/bin/hf download ")
        ]
        downloads = {args[2]: args for args in commands}
        command, _ = worker.server_command()
        list(worker.parse_pdfs([], Path(cache) / "output"))

    paths = {
        "server": Path(command[command.index("--model_dir") + 1]),
        "layout": Path(factory.call_args.kwargs["layout_detection_model_dir"]),
    }
    assert set(downloads) == {
        "PaddlePaddle/PP-DocLayoutV3",
        "PaddlePaddle/PaddleOCR-VL-1.6",
    }
    args = downloads[repository]
    # Each model is pinned as a whole repository and matches its runtime consumer.
    assert "--include" not in args
    assert "--exclude" not in args
    assert re.fullmatch(r"[0-9a-f]{40}", args[args.index("--revision") + 1])
    assert Path(args[args.index("--local-dir") + 1]) == paths[consumer]
