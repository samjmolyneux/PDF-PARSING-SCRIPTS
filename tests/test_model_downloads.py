"""Check image download destinations against both Paddle consumers, without a GPU."""

import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from workers import run as worker

ROOT = Path(__file__).resolve().parents[1]


class ModelDownloadTests(unittest.TestCase):
    def test_pinned_downloads_match_server_and_layout_client_paths(self):
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
            patch.dict(
                sys.modules,
                {
                    "paddleocr": SimpleNamespace(PaddleOCRVL=factory),
                },
            ),
            patch.dict(
                os.environ,
                {
                    "PADDLE_PDX_CACHE_HOME": cache,
                    "PADDLE_SERVER_PYTHON": sys.executable,
                    "PADDLE_SERVER_CLI": __file__,
                },
            ),
        ):
            downloads = {}
            for line in lines:
                if line.startswith("RUN /opt/client/bin/hf download "):
                    args = shlex.split(os.path.expandvars(line.removeprefix("RUN ")))
                    # Download whole repositories: no filename filters or extra options.
                    self.assertEqual(len(args), 7)
                    downloads[args[2]] = (
                        args[args.index("--revision") + 1],
                        Path(args[args.index("--local-dir") + 1]),
                    )
            command, _ = worker.server_command()
            list(worker.parse_pdfs([], Path(cache) / "output"))

        server_dir = Path(command[command.index("--model_dir") + 1])
        layout_dir = Path(factory.call_args.kwargs["layout_detection_model_dir"])
        self.assertEqual(
            downloads,
            {
                "PaddlePaddle/PP-DocLayoutV3": (
                    "241f8bdfc77a7c7bee915a5057aaee58c235a8d3",
                    layout_dir,
                ),
                "PaddlePaddle/PaddleOCR-VL-1.6": (
                    "c5630abae1d940eafe0697512a0325494b02ab42",
                    server_dir,
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
