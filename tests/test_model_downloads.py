"""Download setup contracts, without transferring weights or contacting Azure."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from admin import download_models
from workers import run as worker


class ModelDownloadTests(unittest.TestCase):
    def test_cli_downloads_pinned_official_repos_in_the_workers_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "output"
            output.mkdir()

            def fetch(*, repo_id, revision, local_dir, **kwargs):
                self.assertTrue(repo_id.startswith(("opendatalab/", "PaddlePaddle/")))
                self.assertRegex(revision, r"^[0-9a-f]{40}$")
                local_dir.mkdir(parents=True)
                (local_dir / "weights.bin").write_bytes(b"downloaded fixture")

            for parser in ("mineru", "paddle"):
                with self.subTest(parser=parser), \
                     patch.object(sys, "argv", ["download_models", "--parser", parser,
                                                "--models-dir", str(root / "models")]), \
                     patch.object(download_models, "snapshot_download", side_effect=fetch) as download, \
                     patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                    download_models.main()
                    self.assertEqual(download.call_count, 2)
                    models = root / "models" / parser
                    worker.configure_models(parser, models, output)
                    if parser == "mineru":
                        config = json.loads((output / "mineru-runtime.json").read_text())
                        for directory in config["models-dir"].values():
                            self.assertTrue((Path(directory) / "weights.bin").is_file())
                    else:
                        self.assertEqual(os.environ["PADDLE_PDX_CACHE_HOME"], str(models))
                        # The server's explicit model_dir matches the downloaded VLM.
                        with patch.dict(os.environ, {
                            "PADDLE_SERVER_PYTHON": sys.executable,
                            "PADDLE_SERVER_CLI": __file__,
                        }):
                            command, _ = worker.server_command()
                        model_dir = Path(command[command.index("--model_dir") + 1])
                        self.assertTrue((model_dir / "weights.bin").is_file())

    def test_failed_download_stops_and_rerun_keeps_the_completed_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = root / "paddle/official_models/PP-DocLayoutV3/weights.bin"

            def fetch(*, local_dir, **kwargs):
                local_dir.mkdir(parents=True, exist_ok=True)
                (local_dir / "weights.bin").write_bytes(b"completed download")

            log = io.StringIO()

            def fail_second(*, local_dir, **kwargs):
                if local_dir.name == "PaddleOCR-VL-1.6":
                    raise OSError("connection lost")
                fetch(local_dir=local_dir)

            with patch.object(download_models, "snapshot_download", side_effect=fail_second), \
                 contextlib.redirect_stdout(log), self.assertRaisesRegex(OSError, "connection lost"):
                download_models.download("paddle", root)
            self.assertNotIn("Download complete", log.getvalue())
            self.assertTrue(completed.is_file())

            def retry(*, local_dir, **kwargs):
                self.assertTrue(completed.is_file())
                fetch(local_dir=local_dir)

            with patch.object(download_models, "snapshot_download", side_effect=retry), \
                 contextlib.redirect_stdout(log):
                download_models.download("paddle", root)
            self.assertIn("Download complete", log.getvalue())


if __name__ == "__main__":
    unittest.main()
