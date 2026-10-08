"""Run the installed console commands outside the repository, without Azure calls."""

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest


class InstalledCliTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()

    def command(self, name, *args):
        suffix = ".exe" if sys.platform == "win32" else ""
        executable = Path(sysconfig.get_path("scripts")) / (name + suffix)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [str(executable), *args],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_help_works_for_both_installed_commands_from_another_folder(self):
        for name in ("run-mineru", "run-paddle"):
            with self.subTest(command=name):
                result = self.command(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"usage: {name}", result.stdout)
                for option in (
                    "--config",
                    "--output",
                    "--resume",
                    "--no-wait",
                    "--browser-login",
                ):
                    self.assertIn(option, result.stdout)
        self.assertFalse(list(self.root.iterdir()))

    def test_missing_input_returns_cli_usage_error(self):
        for name in ("run-mineru", "run-paddle"):
            with self.subTest(command=name):
                result = self.command(name)
                self.assertEqual(result.returncode, 2)
                self.assertIn("required", result.stderr)

    def test_wrong_parser_receipt_recommends_the_other_console_command(self):
        receipt = self.root / "receipt.json"
        receipt.write_text(json.dumps({"parser": "mineru"}))
        result = self.command("run-paddle", "--resume", str(receipt))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Use run-mineru for this receipt", result.stderr)


if __name__ == "__main__":
    unittest.main()
