"""Compare client and worker handling of mixed folders and filename collisions."""

import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_parsers import client
from workers import run as worker

pytestmark = pytest.mark.usefixtures("worker_environment")


@pytest.mark.parametrize("extra", ["nested/b.pdf", "picture.png", "notes.txt"])
def test_client_ignores_other_entries_but_worker_requires_a_pdf_only_batch(
    pdf_folder, make_pdf, output_folder, extra
):
    pdf = make_pdf("a.pdf")
    make_pdf(extra)
    assert client.pdf_files(pdf_folder)[1] == [pdf.resolve()]
    with (
        patch.object(worker.subprocess, "run") as run,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert worker.process_batch("mineru", pdf_folder, output_folder) == 1
    run.assert_not_called()


@pytest.mark.parametrize("target", ["client", "mineru", "paddle"])
def test_case_insensitive_name_collisions_are_rejected(
    pdf_folder, make_pdf, output_folder, read_report, target
):
    entries = [make_pdf("same.pdf"), make_pdf("SAME.pdf")]
    # Simulate a Linux directory listing even on a case-insensitive Mac.
    # Each scan needs a fresh iterator: the worker reads the folder twice.
    with patch.object(Path, "iterdir", side_effect=lambda: iter(entries)):
        if target == "client":
            with pytest.raises(ValueError, match="distinct"):
                client.pdf_files(pdf_folder)
            return
        with (
            patch.object(worker.subprocess, "run") as run,
            patch.object(worker, "start_paddle_server") as start,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = worker.process_batch(target, pdf_folder, output_folder)

    assert code == 1
    run.assert_not_called()
    start.assert_not_called()
    report = read_report()
    assert report["parser"] == target
    assert report["total"] == 2
    assert report["finished"] is False
    assert "PDF filenames must be distinct" in report["fatal_error"]
    assert {item["input"] for item in report["documents"]} == {
        pdf.name for pdf in entries
    }
    assert [item["status"] for item in report["documents"]] == [
        "unconfirmed",
        "unconfirmed",
    ]
    assert (
        "PDF filenames must be distinct" in (output_folder / "parser.log").read_text()
    )
