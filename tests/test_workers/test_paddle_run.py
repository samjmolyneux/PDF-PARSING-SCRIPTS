"""Check Paddle document confirmation and failure reporting in the batch worker."""

import contextlib
import io
from unittest.mock import patch

import pytest

from workers import run as worker

pytestmark = pytest.mark.usefixtures("worker_environment")


def test_paddle_batch_failure_preserves_confirmed_documents_without_retry(
    pdf_folder, make_pdf, output_folder, read_report
):
    a, b = make_pdf("a.pdf"), make_pdf("b.pdf")

    def parse(pdfs, output):
        assert pdfs == [a.resolve(), b.resolve()]
        yield a
        assert read_report()["documents"][0]["status"] == "succeeded"
        raise RuntimeError("native pipeline failed")

    with (
        patch.object(worker, "start_paddle_server", return_value=object()) as start,
        patch.object(worker, "stop_paddle_server") as stop,
        patch.object(worker, "parse_paddle_pdfs", side_effect=parse) as parse_call,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        code = worker.process_batch("paddle", pdf_folder, output_folder)
    assert code == 1
    start.assert_called_once()
    stop.assert_called_once()
    parse_call.assert_called_once()
    assert [d["status"] for d in read_report()["documents"]] == [
        "succeeded",
        "unconfirmed",
    ]
    assert "native pipeline failed" in (output_folder / "parser.log").read_text()


def test_paddle_returning_no_results_cannot_report_success(
    pdf_folder, make_pdf, output_folder, read_report
):
    make_pdf("a.pdf")
    with (
        patch.object(worker, "start_paddle_server", return_value=object()),
        patch.object(worker, "stop_paddle_server"),
        patch.object(worker, "parse_paddle_pdfs", return_value=iter([])),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        assert worker.process_batch("paddle", pdf_folder, output_folder) == 1
    assert "did not confirm" in read_report()["fatal_error"]


def test_server_start_failure_leaves_every_pdf_unconfirmed(
    pdf_folder, make_pdf, output_folder, read_report
):
    make_pdf("a.pdf")
    make_pdf("b.pdf")
    with (
        patch.object(
            worker, "start_paddle_server", side_effect=RuntimeError("missing model")
        ) as start,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        code = worker.process_batch("paddle", pdf_folder, output_folder)
    assert code == 1
    assert start.call_count == 1
    assert not read_report()["finished"]
    assert "missing model" in read_report()["fatal_error"]
    assert len(read_report()["documents"]) == 2
