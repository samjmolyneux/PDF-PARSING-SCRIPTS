"""Check how the client decides whether the downloaded batch succeeded."""

import pytest

from pdf_parsers import client


@pytest.mark.parametrize(
    "status, changes, succeeds",
    [
        pytest.param("Completed", {}, True, id="complete"),
        pytest.param("Failed", {}, False, id="azure-failed"),
        pytest.param("Canceled", {}, False, id="azure-canceled"),
        pytest.param("NotResponding", {}, False, id="azure-not-responding"),
        pytest.param("Completed", {"finished": False}, False, id="unfinished"),
        pytest.param("Completed", {"parser": "paddle"}, False, id="wrong-parser"),
        pytest.param("Completed", {"documents": []}, False, id="no-documents"),
        pytest.param(
            "Completed", {"fatal_error": "disk full"}, False, id="fatal-error"
        ),
        pytest.param("Completed", {"total": 2}, False, id="wrong-total"),
        pytest.param(
            "Completed",
            {"documents": [{"input": "different.pdf", "status": "succeeded"}]},
            False,
            id="wrong-document",
        ),
        pytest.param(
            "Completed",
            {"documents": [{"input": "a.pdf", "status": "failed"}]},
            False,
            id="failed-document",
        ),
    ],
)
def test_success_requires_azure_completion_and_every_submitted_pdf(
    status, changes, succeeds
):
    receipt = {"parser": "mineru", "pdf_count": 1, "pdfs": ["a.pdf"]}
    report = {
        "parser": "mineru",
        "total": 1,
        "finished": True,
        "documents": [{"input": "a.pdf", "status": "succeeded"}],
        **changes,
    }
    assert client.report_success(report, receipt, status) is succeeds
