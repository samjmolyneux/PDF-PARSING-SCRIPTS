"""Submit two licensed papers to each deployed parser and inspect its exports."""

import html
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pdf_parsers.client import main

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
PDFS = ROOT / "tests/data/pdfs"
EXPECTED = {
    "Young_2008": {
        "pages": 10,
        "text": [
            "ACTRN012607000091404",
            "Patient acceptance of the intervention",
            "Practical statistics for medical research",
        ],
        # Table 1, PDF page 6: Intervention and Control columns, respectively.
        "row": ["Male", "87 (51)", "61 (41)"],
    },
    "Vander_2016": {
        "pages": 11,
        "text": [
            "NCT01592695",
            "Receipt of telephone counseling",
            "Interviewing older adults",
        ],
        # Table 1, PDF page 7: Quitline, Tailored intervention, and Total columns.
        "row": ["Age", "58.5 (8.8)", "55.1 (11.5)", "56.8 (10.3)"],
    },
}


def normalise(text):
    """Allow spacing, punctuation and HTML styling to vary between parsers."""
    text = html.unescape(re.sub(r"</?[a-zA-Z][^>]*>", " ", text))
    return re.sub(r"[^a-z0-9.]", "", text.casefold())


@pytest.fixture(scope="module", params=["paddle", "mineru"])
def parsed_batch(request):
    """Submit once per parser; share that batch across the checks in this run."""
    parser = request.param
    config = request.config.getoption("--azure-config").resolve()
    assert config.is_file(), (
        f"Workspace config missing: {config}; supply --azure-config PATH"
    )
    output = (
        ROOT
        / "results/integration"
        / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        / parser
    )
    print(f"\nRunning {parser} on both PDFs. Results: {output}", flush=True)
    assert (
        main(parser, [str(PDFS), "--config", str(config), "--output", str(output)]) == 0
    )
    return parser, output


def test_batch_report(parsed_batch):
    parser, output = parsed_batch
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["parser"] == parser
    assert report["finished"] and not report["fatal_error"]
    assert {item["input"] for item in report["documents"]} == {
        f"{name}.pdf" for name in EXPECTED
    }
    assert all(item["status"] == "succeeded" for item in report["documents"])


@pytest.mark.parametrize(
    "name, phrase",
    [
        pytest.param(name, phrase, id=f"{name}-{phrase}")
        for name, expected in EXPECTED.items()
        for phrase in expected["text"]
    ],
)
def test_extracted_text(parsed_batch, name, phrase):
    _, output = parsed_batch
    files = list((output / "documents" / name).rglob("*.md"))
    assert files, f"Missing Markdown export for {name}"
    markdown = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert normalise(phrase) in normalise(markdown), f"{name}: missing {phrase!r}"


@pytest.mark.parametrize("name", EXPECTED)
def test_document_exports(parsed_batch, name):
    parser, output = parsed_batch
    expected = EXPECTED[name]
    folder = output / "documents" / name
    markdown_files = list(folder.rglob("*.md"))
    json_files = list(folder.rglob("*.json"))
    assert markdown_files and json_files, f"Missing exports for {name}"
    markdown = "\n".join(path.read_text(encoding="utf-8") for path in markdown_files)
    rows = [
        [
            normalise(cell)
            for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.I | re.S)
        ]
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", markdown, re.I | re.S)
    ]
    expected_row = [normalise(cell) for cell in expected["row"]]
    assert any(row == expected_row for row in rows), (
        f"{name}: expected table row {expected['row']}"
    )
    exports = {
        path.name: json.loads(path.read_text(encoding="utf-8")) for path in json_files
    }
    if parser == "paddle":
        result = exports[f"{name}_res.json"]
        assert result["page_count"] == expected["pages"]
        assert len(result["layout_det_res"]) == expected["pages"]
        word_files = list(folder.rglob("*.docx"))
        assert word_files, f"Missing Word export for {name}"
        for path in word_files:
            with zipfile.ZipFile(path) as document:
                assert document.testzip() is None
                assert "word/document.xml" in document.namelist()
    else:
        pages = exports[f"{name}_middle.json"]["pdf_info"]
        assert [page["page_idx"] for page in pages] == list(range(expected["pages"]))
