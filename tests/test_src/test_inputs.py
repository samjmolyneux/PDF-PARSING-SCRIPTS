"""Check local PDF discovery, filenames and linked-file handling."""

import pytest

from pdf_parsers import client


def test_flat_pdf_folder_accepts_spaces_and_uppercase_extension(pdf_folder, make_pdf):
    files = [make_pdf("a.pdf"), make_pdf("space name.PDF")]
    folder, found = client.pdf_files(pdf_folder)
    assert found == [p.resolve() for p in files]
    assert folder == pdf_folder.resolve()


def test_linked_pdf_is_rejected(tmp_path, pdf_folder):
    outside = tmp_path / "outside.pdf"
    outside.write_text("external")
    try:
        (pdf_folder / "linked.pdf").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(ValueError, match="linked PDFs"):
        client.pdf_files(pdf_folder)
