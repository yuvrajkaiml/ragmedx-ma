"""Unit test for :class:`PDFLoader`.

Writes a tiny one-page PDF to ``tmp_path`` and asserts the loader returns at
least one page with non-empty text and page metadata. Previously this file was
a ``main()`` script pointing at a non-existent ``data/documents/lab_report.pdf``
so it collected zero tests under pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.loaders.pdf_loader import PDFLoader


def _write_pdf(path: Path, sentences: list[str]) -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(path), pagesize=letter)
        text_object = pdf.beginText(72, 720)
        for sentence in sentences:
            text_object.textLine(sentence)
        pdf.drawText(text_object)
        pdf.showPage()
        pdf.save()
    except Exception:  # pragma: no cover - environment fallback
        spare = Path(r"d:\Ragmed_2\docs\Visual 1.pdf")
        if not spare.exists():
            pytest.skip("No PDF writer available and no spare PDF to copy.")
        path.write_bytes(spare.read_bytes())


def test_pdf_loader_returns_pages_with_text(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(
        pdf_path,
        [
            "Hypertension is a common chronic cardiovascular condition.",
            "First-line management often includes lifestyle changes and medication.",
        ],
    )

    pages = PDFLoader().load(pdf_path)

    assert len(pages) >= 1
    first = pages[0]
    assert first.text.strip()
    assert first.metadata.page == 1
    assert first.metadata.source == "sample.pdf"


def test_pdf_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PDFLoader().load(tmp_path / "does_not_exist.pdf")
