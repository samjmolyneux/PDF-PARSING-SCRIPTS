"""One native Paddle batch; yield each PDF after its merged exports are saved."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


def parse_pdfs(pdfs: Sequence[Path], output: Path) -> Iterator[Path]:
    # Load Paddle only for this parser; the MinerU image does not need it.
    from paddleocr import PaddleOCRVL  # noqa: PLC0415

    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:8118/v1",
        vl_rec_api_model_name="PaddleOCR-VL-1.6-0.9B",
        layout_detection_model_dir=str(
            Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "official_models/PP-DocLayoutV3"
        ),
        use_doc_unwarping=False,
        use_chart_recognition=True,
        use_ocr_for_image_block=True,
        format_block_content=True,
        device="gpu:0",
    )
    pages = []
    # Pass every PDF at once so Paddle can batch across document boundaries.
    # predict() would collect the entire submission in memory; predict_iter streams it.
    for page in pipeline.predict_iter(input=[str(pdf) for pdf in pdfs]):
        pdf = Path(page["input_path"])
        if (
            page["page_index"] != len(pages)
            or page["page_count"] <= len(pages)
            or (
                pages
                and (
                    page["input_path"] != pages[0]["input_path"]
                    or page["page_count"] != pages[0]["page_count"]
                )
            )
        ):
            msg = f"Incomplete or out-of-order Paddle pages: {pdf.name}"
            raise RuntimeError(msg)
        pages.append(page)
        if len(pages) < page["page_count"]:
            continue
        destination = output / pdf.stem
        destination.mkdir(parents=True, exist_ok=True)
        # Restructure one PDF's pages; combining the whole batch would merge documents.
        results = pipeline.restructure_pages(
            res_list=pages,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=True,
        )
        for result in results:
            result.save_to_json(save_path=destination)
            result.save_to_markdown(save_path=destination)
            result.save_to_word(save_path=destination)
        for suffix in (".json", ".md", ".docx"):
            if not any(
                p.stat().st_size for p in destination.rglob(f"*{suffix}") if p.is_file()
            ):
                msg = f"Missing {suffix} export: {pdf.name}"
                raise RuntimeError(msg)
        pages = []
        yield pdf
    if pages:
        msg = f"Incomplete Paddle pages: {pages[0]['input_path']}"
        raise RuntimeError(msg)
