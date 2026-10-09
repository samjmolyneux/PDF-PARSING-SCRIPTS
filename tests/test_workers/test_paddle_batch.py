"""Verify streaming document exports without importing Paddle or running a GPU."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from workers.paddle_batch import parse_pdfs


class TestPaddleBatch:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.pdfs = [self.root / "a.pdf", self.root / "b.pdf"]
        self.output = self.root / "exports"
        self.pipeline = MagicMock()
        self.factory = MagicMock(return_value=self.pipeline)
        monkeypatch.setitem(
            sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=self.factory)
        )
        monkeypatch.setenv("PADDLE_PDX_CACHE_HOME", str(self.root))

        def restructure(*, res_list, **kwargs):
            assert kwargs == dict(
                merge_tables=True, relevel_titles=True, concatenate_pages=True
            )
            assert len({p["input_path"] for p in res_list}) == 1
            content = ",".join(str(page["page_index"]) for page in res_list)
            result = MagicMock()
            for method, suffix in (
                ("json", "json"),
                ("markdown", "md"),
                ("word", "docx"),
            ):

                def save(*, save_path, suffix=suffix):
                    (save_path / f"result.{suffix}").write_text(content)
                    (save_path / "image.png").write_bytes(b"retained image")

                getattr(result, f"save_to_{method}").side_effect = save
            return [result]

        self.pipeline.restructure_pages.side_effect = restructure

    def page(self, document, index=0, count=1):
        return dict(
            input_path=str(self.pdfs[document]), page_index=index, page_count=count
        )

    def test_one_pipeline_and_one_full_batch_stream_into_separate_document_exports(
        self,
    ):
        def pages(*, input):
            assert input == [str(p) for p in self.pdfs]
            yield self.page(0, 0, 2)
            assert not (self.output / "a").exists()
            yield self.page(0, 1, 2)
            # First document must be saved before requesting later pages.
            assert (self.output / "a/result.md").read_text() == "0,1"
            yield self.page(1)

        self.pipeline.predict_iter.side_effect = pages
        assert list(parse_pdfs(self.pdfs, self.output)) == self.pdfs
        self.factory.assert_called_once()
        self.pipeline.predict_iter.assert_called_once()
        self.pipeline.predict.assert_not_called()
        assert self.pipeline.restructure_pages.call_count == 2
        assert (self.output / "b/result.docx").read_text() == "0"
        assert (self.output / "a/image.png").is_file()

    def test_native_exception_retains_prior_exports_and_is_not_retried(self):
        def pages(**kwargs):
            yield self.page(0)
            raise RuntimeError("native VLM error")

        self.pipeline.predict_iter.side_effect = pages
        results = parse_pdfs(self.pdfs, self.output)
        assert next(results) == self.pdfs[0]
        with pytest.raises(RuntimeError, match="native VLM error"):
            next(results)
        self.pipeline.predict_iter.assert_called_once()
        assert (self.output / "a/result.docx").is_file()
        assert not (self.output / "b").exists()

    @pytest.mark.parametrize(
        "page_specs",
        [
            pytest.param([(0, 0, 2)], id="missing-last-page"),
            pytest.param([(0, 0, 3), (0, 2, 3)], id="gap-in-pages"),
            pytest.param([(0, 0, 2), (1, 0, 1)], id="document-changed"),
        ],
    )
    def test_missing_pages_cannot_be_exported_as_a_complete_document(self, page_specs):
        self.pipeline.predict_iter.return_value = iter(
            self.page(*spec) for spec in page_specs
        )
        with pytest.raises(RuntimeError, match="Incomplete"):
            list(parse_pdfs(self.pdfs, self.output))
        self.pipeline.restructure_pages.assert_not_called()

    @pytest.mark.parametrize(
        "method, suffix", [("json", ".json"), ("markdown", ".md"), ("word", ".docx")]
    )
    def test_missing_exports_do_not_yield_a_successful_document(self, method, suffix):
        self.pipeline.predict_iter.return_value = iter([self.page(0)])
        result = self.pipeline.restructure_pages(
            res_list=[self.page(0)],
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=True,
        )[0]
        getattr(result, f"save_to_{method}").side_effect = None
        self.pipeline.restructure_pages.side_effect = None
        self.pipeline.restructure_pages.return_value = [result]
        with pytest.raises(RuntimeError, match=f"Missing {suffix} export"):
            list(parse_pdfs(self.pdfs, self.output))
