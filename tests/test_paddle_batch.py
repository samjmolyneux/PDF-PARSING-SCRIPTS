"""Verify streaming document exports without importing Paddle or running a GPU."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from workers.paddle_batch import parse_pdfs


class PaddleBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pdfs = [self.root / "a.pdf", self.root / "b.pdf"]
        self.output = self.root / "exports"
        self.pipeline = MagicMock()
        self.factory = MagicMock(return_value=self.pipeline)
        modules = patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCRVL=self.factory)})
        modules.start()
        self.addCleanup(modules.stop)
        environment = patch.dict("os.environ", {"PADDLE_PDX_CACHE_HOME": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

        def restructure(*, res_list, **kwargs):
            self.assertEqual(kwargs, dict(merge_tables=True, relevel_titles=True, concatenate_pages=True))
            self.assertEqual(len({p["input_path"] for p in res_list}), 1)
            content = ",".join(str(page["page_index"]) for page in res_list)
            result = MagicMock()
            for method, suffix in (("json", "json"), ("markdown", "md"), ("word", "docx")):
                def save(*, save_path, suffix=suffix):
                    (save_path / f"result.{suffix}").write_text(content)
                    (save_path / "image.png").write_bytes(b"retained image")
                getattr(result, f"save_to_{method}").side_effect = save
            return [result]
        self.pipeline.restructure_pages.side_effect = restructure

    def page(self, document, index=0, count=1):
        return dict(input_path=str(self.pdfs[document]), page_index=index, page_count=count)

    def test_one_pipeline_and_one_full_batch_stream_into_separate_document_exports(self):
        def pages(*, input):
            self.assertEqual(input, [str(p) for p in self.pdfs])
            yield self.page(0, 0, 2)
            self.assertFalse((self.output / "a").exists())
            yield self.page(0, 1, 2)
            # First document must be saved before requesting later pages.
            self.assertEqual((self.output / "a/result.md").read_text(), "0,1")
            yield self.page(1)
        self.pipeline.predict_iter.side_effect = pages
        self.assertEqual(list(parse_pdfs(self.pdfs, self.output)), self.pdfs)
        self.factory.assert_called_once()
        self.pipeline.predict_iter.assert_called_once()
        self.pipeline.predict.assert_not_called()
        self.assertEqual(self.pipeline.restructure_pages.call_count, 2)
        self.assertEqual((self.output / "b/result.docx").read_text(), "0")
        self.assertTrue((self.output / "a/image.png").is_file())

    def test_native_exception_retains_prior_exports_and_is_not_retried(self):
        def pages(**kwargs):
            yield self.page(0)
            raise RuntimeError("native VLM error")
        self.pipeline.predict_iter.side_effect = pages
        results = parse_pdfs(self.pdfs, self.output)
        self.assertEqual(next(results), self.pdfs[0])
        with self.assertRaisesRegex(RuntimeError, "native VLM error"):
            next(results)
        self.pipeline.predict_iter.assert_called_once()
        self.assertTrue((self.output / "a/result.docx").is_file())
        self.assertFalse((self.output / "b").exists())

    def test_missing_pages_cannot_be_exported_as_a_complete_document(self):
        for pages in ([self.page(0, 0, 2)],
                      [self.page(0, 0, 3), self.page(0, 2, 3)],
                      [self.page(0, 0, 2), self.page(1)]):
            with self.subTest(pages=pages):
                self.pipeline.predict_iter.return_value = iter(pages)
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    list(parse_pdfs(self.pdfs, self.output))
        self.pipeline.restructure_pages.assert_not_called()

    def test_missing_exports_do_not_yield_a_successful_document(self):
        self.pipeline.predict_iter.return_value = iter([self.page(0)])
        self.pipeline.restructure_pages.side_effect = None
        self.pipeline.restructure_pages.return_value = [MagicMock()]
        with self.assertRaisesRegex(RuntimeError, "Missing .json export"):
            list(parse_pdfs(self.pdfs, self.output))


if __name__ == "__main__":
    unittest.main()
