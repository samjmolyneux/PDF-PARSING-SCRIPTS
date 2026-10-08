"""Upload PDFs, invoke the PaddleOCR-VL deployment and download its exports."""

from .client import main as run


def main():
    return run("paddle")


if __name__ == "__main__":
    raise SystemExit(main())
