"""Upload PDFs, invoke the MinerU deployment and download its exports."""
from .client import main as run


def main():
    return run("mineru")


if __name__ == "__main__":
    raise SystemExit(main())
