# PDF parsing on Azure ML

[![Tests](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/samjmolyneux/PDF-PARSING-SCRIPTS/badges/coverage.svg)](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/tests.yml)
[![Pre-commit](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/pre-commit.yml?query=branch%3Amain)
[![Docs](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/actions/workflows/docs.yml/badge.svg?branch=main)](https://samjmolyneux.github.io/PDF-PARSING-SCRIPTS/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)

Run PaddleOCR-VL or MinerU on a local folder of PDFs. The commands upload your
PDFs, run the parser on Azure, and download its exports. Accepted jobs continue
running after you disconnect.

[run-guide]: https://samjmolyneux.github.io/PDF-PARSING-SCRIPTS/RUNNING/
[deploy-guide]: https://samjmolyneux.github.io/PDF-PARSING-SCRIPTS/DEPLOYMENT/

## Start here

- **[Run parsers][run-guide]:** install the client and submit PDFs to an
  existing deployment. This is the guide for most people.
- **[Deploy parsers][deploy-guide]:** configure the compute and parser
  deployments in an Azure ML workspace. This is for administrators.
