# PDF parsing on Azure ML

Run PaddleOCR-VL or MinerU on a local folder of PDFs. The commands upload your
PDFs, run the parser on Azure, and download its exports. Accepted jobs continue
running after you disconnect.

## Start here

- **[Run parsers](docs/RUNNING.md):** install the client and submit PDFs to an
  existing deployment. This is the guide for most people.
- **[Deploy parsers](docs/DEPLOYMENT.md):** configure the compute and parser
  deployments in an Azure ML workspace. This is for administrators.

The [documentation home page](docs/index.md) explains which route to take.
The supplied `config.json` points to the team's `EPPI_DEV` workspace.

## Preview the documentation

The documentation uses the same Read the Docs theme and copy buttons as
[Flowde](https://github.com/EPPI-Centre/Flowde). From the repository root,
with your Python environment active:

```bash
python -m pip install --group docs
python -m mkdocs serve
```

Open <http://127.0.0.1:8000>. The preview updates as you edit the Markdown files.
Installing dependency groups requires pip 25.1 or later.

To check links and build the static site locally:

```bash
python -m mkdocs build --strict
```

The generated files go in `site/`. These commands do not publish the site.

## Local development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

The tests check local behaviour with mocked Azure services. Environment builds
and GPU inference need a separate test in Azure.
