# PDF parsing on Azure ML

Run **PaddleOCR-VL** or **MinerU** on a folder of PDFs. Your computer uploads the
files and downloads the results; Azure runs the parser on a GPU. Once Azure
accepts the job, processing continues even if you close your laptop.

## Start here

### I want to parse PDFs

Use an existing deployment. Install the client, choose your PDF folder, and run
`run-paddle` or `run-mineru`.

**[Go to Run parsers →](RUNNING.md#quick-start)**

This is the route for most people. The supplied configuration uses the team's
`EPPI_DEV` workspace. You need access to that workspace; if your team uses
another deployment, ask its administrator for the configuration file.

### I want to deploy parsers in a workspace

Set up the shared GPU compute and deploy one or both parsers in your Azure ML
workspace. You configure the workspace details and run two administrator
scripts.

**[Go to Deploy parsers →](DEPLOYMENT.md)**

This is usually a one-time setup for each workspace. The workspace must already
exist. The compute cluster scales down to zero nodes when idle, while the
endpoint remains available for future submissions.
