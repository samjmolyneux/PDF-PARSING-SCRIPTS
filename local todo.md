# Local todo

- [ ] Validate Paddle's new image and client/server separation on Azure.

  Local definitions now use `environments/paddle/Dockerfile` to install Conda and
  create the client environment at `/opt/client` from the existing `conda.yml`.
  `pdf-paddle:3` builds this image without adding Azure's managed Conda environment.
  Two launch scripts select the environments when the job starts:

  ```text
  Client: activate /opt/client, then exec python run.py ...
  Server: deactivate all Conda environments in a separate shell,
          then exec /usr/local/bin/python ... genai_server ...
  ```

  Local checks with real Conda verified activation/deactivation, client isolation,
  argument forwarding and process shutdown, using temporary copies of the launch
  scripts with the container paths mapped to local test paths.

  The image has not been built or run during the local implementation. Verify
  the Azure image build, client execution and server startup/inference with the
  two-PDF test, including server shutdown. The launch scripts now replace
  `workers/run.py::server_environment()`; verify that Conda deactivation restores
  the server's library paths correctly in the actual Azure image.

  Reference: [Azure ML environments from Docker images](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-environments-v2?view=azureml-api-2#create-an-environment-from-a-docker-image).

- [ ] Build and smoke-test both images with bundled models.

  This replaces the earlier proposal to register a preloaded Hugging Face cache
  and repair it online during jobs. MinerU now uses its own downloader during the
  image build and reads the generated local config at runtime. MinerU uses the
  base image's Python directly, without Conda. Paddle downloads
  its two pinned repositories during its image build. No separate Azure model
  assets are passed to the new pipelines.

  Validate `pdf-mineru:5` with pipeline 5 and `pdf-paddle:5` with pipeline 6.
  Confirm both PDFs in `Data/pdf-test` parse successfully and the workers use
  their image model paths. Neither new image has been built or run by this local
  change. MinerU's fresh builds may resolve newer upstream weight revisions;
  jobs reuse the weights already in their built image.

- [ ] Remove PDF files from the repository.
- [ ] Decide whether to remove `config.json` from the repository.
- [ ] Remove `local todo.md` from the repository, keeping it locally.
- [ ] Add tox to the test dependency group and configure it to run the tests
      across supported Python versions, following Flowde's setup.
