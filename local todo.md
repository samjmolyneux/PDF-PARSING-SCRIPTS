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

- [ ] Let MinerU reuse a preloaded Hugging Face cache and download missing or updated files.

  Keep our explicit seven supporting-model groups plus the VLM in
  `admin/download_models.py`. This list should control the initial downloads;
  MinerU should retain responsibility for choosing models at runtime.

  Proposed design:

  1. Download the existing pinned repositories and patterns into a standard
     Hugging Face cache, rather than the current `local_dir` layout.
  2. Register that cache with Azure, preserving its metadata and ensuring its
     snapshot files remain usable after upload and download.
  3. Make a writable copy available to each job and set `HF_HUB_CACHE` before
     launching MinerU.
  4. Use `MINERU_MODEL_SOURCE=huggingface`, remove MinerU's offline restriction
     and forced local model paths, and let its normal downloader reuse cached
     files or fetch anything missing or updated.

  Initial revision pins would apply to the preloaded cache only; runtime model
  revisions could change. Downloads made during a job would not automatically
  update the registered Azure asset. Leave Paddle's model handling unchanged.

  Validate cache reuse, missing-file downloads and cache portability before
  deployment. Update the tests and deployment instructions to explain the new
  behaviour and its requirement for Hugging Face access.
