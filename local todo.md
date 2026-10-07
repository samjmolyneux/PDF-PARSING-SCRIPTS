# Local todo

- [ ] Simplify Paddle's client/server environment separation.

  Prefer avoiding global activation of the client environment, so the server does
  not need to undo its settings at startup.

  Proposed design:

  1. Build an image containing Paddle's server installation and a separate Python
     virtual environment for the client at `/opt/client`.
  2. Configure Azure ML to use that image directly, without adding its managed
     Conda environment.
  3. Launch both programs through explicit interpreter paths:

     ```text
     Client: /opt/client/bin/python run.py ...
     Server: /usr/local/bin/python ... genai_server ...
     ```

  The client virtual environment would be used through its interpreter path,
  without activation. Each interpreter would find its own installed packages,
  while both processes inherit the image's normal environment variables. This
  would remove the client Conda activation state that `server_environment()`
  currently cleans up.

  Building and validating the image requires additional work, but makes the
  separation explicit at build time and simplifies the runtime code. Validate
  both client execution and server startup/inference before removing the helper.

  Using `conda deactivate` in a server-launch shell is a smaller possible change
  to the current setup. Prefer the explicit image-and-interpreter design for the
  final implementation, rather than depending on activation history or manually
  reconstructing environment settings.

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
