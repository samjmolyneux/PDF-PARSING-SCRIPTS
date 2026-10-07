# Start with the official Paddle vLLM server image, pinned to an exact version.
FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server@sha256:5713fd30ab76094b7b6a20d95fd8e26fa9dc452bcc90ccb16f1fb056bd2a0f4d

# Install software as the container's administrator.
USER root

# Download a fixed Miniforge release, which supplies Conda.
ADD https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-Linux-x86_64.sh /tmp/miniforge.sh

# -b installs without prompts; -p chooses where Conda itself is installed.
RUN bash /tmp/miniforge.sh -b -p /opt/conda

# Copy the client package list from the build folder into the image.
COPY conda.yml /tmp/client-conda.yml

# Create the client environment at a known path, then clear Conda's caches.
# workers/start_paddle_client.sh activates this environment when the job starts.
RUN /opt/conda/bin/conda env create \
        --prefix /opt/client \
        --file /tmp/client-conda.yml \
    && /opt/conda/bin/conda clean --all --yes
