# Use the same vLLM/CUDA base as the official MinerU Dockerfile.
FROM vllm/vllm-openai:v0.21.0

USER root

# OpenCV library and Noto fonts, matching the official MinerU Dockerfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
    && fc-cache -fv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The base supplies vLLM, PyTorch and Transformers. Install pinned MinerU;
# pip reuses compatible dependencies and adjusts any that MinerU requires.
RUN python3 -m pip install --no-cache-dir 'mineru[core]==3.3.1'

# Let this installed MinerU version choose and download all of its models.
# Keep both its Hugging Face cache and generated config at fixed image paths.
ENV HF_HOME=/opt/mineru-models/huggingface \
    MINERU_TOOLS_CONFIG_JSON=/opt/mineru-models/mineru.json
RUN mkdir -p /opt/mineru-models \
    && mineru-models-download -s huggingface -m all

# Parsing uses the local paths written by the downloader above.
ENV MINERU_MODEL_SOURCE=local

# Run the supplied command, rather than the base image's default `vllm serve`.
# Azure supplies `python3 run.py ...` through the command component.
ENTRYPOINT []
