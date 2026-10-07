#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh

# Undo all inherited Conda activations in this shell only, including base.
while [[ ${CONDA_SHLVL:-0} -gt 0 ]]; do
    conda deactivate
done

# The worker supplies the original server Python, its CLI and arguments.
# Replacing this shell lets the worker track and stop the server directly.
exec "$@"
