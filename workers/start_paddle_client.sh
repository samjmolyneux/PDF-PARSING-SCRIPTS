#!/usr/bin/env bash
set -e

# Enable Conda's shell commands, then select the client environment.
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/client

# Replace this shell with the worker, preserving every supplied argument.
exec python "$(dirname "$0")/run.py" "$@"
