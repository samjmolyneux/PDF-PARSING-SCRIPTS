#!/usr/bin/env bash
# Administrator only. Builds/registers environments and deploys batch components.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--apply" ]]; then
  echo "Usage: bash admin/deploy.sh --apply"
  echo "Registers YAML environments, creates components,"
  echo "and deploys the two batch pipelines. Run setup_storage.sh and register models first."
  exit 0
fi
read_config() { python -c 'import json,sys; print(json.load(open("config.json"))[sys.argv[1]])' "$1"; }
PARSER_SUBSCRIPTION="$(read_config subscription_id)"
PARSER_GROUP="$(read_config resource_group)"
PARSER_WORKSPACE="$(read_config workspace)"
PARSER_ENDPOINT="$(read_config endpoint)"
ML_SCOPE=(--subscription "$PARSER_SUBSCRIPTION" --resource-group "$PARSER_GROUP" --workspace-name "$PARSER_WORKSPACE")

python admin/register_environments.py --apply
for PARSER in mineru paddle; do
  az ml component create -f "azure/$PARSER-pipeline.yml" "${ML_SCOPE[@]}" --output none
done
az ml batch-endpoint create -f azure/endpoint.yml --name "$PARSER_ENDPOINT" "${ML_SCOPE[@]}" --output none
for PARSER in mineru paddle; do
  az ml batch-deployment create -f "azure/$PARSER-deployment.yml" --endpoint-name "$PARSER_ENDPOINT" "${ML_SCOPE[@]}" --output none
done
echo "Deployment creation commands completed. Verify provisioning/image build status, then run the documented smoke test."
