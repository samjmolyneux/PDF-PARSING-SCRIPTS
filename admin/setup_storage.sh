#!/usr/bin/env bash
# Administrator only. Creates resources/role assignments ONLY with --apply.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--apply" || -z "${2:-}" ]]; then
  echo "Usage: bash admin/setup_storage.sh --apply TEAM_ENTRA_GROUP_OBJECT_ID"
  echo "Creates the A100 cluster (min 0) and grants team/compute access"
  echo "to the workspace's existing default Blob datastore. No parsing job or automatic deletion."
  exit 0
fi
PARSER_TEAM_GROUP="$2"
read_config() { python -c 'import json,sys; print(json.load(open("config.json"))[sys.argv[1]])' "$1"; }
PARSER_SUBSCRIPTION="$(read_config subscription_id)"
PARSER_GROUP="$(read_config resource_group)"
PARSER_WORKSPACE="$(read_config workspace)"
ML_SCOPE=(--subscription "$PARSER_SUBSCRIPTION" --resource-group "$PARSER_GROUP" --workspace-name "$PARSER_WORKSPACE")
PARSER_STORAGE_ID="$(az ml workspace show --name "$PARSER_WORKSPACE" --resource-group "$PARSER_GROUP" --subscription "$PARSER_SUBSCRIPTION" --query storage_account -o tsv)"
PARSER_WORKSPACE_ID="$(az ml workspace show --name "$PARSER_WORKSPACE" --resource-group "$PARSER_GROUP" --subscription "$PARSER_SUBSCRIPTION" --query id -o tsv)"
PARSER_ACCOUNT="${PARSER_STORAGE_ID##*/}"
PARSER_CONTAINER="$(python - "$PARSER_ACCOUNT" <<'PY'
import json, sys
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
with open("config.json") as file:
    config = json.load(file)
client = MLClient(AzureCliCredential(tenant_id=config["tenant_id"]), config["subscription_id"],
                  config["resource_group"], config["workspace"])
store = client.datastores.get_default()
if store.type != "azure_blob" or store.account_name != sys.argv[1]:
    raise ValueError("Default datastore must use the workspace Blob account; configure its roles manually otherwise.")
print(store.container_name)
PY
)"
PARSER_CONTAINER_SCOPE="$PARSER_STORAGE_ID/blobServices/default/containers/$PARSER_CONTAINER"

az ml compute create -f azure/compute.yml "${ML_SCOPE[@]}" --output none
PARSER_COMPUTE_IDENTITY="$(az ml compute show -n pdf-parsers-a100 "${ML_SCOPE[@]}" --query identity.principal_id -o tsv)"
if [[ -z "$PARSER_COMPUTE_IDENTITY" || "$PARSER_COMPUTE_IDENTITY" == "None" ]]; then
  echo "The cluster has no system-assigned identity; check compute provisioning." >&2
  exit 1
fi

az role assignment create --assignee-object-id "$PARSER_TEAM_GROUP" --assignee-principal-type Group --role "AzureML Data Scientist" --scope "$PARSER_WORKSPACE_ID" --output none
az role assignment create --assignee-object-id "$PARSER_TEAM_GROUP" --assignee-principal-type Group --role "Storage Blob Data Contributor" --scope "$PARSER_CONTAINER_SCOPE" --output none
az role assignment create --assignee-object-id "$PARSER_COMPUTE_IDENTITY" --assignee-principal-type ServicePrincipal --role "Storage Blob Data Reader" --scope "$PARSER_STORAGE_ID" --output none
az role assignment create --assignee-object-id "$PARSER_COMPUTE_IDENTITY" --assignee-principal-type ServicePrincipal --role "Storage Blob Data Contributor" --scope "$PARSER_CONTAINER_SCOPE" --output none

echo "Storage, compute and access configured. Allow time for role assignments to propagate."
