# Deploy parsers

Set up PaddleOCR-VL, MinerU, or both in an **existing Azure ML workspace**.
You configure the workspace details, run `setup_compute.py` once for the shared
GPU cluster, then run `deploy_parsers.py` for the parsers you want.

If you only want to submit PDFs to an existing deployment, use
[Run parsers](RUNNING.md#quick-start).

## 1. Install the administrator tools

Use an active Python 3.10+ environment. To create a new Conda environment:

```bash
conda create --name pdf-parsers python=3.12 -y
conda activate pdf-parsers
```

Download the repository and install the package and Azure CLI:

```bash
git clone https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS.git
cd PDF-PARSING-SCRIPTS
python -m pip install . azure-cli
```

If you already have the repository, run the install command from its directory.
Keep that directory as your working directory for the remaining commands.
The administrator scripts use Azure's Python SDK and an Azure CLI sign-in.

## 2. Configure the workspace {#configure-the-workspace}

Edit the existing `config.json`. For a different workspace, replace all five
values with your workspace details. This example shows the required fields:

```json
{
  "subscription_id": "YOUR_SUBSCRIPTION_ID",
  "resource_group": "YOUR_RESOURCE_GROUP",
  "workspace": "YOUR_WORKSPACE_NAME",
  "tenant_id": "YOUR_TENANT_ID",
  "endpoint": "YOUR_UNIQUE_ENDPOINT_NAME"
}
```

| Field             | What to enter                                                                                               |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| `subscription_id` | The Azure subscription containing your workspace.                                                           |
| `resource_group`  | The resource group containing your workspace.                                                               |
| `workspace`       | The existing Azure ML workspace name.                                                                       |
| `tenant_id`       | Your Microsoft Entra tenant ID, used for sign-in.                                                           |
| `endpoint`        | A name for this batch endpoint, unique within its Azure region. Use lowercase letters, numbers and hyphens. |

You can find the workspace identifiers in the Azure portal's workspace overview
and the tenant ID in Microsoft Entra ID. The supplied file already targets
`EPPI_DEV`; leave those values in place only when that is your intended workspace.

Review [azure/compute.yml](https://github.com/samjmolyneux/PDF-PARSING-SCRIPTS/blob/main/azure/compute.yml).
It currently specifies `westeurope` and `Standard_NC24ads_A100_v4`: one A100 and
24 vCPUs per node. For a workspace in another region, set `location` appropriately
and confirm that the VM size and quota are available there. Your account needs
permission to create compute and deploy assets in the workspace.

## 3. Sign in

Replace `YOUR_TENANT_ID` with the value from your configuration:

```bash
az login --tenant YOUR_TENANT_ID
```

The administrator scripts use this session. They take the subscription and
workspace from `config.json`, so you do not need to set Azure CLI workspace
defaults.

## 4. Set up the compute

```bash
python admin/setup_compute.py --apply
```

This creates or updates `pdf-parsers-a100` from `azure/compute.yml` and waits for
Azure to finish.

By default, the script reads `config.json` from the repository root. To use a
configuration file in another location, pass `--config`:

```bash
python admin/setup_compute.py --config "./other-workspace.json" --apply
```

## 5. Deploy the parsers

To deploy both parsers:

```bash
python admin/deploy_parsers.py --parser both --apply
```

The script registers the selected environments and pipelines, creates or
updates the endpoint named in your configuration, and attaches the selected
deployments. It does not submit a PDF parsing job.

`--parser` defaults to `both`. To deploy **only Paddle** instead:

```bash
python admin/deploy_parsers.py --parser paddle --apply
```

Or to deploy **only MinerU**:

```bash
python admin/deploy_parsers.py --parser mineru --apply
```

You can deploy the other parser later. Selecting one parser leaves the other
deployment in place. The Azure deployment names are `paddle-vl` and `mineru`;
the client commands select the appropriate deployment automatically.

This script also defaults to the repository's `config.json`. Use `--config` to
select another file, using the same configuration as for compute setup:

```bash
python admin/deploy_parsers.py --parser paddle --config "./other-workspace.json" --apply
```
