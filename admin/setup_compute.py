"""Preview or apply the compute YAML to the existing Azure ML workspace."""
import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient, load_compute
from azure.identity import AzureCliCredential

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=ROOT / "config.json")
    cli.add_argument("--apply", action="store_true", help="Apply using your az login session")
    args = cli.parse_args()

    compute = load_compute(ROOT / "azure/compute.yml")
    print(f"Compute: {compute.name}\n  VM size: {compute.size}")
    print(f"  Nodes: minimum {compute.min_instances}, maximum {compute.max_instances}")
    if not args.apply:
        print("Preview only. Add --apply to configure compute; no Azure connection was made.")
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    client = MLClient(
        AzureCliCredential(tenant_id=config["tenant_id"]),
        config["subscription_id"],
        config["resource_group"],
        config["workspace"],
    )
    print("Applying compute configuration; waiting for Azure...", flush=True)
    client.compute.begin_create_or_update(compute).result()
    print(f"Compute configured: {compute.name}. No parsing job was submitted.")


if __name__ == "__main__":
    main()
