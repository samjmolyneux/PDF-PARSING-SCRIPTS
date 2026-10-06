"""Preview or upload downloaded weights as versioned Azure ML model assets."""
import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient, load_model
from azure.identity import AzureCliCredential

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parser", choices=["mineru", "paddle", "both"], default="both")
    cli.add_argument("--config", type=Path, default=ROOT / "config.json")
    cli.add_argument("--apply", action="store_true", help="Upload/register using your az login session")
    args = cli.parse_args()
    parsers = ("mineru", "paddle") if args.parser == "both" else (args.parser,)
    models = [load_model(ROOT / f"models/{parser}.yml") for parser in parsers]
    for model in models:
        print(f"Model: {model.name}:{model.version}\n  Model directory: {model.path}")
    if not args.apply:
        print("Preview only. Add --apply to register; no Azure connection was made.")
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    client = MLClient(AzureCliCredential(tenant_id=config["tenant_id"]), config["subscription_id"],
                      config["resource_group"], config["workspace"])
    for model in models:
        registered = client.models.create_or_update(model)
        print(f"Registered {registered.name}:{registered.version}")


if __name__ == "__main__":
    main()
