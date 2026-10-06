"""Preview or register the YAML environments; never submit a job."""
import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient, load_environment
from azure.identity import AzureCliCredential

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parser", choices=["mineru", "paddle", "both"], default="both")
    cli.add_argument("--config", type=Path, default=ROOT / "config.json")
    cli.add_argument("--apply", action="store_true", help="Register in Azure using your az login session")
    args = cli.parse_args()
    parsers = ("mineru", "paddle") if args.parser == "both" else (args.parser,)
    environments = [load_environment(ROOT / f"environments/{parser}/environment.yml") for parser in parsers]
    for parser, environment in zip(parsers, environments):
        print(f"Environment: {environment.name}:{environment.version}\n  Image: {environment.image}")
        print(f"  Conda file: {ROOT / 'environments' / parser / 'conda.yml'}")
    if not args.apply:
        print("Preview only. Add --apply to register; no Azure connection was made.")
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    client = MLClient(AzureCliCredential(tenant_id=config["tenant_id"]), config["subscription_id"],
                      config["resource_group"], config["workspace"])
    for environment in environments:
        registered = client.environments.create_or_update(environment)
        print(f"Registered {registered.name}:{registered.version}; check image build status in Azure ML.")


if __name__ == "__main__":
    main()
