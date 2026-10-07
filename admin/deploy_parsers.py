"""Register environments and pipelines, then deploy the selected PDF parsers."""
import argparse
import json
from pathlib import Path

from azure.ai.ml import MLClient, load_batch_endpoint, load_component, load_environment
# The public load_batch_deployment loader only handles model deployments.
from azure.ai.ml.entities._load_functions import load_pipeline_component_batch_deployment
from azure.identity import AzureCliCredential

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parser", choices=["mineru", "paddle", "both"], default="both")
    cli.add_argument("--config", type=Path, default=ROOT / "config.json")
    cli.add_argument("--apply", action="store_true", help="Deploy using your az login session")
    args = cli.parse_args()
    parsers = ("mineru", "paddle") if args.parser == "both" else (args.parser,)

    definitions = []
    for parser in parsers:
        environment = load_environment(ROOT / f"environments/{parser}/environment.yml")
        pipeline = load_component(ROOT / f"azure/{parser}-pipeline.yml")
        deployment = load_pipeline_component_batch_deployment(ROOT / f"azure/{parser}-deployment.yml")
        definitions.append((environment, pipeline, deployment))
        print(f"Parser: {parser}\n  Environment: {environment.name}:{environment.version}")
        print(f"  Pipeline: {pipeline.name}:{pipeline.version}\n  Deployment: {deployment.name}")
        print(f"  Compute: {deployment.settings['default_compute']}")
    endpoint = load_batch_endpoint(ROOT / "azure/endpoint.yml")
    if not args.apply:
        print(f"Preview only. Add --apply to deploy using {args.config}; no Azure connection was made.")
        return

    config = json.loads(args.config.read_text(encoding="utf-8"))
    client = MLClient(
        AzureCliCredential(tenant_id=config["tenant_id"]),
        config["subscription_id"],
        config["resource_group"],
        config["workspace"],
    )

    # 1. Register the selected software environments.
    for environment, _, _ in definitions:
        print(f"Registering environment: {environment.name}:{environment.version}", flush=True)
        client.environments.create_or_update(environment)

    # 2. Register the pipelines, including their worker code.
    for _, pipeline, _ in definitions:
        print(f"Registering pipeline: {pipeline.name}:{pipeline.version}", flush=True)
        client.components.create_or_update(pipeline)

    # 3. Create or update the shared endpoint.
    endpoint.name = config["endpoint"]
    print(f"Configuring endpoint: {endpoint.name}", flush=True)
    client.batch_endpoints.begin_create_or_update(endpoint).result()

    # 4. Create or update each selected parser's deployment.
    for _, _, deployment in definitions:
        deployment.endpoint_name = endpoint.name
        print(f"Deploying parser: {deployment.name}", flush=True)
        client.batch_deployments.begin_create_or_update(deployment).result()
    print("Parser deployments configured. Check environment build status, then run the smoke test.")
    print("No parsing job was submitted.")


if __name__ == "__main__":
    main()
