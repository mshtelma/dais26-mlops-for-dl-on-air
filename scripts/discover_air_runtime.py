"""Discover Databricks AI Runtime (AIR) spark-versions and H100 node types."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime

AIR_PATTERN = re.compile(r"ai[-_]?runtime", re.IGNORECASE)
H100_PATTERN = re.compile(r"h100", re.IGNORECASE)
OUTPUT_FILE = ".air-discovery.json"


def run_databricks(args: list[str]) -> dict:
    """Run a Databricks CLI command and return parsed JSON output."""
    cmd = ["databricks", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: Databricks CLI not found.\n"
            "Install it with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh\n"
            "or via Homebrew:\n"
            "  brew tap databricks/tap && brew install databricks",
            file=sys.stderr,
        )
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        stderr_lower = (exc.stderr or "").lower()
        if any(
            token in stderr_lower
            for token in ("auth", "unauthorized", "unauthenticated", "403", "401")
        ):
            print(
                "ERROR: Authentication failed.\n"
                "Run `databricks auth login --host <YOUR_WORKSPACE>` first.",
                file=sys.stderr,
            )
            sys.exit(3)
        print(
            f"ERROR: Databricks CLI returned non-zero exit code {exc.returncode}.\n"
            f"stderr: {exc.stderr}",
            file=sys.stderr,
        )
        sys.exit(3)
    return json.loads(result.stdout)


def filter_air_spark_versions(data: dict) -> list[str]:
    """Return spark version keys that match the AIR pattern."""
    versions = data.get("versions", [])
    matched = []
    for entry in versions:
        key = entry.get("key", "")
        name = entry.get("name", "")
        if AIR_PATTERN.search(key) or AIR_PATTERN.search(name):
            matched.append(key)
    return matched


def filter_h100_node_types(data: list) -> list[str]:
    """Return node_type_ids where node_type_id or instance_type_id contains h100."""
    matched = []
    for node in data:
        node_type_id = node.get("node_type_id", "")
        instance_type_id = node.get("instance_type_id", "")
        if H100_PATTERN.search(node_type_id) or H100_PATTERN.search(instance_type_id):
            matched.append(node_type_id)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover Databricks AI Runtime spark-versions and H100 node types.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what queries would run without calling subprocess (for CI smoke).",
    )
    args = parser.parse_args()

    spark_versions_cmd = ["clusters", "spark-versions", "--output", "json"]
    node_types_cmd = ["clusters", "list-node-types", "--output", "json"]

    if args.dry_run:
        print("DRY-RUN mode: no subprocess calls will be made.\n")
        print("Would run:")
        print(f"  databricks {' '.join(spark_versions_cmd)}")
        print(f"  databricks {' '.join(node_types_cmd)}")
        print(
            "\nWould write results to:",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                OUTPUT_FILE,
            ),
        )
        sys.exit(0)

    print("Querying Databricks CLI for AIR spark-versions...")
    spark_data = run_databricks(spark_versions_cmd)
    air_versions = filter_air_spark_versions(spark_data)

    if not air_versions:
        print(
            "AI Runtime not available. PIVOT: use `-t dev_non_air` target with "
            "standard DBR ML cluster (g5.12xlarge or equiv).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Querying Databricks CLI for node types...")
    node_data = run_databricks(node_types_cmd)
    node_list = node_data if isinstance(node_data, list) else node_data.get("node_types", [])
    h100_nodes = filter_h100_node_types(node_list)

    if not h100_nodes:
        print(
            "No H100 found in node-types. Check `databricks clusters list-node-types` manually; "
            "AIR may use a different naming convention.",
            file=sys.stderr,
        )
        sys.exit(1)

    discovered_at = datetime.now(tz=UTC).isoformat()
    workspace_host = os.environ.get("DATABRICKS_HOST", "")

    result = {
        "air_spark_version": air_versions[0],
        "air_node_type_id": h100_nodes[0],
        "discovered_at": discovered_at,
        "workspace_host": workspace_host,
    }

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(repo_root, OUTPUT_FILE)
    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print("\n--- AIR Discovery Summary ---")
    print(f"AIR spark versions found:  {air_versions}")
    print(f"H100 node types found:     {h100_nodes}")
    print(f"Selected spark version:    {result['air_spark_version']}")
    print(f"Selected node type:        {result['air_node_type_id']}")
    print(f"Discovered at:             {result['discovered_at']}")
    print(f"Workspace host:            {result['workspace_host'] or '(not set)'}")
    print(f"\nOutput written to: {output_path}")
    print("\n--- JSON Output ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
