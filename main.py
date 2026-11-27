"""Run the configured general agent against an explicit local workspace."""

import argparse
from pathlib import Path

from agent_factory import AgentFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="General file and planning agent")
    parser.add_argument("request", help="Task to perform")
    parser.add_argument("--workspace", required=True, type=Path, help="Existing workspace directory")
    parser.add_argument("--agent", default="general")
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    if not workspace.is_dir():
        parser.error("--workspace must be a directory")
    agent = AgentFactory.create_by_name(args.agent)
    result = agent.invoke(
        {"messages": [("user", args.request)]},
        config={"configurable": {"workspace_root": str(workspace)}, "recursion_limit": 50},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
