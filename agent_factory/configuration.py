"""Read the declarations belonging to one named agent."""

from pathlib import Path
import re
import tomllib
from typing import Any

from .contracts import AgentConfigurationError, AgentDefinition, ToolSpec


def _component(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("ref"), str) or not value["ref"].strip():
        raise ValueError(f"{label} requires a table with a non-empty ref")
    return value


def _tool_spec(name: str, value: Any) -> ToolSpec:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict) and isinstance(value.get("ref"), str) and value["ref"].strip():
        return value
    raise ValueError("tools must map names to Python references or tables with a non-empty ref")


def _load_defaults(config_dir: Path) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    defaults_path = config_dir / "defaults.toml"
    if not defaults_path.is_file():
        return (), {}
    try:
        raw = tomllib.loads(defaults_path.read_text(encoding="utf-8-sig"))
        unknown = raw.keys() - {"middleware", "metadata"}
        if unknown:
            raise ValueError(f"Unknown defaults config fields: {sorted(unknown)}")
        middleware = raw.get("middleware", [])
        if not isinstance(middleware, list):
            raise ValueError("defaults.middleware must be an array of tables")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("defaults.metadata must be a table")
        return (
            tuple(_component(item, f"defaults.middleware[{i}]") for i, item in enumerate(middleware)),
            dict(metadata),
        )
    except (OSError, ValueError) as exc:
        raise AgentConfigurationError(f"defaults configuration failed: {exc}") from exc


def load_definition(config_dir: Path, name: str) -> AgentDefinition:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise AgentConfigurationError(f"Invalid agent name {name!r}")
    directory = config_dir / "agents" / name
    if not directory.is_dir():
        raise AgentConfigurationError(f"Unknown agent {name!r}")
    default_middleware, default_metadata = _load_defaults(config_dir)
    try:
        raw = tomllib.loads((directory / "config.toml").read_text(encoding="utf-8-sig"))
        unknown = raw.keys() - {"model", "tools", "middleware", "skills", "metadata"}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        model = _component(raw.get("model"), "model")
        raw_tools = raw.get("tools", {})
        if not isinstance(raw_tools, dict):
            raise ValueError("tools must map names to Python references or tables with a non-empty ref")
        tools = {tool_name: _tool_spec(tool_name, spec) for tool_name, spec in raw_tools.items()}
        middleware = raw.get("middleware", [])
        if not isinstance(middleware, list):
            raise ValueError("middleware must be an array of tables")
        skills = raw.get("skills", [])
        if not isinstance(skills, list) or any(
            not isinstance(path, str) or not path.strip() or Path(path).anchor for path in skills
        ):
            raise ValueError("skills must be a list of relative paths")
        skill_paths = tuple((directory / path).resolve() for path in skills)
        if len(set(skill_paths)) != len(skill_paths):
            raise ValueError("duplicate skill paths")
        agent_metadata = raw.get("metadata", {})
        if not isinstance(agent_metadata, dict):
            raise ValueError("metadata must be a table")
        merged_metadata = {**default_metadata, **agent_metadata}
        agent_middleware = tuple(_component(item, f"middleware[{i}]") for i, item in enumerate(middleware))
        return AgentDefinition(
            name=name,
            model=model,
            prompt_path=directory / "SOUL.md",
            tools=tools,
            middleware=default_middleware + agent_middleware,
            skill_paths=skill_paths,
            metadata=merged_metadata,
        )
    except (OSError, ValueError) as exc:
        raise AgentConfigurationError(f"agent {name!r} configuration failed: {exc}") from exc
