"""Workspace-scoped shell execution tool."""

from pathlib import Path
import subprocess

from langchain_core.runnables import RunnableConfig

MAX_TIMEOUT_SECONDS = 120
MAX_OUTPUT_CHARS = 32_000


def _workspace_root(config: RunnableConfig) -> Path:
    root_value = config.get("configurable", {}).get("workspace_root")
    if not root_value:
        raise ValueError("Caller must set configurable.workspace_root")
    root = Path(root_value).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace_root must be a directory")
    return root


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"


def bash(command: str, config: RunnableConfig, timeout: int = 30) -> str:
    """Execute a shell command inside the workspace directory and return exit code and output."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be between 1 and {MAX_TIMEOUT_SECONDS} seconds")

    root = _workspace_root(config)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Command timed out after {timeout}s") from exc

    parts = [f"Exit code: {completed.returncode}"]
    if completed.stdout:
        parts.append(f"STDOUT:\n{completed.stdout.rstrip()}")
    if completed.stderr:
        parts.append(f"STDERR:\n{completed.stderr.rstrip()}")
    return _truncate("\n".join(parts))
