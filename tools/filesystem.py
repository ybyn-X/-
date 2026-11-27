"""UTF-8 file tools scoped to the caller's explicitly supplied workspace."""

import os
from pathlib import Path
import tempfile

from langchain_core.runnables import RunnableConfig

MAX_BYTES = 1_000_000


def _path(path: str, config: RunnableConfig) -> Path:
    root_value = config.get("configurable", {}).get("workspace_root")
    if not root_value:
        raise ValueError("Caller must set configurable.workspace_root")
    root = Path(root_value).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace_root must be a directory")
    relative = Path(path)
    if relative.is_absolute() or relative.drive or relative.root or ".." in relative.parts or ":" in path:
        raise ValueError("Use a relative path within the workspace")
    target = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or current.is_junction():
            raise ValueError("Symbolic links and junctions are not supported")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes the workspace")
    return target


def _read(target: Path) -> str:
    with target.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("File exceeds the 1 MB text limit")
    return data.decode("utf-8-sig")


def _write(target: Path, content: str, overwrite: bool) -> None:
    data = content.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError("Content exceeds the 1 MB text limit")
    if not overwrite:
        with target.open("xb") as stream:
            stream.write(data)
        return
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".agent-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def list_directory(path: str, config: RunnableConfig) -> str:
    """List up to 200 immediate entries of a workspace directory. Use '.' for root."""
    entries = []
    for index, item in enumerate(_path(path, config).iterdir()):
        if index == 200:
            entries.append("[truncated: more than 200 entries]")
            break
        kind = "link" if item.is_symlink() or item.is_junction() else "dir" if item.is_dir() else "file"
        entries.append(f"{kind}: {item.name}")
    return "\n".join(sorted(entries)) or "[empty directory]"


def read_file(path: str, config: RunnableConfig) -> str:
    """Read a UTF-8 workspace file, limited to 1 MB. Paths must be relative."""
    return _read(_path(path, config))


def write_file(path: str, content: str, config: RunnableConfig, overwrite: bool = False) -> str:
    """Write a UTF-8 file in an existing workspace directory. Existing files require overwrite=True."""
    _write(_path(path, config), content, overwrite)
    return f"Wrote {path} ({len(content.encode('utf-8'))} bytes)"


def str_replace(path: str, old: str, new: str, config: RunnableConfig) -> str:
    """Replace exactly one occurrence in a UTF-8 file. Read first; ambiguous or missing matches fail."""
    target = _path(path, config)
    content = _read(target)
    if not old or content.count(old) != 1:
        raise ValueError("old must be non-empty and match exactly once")
    _write(target, content.replace(old, new, 1), True)
    return f"Updated {path}"
