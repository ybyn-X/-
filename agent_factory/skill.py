"""Skill 文件读取、元数据解析和系统提示词注入内容。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

@dataclass(frozen=True)
class SkillDefinition:
    """表示一个由 ``SKILL.md`` 提供的可注入能力说明。"""

    name: str
    path: Path
    content: str
    metadata: dict[str, Any]


def load_skill(path: Path) -> SkillDefinition:
    content = path.read_text(encoding="utf-8-sig")
    metadata, body = _parse_skill_document(content)
    name = metadata.setdefault("name", path.parent.name)
    metadata.setdefault("description", _first_heading(body))
    return SkillDefinition(name=name, path=path, content=body.strip(), metadata=metadata)


def _parse_skill_document(content: str) -> tuple[dict[str, Any], str]:
    """解析 ``---`` 包裹的 TOML front matter，缺失时保留原文。"""
    if not content.startswith("---"):
        return {}, content
    _, separator, remainder = content.partition("\n")
    if not separator:
        return {}, content
    end = remainder.find("\n---")
    if end < 0:
        return {}, content
    header = remainder[:end]
    body = remainder[end + len("\n---") :].lstrip("\r\n")
    try:
        metadata = tomllib.loads(header)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid Skill front matter: {exc}") from exc
    return metadata, body


def _first_heading(content: str) -> str:
    """从 Skill 正文首个 Markdown 标题提取默认描述。"""
    for line in content.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""

