"""统一解析组件并调用官方 ``create_agent`` 完成 Agent 组装。"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel

from ._loading import construct
from .agent import LangChainAgent
from .configuration import load_definition
from .contracts import Agent, AgentCapabilities, AgentConfigurationError, AgentDefinition
from .skill import load_skill
from .tool import ToolImplementation, load_tool


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class AgentFactory:
    """从命名目录组装可调用 Agent，对外隐藏底层框架实现。"""

    def __init__(self, *, config_dir: Path | None = None):
        """初始化工厂；未指定时使用项目内置的默认配置目录。"""
        self._config_dir = Path(config_dir or DEFAULT_CONFIG_DIR).resolve()

    def list_agents(self) -> tuple[str, ...]:
        """列出当前配置目录下已注册的 Agent 名称。"""
        agents_dir = self._config_dir / "agents"
        if not agents_dir.is_dir():
            return ()
        names = [
            entry.name
            for entry in agents_dir.iterdir()
            if entry.is_dir() and (entry / "config.toml").is_file()
        ]
        return tuple(sorted(names))

    def describe(self, agent_name: str) -> AgentCapabilities:
        """在不实例化模型和图的情况下，静态分析 Agent 的能力边界。"""
        definition = load_definition(self._config_dir, agent_name)
        deferred: list[str] = []
        for spec in definition.middleware:
            names = spec.get("deferred_names")
            if isinstance(names, list):
                deferred.extend(str(item) for item in names)
        return AgentCapabilities(
            name=definition.name,
            model_ref=str(definition.model["ref"]),
            tools=tuple(definition.tools.keys()),
            deferred_tools=tuple(dict.fromkeys(deferred)),
            middleware=tuple(str(spec["ref"]) for spec in definition.middleware),
            skills=tuple(path.parent.name if path.name == "SKILL.md" else path.name for path in definition.skill_paths),
            metadata=dict(definition.metadata),
        )

    def create(self, agent_name: str, *, model_name: Any = None) -> Agent:
        """按名称读取配置并组装 Agent。"""
        definition = load_definition(self._config_dir, agent_name)
        return self._assemble(definition, model_name)

    @classmethod
    def create_by_name(
        cls,
        agent_name: str,
        *,
        model_name: Any = None,
        config_dir: Path | None = None,
    ) -> Agent:
        """读取 config/agents/<name>/config.toml 和 SOUL.md 并组装 Agent。"""
        return cls(config_dir=config_dir).create(agent_name, model_name=model_name)

    def _assemble(self, definition: AgentDefinition, model: Any) -> Agent:
        with _assembly_stage(definition.name, "model configuration"):
            if model is None:
                model = construct(definition.model, BaseChatModel)
            elif not isinstance(model, BaseChatModel):
                raise TypeError("model_name must be a BaseChatModel instance")

        with _assembly_stage(definition.name, "prompt configuration"):
            prompt = definition.prompt_path.read_text(encoding="utf-8-sig").strip()
            if not prompt:
                raise ValueError(f"{definition.prompt_path.name} must not be empty")
            if definition.skill_paths:
                skills = [load_skill(path) for path in definition.skill_paths]
                sections = [f"### {skill.name}\n{skill.content}" for skill in skills]
                prompt += "\n\n## Enabled skills\n\n" + "\n\n".join(sections)

        with _assembly_stage(definition.name, "tool configuration"):
            tools = [load_tool(name, spec) for name, spec in definition.tools.items()]

        with _assembly_stage(definition.name, "middleware configuration"):
            middlewares = [construct(spec, AgentMiddleware) for spec in definition.middleware]

        with _assembly_stage(definition.name, "create_agent assembly"):
            _check_tool_names(tools, middlewares)
            graph = create_agent(
                model=model,
                system_prompt=prompt,
                tools=tools,
                middleware=middlewares,
                name=definition.name,
            )
        return LangChainAgent(graph, name=definition.name)


@contextmanager
def _assembly_stage(agent_name: str, stage: str) -> Iterator[None]:
    """为底层异常补充 Agent 名称和失败阶段。"""
    try:
        yield
    except Exception as exc:
        raise AgentConfigurationError(f"agent {agent_name!r} {stage} failed: {exc}") from exc


def _check_tool_names(
    tools: Sequence[ToolImplementation],
    middlewares: Sequence[AgentMiddleware],
) -> None:
    all_tools = list(tools)
    for middleware in middlewares:
        all_tools.extend(getattr(middleware, "tools", ()))

    seen = set()
    for tool in all_tools:
        name = tool.get("name") if isinstance(tool, dict) else tool.name
        if name is None:
            continue
        if name in seen:
            raise ValueError(f"Duplicate tool name {name!r} across tools and middleware")
        seen.add(name)
