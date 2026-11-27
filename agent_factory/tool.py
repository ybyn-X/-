"""Build the tools explicitly declared by an agent."""

from collections.abc import Mapping
from typing import Any, TypeAlias

from langchain_core.tools import BaseTool, StructuredTool

from tools import CONFIGURED_TOOL_TAG
from ._loading import load_symbol

ToolImplementation: TypeAlias = BaseTool | dict[str, Any]


def load_tool(name: str, spec: str | Mapping[str, Any]) -> ToolImplementation:
    if isinstance(spec, str):
        implementation = load_symbol(spec)
    else:
        options = dict(spec)
        ref = options.pop("ref")
        target = load_symbol(ref)
        if options:
            if not callable(target) or isinstance(target, BaseTool):
                raise TypeError(f"tools.{name} with parameters must reference a callable factory")
            implementation = target(**options)
        else:
            implementation = target

    if isinstance(implementation, dict):
        # Provider-native tools use the provider's schema and identifiers.
        return implementation
    if isinstance(implementation, BaseTool):
        tags = [*(implementation.tags or []), CONFIGURED_TOOL_TAG]
        return implementation.model_copy(update={"name": name, "tags": tags})
    if not callable(implementation):
        raise TypeError(f"tools.{name} must reference a callable, BaseTool or tool dict")
    return StructuredTool.from_function(
        implementation,
        name=name,
        description=implementation.__doc__ or f"Tool {name}",
        tags=[CONFIGURED_TOOL_TAG],
    )
