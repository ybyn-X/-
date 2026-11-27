"""Search tools selected by the current Agent."""

import json

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from . import CONFIGURED_TOOL_TAG


@tool
def tool_research(query: str, runtime: ToolRuntime) -> Command:
    """Discover registered tools by name/description substring, or select:name1,name2.

    Matching tools become callable on the next model turn. This does not execute them.
    """
    available = {
        item.name: item for item in runtime.tools
        if item.name != tool_research.name and CONFIGURED_TOOL_TAG in (item.tags or ())
    }
    query = query.strip()
    if query.startswith("select:"):
        wanted = {name.strip() for name in query[7:].split(",")}
        matched = [name for name in available if name in wanted]
    else:
        matched = [
            name for name, item in available.items()
            if query and (query.casefold() in name.casefold() or query.casefold() in item.description.casefold())
        ][:5]
    content = json.dumps(matched, ensure_ascii=False) if matched else f"No tools found matching: {query}"
    return Command(update={
        "promoted_tools": matched,
        "messages": [ToolMessage(content=content, name="tool_research", tool_call_id=runtime.tool_call_id)],
    })
