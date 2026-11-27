"""记录每次模型调用时的消息数量。"""

import logging

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)


class ModelLoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        """记录模型请求中的消息数量，然后继续默认执行。"""
        logger.info("Model input contains %d messages", len(state["messages"]))
        return None

    async def abefore_model(self, state, runtime):
        """异步记录模型请求中的消息数量，然后继续默认执行。"""
        logger.info("Model input contains %d messages", len(state["messages"]))
        return None
