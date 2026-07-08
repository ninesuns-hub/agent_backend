import logging
from typing import Callable, Dict

from agent_core.skills.vision import analyze_uploaded_image
from .base import Tool

logger = logging.getLogger(__name__)


def create_image_understanding_tool(context_getter: Callable[[], Dict]):
    """创建图片识图工具，从 request_context 读取 image_path"""

    def wrapper(question: str) -> str:
        ctx = context_getter() or {}
        image_path = ctx.get("image_path")
        if not image_path:
            return "当前会话没有可分析的图片。请确认学生已上传图片。"
        logger.info(f"识图工具调用: {image_path}")
        return analyze_uploaded_image(image_path, question)

    return Tool(
        name="analyze_uploaded_image",
        func=wrapper,
        description=(
            "当学生上传了图片（图论题、题目截图、手写笔记等）时必须优先调用。"
            "输入应为针对图片的具体问题或分析要求，例如「识别图中的顶点与边」或「转写题目文字」。"
        ),
    )
