"""识图 Skill：离散数学图论、题目截图等"""
import base64
import logging
import mimetypes
import os

from openai import OpenAI
from agent_core.config.settings import settings

logger = logging.getLogger(__name__)

GRAPH_PROMPT = """你是一位离散数学课程助教，擅长阅读图片中的数学内容，尤其是图论、树、证明、集合与逻辑相关题目。

请仔细观察图片，用中文输出：
1. 图片类型（如：图论示意图 / 题目截图 / 手写笔记 / 其他）
2. 图中可见的文字、符号、标注（尽量逐字转写）
3. 若是图论题：顶点数量、边、是否定向、是否标注权重、图的大致结构
4. 学生可能想问什么、涉及哪些知识点
5. 任何有助于后续解题的关键细节

要求客观描述，不要编造看不清的内容；看不清处请说明。"""


def _resolve_image_path(image_path: str) -> str:
    """支持绝对路径或 chat_images 下的相对路径（如 1/abc.png）。"""
    normalized = image_path.replace("\\", "/")
    if os.path.isfile(image_path):
        return image_path
    candidate = os.path.join(settings.CHAT_IMAGES_DIR, normalized)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"图片文件不存在: {image_path}")


def analyze_uploaded_image(image_path: str, user_question: str = "") -> str:
    api_key = settings.VISION_API_KEY or settings.EMBED_API_KEY
    base_url = settings.VISION_BASE_URL or settings.EMBED_BASE_URL
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        abs_path = _resolve_image_path(image_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return f"图片文件未找到（{image_path}）。请重新上传或改用文字描述题目。"

    mime, _ = mimetypes.guess_type(abs_path)
    mime = mime or "image/jpeg"
    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    question_hint = user_question.strip() or "请分析这张图片中的离散数学相关内容。"
    user_content = [
        {"type": "text", "text": f"{GRAPH_PROMPT}\n\n学生附言：{question_hint}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]

    try:
        response = client.chat.completions.create(
            model=settings.VISION_MODEL_NAME,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1200,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"图片理解失败: {e}")
        return f"图片识别暂时不可用（{e}）。请尝试用文字描述题目，或稍后重试。"
