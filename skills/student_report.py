"""学情报告 Skill"""
import json
import logging
import re
import time
from typing import Any, Dict, List

from agent_core.config.settings import settings
from agent_core.llm_runtime import get_chat_client, thinking_extra_body

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _format_dialogue(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role_label = "学生" if m["role"] == "user" else "助教"
        lines.append(f"[{role_label}] {m['content']}")
    return "\n\n".join(lines)


def generate_student_report(
    student_name: str,
    class_name: str,
    messages: List[dict],
    previous_summary: str | None = None,
) -> Dict[str, Any]:
    if not messages:
        return {
            "summary": (
                f"你好，这是关于{student_name}同学的学习情况简要说明。\n\n"
                f"目前还没有看到 TA 与 AI 助教的对话记录，暂时无法做深入分析。"
                f"建议鼓励同学课后多向助教提问，哪怕是从一个小问题开始也好。"
            ),
            "stats": {
                "message_count": 0,
                "question_count": 0,
                "topics": [],
                "mastery_level": "未知",
                "weak_points": [],
                "suggestions": ["鼓励学生使用 AI 助教进行课后答疑"],
            },
        }

    dialogue = _format_dialogue(messages)
    previous_context = (
        f"\n上一版完整学情报告：\n{previous_summary}\n"
        if previous_summary else ""
    )
    prompt = f"""你是一位离散数学课程的学习分析专家。请根据以下学生与 AI 助教的对话记录，撰写一份学情报告。

学生姓名：{student_name}
班级：{class_name}
对话条数：{len(messages)}
{previous_context}

{"上一版报告之后的新增对话" if previous_summary else "对话记录"}：
{dialogue}

请输出 JSON：
{{
  "summary_text": "学情报告正文（纯文本）",
  "stats": {{
    "message_count": 对话总条数,
    "question_count": 学生提问次数,
    "topics": ["知识点1", "知识点2"],
    "mastery_level": "较好/一般/需加强",
    "weak_points": ["薄弱点1"],
    "suggestions": ["建议1", "建议2"]
  }}
}}

正文写作要求（必须严格遵守）：
1. 使用纯文本，禁止 Markdown：不要用 #、##、-、*、**、> 等任何格式符号
2. 分段落书写，语气自然亲切，像助教在向任课教师口头汇报，有交流感
3. 用小标题引导即可，例如「学习概况：」「涉及知识点：」，标题独占一行，正文另起段落
4. 列举内容用「1. 2. 3.」编号，每条建议单独成段，不要用破折号列表
5. 须覆盖：学习概况、涉及知识点、掌握情况、薄弱环节、学习建议
6. 基于对话客观分析，不编造未出现的知识点；样本少时如实说明
7. 如果提供了上一版报告，必须在保留仍然有效的信息基础上融合新增对话，
   输出一份可独立阅读的完整新报告，不能只写变化摘要"""

    client = get_chat_client()
    started_at = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是严谨的教学分析助手，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            extra_body=thinking_extra_body(False),
        )
        data = json.loads(response.choices[0].message.content)
        summary = data.get("summary_text") or data.get("summary_markdown", "")
        usage = getattr(response, "usage", None)
        logger.info(
            "学情报告模型调用完成 elapsed_ms=%.2f input_messages=%s "
            "incremental=%s prompt_tokens=%s completion_tokens=%s",
            (time.perf_counter() - started_at) * 1000,
            len(messages),
            bool(previous_summary),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return {"summary": _clean_text(summary), "stats": data.get("stats", {})}
    except Exception as e:
        logger.exception("学情报告生成失败 error_type=%s", type(e).__name__)
        raise RuntimeError("学情报告生成服务暂时不可用") from e
