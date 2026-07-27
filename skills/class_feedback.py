"""班级学情反馈 Skill"""
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


def _format_reports(reports: List[dict]) -> str:
    if not reports:
        return "（暂无学生学情报告）"
    lines = []
    for r in reports:
        lines.append(f"{r['student_name']}：\n{r.get('summary_excerpt', r.get('summary', ''))[:800]}")
    return "\n\n".join(lines)


def _format_stats(stats: List[dict]) -> str:
    if not stats:
        return "（暂无活跃度数据）"
    return "\n".join(f"{s['student_name']}: {s['message_count']} 条对话" for s in stats)


def generate_class_feedback(
    class_name: str,
    student_reports: List[dict],
    class_message_stats: List[dict],
) -> Dict[str, Any]:
    if not student_reports and not class_message_stats:
        return {
            "summary": (
                f"关于{class_name}的班级学情，目前还没有收集到足够的互动数据。\n\n"
                f"建议先鼓励同学们多与 AI 助教交流，积累一段时间后再来看整体情况，会更有参考价值。"
            ),
            "stats": {
                "student_count": 0,
                "active_students": 0,
                "common_topics": [],
                "class_weak_points": [],
                "teaching_suggestions": ["鼓励学生使用 AI 助教进行课后练习"],
            },
        }

    prompt = f"""你是一位离散数学课程的教学顾问。请根据班级学生的学情数据，为任课教师撰写班级学情反馈。

班级：{class_name}

各学生学情摘要：
{_format_reports(student_reports)}

学生活跃度统计：
{_format_stats(class_message_stats)}

请输出 JSON：
{{
  "summary_text": "班级学情反馈正文（纯文本）",
  "stats": {{
    "student_count": 学生总数,
    "active_students": 有对话记录的学生数,
    "common_topics": ["班级共性知识点"],
    "class_weak_points": ["班级薄弱点"],
    "teaching_suggestions": ["教学建议1", "教学建议2"]
  }}
}}

正文写作要求（必须严格遵守）：
1. 纯文本，禁止 Markdown：不要用 #、##、-、*、** 等符号
2. 分段落、语气自然，像教学顾问与教师面对面交流
3. 用小标题引导，如「班级整体情况：」「共性问题：」「教学建议：」
4. 列举用「1. 2. 3.」编号，每条单独成段
5. 从教师视角给出可落地的建议；数据不足时如实说明"""

    client = get_chat_client()
    started_at = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是严谨的教学顾问，只输出合法 JSON。"},
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
            "班级学情模型调用完成 elapsed_ms=%.2f student_count=%s "
            "prompt_tokens=%s completion_tokens=%s",
            (time.perf_counter() - started_at) * 1000,
            len(student_reports),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )
        return {"summary": _clean_text(summary), "stats": data.get("stats", {})}
    except Exception as e:
        logger.exception("班级学情反馈生成失败 error_type=%s", type(e).__name__)
        raise RuntimeError("班级学情反馈生成服务暂时不可用") from e
