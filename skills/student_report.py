"""学情报告 Skill"""
import json
import logging
import re
from typing import Any, Dict, List

from openai import OpenAI
from agent_core.config.settings import settings

logger = logging.getLogger(__name__)

_TRAILING_ADVICE_SECTION = re.compile(
    r"(?:^|\n)\s*(?:学习建议|教学建议)\s*[：:]\s*\n?.*$",
    re.DOTALL,
)


def _clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_trailing_advice_section(text: str) -> str:
    return _TRAILING_ADVICE_SECTION.sub("", text).strip()


def _normalize_stats(stats: dict) -> dict:
    normalized = dict(stats or {})
    student_suggestions = (
        normalized.get("student_suggestions")
        or normalized.get("suggestions")
        or []
    )
    teaching_suggestions = normalized.get("teaching_suggestions") or []
    if not isinstance(student_suggestions, list):
        student_suggestions = []
    if not isinstance(teaching_suggestions, list):
        teaching_suggestions = []
    student_suggestions = [
        str(item).strip() for item in student_suggestions if str(item).strip()
    ]
    teaching_suggestions = [
        str(item).strip() for item in teaching_suggestions if str(item).strip()
    ]
    normalized["student_suggestions"] = student_suggestions
    normalized["teaching_suggestions"] = teaching_suggestions
    normalized["suggestions"] = list(student_suggestions)
    return normalized


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
) -> Dict[str, Any]:
    if not messages:
        return {
            "summary": (
                f"你好，这是关于{student_name}同学的学习情况简要说明。\n\n"
                f"目前还没有看到 TA 与 AI 助教的对话记录，暂时无法做深入分析。"
            ),
            "stats": {
                "message_count": 0,
                "question_count": 0,
                "topics": [],
                "mastery_level": "未知",
                "weak_points": [],
                "student_suggestions": ["你可以使用 AI 助教进行课后答疑"],
                "teaching_suggestions": ["鼓励学生使用 AI 助教进行课后答疑"],
                "suggestions": ["你可以使用 AI 助教进行课后答疑"],
            },
        }

    dialogue = _format_dialogue(messages)
    prompt = f"""你是一位离散数学课程的学习分析专家。请根据以下学生与 AI 助教的对话记录，撰写一份学情报告。

学生姓名：{student_name}
班级：{class_name}
对话条数：{len(messages)}

对话记录：
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
    "student_suggestions": ["直接面向学生、可由学生自行执行的建议1", "建议2"],
    "teaching_suggestions": ["面向教师、可落实到讲解或练习安排的建议1", "建议2"],
    "suggestions": ["与 student_suggestions 完全一致的兼容字段"]
  }}
}}

正文写作要求（必须严格遵守）：
1. 使用纯文本，禁止 Markdown：不要用 #、##、-、*、**、> 等任何格式符号
2. 分段落书写，语气自然、客观，学生本人和任课教师阅读时都容易理解
3. 用小标题引导即可，例如「学习概况：」「涉及知识点：」，标题独占一行，正文另起段落
4. 列举内容用「1. 2. 3.」编号，不要用破折号列表
5. 正文只覆盖：学习概况、涉及知识点、掌握情况、薄弱环节
6. summary_text 中禁止出现「学习建议」「教学建议」标题或任何建议列表，建议只能写入 stats
7. student_suggestions 必须直接面向学生，使用「你可以」等表达，禁止出现「布置、教师、检验学生」等教学操作
8. teaching_suggestions 必须面向教师，给出可落地的讲解、例题、练习或检查方式
9. suggestions 必须与 student_suggestions 完全一致，用于兼容旧接口
10. 基于对话客观分析，不编造未出现的知识点；样本少时如实说明"""

    client = OpenAI(
        api_key=settings.CHAT_API_KEY,
        base_url=settings.CHAT_BASE_URL,
        timeout=120.0,
    )
    try:
        response = client.chat.completions.create(
            model=settings.CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是严谨的教学分析助手，只输出合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        data = json.loads(response.choices[0].message.content)
        summary = data.get("summary_text") or data.get("summary_markdown", "")
        summary = _strip_trailing_advice_section(_clean_text(summary))
        return {"summary": summary, "stats": _normalize_stats(data.get("stats", {}))}
    except Exception as e:
        logger.exception("学情报告生成失败 error_type=%s", type(e).__name__)
        raise RuntimeError("学情报告生成服务暂时不可用") from e
