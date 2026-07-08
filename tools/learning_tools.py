import logging
import re
from typing import Callable

from .base import Tool

logger = logging.getLogger(__name__)


def _parse_kv_input(tool_input: str) -> dict:
    result = {}
    for part in re.split(r"[;,\n]", tool_input.strip()):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def create_student_report_tool(handler: Callable[[int, int], str]):
    """
    handler(student_id, class_id) -> 学情报告正文（供 Agent Observation）
 由后端注入 DB 查询与持久化逻辑
    """

    def wrapper(tool_input: str) -> str:
        params = _parse_kv_input(tool_input)
        try:
            student_id = int(params.get("student_id", 0))
            class_id = int(params.get("class_id", 0))
        except ValueError:
            return "参数格式错误。请使用：student_id=学号ID,class_id=班级ID"
        if not student_id or not class_id:
            return "缺少 student_id 或 class_id。格式：student_id=1,class_id=2"
        logger.info(f"学情报告工具: student={student_id}, class={class_id}")
        return handler(student_id, class_id)

    return Tool(
        name="generate_student_learning_report",
        func=wrapper,
        description=(
            "为教师生成某位学生的学情报告。输入格式：student_id=学生用户ID,class_id=班级ID。"
            "适用于教师想了解某学生在班级中的学习互动与掌握情况时。"
        ),
    )


def create_class_feedback_tool(handler: Callable[[int], str]):
    """handler(class_id) -> 班级学情反馈正文"""

    def wrapper(tool_input: str) -> str:
        params = _parse_kv_input(tool_input)
        try:
            class_id = int(params.get("class_id", 0))
        except ValueError:
            return "参数格式错误。请使用：class_id=班级ID"
        if not class_id:
            if tool_input.strip().isdigit():
                class_id = int(tool_input.strip())
            else:
                return "缺少 class_id。格式：class_id=1"
        logger.info(f"班级学情反馈工具: class={class_id}")
        return handler(class_id)

    return Tool(
        name="generate_class_learning_feedback",
        func=wrapper,
        description=(
            "为教师生成整个班级的学情反馈与教学建议。输入格式：class_id=班级ID。"
            "适用于教师需要班级整体学情概览时。"
        ),
    )
