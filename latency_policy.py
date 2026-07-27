"""Conservative routing policy for latency-sensitive chat requests."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LatencyDecision:
    fast_action: str | None
    thinking_enabled: bool
    reason: str


_COMPLEX_PATTERNS = (
    r"证明",
    r"推导",
    r"反例",
    r"归纳法",
    r"充要条件",
    r"必要性",
    r"充分性",
    r"为什么.*成立",
    r"\b(?:prove|proof|derive|counterexample|induction|necessary and sufficient)\b",
)

_MULTI_INTENT_PATTERNS = (
    r"同时",
    r"另外",
    r"并且",
    r"以及.*(?:作业|成绩|老师|课表|考试)",
    r"(?:作业|成绩|老师|课表|考试).*(?:定义|定理|证明|图论|集合|关系)",
    r"\b(?:and also|as well as)\b",
)

_ADMIN_PATTERNS = (
    r"授课教师",
    r"老师(?:是谁|信息|联系方式)",
    r"评分标准",
    r"成绩构成",
    r"考试规定",
    r"课程安排",
    r"教学大纲",
    r"课表",
    r"上课时间",
    r"上课地点",
    r"答疑时间",
    r"课程编号",
    r"课程学分",
    r"课堂规定",
    r"课程网站",
    r"\b(?:teacher|grading|schedule|office hours?|course credit)\b",
)

_KNOWLEDGE_PATTERNS = (
    r"集合",
    r"命题",
    r"逻辑",
    r"真值表",
    r"谓词",
    r"量词",
    r"关系",
    r"函数",
    r"映射",
    r"单射",
    r"满射",
    r"双射",
    r"偏序",
    r"哈斯",
    r"图论",
    r"有向图",
    r"无向图",
    r"顶点",
    r"边集",
    r"路径",
    r"回路",
    r"欧拉",
    r"哈密顿",
    r"树",
    r"递归",
    r"组合",
    r"排列",
    r"鸽巢",
    r"布尔代数",
    r"离散数学",
    r"\b(?:set|logic|predicate|quantifier|relation|function|injective|"
    r"surjective|bijective|poset|hasse|graph|vertex|edge|path|cycle|"
    r"euler|hamilton|tree|recurrence|combinatorics|boolean algebra)\b",
)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def decide_latency_path(user_input: str, *, has_image: bool = False) -> LatencyDecision:
    text = " ".join((user_input or "").strip().split())
    if has_image:
        return LatencyDecision(None, True, "image_requires_agent")
    if not text:
        return LatencyDecision(None, False, "empty")
    if len(text) > 220 or _matches(_COMPLEX_PATTERNS, text):
        return LatencyDecision(None, True, "complex_reasoning")
    if _matches(_MULTI_INTENT_PATTERNS, text):
        return LatencyDecision(None, True, "multiple_intents")

    admin = _matches(_ADMIN_PATTERNS, text)
    knowledge = _matches(_KNOWLEDGE_PATTERNS, text)
    if admin and not knowledge:
        return LatencyDecision("query_course_admin", False, "clear_course_admin")
    if knowledge and not admin:
        return LatencyDecision(
            "query_lecture_knowledge",
            False,
            "clear_course_knowledge",
        )
    return LatencyDecision(None, False, "general_or_ambiguous")
