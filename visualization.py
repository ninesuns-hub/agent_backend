"""Deterministic policy for proactive Mermaid examples."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class VisualizationDecision:
    required: bool
    reason: str


_OPT_OUT_PATTERNS = (
    r"不要(?:画图|图示|可视化)",
    r"不需要(?:画图|图示|可视化)",
    r"无需(?:画图|图示|可视化)",
    r"只要文字",
    r"纯文字",
    r"\b(?:no|without)\s+(?:a\s+)?diagram\b",
    r"\btext[- ]only\b",
)

_EXPLICIT_PATTERNS = (
    r"画(?:一张|个|出)?图",
    r"用图(?:说明|解释|展示|举例)",
    r"图示",
    r"示意图",
    r"可视化",
    r"\bmermaid\b",
    r"\b(?:draw|visuali[sz]e|diagram)\b",
)

_ADMIN_PATTERNS = (
    r"第几周",
    r"哪一周",
    r"课程安排",
    r"课表",
    r"成绩",
    r"评分",
    r"占(?:比|多少)",
    r"授课教师",
    r"答疑时间",
    r"作业截止",
    r"\b(?:schedule|grading|grade|office hours?)\b",
)

_PROOF_OR_CALCULATION_PATTERNS = (
    r"^\s*(?:请)?证明",
    r"如何证明",
    r"给出证明",
    r"\bprove\b",
    r"^\s*(?:请)?计算",
    r"求解",
    r"\bcalculate\b",
)

_STRUCTURAL_PATTERNS = (
    # Graphs and trees.
    r"图论",
    r"(?:什么是|解释|介绍|定义)图(?:[？?]|$|的)",
    r"图的(?:定义|结构|性质|表示)",
    r"有向图",
    r"无向图",
    r"二分图",
    r"平面图",
    r"完全图",
    r"连通图",
    r"顶点",
    r"边集",
    r"邻接",
    r"路径",
    r"回路",
    r"欧拉",
    r"哈密顿",
    r"图着色",
    r"生成树",
    r"二叉树",
    r"根树",
    r"树结构",
    r"树(?:是什么|的定义|结构|与|和|有哪些|中)",
    r"\b(?:graph|graph theory|directed graph|undirected graph|vertex|vertices|edge set|path|cycle|euler|hamilton|coloring|tree|spanning tree|binary tree|rooted tree)\b",
    # Relations, orders, and mappings.
    r"二元关系",
    r"关系图",
    r"关系(?:是什么|的定义|的性质|矩阵|表示)",
    r"自反性",
    r"对称性",
    r"反对称性",
    r"传递性",
    r"等价关系",
    r"等价类",
    r"偏序",
    r"哈斯",
    r"\bhasse\b",
    r"\b(?:binary relation|equivalence relation|partial order|poset|reflexiv|symmetric|transitive)\b",
    r"单射",
    r"满射",
    r"双射",
    r"映射关系",
    r"\b(?:injective|surjective|bijective|mapping diagram)\b",
    # Set and state structures that benefit from a diagram.
    r"集合运算",
    r"并集",
    r"交集",
    r"差集",
    r"补集",
    r"文氏图",
    r"\bvenn\b",
    r"状态图",
    r"状态转移",
    r"流程结构",
    r"\b(?:state diagram|state transition|flow structure)\b",
)


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def decide_visualization(user_input: str) -> VisualizationDecision:
    """Return whether the question should proactively include Mermaid."""
    normalized = " ".join((user_input or "").strip().split())
    if not normalized:
        return VisualizationDecision(False, "empty")
    if _matches_any(_OPT_OUT_PATTERNS, normalized):
        return VisualizationDecision(False, "user_opt_out")
    if _matches_any(_EXPLICIT_PATTERNS, normalized):
        return VisualizationDecision(True, "explicit_request")
    if _matches_any(_ADMIN_PATTERNS, normalized):
        return VisualizationDecision(False, "course_admin")
    if _matches_any(_PROOF_OR_CALCULATION_PATTERNS, normalized):
        return VisualizationDecision(False, "proof_or_calculation")
    if _matches_any(_STRUCTURAL_PATTERNS, normalized):
        return VisualizationDecision(True, "structural_concept")
    return VisualizationDecision(False, "not_applicable")


def visualization_instruction(decision: VisualizationDecision) -> str:
    if decision.required:
        return (
            "本问题适合通过结构图帮助理解。最终回答必须主动包含至少一个"
            "具体、可读、与正文一致的 fenced Mermaid 示例；先用文字解释，"
            "再用图展示具体对象或关系，不能只给装饰性图表。"
        )
    return (
        "本问题不强制配图。只有图表确实能提升理解时才使用 Mermaid，"
        "不要为了形式机械添加图表。"
    )
