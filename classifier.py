import json
from openai import OpenAI
from typing import List, Protocol

class ClassifierConfig(Protocol):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    MODEL_NAME: str
    COURSE_ADMIN_KEYWORDS: List[str]
    COURSE_CONTENT_KEYWORDS: List[str]

INTENT_ADMIN = "admin"
INTENT_CONTENT = "content"
INTENT_KNOWLEDGE = "knowledge"
INTENT_IRRELEVANT = "irrelevant"

_CLASSIFY_PROMPT = """你是一个意图分类器，负责判断学生问题属于哪一类。

四种类别定义：
1. admin：课程事务类（考试、作业、成绩、上课地点、老师联系方式等）
2. content：教学内容类（课件、讲义、PPT中的具体题目或章节内容）
3. knowledge：通用知识点类（离散数学概念解释、定理证明、不依赖课件的数学问题）
4. irrelevant：与课程完全无关的话题

要求：
- 只返回 JSON，不要任何其他文字
- 格式：{"intent": "admin"} 或 {"intent": "admin,knowledge"}
"""

def classify(question: str, config: ClassifierConfig) -> str:
    try:
        return _classify_by_llm(question, config)
    except Exception:
        return _classify_by_keywords(question, config)

def _classify_by_llm(question: str, config: ClassifierConfig) -> str:
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        max_tokens=80,
        temperature=0,
        messages=[
            {"role": "system", "content": _CLASSIFY_PROMPT},
            {"role": "user", "content": f"请对以下问题分类：{question}"},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Handle potential markdown code blocks in LLM response
    if raw.startswith("```json"):
        raw = raw[7:-3].strip()
    elif raw.startswith("```"):
        raw = raw[3:-3].strip()
        
    result = json.loads(raw)
    intent = result.get("intent", "").strip()
    
    valid = {INTENT_ADMIN, INTENT_CONTENT, INTENT_KNOWLEDGE, INTENT_IRRELEVANT}
    first_intent = intent.split(",")[0].strip()
    if first_intent in valid:
        return first_intent
    return _classify_by_keywords(question, config)

def _classify_by_keywords(question: str, config: ClassifierConfig) -> str:
    for kw in config.COURSE_ADMIN_KEYWORDS:
        if kw in question: return INTENT_ADMIN
    for kw in config.COURSE_CONTENT_KEYWORDS:
        if kw in question: return INTENT_CONTENT
    
    MATH_KEYWORDS = ["什么是", "怎么", "如何", "证明", "计算", "解释"]
    for kw in MATH_KEYWORDS:
        if kw in question: return INTENT_KNOWLEDGE
        
    return INTENT_IRRELEVANT
