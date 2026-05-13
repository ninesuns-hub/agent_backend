from openai import OpenAI
from typing import List, Dict, Any, Callable, Protocol
from .base_agent import BaseAgent
from .classifier import (
    classify,
    INTENT_ADMIN, INTENT_CONTENT, INTENT_KNOWLEDGE, INTENT_IRRELEVANT,
    ClassifierConfig
)

class AgentConfig(ClassifierConfig, Protocol):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    MODEL_NAME: str
    MAX_TOKENS: int
    SYSTEM_PROMPT: str
    SIMILARITY_THRESHOLD: float

class ChatAgent(BaseAgent):
    def __init__(
        self, 
        config: AgentConfig,
        admin_query_tool: Callable[[str], str | None],
        vector_query_tool: Callable[[str], List[Dict[str, Any]]]
    ) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        self.admin_query_tool = admin_query_tool
        self.vector_query_tool = vector_query_tool

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "请输入你的问题～"

        reply = self._build_response(user_input)
        if reply:
            self.memory.add("user", user_input)
            self.memory.add("assistant", reply)
        return reply

    def _build_response(self, user_input: str) -> str:
        intent = classify(user_input, self.config)
        
        if intent == INTENT_ADMIN:
            return self._handle_admin(user_input)
        elif intent == INTENT_CONTENT:
            return self._handle_content(user_input)
        elif intent == INTENT_KNOWLEDGE:
            return self._handle_knowledge(user_input)
        else:
            return self._handle_irrelevant(user_input)

    def _handle_admin(self, user_input: str) -> str:
        result = self.admin_query_tool(user_input)
        if result:
            return f"【课程信息】\n{result}"
        return "抱歉，我暂时没有找到关于这个问题的确切信息。建议直接联系老师：kejiwei@tongji.edu.cn"

    def _handle_content(self, user_input: str) -> str:
        results = self.vector_query_tool(user_input)
        results = [r for r in results if r.get("similarity", 0) >= self.config.SIMILARITY_THRESHOLD]

        if not results:
            return "在课件和讲义中未找到直接相关内容。你可以换个关键词，或者直接问我通用知识。"

        context = "\n\n".join([f"【来源：{r['source_file']} 第{r['page']}页】\n{r['text']}" for r in results])
        prompt = f"请根据以下课程材料回答问题：{user_input}\n\n材料：\n{context}"
        return self._call_llm(prompt)

    def _handle_knowledge(self, user_input: str) -> str:
        messages = (
            [{"role": "system", "content": self.config.SYSTEM_PROMPT}]
            + self.memory.get_all()
            + [{"role": "user", "content": user_input}]
        )
        return self._call_llm_with_messages(messages)

    def _handle_irrelevant(self, user_input: str) -> str:
        return "这个问题好像超出了离散数学课程的范围。我擅长回答课程安排、课件内容和离散数学知识点。"

    def _call_llm(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_llm_with_messages(messages)

    def _call_llm_with_messages(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self.config.MODEL_NAME,
            max_tokens=self.config.MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content
