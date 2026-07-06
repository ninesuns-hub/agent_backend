import re
import logging
from openai import OpenAI
from typing import List, Dict, Any, Protocol
from .base_agent import BaseAgent
from .tools import Tool
from .prompts import REACT_PROMPT

# 获取模块级日志记录器
logger = logging.getLogger(__name__)

class AgentConfig(Protocol):
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    MODEL_NAME: str
    MAX_TOKENS: int
    SYSTEM_PROMPT: str

class ReactAgent(BaseAgent):
    def __init__(
        self, 
        config: Any,
        tools: List[Tool]
    ) -> None:
        super().__init__()
        self.config = config
        self._client = OpenAI(
            api_key=config.CHAT_API_KEY,
            base_url=config.CHAT_BASE_URL,
        )
        self.tools = {t.name: t for t in tools}
        self.max_iterations = 5
        self.last_observations = [] # 新增：记录最近一次对话的观察结果

    def stream_chat(self, user_input: str):
        user_input = user_input.strip()
        if not user_input:
            yield "请输入你的问题～"
            return

        self.last_observations = [] # 重置记录
        logger.info(f"收到用户输入: {user_input}")
        try:
            yield from self._build_stream_response(user_input)
            logger.info("Agent 响应生成完毕")
        except Exception as e:
            logger.error(f"Agent 运行出错: {str(e)}", exc_info=True)
            yield f"抱歉，处理您的请求时出现了错误: {str(e)}"

    def _build_stream_response(self, user_input: str):
        scratchpad = ""
        tool_names = ", ".join(self.tools.keys())
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

        for i in range(self.max_iterations):
            logger.debug(f"开始第 {i+1} 轮迭代")
            prompt = REACT_PROMPT.format(
                input=user_input,
                tool_names=tool_names,
                tool_descriptions=tool_descriptions,
                agent_scratchpad=scratchpad
            )

            response_text = self._call_llm(prompt)
            
            if not response_text:
                if i == 0: yield "抱歉，我暂时无法思考这个问题，请稍后再试。"
                break

            thought, action, action_input, final_answer = self._parse_output(response_text)
            
            if final_answer:
                chunk_size = 8
                for i in range(0, len(final_answer), chunk_size):
                    yield final_answer[i:i + chunk_size]
                return
            
            if action:
                if action in self.tools:
                    try:
                        observation = self.tools[action].run(action_input)
                        if action == "query_lecture_knowledge":
                            self.last_observations.append(observation)
                    except Exception as e:
                        observation = f"执行工具时出错: {str(e)}"
                else:
                    observation = f"错误：工具 '{action}' 不存在。"
                
                scratchpad += f"\n<thought>{thought or '继续处理'}</thought>\n<action>{action}</action>\n<input>{action_input}</input>\nObservation: {observation}\n"
            else:
                yield response_text
                return

        yield "抱歉，我经过多次尝试仍无法得出结论。"

    def _parse_output(self, text: str):
        def extract_tag(tag_name, default=None):
            start_tag = f"<{tag_name}>"
            end_tag = f"</{tag_name}>"
            if start_tag in text and end_tag in text:
                return text[text.find(start_tag) + len(start_tag) : text.find(end_tag)].strip()
            return default

        thought = extract_tag("thought")
        action = extract_tag("action")
        action_input = extract_tag("input")
        final_answer = extract_tag("answer")

        if thought:
            logger.debug(f"Agent Thought: {thought}")

        if final_answer:
            return thought, None, None, final_answer
        
        if action and action_input:
            return thought, action, action_input, None
        
        if not action and text.strip():
            clean_text = re.sub(r"<[^>]+>", "", text).strip()
            if clean_text:
                return thought, None, None, clean_text

        return thought, None, None, None

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.config.CHAT_MODEL_NAME,
                max_tokens=self.config.MAX_TOKENS,
                temperature=0,
                messages=messages,
                # 移除 stop 序列，让模型完整输出标签
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise e
