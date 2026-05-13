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

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "请输入你的问题～"

        logger.info(f"收到用户输入: {user_input}")
        try:
            response = self._build_response(user_input)
            logger.info("Agent 响应生成完毕")
            return response
        except Exception as e:
            logger.error(f"Agent 运行出错: {str(e)}", exc_info=True)
            return f"抱歉，处理您的请求时出现了错误: {str(e)}"

    def _build_response(self, user_input: str) -> str:
        scratchpad = ""
        tool_names = ", ".join(self.tools.keys())
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])

        for i in range(self.max_iterations):
            logger.debug(f"开始第 {i+1} 轮迭代")
            # 1. 构造 Prompt
            prompt = REACT_PROMPT.format(
                input=user_input,
                tool_names=tool_names,
                tool_descriptions=tool_descriptions,
                agent_scratchpad=scratchpad
            )

            # 2. 调用 LLM
            response_text = self._call_llm(prompt)
            if not response_text:
                logger.warning("LLM 返回了空响应")
                if i == 0:
                    return "抱歉，我暂时无法思考这个问题，请稍后再试。"
                break

            logger.debug(f"LLM 输出: \n{response_text}")
            
            # 3. 解析输出
            thought, action, action_input, final_answer = self._parse_output(response_text)
            
            if final_answer:
                logger.info(f"得出最终答案: {final_answer}")
                return final_answer
            
            if action:
                # 4. 执行工具
                if action in self.tools:
                    logger.info(f"调用工具: [{action}] | 输入: {action_input}")
                    try:
                        observation = self.tools[action].run(action_input)
                        logger.info(f"工具观测结果 (前100字符): {str(observation)[:100]}...")
                        logger.debug(f"工具观测结果: {observation}")
                    except Exception as e:
                        observation = f"执行工具时出错: {str(e)}"
                        logger.error(f"工具 [{action}] 执行失败: {str(e)}")
                else:
                    observation = f"错误：工具 '{action}' 不存在。请从 [{tool_names}] 中选择。"
                    logger.warning(f"模型尝试调用不存在的工具: {action}")
                
                scratchpad += f"\n<thought>{thought or '继续处理'}</thought>\n<action>{action}</action>\n<input>{action_input}</input>\nObservation: {observation}\n"
            else:
                logger.warning("模型未按格式输出 Action 或 Final Answer，尝试直接返回内容")
                return response_text

        logger.warning(f"达到最大迭代次数 ({self.max_iterations})，未能得出最终答案")
        return "抱歉，我经过多次尝试仍无法得出结论。请尝试换种方式提问。"

    def _parse_output(self, text: str):
        """
        使用标签提取逻辑，比正则更鲁棒
        """
        def extract_tag(tag_name, default=None):
            start_tag = f"<{tag_name}>"
            end_tag = f"</{tag_name}>"
            if start_tag in text and end_tag in text:
                return text[text.find(start_tag) + len(start_tag) : text.find(end_tag)].strip()
            return default

        # 提取各个组件
        thought = extract_tag("thought")
        action = extract_tag("action")
        action_input = extract_tag("input")
        final_answer = extract_tag("answer")

        # 记录思考过程
        if thought:
            logger.debug(f"Agent Thought: {thought}")

        if final_answer:
            return thought, None, None, final_answer
        
        if action and action_input:
            return thought, action, action_input, None
        
        # 兜底逻辑：如果模型没打标签但输出了内容，尝试将其视为最终回答
        if not action and text.strip():
            clean_text = re.sub(r"<[^>]+>", "", text).strip() # 移除所有残余标签
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
