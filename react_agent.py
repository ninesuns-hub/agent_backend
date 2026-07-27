import re
import logging
import time
import json
import threading
from typing import Iterator, List, Dict, Any, Protocol
from .base_agent import BaseAgent
from .tools import Tool
from .prompts import REACT_PROMPT
from .latency_policy import decide_latency_path
from .llm_runtime import get_chat_client, thinking_extra_body
from .visualization import (
    VisualizationDecision,
    decide_visualization,
    visualization_instruction,
)

# 获取模块级日志记录器
logger = logging.getLogger(__name__)


class AnswerStreamExtractor:
    """Incrementally expose only text inside an <answer> element."""

    _OPEN = "<answer>"
    _CLOSE = "</answer>"

    def __init__(self) -> None:
        self.buffer = ""
        self.answer_started = False
        self.answer_finished = False

    @staticmethod
    def _partial_suffix_length(text: str, token: str) -> int:
        max_length = min(len(text), len(token) - 1)
        for length in range(max_length, 0, -1):
            if text.endswith(token[:length]):
                return length
        return 0

    def feed(self, text: str) -> List[str]:
        if not text or self.answer_finished:
            return []
        self.buffer += text
        output: List[str] = []

        if not self.answer_started:
            marker_index = self.buffer.find(self._OPEN)
            if marker_index < 0:
                # Retain the full pre-answer response for normal ReAct parsing.
                return output
            self.answer_started = True
            self.buffer = self.buffer[marker_index + len(self._OPEN):]

        close_index = self.buffer.find(self._CLOSE)
        if close_index >= 0:
            if close_index:
                output.append(self.buffer[:close_index])
            self.buffer = ""
            self.answer_finished = True
            return output

        held = self._partial_suffix_length(self.buffer, self._CLOSE)
        safe_length = len(self.buffer) - held
        if safe_length:
            output.append(self.buffer[:safe_length])
            self.buffer = self.buffer[safe_length:]
        return output

    def finish(self) -> List[str]:
        if self.answer_started and not self.answer_finished and self.buffer:
            remaining = self.buffer
            self.buffer = ""
            return [remaining]
        return []


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
        self._client = get_chat_client(config)
        self.tools = {t.name: t for t in tools}
        self.max_iterations = 5
        self._request_state = threading.local()
        self.request_context = {}
        self.last_observations = []

    @property
    def request_context(self) -> Dict[str, Any]:
        return getattr(self._request_state, "request_context", {})

    @request_context.setter
    def request_context(self, value: Dict[str, Any]) -> None:
        self._request_state.request_context = value

    @property
    def last_observations(self) -> List[Any]:
        return getattr(self._request_state, "last_observations", [])

    @last_observations.setter
    def last_observations(self, value: List[Any]) -> None:
        self._request_state.last_observations = value

    def stream_chat(self, user_input: str, request_context: Dict[str, Any] | None = None):
        for event in self.stream_events(user_input, request_context=request_context):
            if event.get("type") == "content":
                yield event.get("delta", "")
            elif event.get("type") == "error":
                yield event.get("message", "抱歉，处理您的请求时出现了错误。")

    def stream_events(
        self,
        user_input: str,
        request_context: Dict[str, Any] | None = None,
    ) -> Iterator[Dict[str, Any]]:
        self.request_context = request_context or {}
        user_input = user_input.strip()
        if not user_input:
            yield {"type": "content", "delta": "请输入你的问题～"}
            return

        self.last_observations = []
        request_id = self.request_context.get("request_id", "-")
        visual_decision = (
            VisualizationDecision(False, "welcome")
            if self.request_context.get("welcome")
            else decide_visualization(user_input)
        )
        self.request_context["visualization_required"] = visual_decision.required
        self.request_context["visualization_reason"] = visual_decision.reason
        latency_decision = decide_latency_path(
            user_input,
            has_image=bool(self.request_context.get("image_path")),
        )
        adaptive_thinking = getattr(
            self.config,
            "CHAT_ADAPTIVE_THINKING_ENABLED",
            True,
        )
        self.request_context["thinking_enabled"] = (
            latency_decision.thinking_enabled if adaptive_thinking else True
        )
        self.request_context["latency_route"] = latency_decision.reason
        started_at = time.perf_counter()
        logger.info(
            "收到用户输入 request_id=%s input_length=%s visualization_required=%s "
            "visualization_reason=%s latency_route=%s thinking_enabled=%s",
            request_id,
            len(user_input),
            visual_decision.required,
            visual_decision.reason,
            latency_decision.reason,
            self.request_context["thinking_enabled"],
        )
        try:
            yield {"type": "status", "stage": "understanding"}
            fast_path_enabled = getattr(
                self.config,
                "CHAT_FAST_PATH_ENABLED",
                False,
            )
            if (
                fast_path_enabled
                and latency_decision.fast_action
                and latency_decision.fast_action in self.tools
            ):
                yield from self._build_fast_response(
                    user_input,
                    latency_decision.fast_action,
                )
            else:
                yield from self._build_stream_response(user_input)
            self._log_timing(
                request_id,
                "agent_total",
                started_at,
            )
            logger.info("Agent 响应生成完毕")
        except Exception as e:
            logger.error(f"Agent 运行出错: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "message": f"抱歉，处理您的请求时出现了错误: {str(e)}",
            }

    def _build_fast_response(
        self,
        user_input: str,
        action: str,
    ) -> Iterator[Dict[str, Any]]:
        """Run one unambiguous tool and stream a terminal answer directly."""
        request_id = self.request_context.get("request_id", "-")
        yield {"type": "status", "stage": self._status_for_action(action)}
        tool_started_at = time.perf_counter()
        try:
            observation = self.tools[action].run(user_input)
            if action == "query_lecture_knowledge":
                self.last_observations.append(observation)
        except Exception as exc:
            logger.warning(
                "快路径工具失败，回退 ReAct request_id=%s tool=%s error=%s",
                request_id,
                action,
                type(exc).__name__,
            )
            self._log_timing(
                request_id,
                "fast_path_tool",
                tool_started_at,
                tool=action,
                success=False,
            )
            yield {"type": "status", "stage": "organizing"}
            yield from self._build_stream_response(user_input)
            return

        self._log_timing(
            request_id,
            "fast_path_tool",
            tool_started_at,
            tool=action,
            success=True,
        )
        yield {"type": "status", "stage": "organizing"}
        prompt = self._direct_answer_prompt(
            user_input,
            action,
            observation or "没有查询到可用信息。",
        )
        answer_parts: List[str] = []
        llm_started_at = time.perf_counter()
        first_content_at = None
        for delta in self._call_direct_llm_stream(prompt):
            if first_content_at is None:
                first_content_at = time.perf_counter()
                self._log_timing(
                    request_id,
                    "direct_first_answer_token",
                    llm_started_at,
                    tool=action,
                )
            answer_parts.append(delta)
            yield {"type": "content", "delta": delta}
        self._log_timing(
            request_id,
            "direct_answer",
            llm_started_at,
            tool=action,
        )
        yield from self._ensure_visualization(
            user_input,
            "".join(answer_parts),
        )

    def _direct_answer_prompt(
        self,
        user_input: str,
        action: str,
        observation: str,
    ) -> str:
        source_requirement = (
            "如果资料中提供了文件名和页码，请在相关小节开头使用“**来源**”"
            "列表准确标注；没有命中资料时不要编造来源。"
            if action == "query_lecture_knowledge"
            else "只使用查询结果中确实存在的课程信息，不要编造。"
        )
        decision = VisualizationDecision(
            bool(self.request_context.get("visualization_required")),
            str(self.request_context.get("visualization_reason", "")),
        )
        return (
            "工具调用已经完成。请直接给出最终回答，不要输出 thought、action、"
            "input、answer 等控制标签，也不要再次决定或调用工具。\n"
            "保持现有回答深度，不缩短成简略答案；先给结论，再清晰解释并举例。"
            "数学表达使用标准 LaTeX，排版使用 Markdown。\n"
            f"{source_requirement}\n"
            f"{visualization_instruction(decision)}\n\n"
            f"用户问题：\n{user_input}\n\n"
            f"工具结果：\n{observation}"
        )

    def _build_stream_response(self, user_input: str):
        scratchpad = ""
        tool_names = ", ".join(self.tools.keys())
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        request_id = self.request_context.get("request_id", "-")

        for i in range(self.max_iterations):
            logger.debug(f"开始第 {i+1} 轮迭代")
            # 不用 str.format：scratchpad/用户输入里常有 LaTeX {bmatrix} 等花括号，会触发 KeyError
            prompt = (
                REACT_PROMPT
                .replace("{input}", user_input)
                .replace("{tool_names}", tool_names)
                .replace("{tool_descriptions}", tool_descriptions)
                .replace("{agent_scratchpad}", scratchpad)
                .replace(
                    "{visualization_instruction}",
                    visualization_instruction(
                        VisualizationDecision(
                            bool(self.request_context.get("visualization_required")),
                            str(self.request_context.get("visualization_reason", "")),
                        )
                    ),
                )
            )

            llm_started_at = time.perf_counter()
            response_text = ""
            streamed_answer = False
            streamed_answer_parts: List[str] = []
            extractor = AnswerStreamExtractor()
            first_answer_at = None
            for raw_delta in self._call_llm_stream(prompt):
                response_text += raw_delta
                for answer_delta in extractor.feed(raw_delta):
                    if answer_delta:
                        if first_answer_at is None:
                            first_answer_at = time.perf_counter()
                            logger.info(
                                json.dumps(
                                    {
                                        "event": "chat_timing",
                                        "request_id": request_id,
                                        "stage": "llm_first_answer_token",
                                        "iteration": i + 1,
                                        "elapsed_ms": round(
                                            (first_answer_at - llm_started_at) * 1000,
                                            2,
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        streamed_answer = True
                        streamed_answer_parts.append(answer_delta)
                        yield {"type": "content", "delta": answer_delta}
            for answer_delta in extractor.finish():
                if answer_delta:
                    streamed_answer = True
                    streamed_answer_parts.append(answer_delta)
                    yield {"type": "content", "delta": answer_delta}
            self._log_timing(
                request_id,
                "llm_iteration",
                llm_started_at,
                iteration=i + 1,
            )
            
            if not response_text:
                if i == 0:
                    yield {
                        "type": "content",
                        "delta": "抱歉，我暂时无法思考这个问题，请稍后再试。",
                    }
                break

            thought, action, action_input, final_answer = self._parse_output(response_text)
            
            if final_answer:
                if not streamed_answer:
                    streamed_answer_parts.append(final_answer)
                    yield {"type": "content", "delta": final_answer}
                yield from self._ensure_visualization(
                    user_input,
                    "".join(streamed_answer_parts) or final_answer,
                )
                return
            
            if action:
                if action in self.tools:
                    yield {
                        "type": "status",
                        "stage": self._status_for_action(action),
                    }
                    tool_started_at = time.perf_counter()
                    try:
                        observation = self.tools[action].run(action_input)
                        if action in ("query_lecture_knowledge", "analyze_uploaded_image"):
                            self.last_observations.append(observation)
                    except Exception as e:
                        observation = f"执行工具时出错: {str(e)}"
                    self._log_timing(
                        request_id,
                        "tool",
                        tool_started_at,
                        iteration=i + 1,
                        tool=action,
                    )
                else:
                    observation = f"错误：工具 '{action}' 不存在。"
                
                scratchpad += f"\n<thought>{thought or '继续处理'}</thought>\n<action>{action}</action>\n<input>{action_input}</input>\nObservation: {observation}\n"
                yield {"type": "status", "stage": "organizing"}
            else:
                if not streamed_answer:
                    yield {"type": "content", "delta": response_text}
                answer_text = "".join(streamed_answer_parts) or response_text
                yield from self._ensure_visualization(user_input, answer_text)
                return

        yield {"type": "content", "delta": "抱歉，我经过多次尝试仍无法得出结论。"}

    @staticmethod
    def _status_for_action(action: str) -> str:
        return {
            "query_lecture_knowledge": "retrieving",
            "query_course_admin": "querying_course",
            "analyze_uploaded_image": "analyzing_image",
            "generate_student_learning_report": "analyzing_learning",
            "generate_class_learning_feedback": "analyzing_learning",
        }.get(action, "using_tool")

    @staticmethod
    def _log_timing(
        request_id: str,
        stage: str,
        started_at: float,
        **fields: Any,
    ) -> None:
        logger.info(
            json.dumps(
                {
                    "event": "chat_timing",
                    "request_id": request_id,
                    "stage": stage,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    **fields,
                },
                ensure_ascii=False,
            )
        )

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

    def _ensure_visualization(
        self,
        user_input: str,
        answer_text: str,
    ) -> Iterator[Dict[str, Any]]:
        required = bool(self.request_context.get("visualization_required"))
        present = "```mermaid" in answer_text.lower()
        request_id = self.request_context.get("request_id", "-")
        self.request_context["visualization_present"] = present
        self.request_context["visual_supplement_used"] = False

        if not required or present:
            logger.info(
                json.dumps(
                    {
                        "event": "visual_compliance",
                        "request_id": request_id,
                        "visualization_required": required,
                        "visualization_present": present,
                        "visual_supplement_used": False,
                    },
                    ensure_ascii=False,
                )
            )
            return

        yield {"type": "status", "stage": "generating_visual"}
        started_at = time.perf_counter()
        try:
            supplement = self._generate_visual_supplement(user_input, answer_text)
            if supplement:
                self.request_context["visualization_present"] = True
                self.request_context["visual_supplement_used"] = True
                yield {"type": "content", "delta": supplement}
            self._log_timing(
                request_id,
                "visual_supplement",
                started_at,
                success=bool(supplement),
            )
        except Exception as exc:
            logger.warning(
                "补充 Mermaid 图示失败 request_id=%s error=%s",
                request_id,
                type(exc).__name__,
            )
            self._log_timing(
                request_id,
                "visual_supplement",
                started_at,
                success=False,
                error_type=type(exc).__name__,
            )
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "visual_compliance",
                        "request_id": request_id,
                        "visualization_required": required,
                        "visualization_present": bool(
                            self.request_context.get("visualization_present")
                        ),
                        "visual_supplement_used": bool(
                            self.request_context.get("visual_supplement_used")
                        ),
                    },
                    ensure_ascii=False,
                )
            )

    def _generate_visual_supplement(
        self,
        user_input: str,
        answer_text: str,
    ) -> str:
        prompt = (
            "请仅根据下面的问题和已有回答，生成一张用于辅助理解的紧凑 Mermaid 图。"
            "图必须展示一个具体例子，不得引入已有回答之外的新结论或来源。"
            "只输出一个 fenced mermaid 代码块，不要输出解释文字。"
            "节点和子图使用英文 ID；中文及特殊字符标签放在双引号中；"
            "标签必须是与正文一致、有意义的中文或英文，禁止乱码或占位文本；"
            "标签内不用 ASCII 箭头；多个独立子图按阅读顺序紧凑排列。\n\n"
            f"问题：{user_input[:3000]}\n\n"
            f"已有回答：{answer_text[:8000]}"
        )
        response = self._client.chat.completions.create(
            model=self.config.CHAT_MODEL_NAME,
            max_tokens=min(self.config.MAX_TOKENS, 1200),
            temperature=0,
            extra_body=thinking_extra_body(False),
            messages=[
                {
                    "role": "system",
                    "content": "你是 Mermaid 图示生成器，只输出安全、可渲染的 Mermaid 源码。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        match = re.search(
            r"```mermaid\s*([\s\S]*?)```",
            content,
            flags=re.IGNORECASE,
        )
        source = (match.group(1) if match else content).strip()
        if not source or "```" in source:
            return ""
        return f"\n\n### 图示示例\n\n```mermaid\n{source}\n```"

    def _call_llm_stream(self, prompt: str) -> Iterator[str]:
        messages = [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        thinking_enabled = bool(
            self.request_context.get("thinking_enabled", True)
        )
        try:
            response = self._client.chat.completions.create(
                model=self.config.CHAT_MODEL_NAME,
                max_tokens=self.config.MAX_TOKENS,
                temperature=0,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=thinking_extra_body(thinking_enabled),
                # 移除 stop 序列，让模型完整输出标签
            )
            reasoning_chars = 0
            usage = None
            for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning_chars += len(
                    getattr(delta, "reasoning_content", None) or ""
                )
                content = getattr(delta, "content", None)
                if content:
                    yield content
            self._log_llm_usage(
                usage,
                thinking_enabled=thinking_enabled,
                reasoning_chars=reasoning_chars,
                stage="react",
            )
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise

    def _call_direct_llm_stream(self, prompt: str) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self.config.CHAT_MODEL_NAME,
                max_tokens=self.config.MAX_TOKENS,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                stream_options={"include_usage": True},
                extra_body=thinking_extra_body(False),
            )
            usage = None
            for chunk in response:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
            self._log_llm_usage(
                usage,
                thinking_enabled=False,
                reasoning_chars=0,
                stage="direct_answer",
            )
        except Exception as exc:
            logger.error("直接回答模型调用失败: %s", exc)
            raise

    def _log_llm_usage(
        self,
        usage: Any,
        *,
        thinking_enabled: bool,
        reasoning_chars: int,
        stage: str,
    ) -> None:
        payload = {
            "event": "chat_llm_usage",
            "request_id": self.request_context.get("request_id", "-"),
            "stage": stage,
            "thinking_enabled": thinking_enabled,
            "reasoning_chars": reasoning_chars,
        }
        if usage is not None:
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                value = getattr(usage, field, None)
                if value is not None:
                    payload[field] = value
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(details, "cached_tokens", None)
            if cached_tokens is not None:
                payload["cached_tokens"] = cached_tokens
        logger.info(json.dumps(payload, ensure_ascii=False))
