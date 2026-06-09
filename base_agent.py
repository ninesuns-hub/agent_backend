from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def stream_chat(self, user_input: str):
        """流式聊天接口，由子类实现"""
        pass

    def reset(self) -> None:
        """重置 Agent 状态"""
        pass
