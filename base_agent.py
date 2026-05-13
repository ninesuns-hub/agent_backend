from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def chat(self, user_input: str) -> str:
        pass

    @abstractmethod
    def _build_response(self, user_input: str) -> str:
        pass

    def reset(self) -> None:
        pass
