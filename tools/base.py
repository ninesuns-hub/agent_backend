from typing import Callable, Any

class Tool:
    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        accepts_context: bool = False,
    ):
        self.name = name
        self.func = func
        self.description = description
        self.accepts_context = accepts_context

    def run(self, tool_input: str, context: dict | None = None) -> str:
        try:
            if self.accepts_context:
                return str(self.func(tool_input, context or {}))
            return str(self.func(tool_input))
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"
