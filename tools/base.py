from typing import Callable, Any

class Tool:
    def __init__(self, name: str, func: Callable[[str], Any], description: str):
        self.name = name
        self.func = func
        self.description = description

    def run(self, tool_input: str) -> str:
        try:
            return str(self.func(tool_input))
        except Exception as e:
            return f"Error executing tool {self.name}: {str(e)}"
