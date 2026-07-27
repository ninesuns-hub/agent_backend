from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentRunContext:
    request_id: str = "-"
    user_id: Optional[int] = None
    user_role: Optional[str] = None
    class_id: Optional[int] = None
    conversation_id: Optional[int] = None
    image_path: Optional[str] = None
    welcome: bool = False
    history_messages: list[dict] = field(default_factory=list)
    summary_text: str = ""
    memories: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Optional[dict]) -> "AgentRunContext":
        source = value or {}
        fields = cls.__dataclass_fields__
        return cls(**{key: source[key] for key in fields if key in source})

    def as_tool_context(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "class_id": self.class_id,
            "conversation_id": self.conversation_id,
            "image_path": self.image_path,
        }


@dataclass
class AgentRunState:
    observations: list[str] = field(default_factory=list)
    visualization_required: bool = False
    visualization_reason: str = ""
    visualization_present: bool = False
    visual_supplement_used: bool = False
