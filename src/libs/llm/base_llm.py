"""Abstract base class for LLM providers.

All LLM implementations (Azure, OpenAI, Ollama, DeepSeek, etc.) must
inherit from :class:`BaseLLM` and implement the :meth:`chat` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Response from an LLM chat call."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
        }


class BaseLLM(ABC):
    """Abstract base for all LLM providers.

    Subclasses must implement :meth:`chat`.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation messages (system/user/assistant).
            temperature: Override default temperature if provided.
            max_tokens: Override default max_tokens if provided.
            trace: Optional TraceContext for observability.

        Returns:
            ChatResponse with generated content.
        """
        ...

    def chat_str(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
        """Convenience: single-turn chat that returns just the text."""
        msgs: list[ChatMessage] = []
        if system:
            msgs.append(ChatMessage(role="system", content=system))
        msgs.append(ChatMessage(role="user", content=prompt))
        return self.chat(msgs, **kwargs).content
