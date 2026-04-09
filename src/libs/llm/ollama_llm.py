"""Ollama LLM provider using the openai SDK pointed at a local Ollama server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse

if TYPE_CHECKING:
    from core.settings import Settings

_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaLLM(BaseLLM):
    """LLM provider backed by a local Ollama server.

    Uses the OpenAI SDK pointed at the Ollama OpenAI-compatible endpoint.

    Reads configuration from ``settings.llm``:
      - ``model``: Model name (e.g. ``llama3``).
      - ``base_url``: Ollama server URL (default ``http://localhost:11434/v1``).
      - ``temperature``: Default sampling temperature.
      - ``max_tokens``: Default max completion tokens.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

        base_url = settings.llm.base_url or _OLLAMA_DEFAULT_BASE_URL
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama does not require a real key
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Ollama.

        Args:
            messages: Conversation messages.
            temperature: Override default temperature if provided.
            max_tokens: Override default max_tokens if provided.
            trace: Optional TraceContext for observability.

        Returns:
            ChatResponse with generated content.

        Raises:
            RuntimeError: If the API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[m.to_dict() for m in messages],
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama chat request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResponse(
            content=content,
            model=response.model or self._model,
            usage=usage,
        )
