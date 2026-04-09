"""OpenAI LLM provider using the openai SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse

if TYPE_CHECKING:
    from core.settings import Settings


class OpenAILLM(BaseLLM):
    """LLM provider backed by the OpenAI API.

    Reads configuration from ``settings.llm``:
      - ``api_key``: OpenAI API key.
      - ``model``: Model name (e.g. ``gpt-4o``).
      - ``temperature``: Default sampling temperature.
      - ``max_tokens``: Default max completion tokens.
      - ``base_url``: Optional custom base URL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

        client_kwargs: dict[str, Any] = {}
        if settings.llm.api_key:
            client_kwargs["api_key"] = settings.llm.api_key
        if settings.llm.base_url:
            client_kwargs["base_url"] = settings.llm.base_url

        self._client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI.

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
            raise RuntimeError(f"OpenAI chat request failed: {exc}") from exc

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
