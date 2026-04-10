"""Doubao LLM provider using the openai SDK with Volcengine Ark base URL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse

if TYPE_CHECKING:
    from core.settings import Settings

_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class DoubaoLLM(BaseLLM):
    """LLM provider backed by the Doubao (Volcengine Ark) API.

    Uses the OpenAI SDK pointed at the Volcengine Ark API endpoint.

    Reads configuration from ``settings.llm``:
      - ``api_key``: Volcengine API key.
      - ``model``: Model name (e.g. ``doubao-seed-2-0-pro-260215``) or endpoint ID.
      - ``temperature``: Default sampling temperature.
      - ``max_tokens``: Default max completion tokens.
      - ``base_url``: Optional override for the Ark API base URL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

        base_url = settings.llm.base_url or _DOUBAO_BASE_URL
        self._client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=base_url,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Doubao.

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
            raise RuntimeError(f"Doubao chat request failed: {exc}") from exc

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
