"""Azure OpenAI LLM provider using the openai SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AzureOpenAI

from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse

if TYPE_CHECKING:
    from core.settings import Settings


class AzureLLM(BaseLLM):
    """LLM provider backed by Azure OpenAI.

    Reads configuration from ``settings.llm``:
      - ``azure_endpoint``: Azure endpoint URL.
      - ``api_key``: Azure API key.
      - ``api_version``: Azure API version string.
      - ``deployment_name``: Azure deployment name (used as model).
      - ``temperature``: Default sampling temperature.
      - ``max_tokens``: Default max completion tokens.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.llm.deployment_name or settings.llm.model
        self._temperature = settings.llm.temperature
        self._max_tokens = settings.llm.max_tokens

        self._client = AzureOpenAI(
            azure_endpoint=settings.llm.azure_endpoint,
            api_key=settings.llm.api_key,
            api_version=settings.llm.api_version,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        """Send a chat completion request to Azure OpenAI.

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
            raise RuntimeError(f"Azure OpenAI chat request failed: {exc}") from exc

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
