"""Ollama embedding provider using the openai SDK pointed at a local Ollama server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings

_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaEmbedding(BaseEmbedding):
    """Embedding provider backed by a local Ollama server.

    Uses the OpenAI SDK pointed at the Ollama OpenAI-compatible endpoint.

    Reads configuration from ``settings.embedding``:
      - ``model``: Embedding model name (e.g. ``nomic-embed-text``).
      - ``base_url``: Ollama server URL (default ``http://localhost:11434/v1``).
      - ``dimensions``: Desired vector dimensionality.
      - ``batch_size``: Maximum texts per API call.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._batch_size = settings.embedding.batch_size

        base_url = getattr(settings.embedding, "base_url", "") or _OLLAMA_DEFAULT_BASE_URL
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama does not require a real key
        )

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimensions

    def embed(
        self,
        texts: list[str],
        *,
        trace: Any | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        If the number of texts exceeds ``batch_size``, the request is split
        into multiple calls.

        Args:
            texts: List of strings to embed.
            trace: Optional TraceContext for observability.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If texts is empty.
            RuntimeError: If the API call fails.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as exc:
                raise RuntimeError(f"Ollama embedding request failed: {exc}") from exc

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
