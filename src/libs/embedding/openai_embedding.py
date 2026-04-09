"""OpenAI embedding provider using the openai SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings


class OpenAIEmbedding(BaseEmbedding):
    """Embedding provider backed by the OpenAI API.

    Reads configuration from ``settings.embedding``:
      - ``api_key``: OpenAI API key.
      - ``model``: Embedding model name (e.g. ``text-embedding-3-small``).
      - ``dimensions``: Desired vector dimensionality.
      - ``batch_size``: Maximum texts per API call.
      - ``base_url``: Optional custom base URL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._batch_size = settings.embedding.batch_size

        client_kwargs: dict[str, Any] = {}
        if settings.embedding.api_key:
            client_kwargs["api_key"] = settings.embedding.api_key

        self._client = OpenAI(**client_kwargs)

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
                    dimensions=self._dimensions,
                )
            except Exception as exc:
                raise RuntimeError(f"OpenAI embedding request failed: {exc}") from exc

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
