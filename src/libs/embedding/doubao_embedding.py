"""Doubao embedding provider using the openai SDK with Volcengine Ark base URL."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import OpenAI

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings

_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class DoubaoEmbedding(BaseEmbedding):
    """Embedding provider backed by the Doubao (Volcengine Ark) API.

    Uses the OpenAI SDK pointed at the Volcengine Ark API endpoint.

    Reads configuration from ``settings.embedding``:
      - ``api_key``: Volcengine API key.
      - ``model``: Embedding model name (e.g. ``doubao-embedding-vision-251215``) or endpoint ID.
      - ``dimensions``: Desired vector dimensionality.
      - ``batch_size``: Maximum texts per API call.
      - ``base_url``: Optional override for the Ark API base URL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._batch_size = settings.embedding.batch_size

        base_url = settings.embedding.base_url or _DOUBAO_BASE_URL
        self._client = OpenAI(
            api_key=settings.embedding.api_key,
            base_url=base_url,
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
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "input": batch,
                }
                if self._dimensions:
                    kwargs["dimensions"] = self._dimensions

                response = self._client.embeddings.create(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"Doubao embedding request failed: {exc}") from exc

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
