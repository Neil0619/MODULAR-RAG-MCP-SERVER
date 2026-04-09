"""Azure OpenAI embedding provider using the openai SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AzureOpenAI

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings


class AzureEmbedding(BaseEmbedding):
    """Embedding provider backed by Azure OpenAI.

    Reads configuration from ``settings.embedding``:
      - ``azure_endpoint``: Azure endpoint URL.
      - ``api_key``: Azure API key.
      - ``api_version``: Azure API version string.
      - ``deployment_name``: Azure deployment name (used as model).
      - ``dimensions``: Desired vector dimensionality.
      - ``batch_size``: Maximum texts per API call.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.embedding.deployment_name or settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._batch_size = settings.embedding.batch_size

        self._client = AzureOpenAI(
            azure_endpoint=settings.embedding.azure_endpoint,
            api_key=settings.embedding.api_key,
            api_version=settings.embedding.api_version,
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
                    dimensions=self._dimensions,
                )
            except Exception as exc:
                raise RuntimeError(f"Azure embedding request failed: {exc}") from exc

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
