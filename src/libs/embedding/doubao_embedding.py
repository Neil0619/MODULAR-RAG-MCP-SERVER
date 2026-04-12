"""Doubao embedding provider using the openai SDK with Volcengine Ark base URL.

Supports both text-only and multimodal (vision) embedding models:
- Text models (e.g. ``doubao-embedding-large-250515``) use the standard
  ``/api/v3/embeddings`` endpoint via the OpenAI SDK.
- Vision models (e.g. ``doubao-embedding-vision-251215``) use the
  ``multimodal_embeddings`` API via the ``volcenginesdkarkruntime`` SDK.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openai import OpenAI
from volcenginesdkarkruntime import Ark

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import Settings

_DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

logger = logging.getLogger("rag.embedding.doubao")


class DoubaoEmbedding(BaseEmbedding):
    """Embedding provider backed by the Doubao (Volcengine Ark) API.

    Reads configuration from ``settings.embedding``:
      - ``api_key``: Volcengine API key.
      - ``model``: Embedding model name or endpoint ID.
      - ``dimensions``: Desired vector dimensionality.
      - ``batch_size``: Maximum texts per API call.
      - ``base_url``: Optional override for the Ark API base URL.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.embedding.model
        self._dimensions = settings.embedding.dimensions
        self._batch_size = settings.embedding.batch_size
        self._is_vision = "vision" in self._model.lower()

        self._api_key = settings.embedding.api_key
        self._base_url = settings.embedding.base_url or _DOUBAO_BASE_URL

        logger.info(
            "DoubaoEmbedding init: model=%s, is_vision=%s, base_url=%s, "
            "dimensions=%s, batch_size=%d",
            self._model, self._is_vision, self._base_url,
            self._dimensions, self._batch_size,
        )

        if self._is_vision:
            self._ark_client = Ark(api_key=self._api_key, base_url=self._base_url)
        else:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
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

        logger.info(
            "embed called: model=%s, is_vision=%s, text_count=%d",
            self._model, self._is_vision, len(texts),
        )

        if self._is_vision:
            return self._embed_multimodal(texts)
        return self._embed_text(texts)

    def _embed_text(self, texts: list[str]) -> list[list[float]]:
        """Standard text embedding via OpenAI SDK."""
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            logger.info(
                "Text embedding batch: model=%s, batch_size=%d, progress=%d/%d",
                self._model, len(batch), start + len(batch), len(texts),
            )
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "input": batch,
                }
                if self._dimensions:
                    kwargs["dimensions"] = self._dimensions

                response = self._client.embeddings.create(**kwargs)
            except Exception as exc:
                logger.error("Text embedding failed: model=%s, error=%s", self._model, exc)
                raise RuntimeError(f"Doubao embedding request failed: {exc}") from exc

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(
                "Text embedding batch done: %d vectors returned, dim=%d",
                len(batch_embeddings),
                len(batch_embeddings[0]) if batch_embeddings else 0,
            )

        return all_embeddings

    def _embed_multimodal(self, texts: list[str]) -> list[list[float]]:
        """Multimodal embedding via ``volcenginesdkarkruntime`` SDK.

        Calls ``client.multimodal_embeddings.create()`` following the
        official Volcengine Ark documentation for ``doubao-embedding-vision``.
        """
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            logger.info(
                "Multimodal embedding batch: model=%s, batch_size=%d, progress=%d/%d",
                self._model, len(batch), start + len(batch), len(texts),
            )

            for text_idx, text in enumerate(batch):
                input_item = {"type": "text", "text": text}

                try:
                    resp = self._ark_client.multimodal_embeddings.create(
                        model=self._model,
                        encoding_format="float",
                        input=[input_item],
                    )
                except Exception as exc:
                    logger.error(
                        "Multimodal embedding failed: model=%s, text_idx=%d, error=%s",
                        self._model, text_idx, exc,
                    )
                    raise RuntimeError(
                        f"Doubao multimodal embedding request failed: {exc}"
                    ) from exc

                embedding = resp.data.embedding
                # Flatten in case the API returns a nested structure
                if isinstance(embedding, list):
                    vec = []
                    for item in embedding:
                        if isinstance(item, list):
                            vec.extend(item)
                        else:
                            vec.append(item)
                else:
                    vec = list(embedding)

                all_embeddings.append(vec)

            logger.info(
                "Multimodal embedding batch done: %d vectors returned, dim=%d",
                len(batch),
                len(all_embeddings[-1]) if all_embeddings else 0,
            )

        return all_embeddings
