"""Unit tests for LLM factory and base class (B1)."""

from dataclasses import dataclass
from typing import Any

import pytest

from core.settings import (
    EmbeddingSettings,
    LLMSettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
)
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse
from libs.llm.llm_factory import LLMFactory

# ---------------------------------------------------------------------------
# Fake LLM for testing factory routing
# ---------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    """A deterministic fake LLM for unit tests."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: Any | None = None,
    ) -> ChatResponse:
        last_user = [m for m in messages if m.role == "user"][-1]
        return ChatResponse(
            content=f"FAKE: {last_user.content}",
            model="fake-model",
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(provider: str, **overrides: Any) -> Settings:
    """Build a minimal Settings with the given LLM provider."""
    llm = LLMSettings(provider=provider, model="test-model", **overrides)
    return Settings(
        llm=llm,
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        vector_store=VectorStoreSettings(backend="chroma"),
        retrieval=RetrievalSettings(sparse_backend="bm25"),
    )


# ---------------------------------------------------------------------------
# ChatMessage / ChatResponse tests
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_to_dict(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.to_dict() == {"role": "user", "content": "hello"}

    def test_roles(self) -> None:
        for role in ("system", "user", "assistant"):
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


class TestChatResponse:
    def test_to_dict(self) -> None:
        resp = ChatResponse(content="hi", model="m1", usage={"prompt_tokens": 5})
        d = resp.to_dict()
        assert d["content"] == "hi"
        assert d["model"] == "m1"
        assert d["usage"]["prompt_tokens"] == 5

    def test_defaults(self) -> None:
        resp = ChatResponse(content="hi")
        assert resp.model == ""
        assert resp.usage == {}
        assert resp.metadata == {}


# ---------------------------------------------------------------------------
# FakeLLM tests
# ---------------------------------------------------------------------------


class TestFakeLLM:
    def test_chat_returns_fake_response(self) -> None:
        llm = FakeLLM(_make_settings("fake"))
        resp = llm.chat([ChatMessage(role="user", content="ping")])
        assert resp.content == "FAKE: ping"

    def test_chat_str_convenience(self) -> None:
        llm = FakeLLM(_make_settings("fake"))
        text = llm.chat_str("hello", system="you are helpful")
        assert text == "FAKE: hello"

    def test_chat_with_system_and_user(self) -> None:
        llm = FakeLLM(_make_settings("fake"))
        resp = llm.chat([
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="what is 2+2"),
        ])
        assert "what is 2+2" in resp.content


# ---------------------------------------------------------------------------
# LLMFactory routing tests
# ---------------------------------------------------------------------------


class TestLLMFactory:
    def test_unknown_provider_raises(self) -> None:
        settings = _make_settings("nonexistent_provider")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create(settings)

    def test_register_and_create_custom_provider(self) -> None:
        """Dynamically register a fake provider and create it."""
        # Register using fully qualified class path
        LLMFactory.register_provider(
            "fake", f"{FakeLLM.__module__}.{FakeLLM.__qualname__}"
        )
        settings = _make_settings("fake")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)
        assert llm.chat_str("test") == "FAKE: test"

    def test_case_insensitive_provider(self) -> None:
        """Provider matching should be case-insensitive."""
        LLMFactory.register_provider(
            "fake", f"{FakeLLM.__module__}.{FakeLLM.__qualname__}"
        )
        settings = _make_settings("FAKE")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, FakeLLM)

    def test_error_message_lists_available_providers(self) -> None:
        settings = _make_settings("bogus")
        with pytest.raises(ValueError, match="Available:") as exc_info:
            LLMFactory.create(settings)
        # Should list at least the built-in providers
        for name in ("openai", "azure", "ollama", "deepseek"):
            assert name in str(exc_info.value)


# ---------------------------------------------------------------------------
# BaseLLM is abstract
# ---------------------------------------------------------------------------


class TestBaseLLMAbstract:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM()  # type: ignore[abstract]
