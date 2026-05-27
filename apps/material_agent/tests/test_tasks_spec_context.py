# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for material_agent.tasks.spec_context."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage

import material_agent.tasks.spec_context as spec_context


class _FakeLLM:
    def __init__(self, content: str = "Component Overview:\nShort summary") -> None:
        self.content = content
        self.calls: list[tuple[list[Any], dict[str, Any] | None]] = []

    def invoke(
        self, messages: list[Any], config: dict[str, Any] | None = None
    ) -> AIMessage:
        self.calls.append((messages, config))
        return AIMessage(content=self.content)


def test_extract_spec_text_by_model_number_requires_existing_store(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        spec_context.extract_spec_text_by_model_number(
            model_number="MODEL_A",
            llm=_FakeLLM(),  # type: ignore[arg-type]
            vector_store_dir=tmp_path / "missing",
        )


def test_extract_spec_text_by_model_number_collects_matching_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store_dir = tmp_path / "vector_store"
    store_dir.mkdir()
    collected: dict[str, Any] = {}

    def fake_collect_documents_from_vector_store(
        store_path: Path, metadata_filter: dict[str, str]
    ) -> list[Any]:
        collected["store_path"] = store_path
        collected["metadata_filter"] = metadata_filter
        return [
            SimpleNamespace(
                text_content="Housing: black ABS plastic with matte finish."
            )
        ]

    monkeypatch.setattr(
        spec_context,
        "collect_documents_from_vector_store",
        fake_collect_documents_from_vector_store,
    )

    llm = _FakeLLM()
    result = spec_context.extract_spec_text_by_model_number(
        model_number="MODEL_A",
        llm=llm,  # type: ignore[arg-type]
        vector_store_dir=store_dir,
    )

    assert result == "Component Overview:\nShort summary"
    assert collected == {
        "store_path": store_dir,
        "metadata_filter": {"filename": "MODEL_A"},
    }
    assert len(llm.calls) == 1
    assert "Housing: black ABS plastic" in llm.calls[0][0][1].content


def test_build_context_snippets_summarizes_oversized_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = iter([200, 40, 40])
    monkeypatch.setattr(
        spec_context,
        "count_tokens_approximately",
        lambda _text: next(counts),
    )

    llm = _FakeLLM()
    snippets = spec_context._build_context_snippets(  # noqa: SLF001 - tests token-threshold behavior
        [SimpleNamespace(text_content="long document text")],
        llm,  # type: ignore[arg-type]
        max_tokens=100,
    )

    assert snippets == ["Component Overview:\nShort summary"]
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == {"max_tokens": 25}


def test_build_context_snippets_summarizes_joined_snippets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = iter([10, 10, 10, 200, 30])
    monkeypatch.setattr(
        spec_context,
        "count_tokens_approximately",
        lambda _text: next(counts),
    )

    llm = _FakeLLM()
    snippets = spec_context._build_context_snippets(  # noqa: SLF001 - tests joined-snippet summarization
        [
            SimpleNamespace(text_content="first short document"),
            SimpleNamespace(text_content="second short document"),
        ],
        llm,  # type: ignore[arg-type]
        max_tokens=100,
    )

    assert snippets == ["Component Overview:\nShort summary"]
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == {"max_tokens": 25}


def test_build_context_snippets_skips_empty_documents() -> None:
    assert spec_context._build_context_snippets([], _FakeLLM()) == []  # noqa: SLF001 - tests empty helper input
    assert (
        spec_context._build_context_snippets(  # noqa: SLF001 - tests empty helper document
            [SimpleNamespace(text_content=None)],
            _FakeLLM(),  # type: ignore[arg-type]
        )
        == []
    )


def test_extract_spec_text_by_model_number_falls_back_to_snippets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store_dir = tmp_path / "vector_store"
    store_dir.mkdir()

    monkeypatch.setattr(
        spec_context,
        "collect_documents_from_vector_store",
        lambda _store_path, _metadata_filter: [
            SimpleNamespace(text_content="Fallback material context.")
        ],
    )

    result = spec_context.extract_spec_text_by_model_number(
        model_number="MODEL_A",
        llm=_FakeLLM(""),  # type: ignore[arg-type]
        vector_store_dir=store_dir,
    )

    assert result == "Fallback material context."
