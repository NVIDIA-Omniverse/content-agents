# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Specification RAG placeholder returning CMF-focused plain text for a model.

Given a `model_number`, this module loads a vector store, collects all
documents with metadata that references the model (e.g., `model_number` or
`filename` containing the query), builds a concise context from their text,
and asks an LLM to produce a plain-text CMF summary per component.

Output is a single `str` containing only textual CMF information (no JSON).
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from world_understanding.functions.knowledge.multimodal_vector_store import (
    collect_documents_from_vector_store,
)

from material_agent.pcba.prompts import PROMPT_EXTRACT_SPEC

logger = logging.getLogger(__name__)


def _summarize_doc_string(doc_string: str, llm: BaseChatModel, max_tokens: int) -> str:
    """Summarize a document string.

    Args:
        doc_string: The document string to summarize (should already be truncated to max_tokens)
        llm: Language model for summarization
        max_tokens: Maximum token limit for both input and LLM output

    Returns:
        Summarized text
    """

    messages = [
        SystemMessage(
            content="You are a technical writer producing CMF summaries. Return plain text only."
        ),
        HumanMessage(content=doc_string),
    ]
    response = llm.invoke(messages, config={"max_tokens": max_tokens})
    return response.content if isinstance(response.content, str) else str(response)


def _truncate_text(text: str, max_tokens: int) -> str:
    return text[: max_tokens * 4]


def _build_context_snippets(
    docs: list[Any], llm: BaseChatModel, max_tokens: int = 128000
) -> list[str]:
    """Create context snippets from documents by joining the text_content of the documents.

    Args:
        docs: List of documents to process
        llm: Language model for summarization
        max_tokens: Maximum token limit for the final context

    Returns:
        List of context snippets, truncated to max_tokens if necessary
    """
    snippets: list[str] = []
    token_threshold = int(max_tokens * 0.95)
    summarization_tokens = int(
        max_tokens * 0.25
    )  # Use 25% of max_tokens for summarization output

    for doc in docs:
        text: str | None = doc.text_content
        if not text:
            continue

        # Summarize individual document if it exceeds max_tokens
        if count_tokens_approximately(text) > token_threshold:
            text = _truncate_text(text, token_threshold)
            text = _summarize_doc_string(text, llm, summarization_tokens)
            logger.warning(
                f"Summarized document text to {count_tokens_approximately(text)} tokens"
            )

        snippets.append(text)

        # Summarize snippets when they exceed max_tokens
        doc_string = "\n\n".join(snippets)
        if count_tokens_approximately(doc_string) > token_threshold:
            doc_string = _truncate_text(doc_string, token_threshold)
            doc_string = _summarize_doc_string(doc_string, llm, summarization_tokens)
            logger.warning(
                f"Summarized document string to {count_tokens_approximately(doc_string)} tokens"
            )
            snippets = [doc_string]

    return snippets


def extract_spec_text_by_model_number(
    model_number: str,
    llm: BaseChatModel,
    vector_store_dir: str | Path,
) -> str:
    """Extract plain-text specification for a given `model_number`.

    Args:
        model_number: Identifier used to filter and retrieve relevant documents
        llm: LLM used to produce a plain-text spec from the retrieved context
        vector_store_dir: Directory containing the saved text vector store

    Returns:
        Text-only specification summary for the model number
    """
    # Validate vector store path
    store_path = Path(vector_store_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"Vector store directory not found: {store_path}")

    logger.info("Extracting specs for model_number='%s'", model_number)

    logger.debug("Preparing to collect documents from vector store: %s", store_path)

    # Collect all documents whose metadata references the model number
    # - `model_number` contains the query (case-insensitive)
    # - OR `filename` contains the query (case-insensitive), per example prompt
    logger.debug("Collecting documents by metadata 'filename' match")
    docs_by_filename = collect_documents_from_vector_store(
        store_path, {"filename": model_number}
    )

    # Build compact context then summarize it for better prompt efficiency
    snippets = _build_context_snippets(docs_by_filename, llm)

    parsing_prompt = PROMPT_EXTRACT_SPEC.format(snippets=snippets)

    messages = [
        SystemMessage(
            content=(
                "You are a technical writer producing CMF summaries. Return plain text only."
            )
        ),
        HumanMessage(content=parsing_prompt),
    ]

    logger.debug("Invoking LLM for specification text generation")
    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response)
    if not content:
        logger.warning("Empty LLM response; returning concatenated snippets")
        return "\n\n".join(snippets)
    return content.strip()
