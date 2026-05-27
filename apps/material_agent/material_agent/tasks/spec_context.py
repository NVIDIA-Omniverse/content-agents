# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Specification RAG helpers for dataset preparation."""

import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from world_understanding.functions.knowledge.multimodal_vector_store import (
    collect_documents_from_vector_store,
)

logger = logging.getLogger(__name__)

_PROMPT_EXTRACT_SPEC = """
You are a technical documentation analyst specializing in component material
identification. Analyze the supplied technical context and extract Color,
Material, and Finish (CMF) information for the described product or component.

Focus only on physical parts that belong to the final product and have useful
CMF evidence. Ignore packaging, shipping materials, storage materials, test
procedures, and purely electrical or dimensional facts unless they help identify
a physical material.

For each relevant part:
- Use clear, descriptive part names.
- Extract material specifications, including grades, alloys, coatings, plating,
  certifications, or ratings when available.
- Extract color, finish, texture, and visible surface properties when available.
- Prefer exact values from the documents over broad guesses.
- Skip parts with insufficient CMF information.

Return plain text using this structure:

Component Overview:
[Briefly describe the component and any identifiers from the context.]

Parts with CMF Information:
Part: [Part Name]
- Material Type: [Detailed material specification]
- Color Details: [Color details, or "Not specified"]
- Surface Finish: [Finish details, or "Not specified"]
- Texture Characteristics: [Texture or material properties, or "Not specified"]

[Repeat for each part with meaningful CMF information.]

Context summary:
{snippets}
"""


def _summarize_doc_string(doc_string: str, llm: BaseChatModel, max_tokens: int) -> str:
    """Summarize a document string."""
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
    """Create context snippets from vector-store documents."""
    snippets: list[str] = []
    token_threshold = int(max_tokens * 0.95)
    summarization_tokens = int(max_tokens * 0.25)

    for doc in docs:
        text: str | None = doc.text_content
        if not text:
            continue

        if count_tokens_approximately(text) > token_threshold:
            text = _truncate_text(text, token_threshold)
            text = _summarize_doc_string(text, llm, summarization_tokens)
            logger.warning(
                "Summarized document text to %s tokens",
                count_tokens_approximately(text),
            )

        snippets.append(text)

        doc_string = "\n\n".join(snippets)
        if count_tokens_approximately(doc_string) > token_threshold:
            doc_string = _truncate_text(doc_string, token_threshold)
            doc_string = _summarize_doc_string(doc_string, llm, summarization_tokens)
            logger.warning(
                "Summarized document string to %s tokens",
                count_tokens_approximately(doc_string),
            )
            snippets = [doc_string]

    return snippets


def extract_spec_text_by_model_number(
    model_number: str,
    llm: BaseChatModel,
    vector_store_dir: str | Path,
) -> str:
    """Extract plain-text CMF specification context for a model identifier."""
    store_path = Path(vector_store_dir)
    if not store_path.exists():
        raise FileNotFoundError(f"Vector store directory not found: {store_path}")

    logger.info("Extracting specs for model_number='%s'", model_number)
    docs_by_filename = collect_documents_from_vector_store(
        store_path, {"filename": model_number}
    )

    snippets = _build_context_snippets(docs_by_filename, llm)
    snippets_text = "\n\n".join(snippets)
    parsing_prompt = _PROMPT_EXTRACT_SPEC.format(snippets=snippets_text)

    messages = [
        SystemMessage(
            content="You are a technical writer producing CMF summaries. Return plain text only."
        ),
        HumanMessage(content=parsing_prompt),
    ]

    response = llm.invoke(messages)
    content = response.content if isinstance(response.content, str) else str(response)
    if not content:
        logger.warning("Empty LLM response; returning concatenated snippets")
        return "\n\n".join(snippets)
    return content.strip()
