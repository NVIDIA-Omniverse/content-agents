# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and repair VLM predictions against the material library.

After the predict step, some VLM predictions may contain material names that
don't exactly match any entry in the material library. This task:

1. Loads predictions JSONL and material library names from config.
2. Checks each prediction's material name against the library.
3. For invalid names: fuzzy-match via difflib.SequenceMatcher (>0.7 → auto-correct).
4. For low-confidence matches: batch LLM repair call.
5. Rewrites predictions JSONL in-place with corrected names.
6. Logs a summary of corrections.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

logger = logging.getLogger(__name__)

# Fuzzy match thresholds
_AUTO_CORRECT_THRESHOLD = 0.7
_LLM_REPAIR_THRESHOLD = 0.4


def _best_fuzzy_match(name: str, valid_names: list[str]) -> tuple[str | None, float]:
    """Find the best fuzzy match for a name in the valid names list.

    Uses SequenceMatcher ratio with a token-containment bonus: if the
    candidate contains every word from the query, it gets a +0.1 boost.
    This prevents short-but-unrelated names from beating longer correct ones
    (e.g. "Gold Polished" beating "Stainless Steel Polished" for "Steel Polished").
    """
    best_match = None
    best_score = 0.0
    name_lower = name.lower()
    name_tokens = set(name_lower.split())
    for valid in valid_names:
        valid_lower = valid.lower()
        score = SequenceMatcher(None, name_lower, valid_lower).ratio()
        # Bonus if the candidate contains all tokens from the query
        valid_tokens = set(valid_lower.split())
        if name_tokens and name_tokens.issubset(valid_tokens):
            score += 0.1
        if score > best_score:
            best_score = score
            best_match = valid
    return best_match, best_score


def _extract_material_name(prediction: dict[str, Any]) -> str | None:
    """Extract material name from a prediction entry."""
    materials = prediction.get("materials")
    if isinstance(materials, dict):
        return materials.get("material")
    return None


def _set_material_name(prediction: dict[str, Any], name: str) -> None:
    """Set material name in a prediction entry."""
    if "materials" not in prediction:
        prediction["materials"] = {}
    prediction["materials"]["material"] = name


class ValidatePredictionsTask(Task):
    """Validate VLM predictions against the material library.

    Input context keys:
        - predictions_path: Path to predictions JSONL file
        - material_names: List of valid material names from library
        - llm_config: Optional LLM config for repair (backend, model, etc.)

    Output context keys:
        - predictions_path: Same path (rewritten in-place)
        - validation_stats: Dict with correction counts
    """

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)

        predictions_path = Path(context["predictions_path"])
        material_names: list[str] = context["material_names"]
        llm_config: dict[str, Any] | None = context.get("llm_config")

        if not predictions_path.exists():
            listener.warning(f"Predictions file not found: {predictions_path}")
            context["validation_stats"] = {"skipped": True}
            return context

        valid_set = set(material_names)
        listener.info(f"Validating predictions against {len(valid_set)} material names")

        # Load predictions
        predictions: list[dict[str, Any]] = []
        with open(predictions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))

        # Classify each prediction
        valid_count = 0
        auto_corrected: list[tuple[int, str, str, float]] = []  # idx, old, new, score
        needs_llm: list[tuple[int, str, str, float]] = []  # idx, old, best_match, score
        no_material: list[int] = []

        for i, pred in enumerate(predictions):
            mat_name = _extract_material_name(pred)
            if mat_name is None:
                no_material.append(i)
                continue
            if mat_name in valid_set:
                valid_count += 1
                continue

            best_match, score = _best_fuzzy_match(mat_name, material_names)
            if best_match and score >= _AUTO_CORRECT_THRESHOLD:
                auto_corrected.append((i, mat_name, best_match, score))
            elif best_match and score >= _LLM_REPAIR_THRESHOLD:
                needs_llm.append((i, mat_name, best_match, score))
            else:
                # Very low match — still try LLM
                needs_llm.append((i, mat_name, best_match or "", score))

        listener.info(
            f"Validation: {valid_count} valid, "
            f"{len(auto_corrected)} auto-correctable, "
            f"{len(needs_llm)} need LLM repair, "
            f"{len(no_material)} missing material field"
        )

        # Apply auto-corrections
        for idx, old_name, new_name, score in auto_corrected:
            _set_material_name(predictions[idx], new_name)
            listener.info(
                f"  Auto-corrected: '{old_name}' -> '{new_name}' (score={score:.2f})"
            )

        # LLM repair for low-confidence matches
        llm_repaired = 0
        llm_failed = 0
        if needs_llm and llm_config:
            repaired = self._llm_repair(
                [(idx, old, best) for idx, old, best, _ in needs_llm],
                material_names,
                llm_config,
                listener,
            )
            for idx, old_name, new_name in repaired:
                if new_name and new_name in valid_set:
                    _set_material_name(predictions[idx], new_name)
                    listener.info(f"  LLM-repaired: '{old_name}' -> '{new_name}'")
                    llm_repaired += 1
                else:
                    # LLM couldn't fix it — use fuzzy best match as fallback
                    _, best, _, score = next(t for t in needs_llm if t[0] == idx)
                    if best and score >= _LLM_REPAIR_THRESHOLD:
                        _set_material_name(predictions[idx], best)
                        listener.warning(
                            f"  LLM failed for '{old_name}', "
                            f"fuzzy fallback: '{best}' (score={score:.2f})"
                        )
                        llm_repaired += 1
                    else:
                        listener.warning(
                            f"  Could not repair: '{old_name}' (no good match)"
                        )
                        llm_failed += 1
        elif needs_llm:
            # No LLM config — use fuzzy best match for all
            listener.warning("No LLM config for repair, using fuzzy fallback only")
            for idx, old_name, best_match, score in needs_llm:
                if best_match and score >= _LLM_REPAIR_THRESHOLD:
                    _set_material_name(predictions[idx], best_match)
                    listener.info(
                        f"  Fuzzy fallback: '{old_name}' -> '{best_match}' "
                        f"(score={score:.2f})"
                    )
                    llm_repaired += 1
                else:
                    listener.warning(
                        f"  Could not repair: '{old_name}' (no good match)"
                    )
                    llm_failed += 1

        # Rewrite predictions file
        with open(predictions_path, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        stats = {
            "total": len(predictions),
            "valid": valid_count,
            "auto_corrected": len(auto_corrected),
            "llm_repaired": llm_repaired,
            "failed": llm_failed,
            "no_material": len(no_material),
        }
        context["validation_stats"] = stats
        listener.info(
            f"Validation complete: {stats['auto_corrected']} auto-corrected, "
            f"{stats['llm_repaired']} LLM-repaired, {stats['failed']} failed"
        )

        # Write validation report
        report = {
            "stats": stats,
            "auto_corrected": [
                {"index": idx, "old": old, "new": new, "score": round(score, 3)}
                for idx, old, new, score in auto_corrected
            ],
            "llm_repaired": [],
            "failed": [],
        }
        # Collect LLM repair results if available
        if needs_llm and llm_config:
            for idx, old_name, best_match, score in needs_llm:
                final_mat = _extract_material_name(predictions[idx])
                if final_mat and final_mat != old_name and final_mat in valid_set:
                    report["llm_repaired"].append(
                        {"index": idx, "old": old_name, "new": final_mat}
                    )
                else:
                    report["failed"].append(
                        {
                            "index": idx,
                            "name": old_name,
                            "best_fuzzy": best_match,
                            "score": round(score, 3),
                        }
                    )
        report_path = predictions_path.parent / "validate_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        listener.info(f"Validation report written to {report_path}")

        return context

    # Maximum items per LLM repair batch to avoid token-limit timeouts
    _LLM_BATCH_SIZE = 30

    def _llm_repair(
        self,
        items: list[tuple[int, str, str]],  # (idx, invalid_name, fuzzy_best)
        valid_names: list[str],
        llm_config: dict[str, Any],
        listener: Any,
    ) -> list[tuple[int, str, str | None]]:
        """Use LLM to repair invalid material names.

        Splits items into batches of _LLM_BATCH_SIZE and runs them in parallel
        using a thread pool. Each batch is a single LLM call.
        Returns list of (idx, old_name, repaired_name_or_None).
        """
        from world_understanding.functions.models.chat_models import (
            create_chat_model_from_config,
        )

        llm = create_chat_model_from_config(llm_config)
        if llm is None:
            listener.warning("No API key for LLM repair — skipping")
            return [(idx, old, None) for idx, old, _ in items]

        # Split into batches
        batches = [
            items[i : i + self._LLM_BATCH_SIZE]
            for i in range(0, len(items), self._LLM_BATCH_SIZE)
        ]

        if len(batches) == 1:
            return self._llm_repair_batch(batches[0], valid_names, llm, listener)

        # Run batches in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = min(len(batches), 8)
        listener.info(
            f"Splitting {len(items)} repairs into {len(batches)} batches "
            f"({max_workers} parallel workers)"
        )

        all_repaired: list[tuple[int, str, str | None]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._llm_repair_batch, batch, valid_names, llm, listener
                ): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    all_repaired.extend(future.result())
                except Exception:
                    logger.exception("LLM repair batch %d failed", batch_idx)
                    batch = batches[batch_idx]
                    all_repaired.extend((idx, old, None) for idx, old, _ in batch)

        return all_repaired

    def _llm_repair_batch(
        self,
        items: list[tuple[int, str, str]],
        valid_names: list[str],
        llm: Any,
        listener: Any,
    ) -> list[tuple[int, str, str | None]]:
        """Run a single LLM repair call for a batch of invalid names."""
        from world_understanding.utils.llm_parsing import (
            extract_json_from_llm_response,
        )

        names_text = "\n".join(
            f'  {i + 1}. "{old}" (closest fuzzy match: "{best}")'
            for i, (_, old, best) in enumerate(items)
        )
        valid_names_text = ", ".join(f'"{n}"' for n in valid_names)

        system_prompt = (
            "You are an expert at matching material names. "
            "Given a list of invalid material names and the valid material library, "
            "map each invalid name to the correct valid name from the library. "
            "Return ONLY a JSON object mapping each invalid name to its corrected name.\n"
            'Example: {"Steel Polished": "Stainless Steel Polished", '
            '"Plastic Pure White": "Plastic White"}'
        )
        user_prompt = (
            f"Valid material names:\n{valid_names_text}\n\n"
            f"Invalid names to fix:\n{names_text}\n\n"
            "Return a JSON object mapping each invalid name to the best matching "
            "valid name from the library above."
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            result = extract_json_from_llm_response(response.content)
        except Exception:
            logger.exception("LLM repair call failed")
            return [(idx, old, None) for idx, old, _ in items]

        if not result or not isinstance(result, dict):
            listener.warning("Failed to parse LLM repair response")
            return [(idx, old, None) for idx, old, _ in items]

        repaired = []
        for idx, old_name, _ in items:
            corrected = result.get(old_name)
            repaired.append((idx, old_name, corrected))
        return repaired
