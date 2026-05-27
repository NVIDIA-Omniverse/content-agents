# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract-neutral planning helpers for the ``look_right`` visual template.

This module prepares the prompt/evidence shape for a future Validation Agent
``look_right`` template. It intentionally does not call a VLM and does not
depend on the unmerged Validation Agent contract layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage

from world_understanding.utils.llm_parsing import (
    dedupe_strings,
    extract_json_from_llm_response,
    extract_labeled_choice,
    extract_labeled_codes,
    extract_labeled_score,
    extract_labeled_value,
)

PathInput = str | Path
EvidenceRole = Literal["reference", "current", "focus"]
EvidenceSourceKind = Literal[
    "direct_image",
    "render_output",
    "reference_image",
    "sampled_video_frame",
    "focused_render",
]
LookRightVerdict = Literal["pass", "fail", "needs_refinement", "warn"]

VISUAL_EVIDENCE_MISSING = "visual.evidence_missing"
VISUAL_PROMPT_MISSING = "visual.prompt_missing"
VISUAL_PROMPT_MISMATCH = "visual.prompt_mismatch"
VISUAL_REFERENCE_MISMATCH = "visual.reference_mismatch"
VISUAL_LOW_CONFIDENCE = "visual.low_confidence"
VISUAL_BLOCKING_DEFECT = "visual.blocking_defect"
VISUAL_JUDGE_UNAVAILABLE = "visual.judge_unavailable"
VISUAL_RENDER_PREFLIGHT_FAILED = "visual.render_preflight_failed"
VISUAL_RENDER_PREFLIGHT_UNAVAILABLE = "visual.render_preflight_unavailable"

_BLOCKING_VISUAL_ISSUE_CODES = {
    VISUAL_BLOCKING_DEFECT,
    VISUAL_PROMPT_MISMATCH,
    VISUAL_REFERENCE_MISMATCH,
}
_LOOK_RIGHT_ISSUE_CODES = {
    VISUAL_BLOCKING_DEFECT,
    VISUAL_PROMPT_MISMATCH,
    VISUAL_REFERENCE_MISMATCH,
    VISUAL_LOW_CONFIDENCE,
}
_RENDER_PREFLIGHT_FAILED_VALUES = {"fail", "failed", "error"}
_RENDER_PREFLIGHT_UNAVAILABLE_VALUES = {"skipped", "warn", "warning"}
_SECTION_NAMES = (
    "Critique",
    "Score",
    "Decision",
    "Issue Codes",
    "Issue Code",
    "Evidence Notes",
    "Evidence Note",
    "Improvement Suggestion",
    "Improvement Suggestions",
    "Recommendation",
    "Recommendations",
    "Suggestion",
    "Suggestions",
)

DEFAULT_LOOK_RIGHT_SYSTEM_PROMPT = (
    "You are an expert visual validation judge for 3D asset evidence."
)

DEFAULT_LOOK_RIGHT_PROMPT_TEMPLATE = """Evaluate whether the current asset evidence satisfies the user task.

User task:
{task_description}

Manual focus prims:
{focus_prim_list}

Reference guidance:
{reference_guidance}

Evidence roles:
- Reference images are optional loose guidance. Use them only when they clearly
  depict the same asset, style, or requested visual target.
- Current asset images are the evidence to judge.
- Focus images are close-ups for user-supplied prim paths. Do not invent or
  validate focus prims.

Evaluation order:
1. Asset identity and major geometry.
2. Required visual attributes named in the task.
3. Reference alignment when references exist.
4. Per-view defect inspection.
5. Confidence based on evidence coverage and render quality.

Multi-image decision rules:
- Inspect every current or focus image independently before the final decision.
- Do not average away defects across views. A required visual attribute that is
  missing, incorrect, or reference-mismatched in any required view is a blocking
  defect unless another view clearly proves it is only occluded, cropped, or
  hidden by lighting.
- Extra views should change the final decision only when they reveal a new
  defect, resolve uncertainty, or prove that a suspected defect is not real.
- Do not escalate NEEDS_REFINEMENT to FAIL only because the same defect is
  visible in more views. Escalate only when added views reveal a materially new
  defect, show that the asset identity/major geometry is wrong, or prove that a
  required visual attribute is broadly absent rather than merely imperfect.
- PASS only when all required visual attributes are satisfied in the available
  current/focus evidence and no blocking defect is visible.
- NEEDS_REFINEMENT when the asset is close but any blocking visual defect or
  reference mismatch remains fixable.
- FAIL when identity, major geometry, or required visual attributes are clearly
  wrong.
- WARN only for evidence limitations or low confidence, not for visible
  blocking defects.

Return exactly this structure:

**Critique:**
[Concise evidence-grounded assessment.]

**Score:** [0-10]

**Decision:** [PASS, FAIL, NEEDS_REFINEMENT, or WARN]

**Issue Codes:** [comma-separated codes from visual.blocking_defect,
visual.prompt_mismatch, visual.reference_mismatch, visual.low_confidence,
or none]

**Evidence Notes:**
[Mention which images/views support the decision, including whether added
views revealed new defects or only confirmed defects already visible.]
"""

DEFAULT_LOOK_RIGHT_FINAL_JUDGE_SYSTEM_PROMPT = (
    "You normalize visual judge critiques into strict validation results."
)

DEFAULT_LOOK_RIGHT_FINAL_JUDGE_PROMPT_TEMPLATE = """Normalize this visual judge response into one JSON object.

This is a text-only normalization step. Do not re-judge images. Use only the
judge response below and the decision rules. If the judge response is ambiguous,
contradictory, or explicitly rejects PASS/APPROVE, choose WARN or
NEEDS_REFINEMENT rather than PASS.

User task:
{task_description}

Existing parser fallback:
{parser_fallback}

Visual judge response:
{judge_response}

Decision rules:
- decision must be one of: pass, fail, needs_refinement, warn.
- PASS means the response clearly accepts the asset and reports no blocking
  visual defect.
- NEEDS_REFINEMENT means the response describes a fixable visual defect or
  explicitly refuses PASS/APPROVE.
- FAIL means identity, major geometry, or required visual attributes are
  clearly wrong.
- WARN means the response is unparseable, low confidence, or evidence-limited.
- issue_codes may include only: visual.blocking_defect, visual.prompt_mismatch,
  visual.reference_mismatch, visual.low_confidence.

Return only JSON with this shape:
{{
  "decision": "pass|fail|needs_refinement|warn",
  "score": 0.0,
  "issue_codes": [],
  "rationale": "brief reason grounded in the judge response",
  "evidence_notes": "optional concise notes"
}}
"""


@dataclass(frozen=True)
class LookRightIssue:
    """Simple issue payload for later mapping into validation result models."""

    code: str
    message: str
    severity: str = "error"
    subject: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "subject": self.subject,
            "details": self.details,
        }


@dataclass(frozen=True)
class LookRightEvidenceImage:
    """One image item passed to a future multimodal judge call."""

    role: EvidenceRole
    path: str
    caption: str
    source_kind: EvidenceSourceKind = "direct_image"
    prim_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence dictionary."""
        return {
            "role": self.role,
            "path": self.path,
            "caption": self.caption,
            "source_kind": self.source_kind,
            "prim_path": self.prim_path,
        }


@dataclass(frozen=True)
class LookRightJudgePlan:
    """Prepared prompt and image-caption pairs for a ``look_right`` judge."""

    task_description: str
    final_prompt: str
    system_prompt: str = DEFAULT_LOOK_RIGHT_SYSTEM_PROMPT
    evidence_images: tuple[LookRightEvidenceImage, ...] = ()
    focus_prim_paths: tuple[str, ...] = ()
    temperature: float = 0.1
    max_tokens: int = 2048
    issues: tuple[LookRightIssue, ...] = ()

    @property
    def image_caption_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return the ordered image-caption pairs expected by existing VLM APIs."""
        return tuple(
            (evidence.caption, evidence.path) for evidence in self.evidence_images
        )

    @property
    def ready_for_judge(self) -> bool:
        """Whether evidence is sufficient and no issue blocks VLM execution."""
        has_current_evidence = any(
            evidence.role in {"current", "focus"} for evidence in self.evidence_images
        )
        has_unavailable_render_evidence = _has_issue(
            self.issues, VISUAL_RENDER_PREFLIGHT_UNAVAILABLE
        ) and _has_generated_render_evidence(self.evidence_images)
        has_blocking_issue = any(
            _is_judge_blocking_issue(issue) for issue in self.issues
        )
        return (
            has_current_evidence
            and not has_blocking_issue
            and not has_unavailable_render_evidence
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan dictionary."""
        return {
            "task_description": self.task_description,
            "system_prompt": self.system_prompt,
            "final_prompt": self.final_prompt,
            "image_caption_pairs": [
                {"caption": caption, "path": path}
                for caption, path in self.image_caption_pairs
            ],
            "evidence_images": [
                evidence.to_dict() for evidence in self.evidence_images
            ],
            "focus_prim_paths": list(self.focus_prim_paths),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "issues": [issue.to_dict() for issue in self.issues],
            "ready_for_judge": self.ready_for_judge,
        }


@dataclass(frozen=True)
class LookRightJudgment:
    """Parsed visual judge response in a contract-neutral shape."""

    raw_response: str
    verdict: LookRightVerdict
    score: float | None
    issue_codes: tuple[str, ...] = ()
    critique: str = ""
    reasoning: str = ""
    evidence_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable judgment dictionary."""
        return {
            "raw_response": self.raw_response,
            "verdict": self.verdict,
            "score": self.score,
            "issue_codes": list(self.issue_codes),
            "critique": self.critique,
            "reasoning": self.reasoning,
            "evidence_notes": self.evidence_notes,
        }


@dataclass(frozen=True)
class LookRightJudgeInvocation:
    """Raw output and model metadata from a live ``look_right`` judge call."""

    raw_response: str
    backend_name: str | None = None
    model_name: str | None = None
    token_usage: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable invocation summary."""

        return {
            "raw_response": self.raw_response,
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "token_usage": dict(self.token_usage) if self.token_usage else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LookRightFinalJudgeResult:
    """Final text-normalized look_right judgment and LLM call metadata."""

    judgment: LookRightJudgment
    raw_response: str | None = None
    backend_name: str | None = None
    model_name: str | None = None
    token_usage: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable final-judge summary."""

        return {
            "judgment": self.judgment.to_dict(),
            "raw_response": self.raw_response,
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "token_usage": dict(self.token_usage) if self.token_usage else None,
            "metadata": dict(self.metadata),
        }


def build_look_right_judge_plan(
    task_description: str | None,
    *,
    current_image_paths: Sequence[PathInput] | None = None,
    render_image_paths: Sequence[PathInput] | None = None,
    sampled_video_frame_paths: Sequence[PathInput] | None = None,
    reference_image_paths: Sequence[PathInput] | None = None,
    focused_image_paths: Mapping[str, Sequence[PathInput]] | None = None,
    focus_prim_paths: Sequence[str] | None = None,
    reference_guidance: str | None = None,
    render_valid_result: Mapping[str, Any] | None = None,
    vlm_available: bool = True,
    prompt_template: str = DEFAULT_LOOK_RIGHT_PROMPT_TEMPLATE,
    system_prompt: str = DEFAULT_LOOK_RIGHT_SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> LookRightJudgePlan:
    """Build the prompt/evidence plan for a future ``look_right`` VLM judge.

    The function mirrors the reusable part of the material-agent judge flow:
    reference images are listed first, current asset evidence follows, and
    manual focus prim paths are passed through without USD validation.
    """
    normalized_task = (task_description or "").strip()
    current_paths = _normalize_paths(current_image_paths)
    render_paths = _normalize_paths(render_image_paths)
    sampled_frame_paths = _normalize_paths(sampled_video_frame_paths)
    reference_paths = _normalize_paths(reference_image_paths)
    focused_paths = _normalize_focused_paths(focused_image_paths)
    normalized_focus_prims = _merge_focus_prim_paths(
        focus_prim_paths,
        tuple(focused_paths.keys()),
    )

    issues: list[LookRightIssue] = []
    if not normalized_task:
        issues.append(
            LookRightIssue(
                code=VISUAL_PROMPT_MISSING,
                message="A visual validation task description is required.",
            )
        )
    if (
        not current_paths
        and not render_paths
        and not sampled_frame_paths
        and not focused_paths
    ):
        issues.append(
            LookRightIssue(
                code=VISUAL_EVIDENCE_MISSING,
                message=(
                    "At least one current asset image, generated render image, "
                    "sampled video frame, or focused image is required for "
                    "look_right judging."
                ),
            )
        )
    if not vlm_available:
        issues.append(
            LookRightIssue(
                code=VISUAL_JUDGE_UNAVAILABLE,
                severity="warning",
                message=(
                    "The look_right VLM judge is unavailable; visual evidence "
                    "can be prepared but cannot be judged."
                ),
                details={"vlm_available": False},
            )
        )

    render_preflight_issue = _render_preflight_issue(render_valid_result)
    if render_preflight_issue is not None:
        issues.append(render_preflight_issue)

    evidence = _build_evidence_images(
        reference_paths,
        current_paths,
        render_paths,
        sampled_frame_paths,
        focused_paths,
    )
    final_prompt = prompt_template.format(
        task_description=normalized_task or "(missing)",
        focus_prim_list=_format_focus_prim_list(normalized_focus_prims),
        reference_guidance=reference_guidance
        or _default_reference_guidance(has_reference_images=bool(reference_paths)),
    )

    return LookRightJudgePlan(
        task_description=normalized_task,
        final_prompt=final_prompt,
        system_prompt=system_prompt,
        evidence_images=tuple(evidence),
        focus_prim_paths=normalized_focus_prims,
        temperature=temperature,
        max_tokens=max_tokens,
        issues=tuple(issues),
    )


def invoke_look_right_judge(
    judge_plan: LookRightJudgePlan,
    vlm: Any,
    **kwargs: Any,
) -> LookRightJudgeInvocation:
    """Invoke an existing VLM with the prepared ``look_right`` judge plan."""

    raw_response = vlm.generate_with_image_caption_pairs(
        image_caption_pairs=list(judge_plan.image_caption_pairs),
        final_prompt=judge_plan.final_prompt,
        system_prompt=judge_plan.system_prompt,
        temperature=judge_plan.temperature,
        max_tokens=judge_plan.max_tokens,
        **kwargs,
    )
    token_usage = getattr(vlm, "last_token_usage", None)
    if token_usage is not None and hasattr(token_usage, "to_dict"):
        token_usage_data = token_usage.to_dict()
    else:
        token_usage_data = None
    return LookRightJudgeInvocation(
        raw_response=str(raw_response),
        backend_name=_safe_optional_attr(vlm, "backend_name"),
        model_name=_safe_optional_attr(vlm, "model_name"),
        token_usage=token_usage_data,
        metadata={
            "image_count": len(judge_plan.image_caption_pairs),
            "temperature": judge_plan.temperature,
            "max_tokens": judge_plan.max_tokens,
        },
    )


def normalize_look_right_judgment(
    response: str,
    *,
    llm_judge: Any | None = None,
    judge_plan: LookRightJudgePlan | None = None,
    pass_threshold: float = 0.7,
    needs_refinement_threshold: float = 0.55,
    temperature: float | None = 0.0,
    max_tokens: int | None = 512,
) -> LookRightFinalJudgeResult:
    """Normalize a VLM judge critique into the final look_right judgment.

    The deterministic parser remains the fallback for clean structured output.
    When an LLM judge is supplied, it normalizes the full critique into strict
    JSON instead of expanding regex parsing into natural-language semantics.
    """

    parser_fallback = parse_look_right_judgment(
        response,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )
    if llm_judge is None:
        return LookRightFinalJudgeResult(
            judgment=parser_fallback,
            metadata={
                "method": "parser",
                "llm_invoked": False,
            },
        )

    prompt = DEFAULT_LOOK_RIGHT_FINAL_JUDGE_PROMPT_TEMPLATE.format(
        task_description=judge_plan.task_description if judge_plan else "(unknown)",
        parser_fallback=parser_fallback.to_dict(),
        judge_response=response.strip(),
    )
    raw_llm_response: str | None = None
    try:
        raw_llm_response = _invoke_text_llm_judge(
            llm_judge,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        payload = extract_json_from_llm_response(
            raw_llm_response,
            expected_keys=["decision", "score", "issue_codes", "rationale"],
        )
        if payload is None:
            raise ValueError("LLM final judge did not return a JSON object")
        judgment = _look_right_judgment_from_final_judge_payload(
            response,
            payload,
            parser_fallback=parser_fallback,
            pass_threshold=pass_threshold,
            needs_refinement_threshold=needs_refinement_threshold,
        )
        return LookRightFinalJudgeResult(
            judgment=judgment,
            raw_response=raw_llm_response,
            backend_name=_safe_optional_attr(llm_judge, "backend_name"),
            model_name=_safe_optional_attr(llm_judge, "model_name"),
            token_usage=_token_usage_dict(getattr(llm_judge, "last_token_usage", None)),
            metadata={
                "method": "llm",
                "llm_invoked": True,
                "parser_fallback_verdict": parser_fallback.verdict,
                "parser_fallback_score": parser_fallback.score,
            },
        )
    except Exception as exc:
        failed_closed = _fail_closed_final_judge(
            response,
            parser_fallback,
            error=exc,
        )
        return LookRightFinalJudgeResult(
            judgment=failed_closed,
            raw_response=raw_llm_response,
            backend_name=_safe_optional_attr(llm_judge, "backend_name"),
            model_name=_safe_optional_attr(llm_judge, "model_name"),
            token_usage=_token_usage_dict(getattr(llm_judge, "last_token_usage", None)),
            metadata={
                "method": "llm_failed",
                "llm_invoked": True,
                "error_type": type(exc).__name__,
                "error": "Final LLM judge failed; raw exception text is redacted.",
                "parser_fallback_verdict": parser_fallback.verdict,
                "parser_fallback_score": parser_fallback.score,
            },
        )


def parse_look_right_judgment(
    response: str,
    *,
    pass_threshold: float = 0.7,
    needs_refinement_threshold: float = 0.55,
) -> LookRightJudgment:
    """Parse a structured ``look_right`` judge response.

    Existing material-agent parsing accepts score/decision text from a VLM. This
    helper keeps the same practical tolerance while returning provisional V1
    verdicts and stable ``visual.*`` issue codes.
    """
    raw_response = response.strip()
    score = _extract_score(raw_response)
    decision = _extract_decision(raw_response)
    issue_codes = list(_extract_issue_codes(raw_response))

    verdict = _initial_verdict_from_judge(
        decision,
        score,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )

    if (decision is None and verdict == "pass") or (
        score is None and verdict in {"pass", "warn"}
    ):
        issue_codes.append(VISUAL_LOW_CONFIDENCE)

    verdict = _apply_verdict_severity_floors(
        verdict,
        score,
        issue_codes,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )

    if verdict == "fail" and not issue_codes:
        issue_codes.append(VISUAL_PROMPT_MISMATCH)
    if verdict == "needs_refinement" and not issue_codes:
        issue_codes.append(VISUAL_LOW_CONFIDENCE)

    return LookRightJudgment(
        raw_response=raw_response,
        verdict=verdict,
        score=score,
        issue_codes=dedupe_strings(issue_codes),
        critique=_extract_section(raw_response, "Critique"),
        reasoning=_summarize_response(raw_response),
        evidence_notes=_extract_section_any(
            raw_response,
            ("Evidence Notes", "Evidence Note"),
        ),
    )


def _invoke_text_llm_judge(
    llm_judge: Any,
    prompt: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
) -> str:
    invoke_kwargs: dict[str, Any] = {}
    if temperature is not None:
        invoke_kwargs["temperature"] = temperature
    if max_tokens is not None:
        invoke_kwargs["max_tokens"] = max_tokens

    if hasattr(llm_judge, "invoke"):
        response = llm_judge.invoke(
            [
                SystemMessage(content=DEFAULT_LOOK_RIGHT_FINAL_JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            **invoke_kwargs,
        )
        return str(getattr(response, "content", response))
    if hasattr(llm_judge, "generate"):
        return str(
            llm_judge.generate(
                prompt=prompt,
                system_prompt=DEFAULT_LOOK_RIGHT_FINAL_JUDGE_SYSTEM_PROMPT,
                **invoke_kwargs,
            )
        )
    raise TypeError("LLM judge must provide invoke(...) or generate(...)")


def _look_right_judgment_from_final_judge_payload(
    raw_response: str,
    payload: Mapping[str, Any],
    *,
    parser_fallback: LookRightJudgment,
    pass_threshold: float,
    needs_refinement_threshold: float,
) -> LookRightJudgment:
    verdict = _coerce_final_judge_verdict(payload.get("decision"))
    score = _coerce_final_judge_score(payload.get("score"), parser_fallback.score)
    issue_codes = list(_coerce_final_judge_issue_codes(payload.get("issue_codes")))
    issue_codes.extend(
        code
        for code in parser_fallback.issue_codes
        if code in _BLOCKING_VISUAL_ISSUE_CODES
    )
    issue_codes = list(dedupe_strings(issue_codes))
    verdict = _apply_verdict_severity_floors(
        verdict,
        score,
        issue_codes,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )
    if verdict == "fail" and not issue_codes:
        issue_codes.append(VISUAL_PROMPT_MISMATCH)
    if verdict in {"needs_refinement", "warn"} and not issue_codes:
        issue_codes.append(VISUAL_LOW_CONFIDENCE)

    rationale = _optional_payload_string(payload, "rationale")
    evidence_notes = _optional_payload_string(payload, "evidence_notes")
    return LookRightJudgment(
        raw_response=raw_response.strip(),
        verdict=verdict,
        score=score,
        issue_codes=dedupe_strings(issue_codes),
        critique=parser_fallback.critique,
        reasoning=rationale or parser_fallback.reasoning,
        evidence_notes=evidence_notes or parser_fallback.evidence_notes,
    )


def _coerce_final_judge_verdict(value: Any) -> LookRightVerdict:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "approve": "pass",
        "continue": "needs_refinement",
        "inconclusive": "warn",
        "needs_refinement": "needs_refinement",
    }
    text = aliases.get(text, text)
    if text not in _VERDICT_SEVERITY:
        raise ValueError(f"Invalid final judge decision: {value!r}")
    return cast(LookRightVerdict, text)


def _coerce_final_judge_score(value: Any, fallback: float | None) -> float | None:
    if value is None:
        return fallback
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid final judge score: {value!r}") from exc
    if score > 1.0:
        score /= 10.0
    return max(0.0, min(score, 1.0))


def _coerce_final_judge_issue_codes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_values: Sequence[Any]
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        raw_values = value
    else:
        raise ValueError(f"Invalid final judge issue_codes: {value!r}")
    issue_codes = [
        str(item).strip().lower()
        for item in raw_values
        if str(item).strip().lower() in _LOOK_RIGHT_ISSUE_CODES
    ]
    return dedupe_strings(issue_codes)


def _optional_payload_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    return str(value).strip()


def _fail_closed_final_judge(
    raw_response: str,
    parser_fallback: LookRightJudgment,
    *,
    error: Exception,
) -> LookRightJudgment:
    verdict = _more_severe_verdict(parser_fallback.verdict, "warn")
    issue_codes = list(parser_fallback.issue_codes)
    issue_codes.append(VISUAL_LOW_CONFIDENCE)
    return LookRightJudgment(
        raw_response=raw_response.strip(),
        verdict=verdict,
        score=parser_fallback.score,
        issue_codes=dedupe_strings(issue_codes),
        critique=parser_fallback.critique,
        reasoning=f"Final LLM judge failed closed: {type(error).__name__}",
        evidence_notes=parser_fallback.evidence_notes,
    )


def _token_usage_dict(token_usage: Any) -> Mapping[str, Any] | None:
    if token_usage is None:
        return None
    if hasattr(token_usage, "to_dict"):
        data = token_usage.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
        return None
    if isinstance(token_usage, Mapping):
        return dict(token_usage)
    return None


def _safe_optional_attr(obj: Any, name: str) -> str | None:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if value is None:
        return None
    return str(value)


def _build_evidence_images(
    reference_paths: tuple[str, ...],
    current_paths: tuple[str, ...],
    render_paths: tuple[str, ...],
    sampled_frame_paths: tuple[str, ...],
    focused_paths: Mapping[str, tuple[str, ...]],
) -> list[LookRightEvidenceImage]:
    evidence: list[LookRightEvidenceImage] = []
    for index, path in enumerate(reference_paths, 1):
        evidence.append(
            LookRightEvidenceImage(
                role="reference",
                path=path,
                caption=f"Reference Image {index}:",
                source_kind="reference_image",
            )
        )
    for index, path in enumerate(current_paths, 1):
        evidence.append(
            LookRightEvidenceImage(
                role="current",
                path=path,
                caption=f"Current Asset Evidence - View {index}:",
                source_kind="direct_image",
            )
        )
    for index, path in enumerate(render_paths, 1):
        evidence.append(
            LookRightEvidenceImage(
                role="current",
                path=path,
                caption=f"Current Render Output - View {index}:",
                source_kind="render_output",
            )
        )
    for index, path in enumerate(sampled_frame_paths, 1):
        evidence.append(
            LookRightEvidenceImage(
                role="current",
                path=path,
                caption=f"Sampled Video Frame - Frame {index}:",
                source_kind="sampled_video_frame",
            )
        )
    for prim_path, paths in focused_paths.items():
        for index, path in enumerate(paths, 1):
            evidence.append(
                LookRightEvidenceImage(
                    role="focus",
                    path=path,
                    caption=f"Focused Asset Evidence - {prim_path} - View {index}:",
                    source_kind="focused_render",
                    prim_path=prim_path,
                )
            )
    return evidence


def _normalize_paths(paths: Sequence[PathInput] | None) -> tuple[str, ...]:
    if paths is None:
        return ()
    return tuple(str(Path(path)) for path in paths)


def _normalize_focused_paths(
    focused_image_paths: Mapping[str, Sequence[PathInput]] | None,
) -> dict[str, tuple[str, ...]]:
    if not focused_image_paths:
        return {}
    normalized: dict[str, tuple[str, ...]] = {}
    for prim_path, paths in focused_image_paths.items():
        if paths:
            normalized[prim_path] = _normalize_paths(paths)
    return normalized


def _merge_focus_prim_paths(
    focus_prim_paths: Sequence[str] | None,
    focused_prim_paths: Sequence[str],
) -> tuple[str, ...]:
    merged: list[str] = []
    if focus_prim_paths:
        merged.extend(focus_prim_paths)
    merged.extend(focused_prim_paths)
    return dedupe_strings(merged)


def _format_focus_prim_list(focus_prim_paths: Sequence[str]) -> str:
    if not focus_prim_paths:
        return "(none supplied)"
    return "\n".join(f"- {path}" for path in focus_prim_paths)


def _default_reference_guidance(*, has_reference_images: bool) -> str:
    if has_reference_images:
        return (
            "Use reference images as supporting evidence, not exact ground "
            "truth unless the task explicitly says they are canonical."
        )
    return "No reference images were supplied; judge against the task description."


def _render_preflight_issue(
    render_valid_result: Mapping[str, Any] | None,
) -> LookRightIssue | None:
    if render_valid_result is None:
        return None

    status = _normalized_result_value(render_valid_result, "status")
    verdict = _normalized_result_value(render_valid_result, "verdict")
    issue_codes = _result_issue_codes(render_valid_result)
    details = {
        "render_valid_status": status or None,
        "render_valid_verdict": verdict or None,
        "render_valid_issue_codes": issue_codes,
    }
    if (
        status in _RENDER_PREFLIGHT_FAILED_VALUES
        or verdict in _RENDER_PREFLIGHT_FAILED_VALUES
    ):
        return LookRightIssue(
            code=VISUAL_RENDER_PREFLIGHT_FAILED,
            message=(
                "render_valid reported failed render evidence; look_right "
                "should not call the VLM judge on failed render inputs."
            ),
            details=details,
        )
    if (
        status in _RENDER_PREFLIGHT_UNAVAILABLE_VALUES
        or verdict in _RENDER_PREFLIGHT_UNAVAILABLE_VALUES
    ):
        return LookRightIssue(
            code=VISUAL_RENDER_PREFLIGHT_UNAVAILABLE,
            severity="warning",
            message=(
                "render_valid did not produce a passing preflight result; "
                "look_right may judge only caller-supplied visual evidence."
            ),
            details=details,
        )
    return None


def _normalized_result_value(result: Mapping[str, Any], key: str) -> str:
    value = result.get(key)
    if value is None:
        return ""
    return str(value).strip().lower()


def _result_issue_codes(result: Mapping[str, Any]) -> list[str]:
    issues = result.get("issues")
    if not isinstance(issues, Sequence) or isinstance(issues, str | bytes | bytearray):
        return []
    codes: list[str] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        code = issue.get("code")
        if code is not None:
            codes.append(str(code))
    return list(dict.fromkeys(codes))


def _is_judge_blocking_issue(issue: LookRightIssue) -> bool:
    return issue.severity == "error" or issue.code == VISUAL_JUDGE_UNAVAILABLE


def _has_issue(issues: Sequence[LookRightIssue], code: str) -> bool:
    return any(issue.code == code for issue in issues)


def _has_generated_render_evidence(
    evidence_images: Sequence[LookRightEvidenceImage],
) -> bool:
    return any(
        evidence.source_kind in {"render_output", "focused_render"}
        for evidence in evidence_images
    )


def _extract_score(response: str) -> float | None:
    return extract_labeled_score(response, "Score")


def _extract_decision(response: str) -> LookRightVerdict | None:
    decision = extract_labeled_choice(
        response,
        "Decision",
        ("fail", "needs_refinement", "warn", "pass"),
        aliases={
            "approve": "pass",
            "continue": "needs_refinement",
            "inconclusive": "warn",
            "needs refinement": "needs_refinement",
        },
    )
    if not decision:
        return None
    return cast(LookRightVerdict, decision)


def _extract_issue_codes(response: str) -> tuple[str, ...]:
    return extract_labeled_codes(
        response,
        ("Issue Codes", "Issue Code"),
        allowed_codes=_LOOK_RIGHT_ISSUE_CODES,
    )


_VERDICT_SEVERITY: Mapping[LookRightVerdict, int] = {
    "pass": 0,
    "warn": 1,
    "needs_refinement": 2,
    "fail": 3,
}


def _initial_verdict_from_judge(
    decision: LookRightVerdict | None,
    score: float | None,
    *,
    pass_threshold: float,
    needs_refinement_threshold: float,
) -> LookRightVerdict:
    if decision in {"fail", "needs_refinement", "warn"}:
        return decision
    return _score_based_verdict(
        score,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )


def _apply_verdict_severity_floors(
    verdict: LookRightVerdict,
    score: float | None,
    issue_codes: Sequence[str],
    *,
    pass_threshold: float,
    needs_refinement_threshold: float,
) -> LookRightVerdict:
    score_floor = _score_based_verdict(
        score,
        pass_threshold=pass_threshold,
        needs_refinement_threshold=needs_refinement_threshold,
    )
    verdict = _more_severe_verdict(verdict, score_floor)
    if _has_blocking_visual_issue(issue_codes):
        verdict = _more_severe_verdict(
            verdict,
            _blocking_defect_verdict(
                score,
                needs_refinement_threshold=needs_refinement_threshold,
            ),
        )
    elif VISUAL_LOW_CONFIDENCE in issue_codes:
        verdict = _more_severe_verdict(verdict, "warn")
    return verdict


def _score_based_verdict(
    score: float | None,
    *,
    pass_threshold: float,
    needs_refinement_threshold: float,
) -> LookRightVerdict:
    if score is None:
        return "warn"
    if score < needs_refinement_threshold:
        return "fail"
    if score < pass_threshold:
        return "needs_refinement"
    return "pass"


def _more_severe_verdict(
    current: LookRightVerdict,
    floor: LookRightVerdict,
) -> LookRightVerdict:
    if _VERDICT_SEVERITY[floor] > _VERDICT_SEVERITY[current]:
        return floor
    return current


def _has_blocking_visual_issue(issue_codes: Sequence[str]) -> bool:
    return any(code in _BLOCKING_VISUAL_ISSUE_CODES for code in issue_codes)


def _blocking_defect_verdict(
    score: float | None,
    *,
    needs_refinement_threshold: float,
) -> LookRightVerdict:
    if score is not None and score < needs_refinement_threshold:
        return "fail"
    return "needs_refinement"


def _extract_section(response: str, section_name: str) -> str:
    return extract_labeled_value(
        response,
        section_name,
        multiline=True,
        boundary_labels=_SECTION_NAMES,
    )


def _extract_section_any(response: str, section_names: Sequence[str]) -> str:
    for section_name in section_names:
        value = _extract_section(response, section_name)
        if value:
            return value
    return ""


def _summarize_response(response: str) -> str:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    summary = " ".join(lines[:3])
    if len(summary) > 240:
        return summary[:237] + "..."
    return summary
