"""Tool 5: generate_med_rec_note.

Orchestrator. Runs tools 1-4 and tool 6 in parallel, assembles a structured
context, and asks the LLM to compose a clinician-ready note with inline
citations to FHIR resource IDs. Every citation is post-validated to prevent
hallucinated references.

Audiences & formats are driven entirely by prompt conditioning, not by code
branching — so adding a new audience only requires a new system-prompt block.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from argus.fhir_client import FhirClient
from argus.llm import get_llm
from argus.logging_setup import get_logger
from argus.schemas import (
    ActionItem,
    CheckInteractionsInput,
    FhirReference,
    GenerateNoteInput,
    GenerateNoteOutput,
    NoteAudience,
    ReconcileInput,
    RenalCheckInput,
    ScreenPatternsInput,
    Severity,
    Warning_,
)
from argus.sharp_context import SharpContext
from argus.tools._common import patient_age_years, patient_display_name, patient_sex
from argus.tools.check_interactions import run as run_check_interactions
from argus.tools.reconcile import run as run_reconcile
from argus.tools.renal_check import run as run_renal_check
from argus.tools.screen_patterns import run as run_screen_patterns

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM schema for structured action items
# ---------------------------------------------------------------------------


class _ActionItemSchema(BaseModel):
    priority: str
    action: str
    reason: str
    owner_role: str
    evidence_refs: list[str] = Field(default_factory=list)


class _ActionsList(BaseModel):
    items: list[_ActionItemSchema] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(request: GenerateNoteInput, context: SharpContext) -> GenerateNoteOutput:
    start = time.perf_counter()
    patient_id = request.patient_id or context.patient_id

    if not patient_id:
        return GenerateNoteOutput(
            note_markdown="_No patient context was provided; note could not be generated._",
            warnings=[Warning_(code="missing_patient_id", message="No patient_id.")],
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    warnings: list[Warning_] = []

    # ---- Fetch patient + run tools in parallel -----------------------------
    async with FhirClient(context) as fhir:
        patient_task = fhir.get_patient(patient_id)

        tool_tasks: list[asyncio.Future] = []
        included = set(request.include_sections)

        if "interactions" in included:
            tool_tasks.append(
                run_check_interactions(
                    CheckInteractionsInput(patient_id=patient_id), context
                )
            )
        if "renal" in included:
            tool_tasks.append(
                run_renal_check(RenalCheckInput(patient_id=patient_id), context)
            )
        if "discrepancies" in included:
            tool_tasks.append(
                run_reconcile(ReconcileInput(patient_id=patient_id), context)
            )
        if "high_risk" in included:
            tool_tasks.append(
                run_screen_patterns(ScreenPatternsInput(patient_id=patient_id), context)
            )

        results = await asyncio.gather(patient_task, *tool_tasks, return_exceptions=True)

    patient = results[0] if not isinstance(results[0], Exception) else {}
    if isinstance(results[0], Exception):
        warnings.append(Warning_(code="patient_fetch_failed", message=str(results[0])))

    tool_results: dict[str, Any] = {}
    idx = 1
    for section in ("interactions", "renal", "discrepancies", "high_risk"):
        if section not in included:
            continue
        r = results[idx]
        idx += 1
        if isinstance(r, Exception):
            warnings.append(
                Warning_(code=f"{section}_tool_failed", message=str(r))
            )
            tool_results[section] = None
        else:
            tool_results[section] = r

    # ---- Collect citations from sub-tool outputs ---------------------------
    all_citations: list[FhirReference] = []
    citation_ids: set[str] = set()
    for result in tool_results.values():
        if result is None:
            continue
        for cite in getattr(result, "citations", []):
            if cite.reference not in citation_ids:
                all_citations.append(cite)
                citation_ids.add(cite.reference)

    # ---- Compose the note -------------------------------------------------
    llm = get_llm()
    note_md, tokens, gen_ms = await _compose_note(
        llm, request, patient, tool_results, all_citations
    )

    # ---- Extract structured action items ----------------------------------
    action_items = await _extract_action_items(
        llm, note_md, tool_results, all_citations, citation_ids
    )

    # ---- Citation post-validation -----------------------------------------
    note_md = _redact_invalid_citations(note_md, citation_ids)

    latency_ms = int((time.perf_counter() - start) * 1000)
    log.info(
        "generate_note.done",
        patient_id=patient_id,
        audience=request.audience.value,
        sections=list(included),
        tokens=tokens,
        latency_ms=latency_ms,
    )

    return GenerateNoteOutput(
        patient_id=patient_id,
        note_markdown=note_md,
        structured_action_items=action_items,
        llm_tokens_used=tokens,
        generation_time_ms=gen_ms,
        warnings=warnings,
        citations=all_citations,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Note composition
# ---------------------------------------------------------------------------


async def _compose_note(
    llm,
    request: GenerateNoteInput,
    patient: dict[str, Any],
    tool_results: dict[str, Any],
    citations: list[FhirReference],
) -> tuple[str, int | None, int | None]:
    if not llm.available:
        return _deterministic_note(request, patient, tool_results), None, None

    audience_guidance = {
        NoteAudience.PHYSICIAN: (
            "Concise, differential-oriented. Medical abbreviations acceptable. "
            "Prioritize actionable items in the first paragraph."
        ),
        NoteAudience.PHARMACIST: (
            "Drug-centric. Include mechanism and PK/PD details where relevant. "
            "Cite guideline sources for every recommendation."
        ),
        NoteAudience.NURSE: (
            "Administration-focused. List monitoring parameters and when to escalate. "
            "Avoid deep pharmacology."
        ),
        NoteAudience.PATIENT: (
            "Plain language at a sixth-grade reading level. No medical jargon. "
            "Use a warm, supportive tone. Explain *why* each change was made."
        ),
    }

    context_json = _context_block(patient, tool_results)
    citation_block = "\n".join(
        f"- [{c.reference}]{f' — {c.label}' if c.label else ''}" for c in citations
    )

    prompt = f"""You are composing a medication reconciliation note.

AUDIENCE: {request.audience.value}
FORMAT: {request.format.value}
LANGUAGE: {request.language}
AUDIENCE GUIDANCE: {audience_guidance.get(request.audience, "")}

ABSOLUTE RULES:
- Every clinical claim must end with a citation marker of the form [ResourceType/id]
- Only cite resource IDs that appear in the AVAILABLE CITATIONS list below
- Do NOT invent patient data; use only values present in the context
- Missing data → write "not available" rather than guessing
- If the language is not English, translate the entire note but keep citation markers unchanged

STRUCTURE for SOAP format:
  ## Subjective
  ## Objective
  ## Assessment
  ## Plan
  (end with a "URGENT ACTIONS" bullet list if any critical items exist)

STRUCTURE for NARRATIVE format: single flowing paragraph followed by Plan bullets.
STRUCTURE for BULLETED format: all sections as bullet lists.
STRUCTURE for TABLE format: use markdown tables where rows are medications/findings.

CONTEXT (JSON):
{context_json}

AVAILABLE CITATIONS:
{citation_block or '(none — no findings to cite)'}

Output the note in Markdown only. No preamble, no closing remarks."""

    t0 = time.perf_counter()
    # 18s timeout — long enough for one Gemini call but short enough that PO's
    # agent loop (typically ~30-60s patience) doesn't think the tool hung. If
    # the call fails or times out we fall through to the deterministic note,
    # which is already structurally complete; the LLM polish is a nice-to-have.
    result = await llm.generate(prompt, timeout_s=18.0, max_retries=1)
    gen_ms = int((time.perf_counter() - t0) * 1000)

    if result is None or not result.text.strip():
        return _deterministic_note(request, patient, tool_results), None, gen_ms

    return result.text.strip(), result.tokens_used, gen_ms


def _context_block(patient: dict[str, Any], tool_results: dict[str, Any]) -> str:
    import json

    def _dump(obj):
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return obj

    payload = {
        "patient": {
            "id": patient.get("id"),
            "name": patient_display_name(patient),
            "age": patient_age_years(patient),
            "sex": patient_sex(patient),
        },
        "interactions": _dump(tool_results.get("interactions")),
        "renal": _dump(tool_results.get("renal")),
        "discrepancies": _dump(tool_results.get("discrepancies")),
        "high_risk": _dump(tool_results.get("high_risk")),
    }
    return json.dumps(payload, default=str, indent=2)


def _deterministic_note(
    request: GenerateNoteInput,
    patient: dict[str, Any],
    tool_results: dict[str, Any],
) -> str:
    """Fallback for when LLM unavailable — mechanically composed from tool outputs."""
    name = patient_display_name(patient) or "Patient"
    age = patient_age_years(patient)
    sex = patient_sex(patient) or "unknown"
    lines = [
        "# Medication Reconciliation Note",
        "",
        f"**Patient**: {name} · Age {age} · Sex {sex} "
        f"· ID [{patient.get('id', 'unknown')}]",
        "",
    ]

    interactions = tool_results.get("interactions")
    if interactions and interactions.interactions:
        lines.append("## Drug-Drug Interactions")
        for i in interactions.interactions[:10]:
            ref = (
                f"[{i.source_fhir_resources[0].reference}]"
                if i.source_fhir_resources
                else ""
            )
            lines.append(
                f"- **{i.drug_a.get('name')} × {i.drug_b.get('name')}** — "
                f"{i.contextual_severity_label.value} "
                f"(score {i.contextual_severity_score:.1f}). "
                f"{i.recommended_action} {ref}"
            )
        lines.append("")

    renal = tool_results.get("renal")
    if renal and renal.renal_function.egfr_ml_min_1_73m2 is not None:
        egfr = renal.renal_function.egfr_ml_min_1_73m2
        stage = renal.renal_function.ckd_stage
        lines.append("## Renal Function")
        ref = ""
        if renal.renal_function.source_creatinine_fhir_reference:
            ref = f"[{renal.renal_function.source_creatinine_fhir_reference.reference}]"
        lines.append(f"- eGFR {egfr} mL/min/1.73m² — CKD stage {stage} {ref}")
        for rec in renal.recommendations[:10]:
            lines.append(
                f"- **{rec.medication.get('name')}** — {rec.recommended_action}: {rec.rationale}"
            )
        lines.append("")

    discrepancies = tool_results.get("discrepancies")
    if discrepancies and discrepancies.discrepancies:
        lines.append("## Home vs Hospital Discrepancies")
        for d in discrepancies.discrepancies:
            home = d.home_medication.rxnorm_ingredient_name if d.home_medication else "—"
            hosp = (
                d.hospital_medication.rxnorm_ingredient_name
                if d.hospital_medication
                else "—"
            )
            lines.append(
                f"- **{d.type.value}**: home={home} · hospital={hosp} · "
                f"{d.intentionality} (conf {d.intentionality_confidence:.2f}). "
                f"{d.recommended_action}"
            )
        lines.append("")

    high_risk = tool_results.get("high_risk")
    if high_risk and high_risk.findings:
        lines.append("## High-Risk Pattern Findings")
        for f in high_risk.findings:
            lines.append(
                f"- **{f.title}** ({f.severity.value}): {f.description} — "
                f"_Alt: {f.recommended_alternative or 'see guideline'}_"
            )
        lines.append("")

    lines.append("---")
    lines.append(
        "_AI-generated clinical decision support. Requires clinician review before "
        "clinical action. Built on synthetic data._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action items extraction
# ---------------------------------------------------------------------------


async def _extract_action_items(
    llm,
    note_md: str,
    tool_results: dict[str, Any],
    citations: list[FhirReference],
    citation_ids: set[str],
) -> list[ActionItem]:
    # Heuristic pass first — fast and always returns something.
    heuristic = _heuristic_action_items(tool_results, citations)
    if not llm.available:
        return heuristic

    prompt = f"""Extract a de-duplicated, prioritized action list from this note:

---
{note_md[:6000]}
---

Return JSON in this exact shape:
{{
  "items": [
    {{
      "priority": "trivial|minor|moderate|major|critical",
      "action": "single actionable imperative sentence",
      "reason": "brief clinical justification",
      "owner_role": "attending|resident|pharmacist|nurse|patient",
      "evidence_refs": ["ResourceType/id", ...]
    }}
  ]
}}

Rules: at most 8 items, highest priority first, no duplicates, evidence_refs
must appear in the note citation markers. If unsure, use "pharmacist" as owner."""

    parsed = await llm.generate_json(prompt, _ActionsList, timeout_s=12.0, max_retries=1)
    if parsed is None:
        return heuristic

    items: list[ActionItem] = []
    for raw in parsed.items:
        try:
            priority = Severity(raw.priority.lower())
        except ValueError:
            priority = Severity.MODERATE
        owner = raw.owner_role.lower()
        if owner not in ("attending", "resident", "pharmacist", "nurse", "patient"):
            owner = "pharmacist"

        # Validate refs against known citations
        validated_refs = []
        for ref_str in raw.evidence_refs:
            if ref_str in citation_ids:
                resource_type, _, resource_id = ref_str.partition("/")
                validated_refs.append(
                    FhirReference(resource_type=resource_type, resource_id=resource_id)
                )
        items.append(
            ActionItem(
                priority=priority,
                action=raw.action.strip(),
                reason=raw.reason.strip(),
                owner_role=owner,  # type: ignore[arg-type]
                evidence=validated_refs,
            )
        )

    # Combine heuristic + LLM, dedupe by action text
    seen_actions: set[str] = set()
    combined: list[ActionItem] = []
    for item in items + heuristic:
        key = item.action.strip().lower()
        if key in seen_actions:
            continue
        seen_actions.add(key)
        combined.append(item)

    # Sort by priority
    prio_order = {
        Severity.CRITICAL: 0,
        Severity.MAJOR: 1,
        Severity.MODERATE: 2,
        Severity.MINOR: 3,
        Severity.TRIVIAL: 4,
    }
    combined.sort(key=lambda x: prio_order.get(x.priority, 5))
    return combined[:12]


def _heuristic_action_items(
    tool_results: dict[str, Any],
    citations: list[FhirReference],
) -> list[ActionItem]:
    items: list[ActionItem] = []

    interactions = tool_results.get("interactions")
    if interactions:
        for i in interactions.interactions:
            if i.contextual_severity_label in (Severity.MAJOR, Severity.CRITICAL):
                items.append(
                    ActionItem(
                        priority=i.contextual_severity_label,
                        action=i.recommended_action,
                        reason=(
                            f"{i.drug_a.get('name')} × {i.drug_b.get('name')}: {i.mechanism}"
                        ),
                        owner_role="pharmacist",
                        evidence=list(i.source_fhir_resources),
                    )
                )

    renal = tool_results.get("renal")
    if renal:
        for r in renal.recommendations:
            if r.recommended_action in ("AVOID", "REDUCE"):
                items.append(
                    ActionItem(
                        priority=r.severity,
                        action=f"{r.recommended_action.title()}: {r.medication.get('name')}",
                        reason=r.rationale,
                        owner_role="pharmacist",
                        evidence=list(citations)[:1],
                    )
                )

    discrepancies = tool_results.get("discrepancies")
    if discrepancies:
        for d in discrepancies.discrepancies:
            if d.intentionality == "likely_unintentional":
                items.append(
                    ActionItem(
                        priority=d.clinical_significance,
                        action=d.recommended_action,
                        reason=d.reasoning,
                        owner_role="attending",
                        evidence=[],
                    )
                )

    high_risk = tool_results.get("high_risk")
    if high_risk:
        for f in high_risk.findings:
            if f.severity in (Severity.MAJOR, Severity.CRITICAL):
                items.append(
                    ActionItem(
                        priority=f.severity,
                        action=(f.recommended_alternative or f.title),
                        reason=f.description,
                        owner_role="pharmacist",
                        evidence=list(f.evidence)[:2],
                    )
                )

    return items[:12]


# ---------------------------------------------------------------------------
# Citation post-validation
# ---------------------------------------------------------------------------


_CITE_PATTERN = re.compile(r"\[([A-Z][A-Za-z]+)/([^\]]+)\]")


def _redact_invalid_citations(note_md: str, valid_ids: set[str]) -> str:
    """Replace any [ResourceType/id] marker that isn't in valid_ids with '[?]'."""

    def _repl(match: re.Match) -> str:
        ref = f"{match.group(1)}/{match.group(2)}"
        return match.group(0) if ref in valid_ids else "[?]"

    return _CITE_PATTERN.sub(_repl, note_md)
