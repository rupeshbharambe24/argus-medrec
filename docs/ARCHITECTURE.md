# Architecture

> Argus is a **Model Context Protocol (MCP) server** plus an **A2A agent** that
> delivers medication reconciliation and safety tooling to the Prompt Opinion
> platform. This doc covers the internals; see `README.md` for quick-start.

## System diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Prompt Opinion Workspace                        │
│                                                                          │
│    ┌────────────┐       ┌───────────────┐     ┌──────────────────┐      │
│    │ Launchpad  │──────▶│ General Agent │────▶│ Argus A2A Agent  │      │
│    │   Chat     │       └───────────────┘     │ (sub #2)         │      │
│    └────────────┘              │              └──────────────────┘      │
│                                │                       │                 │
│                                │  MCP    ┌──────────────────────────┐   │
│                                └────────▶│   Argus MCP (sub #1)     │   │
│                                          │                          │   │
│                                          │  6 composable tools      │   │
│                                          └──────────────────────────┘   │
│                                                     │                    │
│                                                     │ FHIR + SHARP       │
│                                                     ▼                    │
│                                          ┌──────────────────────────┐   │
│                                          │  FHIR server (workspace) │   │
│                                          │  Synthea bundles loaded  │   │
│                                          └──────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ outbound
                                 ▼
                  ┌────────────────────────────┐
                  │ RxNav (NLM)                │
                  │ Gemini 2.5 Flash           │
                  └────────────────────────────┘
```

## Request lifecycle — end to end

A typical `generate_med_rec_note` call traces as:

```
 Prompt Opinion → MCP tool call with SHARP headers
   │
   ├─ extract_sharp_context()   ← pulls FHIR base URL + bearer from headers
   │
   ├─ generate_note.run()
   │    │
   │    ├── get_active_medications.run()              ┐
   │    │     ├── FHIR: MedicationRequest/Statement/Dispense (parallel)
   │    │     ├── RxNav: ingredient normalization (cached in SQLite)
   │    │     └── dedupe by (ingredient, route)
   │    │
   │    ├── check_drug_interactions.run()             │  asyncio.gather
   │    │     ├── FHIR: Patient + Conditions + Labs   │  (all tools run
   │    │     ├── SQLite: base severity lookup        │   in parallel)
   │    │     ├── XGBoost model (or heuristic)        │
   │    │     └── Gemini: patient-specific action     │
   │    │                                             │
   │    ├── renal_dose_check.run()                    │
   │    │     ├── FHIR: Patient + latest creatinine   │
   │    │     ├── CKD-EPI 2021 eGFR                   │
   │    │     └── SQLite: renal_dosing_rules          │
   │    │                                             │
   │    ├── reconcile_home_vs_hospital.run()          │
   │    │     ├── FHIR: home meds + encounter orders  │
   │    │     ├── ingredient-level diff               │
   │    │     └── Gemini: intentionality classifier   │
   │    │                                             │
   │    └── screen_high_risk_patterns.run()           ┘
   │         └── 7 screens, each SQLite lookups
   │
   ├─ aggregate tool outputs
   ├─ Gemini: note composition with citation rules
   ├─ validate every [ResourceType/id] marker
   └─ return GenerateNoteOutput(disclaimer, citations, note_markdown, ...)
```

Every tool output includes:
- A uniform `BaseToolResponse` envelope (disclaimer, citations, warnings, latency_ms)
- Structured, schema-validated data
- A FHIR resource-id citation trail for every clinical claim

## Data flow and trust boundaries

| Trust zone | What flows | Controls |
|------------|-----------|----------|
| Prompt Opinion ↔ Argus MCP | SHARP headers, tool args, tool results | HTTPS; bearer tokens never logged |
| Argus ↔ FHIR server | Patient data, clinical resources | Bearer auth from SHARP; no writes |
| Argus ↔ RxNav | Drug RxCUIs only (no PHI) | Public API; cached locally |
| Argus ↔ Gemini | Structured clinical context | No patient names / MRNs in prompts |

## Stateless by design

- No persistent per-patient state. Every call fetches fresh FHIR data.
- Only caches are: RxNav normalization (drug reference, not PHI) and the reference KB.
- Instances are interchangeable; horizontal scaling works out of the box.

## Performance budget (p95)

| Tool | Target | Dominant cost |
|------|-------:|---------------|
| `get_active_medications`       | 1.5 s | FHIR roundtrip + cache misses on RxNav |
| `check_drug_interactions`      | 3.0 s | Parallel patient-context fetch + LLM action |
| `renal_dose_check`             | 1.0 s | Single FHIR query, in-memory SQLite |
| `reconcile_home_vs_hospital`   | 4.0 s | LLM intentionality pass |
| `generate_med_rec_note`        | 8.0 s | LLM composition is the dominant cost |
| `screen_high_risk_patterns`    | 2.0 s | All 7 screens in parallel |

## ML: DDI contextual severity model

- **Family**: XGBoost regressor, depth 5, 300 estimators
- **Target**: scalar 0–5 severity score
- **Features** (11 total): base severity, age, sex, eGFR, K+, INR, QTc, CKD/hepatic/cardiac flags, polypharmacy count
- **Explainer**: SHAP TreeExplainer — every prediction is decomposed into per-feature contributions that appear in the tool output as `patient_specific_factors`
- **Fallback**: a transparent heuristic when the `.xgb` artifact is not present. Disclosed in tool output via `contextual_severity_source`
- **Training data**: synthetic (generated by `scripts/train_ddi_model.py`). Hackathon-grade and disclosed as such; a production replacement would use real outcome labels.

## SHARP-on-MCP context

Argus implements the [SHARP-on-MCP specification](https://www.sharponmcp.com/)
for receiving FHIR session credentials from the calling agent platform.

**Capability declaration** — Argus advertises the following in its MCP
`initialize` response:

```json
{
  "capabilities": {
    "experimental": { "fhir_context_required": { "value": true } }
  }
}
```

This is what tells compliant agent platforms (PromptOpinion, etc.) to expose
the "Pass FHIR token" toggle to the user and propagate FHIR session credentials
on every subsequent tool call. Without this flag, PromptOpinion shows the
warning *"This MCP server does not support PromptOpinion's FHIR extension"*
and the toggle is hidden.

The flag is injected by monkey-patching the FastMCP low-level server's
`create_initialization_options` so it survives across all transports.

**Headers** — On each `tools/call` request, the platform sends:

```
X-FHIR-Server-URL:    https://workspace/fhir
X-FHIR-Access-Token:  <bearer token>
X-Patient-ID:         <FHIR Patient logical id>     (optional)
X-Encounter-ID:       <FHIR Encounter logical id>   (optional)
```

Legacy `x-sharp-*` aliases are still accepted for backwards compatibility, but
the canonical name always wins when both are present. Header parsing lives in
`argus/sharp_context.py`.

**403 enforcement** — Per spec section 3, tool-call requests missing the
required headers (`X-FHIR-Server-URL`, `X-FHIR-Access-Token`) are rejected
with HTTP 403 by `SharpFhirContextMiddleware` *before* reaching the tool layer.

The middleware uses `Mcp-Session-Id` to discriminate between the `initialize`
handshake (no session id yet, so it passes through unchecked) and follow-up
`tools/call` requests (must carry the SHARP headers). This matches PromptOpinion's
behavior of sending the FHIR headers only on per-call context, not on the
session-establishing initialize.

**Dev mode fallback** — When `ARGUS_ENV=dev`:
- The 403 middleware is bypassed entirely so local smoke tests work without
  a real PromptOpinion in front.
- Missing headers fall back to `ARGUS_FALLBACK_FHIR_BASE_URL` /
  `ARGUS_FALLBACK_FHIR_TOKEN`, then to the HAPI public sandbox.

**Verification** — Run `python scripts/verify_sharp.py [url]` against a
running server; it asserts the `fhir_context_required.value: true` flag is
present in the initialize response.

## Failure behavior

Argus is designed to degrade gracefully, not crash:

- **LLM unavailable** → deterministic templates for notes and actions
- **RxNav down** → medications returned with raw codes + `coverage_score < 1.0` warning
- **One FHIR query fails** → `asyncio.gather` result marked as exception, captured in `warnings`, other results still returned
- **No creatinine on file** → `RenalCheckOutput.missing_data` populated; no eGFR computed
- **Unknown RxNorm code** → kept in list, marked `rxnorm_ingredient_code: null`

## What is *not* in scope

- Writing FHIR resources (no CRUD beyond read)
- Patient-identifying data in logs
- Automated actions (every output requires clinician review)
- Billing / coding (out of competition scope)

## A2A Agent (sub #2) — separate submission

See `a2a_agent/README.md`. The agent is a higher-level orchestration layer
(skills: `run_admission_med_rec`, `run_discharge_med_rec`, `evaluate_new_prescription`,
`explain_medication_concern`) that composes MCP tool calls into clinician-level
abstractions. It is substantively different from the raw MCP and is permitted
as a second submission under the competition rules.
