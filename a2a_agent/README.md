# Argus A2A Agent — Second Submission

> This is the **second marketplace submission** for the hackathon. It is a
> Google ADK-based A2A agent that orchestrates the Argus MCP tools into
> clinician-level workflows.

## Relationship to the MCP submission

The MCP submission (repo root) exposes six **atomic tools**. This agent exposes
four **workflow skills**:

| Skill | What it does | MCP tools used |
|-------|-------------|-----------------|
| `run_admission_med_rec` | Full admission reconciliation in one call | 1 → (2, 3, 6 parallel) → 4 → 5 |
| `run_discharge_med_rec` | Discharge variant + patient handout | 1 → (2, 3, 6) → 5 (patient audience) |
| `evaluate_new_prescription` | Pre-write safety check for a proposed med | 1 + 2 (test-before-prescribe mode) |
| `explain_medication_concern` | Audience-tailored deep-dive on one finding | Re-queries the relevant MCP tool |

This is **substantively different** from the MCP per hackathon rules because it
operates at a workflow abstraction layer — planning, memory, audience shaping —
not at the tool layer.

## Implementation plan

The Prompt Opinion team provides official ADK starter repos that should be
cloned as the starting point rather than building from scratch:

```bash
git clone https://github.com/prompt-opinion/po-adk-python argus_a2a
cd argus_a2a
```

Replace the default agent definition with one that:

1. Loads the Argus MCP as a tool source (by its marketplace URL).
2. Registers the four skills listed above in the agent card.
3. For each skill, implements a planner loop that:
   - Calls MCP tools in parallel where independent
   - Aggregates results into the LLM prompt context
   - Emits a single final response with structured action items
4. Uses the same `GEMINI_API_KEY` from `.env` — no new secrets.
5. Preserves SHARP context passthrough so downstream MCP calls stay authenticated.

## Skill definitions (agent card)

```yaml
agentCard:
  name: Argus Agent — MedRec Copilot
  description: |
    Orchestrates medication reconciliation, drug-interaction analysis, renal
    dose checking, and high-risk pattern screening into complete clinician
    workflows for admission, discharge, and new-prescription scenarios.
  skills:
    - id: run_admission_med_rec
      name: Admission medication reconciliation
      description: |
        Complete med rec workflow for a newly admitted patient. Returns a
        signable note, an ordered action list, and flagged concerns.
      inputs:
        - name: patient_id
          type: string
          required: false
        - name: encounter_id
          type: string
          required: false
      outputs:
        - name: reconciliation_note
          type: markdown
        - name: action_items
          type: array
        - name: severity_summary
          type: object

    - id: run_discharge_med_rec
      name: Discharge medication reconciliation
      description: |
        Discharge variant. Produces a physician-facing summary and a
        patient-facing handout in the patient's preferred language.
      inputs:
        - name: patient_id
          type: string
          required: false
        - name: disposition
          type: string
          description: home | rehab | SNF | hospice | other
          required: true
        - name: patient_language
          type: string
          default: en

    - id: evaluate_new_prescription
      name: New prescription safety check
      description: |
        Pre-write analysis of a proposed medication against the patient's
        current regimen. Returns a verdict (approve | caution | avoid) with
        justification.
      inputs:
        - name: patient_id
          type: string
          required: false
        - name: proposed_rxnorm
          type: string
          required: true
        - name: proposed_dose_value
          type: number
          required: false
        - name: proposed_dose_unit
          type: string
          required: false

    - id: explain_medication_concern
      name: Explain a medication concern
      description: |
        Audience-tailored deep-dive on a specific concern (DDI, Beers flag,
        renal dose, discrepancy). Used as a conversational follow-up.
      inputs:
        - name: concern_summary
          type: string
          required: true
        - name: audience
          type: string
          default: physician
```

## Local dev

```bash
cd argus_a2a
pip install -r requirements.txt
cp .env.example .env
# edit .env: GEMINI_API_KEY + ARGUS_MCP_URL (your ngrok for the MCP)
python -m argus_a2a
```

Then register the agent with Prompt Opinion via Workspace Hub → External
Agents → Add Connection. The workspace will fetch the agent card and
make the skills available to other agents.

## Files to write

The exact implementation depends on the ADK starter's structure, but the
skeleton is three files:

```
argus_a2a/
├── agent.py          # ADK agent definition, registers skills
├── planning.py       # Per-skill orchestration logic (the interesting code)
└── middleware.py     # SHARP context passthrough + MCP URL injection
```

`planning.py` is the heart of this submission. Example pseudocode for
`run_admission_med_rec`:

```python
async def run_admission_med_rec(ctx, patient_id: str | None, encounter_id: str | None):
    mcp = await ctx.get_mcp_client("argus")  # uses cached marketplace listing

    # Step 1: canonical medication list
    meds = await mcp.call("get_active_medications", patient_id=patient_id)

    # Step 2: parallel analyses
    interactions, renal, screens = await asyncio.gather(
        mcp.call("check_drug_interactions", patient_id=patient_id),
        mcp.call("renal_dose_check", patient_id=patient_id),
        mcp.call("screen_high_risk_patterns", patient_id=patient_id),
    )

    # Step 3: reconcile (depends on nothing parallel-safe)
    discrepancies = await mcp.call(
        "reconcile_home_vs_hospital",
        patient_id=patient_id,
        encounter_id=encounter_id,
    )

    # Step 4: compose final note
    note = await mcp.call(
        "generate_med_rec_note",
        patient_id=patient_id,
        audience="physician",
        format="soap",
    )

    return {
        "reconciliation_note": note["note_markdown"],
        "action_items": note["structured_action_items"],
        "severity_summary": interactions["summary"],
        "flagged_concerns": screens["findings"],
    }
```

## Why this earns its own submission

1. **Different abstraction level**: The MCP provides tools; the agent provides workflows.
2. **Planning behavior**: Conditional tool calls, early-termination on missing patient context, parallel fan-out for speed.
3. **Composition across audiences**: Same patient data, different outputs for physician vs. pharmacist vs. patient — done at the agent, not the MCP.
4. **Natural-language interface**: Clinicians type "do admission med rec for this patient," not "call tool 1 then tool 2 then tool 4."

Per the hackathon rules (multiple submissions allowed if substantively
different), this is a legitimate second entry. Two marketplace listings, two
demo videos, two shots at the prize pool from a single codebase.
