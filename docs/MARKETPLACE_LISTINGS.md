# Prompt Opinion Marketplace — Listing Copy

Paste this into your marketplace listings when you publish.

---

## Listing 1: Argus — MedRec Copilot MCP

**Name**: Argus — MedRec Copilot

**Tagline**: Six composable medication safety tools, FHIR-native.

**Category**: Medication management · Clinical decision support · MCP server

**Short description** (150 chars):
Medication reconciliation and safety MCP. Drug interactions, renal dosing, Beers criteria, home-vs-hospital reconciliation, clinician-ready notes.

**Long description**:

Argus exposes six composable clinical tools over MCP:

1. **get_active_medications** — Deduplicated, RxNorm-normalized current medication list. The foundation every other tool depends on.
2. **check_drug_interactions** — Patient-context-aware DDI analysis with SHAP-explained severity ranking. Beats standard rule-based checkers.
3. **renal_dose_check** — eGFR-aware dose adjustments using CKD-EPI 2021 and embedded FDA/KDIGO guidance.
4. **reconcile_home_vs_hospital** — Discrepancy detection with LLM-classified intentionality (intentional vs. needs review).
5. **generate_med_rec_note** — Orchestrator that composes a clinician-ready Markdown note with inline FHIR resource citations.
6. **screen_high_risk_patterns** — Sweeps for Beers criteria, QTc-prolonging combos, opioid+benzo, anticholinergic burden, adherence gaps, pregnancy risk, serotonin syndrome.

**Every output** carries:
- Traceable FHIR resource citations for every clinical claim
- A graduated severity scale and structured action items
- A clinician-in-loop safety disclaimer

**Built on**: FHIR R4, SHARP extension for context propagation, RxNav for drug normalization, XGBoost + SHAP for contextual severity, Gemini for LLM reasoning.

**Data**: Synthetic only (Synthea). No real PHI ever.

**Ideal for**: Any agent working on admission / discharge / transfer medication reconciliation, high-risk prescribing review, pharmacist-facing workflows.

**Tags**: `medication-reconciliation` `drug-interactions` `renal-dosing` `beers-criteria` `pharmacy` `clinical-decision-support` `fhir-r4` `mcp` `synthetic-data`

---

## Listing 2: Argus Agent — A2A Workflow Orchestrator

**Name**: Argus Agent — MedRec Workflows

**Tagline**: Clinician-level med rec workflows, powered by A2A.

**Category**: Clinical agents · Pharmacy workflows · A2A

**Short description** (150 chars):
A2A agent that orchestrates six Argus MCP tools into complete admission, discharge, and new-prescription workflows. Natural language in, actionable outputs out.

**Long description**:

Where the Argus MCP gives you atomic tools, the Argus Agent gives you whole workflows. It speaks four clinician-level skills:

- `run_admission_med_rec(patient_id)` — Runs the full admission reconciliation pipeline: canonical med list, DDI analysis with patient context, renal screening, home-vs-hospital comparison, high-risk pattern screen, and a signable Markdown note.
- `run_discharge_med_rec(patient_id, disposition)` — Discharge variant that includes a patient-language handout in the preferred language.
- `evaluate_new_prescription(patient_id, rxnorm, dose)` — Pre-write safety check for a proposed prescription against the current regimen.
- `explain_medication_concern(concern, audience)` — Deep-dive clinical explanation at physician / pharmacist / nurse / patient levels.

**Under the hood**: plans which MCP tools to call, runs them in parallel where possible, passes SHARP context through, and composes the results with an LLM that is required to cite every FHIR resource it uses.

**Substantively different** from the MCP submission because it operates at the **workflow abstraction level** with planning, memory, and audience-specific composition — not at the tool level.

**Tags**: `a2a` `medication-reconciliation` `clinical-agents` `pharmacy-workflows` `admission` `discharge`
