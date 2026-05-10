# Argus — MedRec Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastMCP 3.x](https://img.shields.io/badge/MCP-FastMCP%203.x-green.svg)](https://gofastmcp.com/)
[![FHIR R4](https://img.shields.io/badge/FHIR-R4-orange.svg)](https://hl7.org/fhir/R4/)
[![SHARP-on-MCP](https://img.shields.io/badge/SHARP--on--MCP-compliant-brightgreen.svg)](https://www.sharponmcp.com/)
[![Tests](https://img.shields.io/badge/tests-51%20passing-success.svg)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> A medication reconciliation and safety **MCP server** for the [Prompt Opinion](https://app.promptopinion.ai/) platform.
> Built for the [Agents Assemble hackathon](https://agents-assemble.devpost.com/).

Argus exposes six composable healthcare tools via the Model Context Protocol (MCP). Any
agent on Prompt Opinion (or any other MCP-compatible host) can pick up these tools to
deliver medication reconciliation, drug-interaction analysis, renal dose checks,
high-risk pattern screening, and clinician-ready note generation — all grounded in FHIR
R4 data with traceable citations.

**Stack**: Python 3.11 · [FastMCP](https://gofastmcp.com/) · FHIR R4 · [RxNav](https://lhncbc.nlm.nih.gov/RxNav/) · Gemini 2.5 Flash-Lite · XGBoost · SHAP

## Demo output (against an 84 yo polypharmacy patient)

```
$ get_active_medications
  → 15 ingredients: lisinopril, simvastatin, warfarin, atorvastatin,
                    metoprolol, amlodipine, clopidogrel, digoxin,
                    aspirin, hydrochlorothiazide, alendronate, ...

$ renal_dose_check
  → eGFR 38.4 mL/min/1.73m² (CKD-EPI 2021), Stage 3b CKD
  → digoxin: REDUCE — clearance ↓ in CKD; suggest 0.125 mg
  → lisinopril: MONITOR — track K+ and creatinine

$ check_drug_interactions
  → warfarin × aspirin — CRITICAL (score 4.7) — bleeding risk
  → amlodipine × simvastatin — MODERATE (score 3.2) — myopathy risk

$ generate_med_rec_note
  → Full SOAP note with [MedicationRequest/abc-123] inline citations
```

---

## Why this exists

Medication errors cause ~7,000 US deaths/year and ~$3.5B in preventable costs. Half happen at
care transitions — admission, transfer, discharge. Joint Commission NPSG.03.06.01 requires
reconciliation at every transition, and every hospital does it badly: on paper, in 15 minutes,
by an overworked intern.

Argus replaces that with a set of tools an AI agent can compose on demand.

## The six tools

| # | Tool | What it does |
|---|------|-------------|
| 1 | `get_active_medications` | Deduplicated, canonical current medication list for a patient. The foundation every other tool depends on. |
| 2 | `check_drug_interactions` | Clinical severity-ranked DDIs with patient-specific context (age, labs, coadministered meds) — beats rule-based checker alarm fatigue. |
| 3 | `renal_dose_check` | eGFR-aware dose-adjustment recommendations using CKD-EPI 2021. |
| 4 | `reconcile_home_vs_hospital` | Home vs. current-encounter discrepancy analysis with intentional-vs-unintentional classification. |
| 5 | `generate_med_rec_note` | Orchestrator — produces a clinician-ready note with inline FHIR resource citations. |
| 6 | `screen_high_risk_patterns` | Beers (elderly), QTc-prolonging combos, opioid+benzo, anticholinergic burden, adherence gap. |

See `docs/ARCHITECTURE.md` for the full spec of each tool.

## Quick start

```bash
# 1. Clone & install
git clone <your-repo> argus
cd argus
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# edit .env — add GEMINI_API_KEY

# 3. Build the reference knowledge base (one-time, ~10s)
python -m argus.reference.build_kb

# 4. Run the MCP server
python -m argus.server
# or: argus-server

# 5. Expose to Prompt Opinion via ngrok (dev)
ngrok http 8080
```

In the Prompt Opinion workspace, go to **Workspace Hub → MCP Servers → Add** and paste the
ngrok URL + `/mcp`. Enable the "Pass FHIR token" option so SHARP context propagates.

> The "Pass FHIR token" toggle only appears once PromptOpinion confirms the server
> advertises the [SHARP-on-MCP](https://www.sharponmcp.com/) `fhir_context_required`
> capability. Argus does this automatically. Confirm with:
> ```bash
> python scripts/verify_sharp.py http://127.0.0.1:8080/mcp
> ```
> Expected output: `[OK] SHARP-on-MCP capability advertised correctly`.

## Project layout

```
argus/
├── argus/                  # Main package
│   ├── server.py           # FastMCP entry point; registers tools
│   ├── config.py           # Settings (pydantic-settings)
│   ├── schemas.py          # Pydantic I/O models for every tool
│   ├── fhir_client.py      # Async FHIR client; handles SHARP context
│   ├── rxnorm.py           # RxNav client + SQLite cache
│   ├── sharp_context.py    # SHARP extension token extraction
│   ├── logging_setup.py    # structlog config
│   ├── tools/              # One module per MCP tool
│   │   ├── get_medications.py
│   │   ├── check_interactions.py
│   │   ├── renal_check.py
│   │   ├── reconcile.py
│   │   ├── generate_note.py
│   │   └── screen_patterns.py
│   ├── ml/                 # ML model wrappers + training artifacts
│   └── reference/          # KB builder + seed data
│       ├── build_kb.py
│       └── data/
├── scripts/                # Operations — synthea gen, training, upload
├── tests/                  # pytest
├── a2a_agent/              # Separate submission — A2A agent wrapper
├── docs/                   # ARCHITECTURE.md, SAFETY.md
├── Dockerfile
├── fly.toml
└── pyproject.toml
```

## Data

Argus uses **only synthetic data** (Synthea-generated FHIR R4 Bundles). No real PHI ever
touches the system. See `SAFETY.md`.

Generate a demo cohort:

```bash
./scripts/generate_synthea.sh
python scripts/upload_to_prompt_opinion.py  # uploads to your PO workspace
```

## Deploy

```bash
# Fly.io (free tier)
fly launch --no-deploy
fly secrets set GEMINI_API_KEY=xxx
fly deploy
```

Or any container platform — see `Dockerfile`.

## Testing

```bash
pytest                       # full suite
pytest -k "test_rxnorm"      # one module
pytest --cov=argus           # coverage
```

## Safety & licensing

- **No PHI**: synthetic data only; any real PHI in inputs is rejected by design.
- **No autonomous writes**: Argus never modifies FHIR resources. It only reads and analyzes.
- **Clinician-in-loop**: every output carries a mandatory disclaimer.
- **License**: Apache 2.0.

## Hackathon submission

This repo produces **two** Prompt Opinion Marketplace listings:

1. **Argus MCP** — the raw MCP server (this package).
2. **Argus Agent** — an A2A agent that composes the MCP into full admission/discharge
   workflows. See `a2a_agent/`.

Both are substantively different and permitted as separate submissions per the rules.

## Demo

3-minute demo: *[YouTube link]* — see `docs/DEMO_SCRIPT.md` for the storyboard.

## Troubleshooting

**"This MCP server does not support PromptOpinion's FHIR extension"**
The server is not advertising `experimental.fhir_context_required` in its
MCP `initialize` response. Make sure you are running the latest Argus and
that `python scripts/verify_sharp.py` succeeds before registering with
PromptOpinion.

**Tool call returns 403 `fhir_context_required`**
Expected — the server is enforcing the SHARP-on-MCP spec. PromptOpinion
sends the FHIR headers automatically once the "Pass FHIR token" toggle is
checked in the MCP server config dialog. Make sure that toggle is on.

**Local smoke tests get 403**
Set `ARGUS_ENV=dev` in your `.env` to bypass the 403 middleware while
running without PromptOpinion in front.

**Port 8080 already in use**
Override with `ARGUS_PORT=8765` (or any free port) and point ngrok at the
same port.
