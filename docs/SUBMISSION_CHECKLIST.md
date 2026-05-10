# Argus — Hackathon Submission Checklist

Everything from "code works locally" to "submitted on Devpost". Work through it in order.

---

## Phase 1 — Deploy to Fly.io (stable URL)

ngrok URLs change every restart and the judging window is May 11–25; you need a permanent endpoint.

### Install flyctl

```powershell
# Windows
iwr https://fly.io/install.ps1 -useb | iex
# Restart PowerShell after install completes so PATH refreshes
```

### One-time setup

```powershell
cd "D:\Projects\Agents Assemble\argus"
fly auth signup     # or: fly auth login (if you already have an account)
```

### Deploy

The `fly.toml` and `Dockerfile` are already configured. Just run:

```powershell
# Set the API key as a Fly secret (NOT in fly.toml — that file is committed)
fly secrets set GEMINI_API_KEY=<your-argus-key>

# Deploy
fly deploy
```

First deploy takes ~3-5 minutes (image build + push + boot). When it finishes, fly prints a URL like `https://argus-mcp.fly.dev`.

### Verify the deployment

```powershell
# Confirm SHARP capability is advertised by the deployed instance
"D:/commonenv/Scripts/python.exe" scripts/verify_sharp.py https://argus-mcp.fly.dev/mcp
```

Expected: `[OK] PromptOpinion FHIR-context extension advertised (8 scopes).`

### Re-register in PromptOpinion with the stable URL

In PO: **Configuration → MCP Servers** → delete the old ngrok-based entry → **Add MCP Server**:

| Field | Value |
|---|---|
| Name | `Argus` |
| URL | `https://argus-mcp.fly.dev/mcp` |
| Transport | Streamable HTTP |
| Auth | None |
| Pass FHIR token | ✅ checked |

Test → all 6 tools list → Save.

Re-run the 5 demo prompts against Dewey to confirm everything works against the deployed instance.

---

## Phase 2 — Marketplace publish (Stage 1 pass/fail gate)

Per the rules, the MCP must be published on the PO Marketplace and discoverable by judges.

1. PO sidebar → **Marketplace Studio** (or "Publish")
2. Click **Publish MCP Server**
3. Fill in fields from `docs/MARKETPLACE_LISTINGS.md` — Listing 1 copy is ready to paste
4. Set the URL to `https://argus-mcp.fly.dev/mcp` (your Fly URL, NOT ngrok)
5. Tags: paste from the listing doc
6. Submit

Wait for the listing to confirm published. Copy the marketplace URL — you'll need it for Devpost.

---

## Phase 3 — Demo video (≤3 minutes, mandatory)

Storyboard is in `docs/DEMO_SCRIPT.md`. Hard rules: max 3:00, English (or English-subtitled), shows the project running inside PromptOpinion, no copyrighted music.

### Recording

Quick + free: **OBS Studio** (https://obsproject.com) or Windows built-in **Game Bar** (Win+G → Capture).

Sequence:

1. **0:00–0:15 — Problem hook**: voiceover over a still — "Joint Commission requires medication reconciliation at every transition. 7,000 deaths/year, half of all errors happen at admission/discharge. Most hospitals do it on paper."
2. **0:15–0:30 — Architecture cut**: show the architecture diagram from `docs/ARCHITECTURE.md` (screenshot it). "Argus is six composable MCP tools, FHIR-native, with an A2A agent on top."
3. **0:30–2:30 — Live workflow** in PromptOpinion's Launchpad with Dewey selected:
   - Type prompt 1 → show the 15-med list
   - Type prompt 2 → show eGFR 38.4, CKD 3b, lisinopril + digoxin recs
   - Type prompt 3 → show warfarin + aspirin critical interaction
   - Type prompt 5 → show the SOAP note rendering with citations
4. **2:30–3:00 — Close**: "Built on synthetic Synthea data. Every output cites FHIR resources. Clinician-in-loop. Ready for any A2A agent on PromptOpinion." Show the marketplace listing as final frame.

### Upload

YouTube (unlisted is fine) or Vimeo. Note the URL.

---

## Phase 4 — Devpost submission

Go to https://agents-assemble.devpost.com → Submit.

### Required fields

| Field | Source |
|---|---|
| Project name | Argus — MedRec Copilot |
| Tagline | "Six composable medication-safety tools for any healthcare AI agent." |
| Description | Paste from `docs/MARKETPLACE_LISTINGS.md` Listing 1 long description, plus a short "Tech stack" + "Built with" line |
| Marketplace URL | (from Phase 2) |
| Demo video | (YouTube/Vimeo URL from Phase 3) |
| Repository URL | Your GitHub repo (push the code first if you haven't) |
| Built with | `python` `fastmcp` `fhir` `gemini` `xgboost` `shap` `synthea` `rxnorm` |

### Optional but recommended

- **Architecture image**: screenshot of the diagram in `docs/ARCHITECTURE.md`
- **Try-it screenshots**: the chat output from your last successful demo run
- **Built with → custom tools**: add `Synthea` and `RxNav` for visibility

---

## Phase 5 — Optional second submission (A2A Agent)

Per rules, multiple substantively-different submissions are allowed. The A2A agent (`a2a_agent/`) is workflow-level vs. the MCP's tool-level — substantively different.

If time permits, scaffold and publish the A2A agent as a second listing using the Listing 2 copy. Two shots at the 13 prize slots from one codebase.

---

## Final pre-submit sanity check

- [ ] `pytest -q` passes (51 tests)
- [ ] `python scripts/verify_sharp.py https://argus-mcp.fly.dev/mcp` prints `[OK]`
- [ ] PromptOpinion registration uses the Fly URL (not ngrok)
- [ ] All 5 demo prompts return real clinical output against Dewey
- [ ] Marketplace listing published and visible to logged-out viewers
- [ ] Demo video uploaded, ≤ 3:00, English audio or captions
- [ ] Devpost submission saved (you can edit until the deadline)
- [ ] No PHI in the codebase, video, or listing — Synthea synthetic only
- [ ] Repository pushed (if including link in submission)
