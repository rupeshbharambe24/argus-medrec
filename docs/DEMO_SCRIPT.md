# Demo Video Script — Argus / MedRec Copilot

> **Target duration**: 2:55 (hard cap 3:00 per hackathon rules)
> **Format**: Screen capture + voiceover; minimal production.
> **Tools**: OBS Studio or Loom for recording; DaVinci Resolve free for edits.

## Structure

```
00:00–00:12  The pain                       12s
00:12–00:35  The product                    23s
00:35–01:00  Setup in Prompt Opinion        25s
01:00–01:45  The workflow (live demo)       45s
01:45–02:15  The differentiators            30s
02:15–02:40  The impact                     25s
02:40–02:55  The close                      15s
```

## Shot list & voiceover

### [00:00–00:12] The pain
**Screen**: Title card "Argus — MedRec Copilot" → still image of an ICU medication cart or EHR med list.
**VO**: *"Medication errors kill about 7,000 Americans every year. Half happen at hospital admission and discharge. Joint Commission requires reconciliation at every transition — and every hospital does it on paper, in fifteen minutes, by an overworked intern."*

### [00:12–00:35] The product
**Screen**: The architecture diagram from `docs/ARCHITECTURE.md`, annotated with callouts to the 6 MCP tools.
**VO**: *"Argus is two submissions on Prompt Opinion: an MCP server with six composable clinical tools, and an A2A agent that chains them into complete admission workflows. Every tool returns evidence-linked outputs. Every clinical claim is traceable to a FHIR resource."*

### [00:35–01:00] Setup
**Screen**: Prompt Opinion launchpad. Select patient "Carlos Martinez, 82, CKD stage 3b, 11 active medications."
**VO**: *"I've loaded ten Synthea patients with realistic polypharmacy. I'm selecting Mr. Martinez — eighty-two years old, stage 3b chronic kidney disease, eleven active medications, on warfarin for atrial fibrillation. I'll launch the general agent and ask it to consult Argus."*

### [01:00–01:45] Workflow (the demo)
**Screen**: Chat input: *"Run complete admission medication reconciliation for this patient."* → streaming response.
**VO**: *"Argus pulls his canonical list — deduplicated from 14 raw FHIR records to 11 ingredients. It runs 55 interaction checks in parallel and flags three. The warfarin-amiodarone pair scored 4.6 because he just added clarithromycin — which triples warfarin's effect acutely. A rule-based checker can't see that compounding; ours does."*

**B-roll overlay**: zoom into the SHAP factor list on-screen.

### [01:45–02:15] Differentiators
**Screen**: Expand the interaction detail showing patient_specific_factors with SHAP values. Then the generated note.
**VO**: *"Every severity score is explained — here's the SHAP breakdown showing age eighty-plus added 0.4, recent INR above range added 0.5. Every claim in the generated note cites a FHIR resource ID. Renal dosing caught his metformin — eGFR 34, needs reduction or avoidance. Beers criteria flagged diphenhydramine. Home-vs-hospital reconciliation found one unintentional omission: his ACE inhibitor was dropped on admission with no documented reason."*

### [02:15–02:40] Impact
**Screen**: The structured action items list. Then toggle the language selector to Spanish and show the patient handout regenerating.
**VO**: *"The output is a reconciliation note the attending can sign, a structured pharmacist action list, and a patient handout in Spanish because Carlos's preferred language is Spanish. Ninety seconds of agent work replaces fifteen minutes of manual chart review — at every admission."*

### [02:40–02:55] Close
**Screen**: The marketplace listing in Prompt Opinion.
**VO**: *"Argus is live in the Prompt Opinion Marketplace as both an MCP server and an A2A agent. Built on FHIR, trained on Synthea, ready to plug into any compliant workspace. Thanks for watching."*

## Recording checklist

- [ ] Test audio levels before the final take — a muddy voice kills momentum
- [ ] Use a clean Prompt Opinion workspace with only the Argus tools visible
- [ ] Pre-load the Carlos Martinez patient and the exact chat message (copy from above)
- [ ] Have the SHAP breakdown and the action items list ready to zoom into
- [ ] Final export: 1080p, 30fps, H.264, under 100 MB
- [ ] Upload to YouTube as **unlisted** until submission day, then flip to public
- [ ] Double-check: total duration ≤ 3:00 (judges are not required to watch beyond)

## Things *not* to say

- "HIPAA-compliant" (say "HIPAA-compatible architecture")
- "Clinically validated" (say "clinical-decision-support prototype")
- Never show any real patient name, MRN, or photo — use only Synthea-generated content

## Backup content for the second submission (A2A Agent)

The A2A agent submission reuses ~70% of this script. Swap the first 12 seconds and the last 15 seconds:

**[00:00–00:12] A2A opener**: *"MCP gives you tools. A2A gives you workflows. Argus Agent is what happens when you ask six specialized tools to work together as one clinician assistant."*

**[02:40–02:55] A2A close**: *"The Argus Agent composes an entire admission workflow from a single natural-language request — and it's invokable from any other A2A-compliant agent in the Prompt Opinion marketplace."*
