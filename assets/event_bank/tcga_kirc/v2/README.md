# TCGA-KIRC event bank v2

V2 is a scale-aware, evidence-tracked candidate ontology for CONCH grounding of
single `448 x 448 @ 20x` H&E patches. It does not replace the legacy v0 files in
the parent directory and it has not yet been reviewed by a genitourinary
pathologist.

## Contents

- `events_candidates.json`: 38 evidence-constrained candidates with provenance,
  observation scale, patch observability, required context, confounders, and
  source basis.
- `curation_decisions.json`: one auditable decision and reason per candidate.
- `events_patch_core.json`: 22 directly patch-observable events used to build the
  v2 CONCH bank.
- `events_context.json`: 8 interface or WSI/anatomic events that must not be mixed
  into the local patch bank.
- `events_reserve.json`: 8 locally visible but redundant or lower-specificity
  reserve candidates.
- `conch_event_bank_v2.pt`: generated frozen text embeddings; ignored by Git.
- `conch_event_bank_v2_summary.json`: generated non-tensor metadata summary.
- `analysis/v0_v2_audit.{json,md}`: generated embedding and optional patch
  activation comparison.

The curation status `machine_curated_pending_pathologist_review` means exactly
that: deterministic checks and computational curation have run, but independent
pathology review has not.

## Build

```bash
CONCH_DEVICE=cuda bash scripts/build_tcga_kirc_event_bank_v2.sh
```

The script validates the 38 candidates, applies the fixed curation decisions,
builds only the 22-event patch-local bank, and compares v0/v2 text embeddings.
It never overwrites the v0 files.

To include patch activation frequency, slide coverage, patient coverage, and
activation-overlap analysis:

```bash
CONCH_DEVICE=cuda \
V2_PATCH_FEATURE_DIR=/data0/lfy_data/Pathology/CONCH/kirc/pt_files \
bash scripts/build_tcga_kirc_event_bank_v2.sh
```

The audit sweeps cosine thresholds `0.20`, `0.30`, `0.40`, and `0.50` because
`0.20` is the current event-gating calibration center but can yield saturated
hard activations. The sparse `0.50` result is used for the detailed primary
table, while every threshold is retained for sensitivity analysis. A slide or
patient is counted as covered when at least 1% of its patches activate the
event. A second, offset-invariant audit calibrates each event at its own 95th
score percentile on a deterministic slide-balanced sample, then compares the
selected patch sets by Jaccard overlap. These definitions and thresholds are
written into the audit JSON.

To train with v2 without changing the v0 default:

```bash
EVENT_BANK_PATH=assets/event_bank/tcga_kirc/v2/conch_event_bank_v2.pt \
DELTA_PATCH_EVENT=0.40 \
bash scripts/run_kirc_event_gated.sh
```

The audit shows that `0.20` is saturated as a hard activation threshold for
this bank. `0.40` is an audit-informed starting point, not a fixed optimum;
include `0.30`, `0.40`, and `0.50` in the training validation sweep and select
without consulting the test fold.

## Regeneration

The strict prompt is
`assets/event_bank/prompts/generate_pathology_events_v2.md`. If candidates are
regenerated through an OpenAI-compatible API, set `LLM_API_KEY`, `LLM_MODEL`, and
optionally `LLM_BASE_URL`, then move the existing candidate file aside and run
the build script. Do not silently reuse the current curation decisions for a
new candidate set: every candidate ID must receive a fresh explicit decision.

## Required expert checkpoint

Before using v2 as a final paper ontology, a genitourinary pathologist should
review definitions, visual/exclusion cues, evidence tiers, and the core/context
split. After that review, record reviewer/date and change status to
`pathologist_reviewed`; until then, all outputs remain experimental candidates.
