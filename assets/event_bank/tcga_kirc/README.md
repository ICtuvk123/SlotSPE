# TCGA-KIRC candidate event bank

`events_raw.json` contains 30 LLM-generated morphology candidates for kidney
renal clear cell carcinoma. They are intentionally marked
`unreviewed_llm_candidates`.

`events_reviewed.json` has passed only the repository's deterministic schema,
forbidden-language, duplicate-ID, and simple token-overlap checks. Its status is
`machine_checked_pending_expert_review`; it is not a claim of pathology-expert
validation.

`conch_event_bank.pt` and its summary are generated locally after gated CONCH
weights are available. Binary embeddings and model checkpoints are ignored by
Git and must never be replaced with random weights.

These parent-directory files are the reproducible legacy **v0** baseline. The
scale-aware v2 workflow lives in [`v2/`](v2/README.md), uses a separate
22-event patch-local core, and does not overwrite v0.
