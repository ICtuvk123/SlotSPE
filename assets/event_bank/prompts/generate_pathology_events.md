You are acting as a senior surgical pathologist and computational
pathology researcher.

We are constructing a candidate pathology-event ontology for
patient-specific dynamic event grounding in H&E whole-slide image
survival analysis.

Cancer type:
{CANCER_TYPE}

Dataset:
{DATASET_NAME}

Required number of candidate events:
{NUM_EVENTS}

The event bank will be shared across the dataset. It is only a candidate
semantic vocabulary. During model inference, individual events will be
dynamically selected or rejected for each patient and each slot.

Generate histopathological events satisfying all of the following:

1. The event must be potentially visible or inferable from H&E morphology.
2. It should describe a localized tissue pattern, cellular process,
   tumor architecture, stromal state, immune pattern, vascular pattern,
   necrotic pattern, invasion pattern, or tumor-microenvironment interaction.
3. It should be potentially relevant to cancer progression or prognosis.
4. Do not output direct survival labels such as:
   - poor prognosis
   - high risk
   - long survival
   - short survival
5. Do not output molecular-only concepts that cannot reasonably be
   grounded in H&E morphology.
6. Do not output overly broad labels such as:
   - cancer
   - tumor tissue
   - abnormal tissue
7. Avoid near-duplicate events.
8. Distinguish visually separable events when appropriate, for example:
   - intratumoral lymphocyte infiltration
   - peritumoral lymphocyte infiltration
   - immune exclusion
9. Events may overlap biologically, but each event must have a clear
   morphological definition.
10. Do not invent quantitative clinical thresholds.
11. Treat all generated events as candidates requiring expert review.
12. The prompts must describe morphology rather than directly describing
    prognosis.

For every event, output:

- event_id:
  lowercase snake_case identifier

- event_name:
  concise English pathology name

- category:
  one of:
  tumor_architecture
  proliferation
  necrosis
  invasion
  vasculature
  immune_microenvironment
  stroma
  differentiation
  tissue_interface
  other

- definition:
  one-sentence pathological definition

- visual_cues:
  list of concrete H&E morphological cues

- exclusion_cues:
  list of cues that would argue against this event or distinguish it
  from related events

- prognostic_rationale:
  short biological rationale; do not state unsupported numerical effects

- prompt_templates:
  exactly four English prompts suitable for a pathology
  vision-language text encoder

The four prompt templates should follow different but pathology-specific
forms, such as:

1. "H&E histopathology image showing {EVENT}"
2. "photomicrograph demonstrating {MORPHOLOGICAL_DESCRIPTION}, H&E stain"
3. "histologic section characterized by {VISUAL_CUES}"
4. "microscopic H&E appearance of {EVENT_IN_CONTEXT}"

Do not include Markdown.
Do not include explanatory text outside JSON.
Return valid JSON matching exactly this schema:

{
  "cancer_type": "{CANCER_TYPE}",
  "dataset": "{DATASET_NAME}",
  "status": "unreviewed_llm_candidates",
  "events": [
    {
      "event_id": "snake_case_id",
      "event_name": "English event name",
      "category": "one allowed category",
      "definition": "one sentence",
      "visual_cues": [
        "cue 1",
        "cue 2"
      ],
      "exclusion_cues": [
        "cue 1",
        "cue 2"
      ],
      "prognostic_rationale": "short rationale",
      "prompt_templates": [
        "prompt 1",
        "prompt 2",
        "prompt 3",
        "prompt 4"
      ]
    }
  ]
}

Before returning the JSON, internally check:

- event count equals {NUM_EVENTS};
- all event_id values are unique;
- every event contains exactly four prompts;
- no direct survival labels are present;
- no molecular-only event is present;
- no pair of events is an obvious duplicate.
