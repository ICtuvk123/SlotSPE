You are acting as a senior genitourinary surgical pathologist and a
computational pathology ontology curator.

Construct a version-2 candidate H&E event ontology for patient-specific
event grounding in whole-slide survival analysis.

Cancer type: {CANCER_TYPE}
Dataset: {DATASET_NAME}
Candidate count: {NUM_EVENTS}
Image unit used by the model: one 448 x 448 pixel H&E patch sampled at 20x.

The output is a candidate vocabulary, not a clinical claim. Generate 35-40
events before later curation to a 20-25 event patch-local core. Apply these
constraints:

1. Prefer a morphology that can be judged directly in one 448 x 448 @ 20x
   patch. A patch-local event must not depend on tumor location, gross
   orientation, slide margin, or another field of view.
2. Use observation_scale exactly as follows:
   - patch_local: cellular, cytologic, architectural, vascular, inflammatory,
     necrotic, or stromal morphology visible within one patch;
   - interface_context: requires a genuine tumor-host boundary or multiple
     adjacent fields;
   - wsi_anatomic_context: requires gross sampling, slide-level orientation,
     or knowledge of renal sinus, perinephric tissue, pseudocapsule, or a major
     vein.
3. Use patch_observable=yes only when a positive call is defensible from one
   patch. Use conditional when the patch may show the morphology but a true
   interface is required. Use no for every wsi_anatomic_context event.
4. Do not convert an anatomic diagnosis into a local visual label. For
   example, tumor plus adipocytes cannot by itself distinguish renal sinus
   fat from perinephric fat.
5. Limit invasion candidates. Put renal sinus invasion, perinephric adipose
   invasion, pseudocapsule penetration, and venous tumor thrombus in the WSI
   catalog rather than the patch-local core.
6. Balance architecture, cytology/differentiation, grade-related morphology,
   necrosis, vasculature, immune microenvironment, and stroma.
7. Avoid synonyms and causally adjacent labels that are visually
   indistinguishable. Give related but separable events the same
   semantic_group so that curation can compare them explicitly.
8. Favor established or cohort-evaluated ccRCC morphology. Use evidence_tier:
   - tier_1_consensus: ISUP/WHO-recognized grading or reportable parameter;
   - tier_2_cohort_supported: morphology evaluated in a peer-reviewed ccRCC
     cohort but not a grading/staging standard;
   - tier_3_exploratory: biologically plausible or lower-specificity candidate.
9. Every source_basis value must reference one source_id in the supplied
   source registry. Do not invent a citation or claim that a source supports a
   feature it did not assess.
10. required_context must state what must be present for a positive visual
    call. confounders must name sampling, processing, mimic, or background
    effects that could create false support.
11. Include exactly four morphology-only English prompt templates. Do not put
    outcome direction, survival language, molecular status, grade number, or
    anatomic location that is unavailable to the patch into encoder prompts.
12. Never output direct labels such as poor prognosis, good prognosis, high
    risk, low risk, long survival, or short survival. Do not invent numerical
    thresholds.
13. Do not output molecular-only concepts, treatment response labels, broad
    labels such as tumor tissue, or findings that require immunohistochemistry.
14. Treat all output as pending independent genitourinary-pathologist review.

Use this source registry exactly; it may be extended only with a real,
verifiable peer-reviewed source:

- isup_prognostic_2013: Delahunt et al., Am J Surg Pathol 2013,
  PMID 24025520. Supports RCC morphotype, nucleolar prominence, extreme
  pleomorphism, sarcomatoid/rhabdoid differentiation, tumor necrosis, and
  microvascular invasion as consensus prognostic parameters.
- isup_staging_2013: Trpkov et al., Am J Surg Pathol 2013,
  PMID 24025521. Supports renal sinus/perinephric fat/major-vein staging
  definitions and the need for directed interface sampling.
- caircc_ontology_2020: Cai et al., EBioMedicine 2020,
  PMID 31859241, PMCID PMC7000318. Supports a 33-parameter ccRCC ontology
  spanning architecture, cytology, and microenvironment in 549 tumors.
- infiltrative_growth_2013: Fukatsu et al., Am J Clin Pathol 2013,
  PMID 24045546. Supports infiltrative versus expansive ccRCC growth pattern.
- tumor_necrosis_2012: Pichler et al., Am J Clin Pathol 2012,
  PMID 22261455. Supports histologic tumor necrosis in ccRCC outcome analysis.
- neutrophils_2021: Tessier-Cloutier et al., J Pathol Clin Res 2021,
  PMID 33665979, PMCID PMC8185362. Supports morphologic scoring of
  tumor-infiltrating neutrophils in ccRCC.
- conch_2024: Lu et al., Nature Medicine 2024, PMID 38504017. Supports the
  pathology vision-language encoder and prompt-ensemble method, not the
  clinical validity of an event.

Return valid JSON only, with no Markdown or explanatory text. Match exactly
this structure (repeat the event object {NUM_EVENTS} times):

{
  "schema_version": "2.0",
  "ontology_version": "tcga_kirc_v2",
  "cancer_type": "{CANCER_TYPE}",
  "dataset": "{DATASET_NAME}",
  "status": "unreviewed_llm_candidates",
  "patch_spec": {
    "width_px": 448,
    "height_px": 448,
    "magnification": 20,
    "stain": "H&E"
  },
  "generation_metadata": {
    "prompt_version": "tcga_kirc_v2.0",
    "generator": "model name and provider",
    "generated_at": "ISO-8601 timestamp"
  },
  "sources": [
    {
      "source_id": "snake_case source id",
      "citation": "short citation",
      "url": "stable PMID, PMCID, or DOI URL",
      "evidence_scope": "what this source supports"
    }
  ],
  "events": [
    {
      "event_id": "lowercase_snake_case_id",
      "event_name": "concise English pathology name",
      "category": "one allowed category",
      "definition": "one-sentence morphologic definition",
      "visual_cues": ["concrete positive H&E cue", "second positive cue"],
      "exclusion_cues": ["specific mimic or negative cue", "second cue"],
      "prognostic_rationale": "short evidence-calibrated biological rationale without outcome direction",
      "evidence_tier": "tier_1_consensus | tier_2_cohort_supported | tier_3_exploratory",
      "observation_scale": "patch_local | interface_context | wsi_anatomic_context",
      "patch_observable": "yes | conditional | no",
      "required_context": ["information required for a defensible positive call"],
      "confounders": ["mimic, artifact, sampling effect, or background effect"],
      "source_basis": ["source_id"],
      "semantic_group": "lowercase_snake_case_group",
      "prompt_templates": ["prompt 1", "prompt 2", "prompt 3", "prompt 4"]
    }
  ]
}

Allowed categories are: tumor_architecture, proliferation, necrosis, invasion,
vasculature, immune_microenvironment, stroma, differentiation,
tissue_interface, and other.

Before returning JSON, internally verify the candidate count, all required
fields and enum values, unique IDs/names, source references, exactly four
prompts per event, patch-scale consistency, limited invasion representation,
and lack of obvious semantic duplicates.
