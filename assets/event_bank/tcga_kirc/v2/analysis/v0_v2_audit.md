# TCGA-KIRC v0/v2 event-bank audit

Generated: 2026-07-18T07:15:37.453280+00:00

- v0 events: 30
- v2 events: 22
- v2 embedding pairs at or above 0.85: 0
- embedding redundancy check: PASS
- offset-invariant activation-overlap check: PASS (maximum Jaccard 0.3752)
- cosine 0.20 saturation warning: YES

## Highest within-v2 embedding similarities

| Event A | Event B | Cosine |
|---|---|---:|
| brisk_mitotic_activity | atypical_mitotic_figures | 0.7938 |
| extreme_nuclear_pleomorphism | syncytial_tumor_giant_cells | 0.7816 |
| abundant_eosinophilic_cytoplasm | cytoplasmic_hyaline_globules | 0.7659 |
| thick_trabecular_insular_architecture | delicate_capillary_network | 0.7605 |
| prominent_nucleoli | cytoplasmic_hyaline_globules | 0.7561 |
| scant_cytoplasm_phenotype | brisk_mitotic_activity | 0.7439 |
| scant_cytoplasm_phenotype | abundant_eosinophilic_cytoplasm | 0.7236 |
| thick_trabecular_insular_architecture | solid_sheet_architecture | 0.7164 |
| solid_sheet_architecture | scant_cytoplasm_phenotype | 0.7125 |
| thick_trabecular_insular_architecture | papillary_pseudopapillary_architecture | 0.7085 |

## Closest v0 event for each v2 event

| v2 event | nearest v0 event | Cosine |
|---|---|---:|
| rhabdoid_differentiation | rhabdoid_differentiation | 0.9077 |
| coagulative_tumor_necrosis | coagulative_tumor_necrosis | 0.9000 |
| microvascular_tumor_invasion | microvascular_tumor_invasion | 0.8918 |
| delicate_capillary_network | delicate_capillary_network | 0.8825 |
| intratumoral_lymphoplasmacytic_infiltrate | intratumoral_lymphocyte_infiltration | 0.8701 |
| sarcomatoid_differentiation | sarcomatoid_spindle_cell_transformation | 0.8646 |
| prominent_nucleoli | prominent_nucleolar_morphology | 0.8371 |
| brisk_mitotic_activity | atypical_mitotic_figures | 0.8332 |
| intratumoral_neutrophil_infiltrate | neutrophil_rich_inflammation | 0.8206 |
| atypical_mitotic_figures | atypical_mitotic_figures | 0.8181 |
| scar_like_stromal_regression | hyalinized_fibrotic_stroma | 0.8017 |
| tubular_acinar_architecture | acinar_tubular_differentiation | 0.7776 |
| microcystic_architecture | cystic_degenerative_change | 0.7571 |
| papillary_pseudopapillary_architecture | papillary_growth_pattern | 0.7526 |
| extreme_nuclear_pleomorphism | marked_tumor_cell_pleomorphism | 0.7497 |
| abundant_eosinophilic_cytoplasm | compact_eosinophilic_tumor_component | 0.7495 |
| solid_sheet_architecture | clear_cell_nested_alveolar_architecture | 0.7356 |
| cytoplasmic_hyaline_globules | prominent_nucleolar_morphology | 0.7322 |
| scant_cytoplasm_phenotype | delicate_capillary_network | 0.7306 |
| syncytial_tumor_giant_cells | marked_tumor_cell_pleomorphism | 0.7291 |
| compact_small_nest_architecture | pseudocapsule_penetration | 0.6773 |
| thick_trabecular_insular_architecture | infiltrative_small_nests_and_cords | 0.6545 |

## Activation-threshold sensitivity

| Cosine threshold | Median patch frequency | Patch-frequency range | Median patient coverage | Max within-v2 Jaccard |
|---:|---:|---:|---:|---:|
| 0.200 | 63.1885% | 11.0083%-87.5232% | 99.8131% | 0.8913 |
| 0.300 | 29.1621% | 2.9857%-72.5014% | 97.5701% | 0.8446 |
| 0.400 | 6.8222% | 0.1447%-45.8863% | 67.2897% | 0.7055 |
| 0.500 | 0.3520% | 0.0037%-16.2706% | 8.2243% | 0.3668 |

## Primary patch activation audit

Scored 2,653,350 patches from 542 slides and 535 patients at cosine >= 0.500.

| v2 event | Patch frequency | Slide coverage | Patient coverage |
|---|---:|---:|---:|
| abundant_eosinophilic_cytoplasm | 16.2706% | 87.6384% | 87.6635% |
| scant_cytoplasm_phenotype | 14.7942% | 81.1808% | 81.1215% |
| delicate_capillary_network | 11.2673% | 78.9668% | 79.2523% |
| cytoplasmic_hyaline_globules | 10.9180% | 80.8118% | 80.5607% |
| prominent_nucleoli | 8.9612% | 81.3653% | 81.3084% |
| microcystic_architecture | 4.2767% | 64.0221% | 64.2991% |
| microvascular_tumor_invasion | 4.1964% | 77.1218% | 76.8224% |
| solid_sheet_architecture | 2.8221% | 43.9114% | 44.4860% |
| intratumoral_neutrophil_infiltrate | 1.9922% | 36.5314% | 36.8224% |
| coagulative_tumor_necrosis | 1.4209% | 27.1218% | 27.2897% |
| scar_like_stromal_regression | 0.7676% | 20.4797% | 20.5607% |
| papillary_pseudopapillary_architecture | 0.3520% | 8.3026% | 8.2243% |
| tubular_acinar_architecture | 0.3342% | 6.2731% | 5.7944% |
| intratumoral_lymphoplasmacytic_infiltrate | 0.3042% | 5.9041% | 5.9813% |
| rhabdoid_differentiation | 0.1028% | 2.9520% | 2.9907% |
| brisk_mitotic_activity | 0.0929% | 2.0295% | 2.0561% |
| sarcomatoid_differentiation | 0.0906% | 1.2915% | 1.3084% |
| extreme_nuclear_pleomorphism | 0.0613% | 2.0295% | 2.0561% |
| thick_trabecular_insular_architecture | 0.0318% | 0.5535% | 0.5607% |
| syncytial_tumor_giant_cells | 0.0227% | 0.3690% | 0.3738% |
| atypical_mitotic_figures | 0.0072% | 0.1845% | 0.1869% |
| compact_small_nest_architecture | 0.0037% | 0.0000% | 0.0000% |

## Highest within-v2 activation overlaps

| Event A | Event B | Activation Jaccard |
|---|---|---:|
| scant_cytoplasm_phenotype | delicate_capillary_network | 0.3668 |
| abundant_eosinophilic_cytoplasm | cytoplasmic_hyaline_globules | 0.3486 |
| prominent_nucleoli | cytoplasmic_hyaline_globules | 0.3034 |
| prominent_nucleoli | abundant_eosinophilic_cytoplasm | 0.2808 |
| scant_cytoplasm_phenotype | abundant_eosinophilic_cytoplasm | 0.2112 |
| microcystic_architecture | scant_cytoplasm_phenotype | 0.1733 |
| microcystic_architecture | delicate_capillary_network | 0.1731 |
| cytoplasmic_hyaline_globules | delicate_capillary_network | 0.1717 |
| coagulative_tumor_necrosis | intratumoral_neutrophil_infiltrate | 0.1688 |
| cytoplasmic_hyaline_globules | microvascular_tumor_invasion | 0.1555 |

## Per-event quantile overlap audit

Per-event thresholds were calibrated at the 95.0% score quantile from 138,752 slide-balanced sampled patches. This removes event-specific cosine offsets when checking whether different prompts select the same patches.

Maximum within-v2 activation Jaccard: 0.3752.

| Event A | Event B | Activation Jaccard |
|---|---|---:|
| thick_trabecular_insular_architecture | delicate_capillary_network | 0.3752 |
| extreme_nuclear_pleomorphism | syncytial_tumor_giant_cells | 0.3396 |
| brisk_mitotic_activity | atypical_mitotic_figures | 0.3380 |
| thick_trabecular_insular_architecture | scant_cytoplasm_phenotype | 0.3116 |
| abundant_eosinophilic_cytoplasm | cytoplasmic_hyaline_globules | 0.2883 |
| solid_sheet_architecture | atypical_mitotic_figures | 0.2826 |
| extreme_nuclear_pleomorphism | rhabdoid_differentiation | 0.2780 |
| thick_trabecular_insular_architecture | atypical_mitotic_figures | 0.2646 |
| coagulative_tumor_necrosis | intratumoral_neutrophil_infiltrate | 0.2609 |
| rhabdoid_differentiation | syncytial_tumor_giant_cells | 0.2588 |
