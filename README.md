# TRACE — Traceable, Reproducible Analysis of Collective Emergent States

**TRACE** is an open, modular pipeline that extracts nonverbal indices of team coordination from multiparty interaction recordings. It processes three modalities — speech turn-taking, facial action units, and shared visual attention — and fuses them into provenance-traced high-level coordination indices.

> Lannuzel T., Biancardi B., Barange M., Buisine S. — *TRACE: A Reproducible Multimodal Pipeline for Extracting Nonverbal Indices of Team Coordination* — CESI LINEACT, Nanterre, France

---

## Overview

The pipeline was developed in the context of a longitudinal study of **collective intelligence (CI)** in triads performing BIM (Building Information Modeling) tasks under two interaction modalities:

- **VR** — immersive virtual reality
- **PC** — 2D Revit interface

**Experimental design:**

| Dimension   | Values                             |
|-------------|------------------------------------|
| Timepoints  | T1, T2                             |
| Modalities  | VR, PC                             |
| Scenarios   | S1 (room layout), S2 (BIM objects) |
| Roles       | Calculator, Modeller, Reader       |
| Groups      | Triads identified as `bimXXX`      |
| Excluded    | bim002, bim032, bim065_2, bim075   |

---

## Multimodal Pipeline

The pipeline follows a strict sequential architecture: modality-specific extractors produce per-group CSV outputs, which are then fused into high-level indices (HLF). Each output value carries a `*_source` provenance column indicating which upstream extractor produced it.

```
Raw data (audio + video + eye-tracking)
         │
         ├─── Speech ──────────────────────────────► audio_features.csv
         │    analyze_audio.py                        ~30 features
         │    compute_audio_features.py               (turn-taking, overlaps, pauses)
         │
         ├─── Gaze ────────────────────────────────► gaze/ALL_metrics_overall.csv
         │    analyze_gaze.py                              ALL_metrics_pairs.csv
         │    analyze_gaze_directional.py            ~25 features
         │                                            (JVA, shared attention, mutual gaze)
         │
         ├─── Face ────────────────────────────────► face_emotion_metrics_all.csv
         │    analyze_aus_group.py                   ~15 features
         │                                            (AUs, facial synchrony, emotions)
         │
         └─── Fusion ──────────────────────────────► high_level_features.csv
              compute_high_level_features.py               high_level_features_audit.csv
                                                     ~50 high-level indices
                                                     + missingness summary
                                                     + *_source provenance columns
                          │
                          └─── Structure analysis ──► results_inv_structure/
                               analyze_inv_structure.py   with_pruning/   (|r| > 0.80 pruned)
                                                          without_pruning/ (all valid features)
                                                          PCA, loadings, redundancy diagnostics
```

### Stage 1 — Speech (turn-taking)

**Script:** `scripts/analyse_inv/speech/analyze_audio.py` + `compute_audio_features.py`

Processes per-speaker WAV files through a 3-pass IPU-to-turn aggregation:
1. Consecutive IPUs (VAD segments) from the same speaker are merged into CA turns
2. Turns shorter than `TURN_MIN_SEC` (1.0 s) are filtered
3. Post-filter re-merge of adjacent same-speaker turns

Key features produced:
- `audio_avg_speaking_turn_duration_s` — mean CA turn duration
- `audio_overlap_speaking_ratio` — ratio of overlapping speech (interruptions)
- `audio_floor_exchange_pause_mean_s` — mean inter-turn gap at floor exchanges
- `audio_speech_equality_*` — Gini/entropy of speaking time across roles
- `audio_turn_frequency_per_min` — floor exchange rate

### Stage 2 — Gaze (shared visual attention)

**Script:** `scripts/analyse_inv/gaze/analyze_gaze.py` + `analyze_gaze_directional.py`

Processes eye-tracking data (VR only for directional gaze; PC has no gaze capture). Computes group-level and dyadic metrics over a sliding window framework.

Key features produced:
- `gaze_mutual_gaze_ratio` — proportion of time ≥2 members share mutual gaze
- `gaze_shared_visual_attention_ratio` — proportion of time ≥2 members fixate the same AOI (200–500 ms window)
- `shared_obj_ratio`, `shared_obj_n_episodes`, `shared_obj_dur_mean_s` — shared object fixation metrics
- `gaze_joint_attention_idx_raw` — composite JVA score (duration × simultaneity)
- `gaze_entropy` — Shannon entropy of fixation distribution across AOIs
- Dyadic gaze-toward-partner ratios per role pair

### Stage 3 — Face (action units & synchrony)

**Script:** `scripts/analyse_inv/face/analyze_aus_group.py`

Processes OpenFace FACS outputs (Action Units intensity + presence). Runs OpenFace via `openface/run_openface_and_export_facs.py`.

Key features produced:
- `face_joy_ratio`, `face_surprise_ratio`, `face_sad_ratio` — proportion of time emotion-AUs are active at group level
- `face_sync_*` — facial synchrony scores across dyads (correlation of AU time series)
- AU presence/intensity aggregates per role

### Stage 4 — High-Level Feature Fusion

**Script:** `scripts/analyse_inv/hlf/compute_high_level_features.py`

Takes the three modality-level CSVs as input. For each high-level index, a `first_valid_series` resolution strategy selects the best available upstream source (handles missing gaze in PC condition). Produces three outputs:

| Output | Description |
|--------|-------------|
| `high_level_features.csv` | Compact dataset — one row per group × condition × scenario × timepoint |
| `high_level_features_audit.csv` | Full audit dataset including all intermediate values and `*_source` columns |
| `hlf_availability_summary.csv` | Per-feature missingness and coverage statistics |

Key high-level indices:
- `hlf_speech_equality` — speaking-time equality (Gini-based, group level)
- `hlf_turn_regulation_idx` — composite turn-taking regularity index
- `hlf_shared_attention_idx` — fused shared visual attention (gaze + proxy)
- `hlf_facial_sync_idx` — group facial synchrony composite
- `hlf_task_focus_ratio` — proportion of attention directed to task objects

### Stage 5 — Structure Analysis (PCA + Redundancy)

**Script:** `scripts/analyse_inv/analyze_inv_structure.py`

Runs hierarchical clustering, correlation matrix, and PCA on the full HLF feature space. Produces two parallel outputs:

- `results_inv_structure/with_pruning/` — PCA after removing redundant features (|r| > 0.80), keeping the higher-priority feature per correlated pair
- `results_inv_structure/without_pruning/` — PCA on all valid features (for comparison)

Priority rules for pruning: (1) fewer missing values, (2) business priority rank in `inv_features_config.py`, (3) shorter canonical name.

---

## Repository Structure

```
Longitudinale/
├── scripts/
│   ├── run_inv.py                        # Main entry point: full INV pipeline
│   ├── run_performance.py                # Main entry point: task performance scoring
│   ├── build_schema.py                   # Pipeline diagram generator
│   │
│   ├── common/                           # Shared utilities (imported by all modules)
│   │   ├── constants.py                  # ROLES, EXCLUDED_GROUPS, PERF_WEIGHTS
│   │   ├── metadata.py                   # Condition/scenario/timepoint extraction
│   │   ├── io_utils.py                   # EU-format CSV reader (sep=";", decimal=",")
│   │   ├── stats.py                      # safe_div, zscore, shannon_entropy, coef_var
│   │   └── temporal.py                   # Sliding windows, episode detection, overlaps
│   │
│   ├── config/
│   │   └── inv_features_config.py        # Single source of truth for all INV features
│   │                                     # (families, priorities, redundancy, core flags)
│   │
│   ├── analyse_inv/
│   │   ├── speech/
│   │   │   ├── analyze_audio.py          # IPU → CA turn aggregation, floor exchanges
│   │   │   └── compute_audio_features.py # Feature derivation from turn-level metrics
│   │   ├── gaze/
│   │   │   ├── analyze_gaze.py           # Group-level gaze: JVA, shared attention
│   │   │   ├── analyze_gaze_directional.py  # Directional gaze toward partners/objects
│   │   │   ├── diagnose_eyetracking.py   # Completeness diagnostic
│   │   │   ├── reconstruct_eyetracking.py   # Missing gaze reconstruction
│   │   │   └── refine_yaw_eyetracking.py    # Yaw correction (MOD-16)
│   │   ├── face/
│   │   │   ├── analyze_aus_group.py      # AU/emotion aggregation at group level
│   │   │   └── openface/
│   │   │       ├── preprocess_pc_videos.py
│   │   │       └── run_openface_and_export_facs.py
│   │   ├── hlf/
│   │   │   ├── compute_high_level_features.py      # Multimodal fusion + provenance
│   │   │   └── compute_high_level_features_old_gaze.py  # Variant (legacy gaze pipeline)
│   │   └── analyze_inv_structure.py      # PCA, clustering, redundancy diagnostics
│   │
│   ├── analyse_performance/              # Statistical analyses of task performance
│   │   ├── analyze_performance_effects.py   # 2×2 ANOVA, t-tests, ANCOVA
│   │   ├── correlation_metrics_final.py
│   │   ├── cronbach_alpha_performance.py
│   │   └── multiple_regression_perf.py
│   │
│   ├── evaluation_performance_scenario/  # Task scoring (M1–M4)
│   │   ├── score_groupes.py              # Aggregates Score_perf_tsk across groups
│   │   ├── S1/                           # Scenario 1 evaluators (VR + PC)
│   │   │   ├── performance_eval_unified.py
│   │   │   ├── PC/eval_consignes_align.py
│   │   │   └── VR/eval_consignes_align.py
│   │   ├── S2/                           # Scenario 2 evaluators (VR + PC)
│   │   │   ├── PC/{eval_consignes_s2,performance_eval}.py
│   │   │   └── VR/{eval_consignes_s2,performance_eval}.py
│   │   ├── corrections/                  # Reference solutions (PC_s1, VR_S1, PC_s2, VR_S2)
│   │   └── script_revit/                 # Dynamo scripts for Revit export
│   │
│   ├── analyse_TCI/
│   │   ├── TCI.py                        # C-factor computation (Woolley et al.)
│   │   └── compute_team_indicators_rields.py  # Riedl indicators (skill, strategy, effort)
│   │
│   ├── analyse_questionnaire/            # Questionnaire reliability & descriptive analysis
│   │   ├── main.py
│   │   └── py/{config,io_read,transform,reliability,descriptives,
│   │            role_tests,plots,item_pruning,g3_context,scenario_modalite}.py
│   │
│   ├── rapport/v2/                       # PDF report generation (canonical publication pipeline)
│   │   ├── main.py                       # Orchestrator (modes: vr_only, pc_vr, inv_vr, pca_vr)
│   │   └── py/                           # 30+ report modules
│   │
│   ├── sem/
│   │   └── pls_sem_vr.py                 # PLS-SEM path analysis (VR)
│   │
│   ├── visualisation_sociale/
│   │   ├── mirage_sociogram.py           # Offline multimodal sociogram (MIRAGE-inspired)
│   │   └── run_all_vr_mirage_sociograms.py
│   │
│   └── utilitaires/                      # Misc tools (Revit adapter, CSV merge, etc.)
│
├── results/                             # Generated outputs — not versioned (see .gitignore)
│
├── ci_multimodal_features_dimensions.csv  # Feature inventory with dimensions and sources
├── INV_METHODES_CALCUL.md               # Detailed computation methods for all INV variables
├── MIRAGE_SOCIOGRAM_NOTES.md            # Technical notes on the multimodal sociogram
├── PROJECT_CONTEXT.md                   # Project decisions, conventions, changelog
└── PROJECT_DOCUMENTATION.md            # Experimental design, populations, data acquisition
```

---

## Quick Start

All commands run from `scripts/`:

```bash
cd Longitudinale/scripts
```

### Full INV pipeline

```bash
# All modalities
python run_inv.py --data-dir ../data_e2 --out-dir ../results/INV

# Speech only
python run_inv.py --data-dir ../data_e2 --inv speech --out-dir ../results/INV

# Recompute HLF from existing modality CSVs
python run_inv.py --hlf-only \
    --speech-csv ../results/INV/audio_features.csv \
    --gaze-group-csv ../results/INV/gaze/ALL_metrics_overall.csv \
    --gaze-pair-csv ../results/INV/gaze/ALL_metrics_pairs.csv \
    --face-csv ../results/INV/face_emotion_metrics_all.csv \
    --out-dir ../results/INV
```

### Task performance scoring

```bash
python run_performance.py --results-dir ../results
```

**Scoring weights:** M1 (rules followed) × 0.5 + M3 (spatial precision) × 0.5 → `Score_perf_tsk`

### PCA structure analysis

```bash
cd analyse_inv
python analyze_inv_structure.py
# Outputs: results/results_inv_structure/with_pruning/ and without_pruning/
```

### PDF report (publication pipeline)

```bash
python rapport/v2/main.py \
    --results-dir ../results \
    --out-dir ../results/rapport_v2
```

| Argument | Description |
|----------|-------------|
| `--mode` | `vr_only` · `pc_vr` · `inv_vr` · `pca_vr` · bundle (default = all) |
| `--inv-analysis-mode` | `pruning` (default) or `no-pruning` |
| `--fdr` | Apply FDR correction per table |
| `--bayes` | Bayesian ICC via MCMC (slow) |

### Multimodal sociogram

```bash
python visualisation_sociale/mirage_sociogram.py \
    --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
    --data-dir ../data_e2 \
    --out-dir ../results/visualisation_sociale/bim073 \
    --snapshot-at 120
```

---

## Configuration

**`scripts/config/inv_features_config.py`** is the single source of truth for all INV features. It defines:

- Feature families (`audio`, `gaze`, `face`, `high_level`)
- Priority ranks for redundancy pruning
- `core` / `core_hl` flags — features displayed in reports
- `REGRESSION_RETAINED_INV_FEATURES` + `REGRESSION_FORCE_INCLUDE` — stepwise regression candidates

Feature selection is intentionally dissociated by analysis context:

| Context | Selection mechanism |
|---------|---------------------|
| Report | `core` / `core_hl` flags + `report_preferred` |
| PCA | `FEATURE_PRIORITY` ranks |
| Regression (v2) | `REGRESSION_RETAINED_INV_FEATURES` ∩ `inv_pruned_features.csv` (kept=1) |

---

## Data Format

- CSV separator: `;` (European format)
- Decimal separator: `,`
- Encoding: UTF-8
- `common/io_utils.py::read_csv_eu()` and `read_csv_smart()` handle these automatically.

---

## Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib librosa openpyxl statsmodels reportlab
```

Python ≥ 3.10 required. OpenFace 2.x must be installed separately for facial AU extraction.

---

## Known Data Issues

| Group | Issue |
|-------|-------|
| bim002 | No performance file, corrupted data |
| bim032 | No performance file, missing audio (Mod/Calc), no Lecteur marker |
| bim065_2 | No performance file |
| bim075 | No audio track, no markers, no performance file |

These groups are excluded automatically via `common.constants.EXCLUDED_GROUPS`.
