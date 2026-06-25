# TRACE - Traceable, Reproducible Analysis of Collective Emergent States

**TRACE** is an open, modular pipeline that extracts nonverbal indices of team coordination from multiparty interaction recordings. It processes three modalities — speech turn-taking, facial action units, and shared visual attention — and fuses them into provenance-traced high-level coordination indices.

> Lannuzel T., Biancardi B., Barange M., Buisine S. — *TRACE: A Reproducible Multimodal Pipeline for Extracting Nonverbal Indices of Team Coordination* — CESI LINEACT, Nanterre, France

---

## Study Context — Collective Intelligence in BIM Collaboration

The pipeline was developed for a longitudinal study of **collective intelligence (CI)** in triads performing collaborative tasks under two interaction modalities:

- **VR** — immersive virtual reality *(BIM task environment — dataset link: coming soon)*
- **PC** — 2D Revit interface *(BIM PC scenarios — dataset link: coming soon)*

The study is grounded in the **Woolley et al. (2010)** collective intelligence framework, extended with Riedl's team indicators (skill, strategy, effort). Nonverbal behaviour is treated as a proxy for emergent coordination states (TMS, TAS, TRS) that mediate group performance.

**Experimental design:**

| Dimension   | Values                               |
|-------------|--------------------------------------|
| Timepoints  | T1, T2 (longitudinal)                |
| Modalities  | VR (immersive), PC (Revit 2D)        |
| Scenarios   | S1 (spatial layout), S2 (BIM objects)|
| Roles       | Three role-based positions per triad |
| Groups      | Triads with a shared group identifier |

### Collective Intelligence Measures

Beyond nonverbal indices, the pipeline includes two modules that operationalise collective intelligence directly:

**`analyse_TCI/TCI.py`** — computes the **c-factor** (Woolley et al., 2010), a general collective intelligence factor extracted from a battery of cognitive tasks administered to each triad. The c-factor predicts group performance independently of individual ability.

**`analyse_TCI/compute_team_indicators_rields.py`** — computes team-level indicators inspired by Riedl et al.:
- **Skill** — mean individual performance on standardised tasks
- **Strategy** — consistency of approach across task types
- **Effort** — activity volume proxy (event count or keystroke length)
- **Congruence** — alignment between individual contributions

These indicators are used alongside INV features in regression and path analyses to model the full input–mediator–output structure of collective intelligence.

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
- `audio_overlap_speaking_ratio` — ratio of overlapping speech
- `audio_floor_exchange_pause_mean_s` — mean inter-turn gap at floor exchanges
- `audio_backchannel_rate_per_min` — backchannel rate (strict 4-filter detection)
- `participation_entropy` — Shannon entropy of speaking-time distribution across roles

### Stage 2 — Gaze (shared visual attention)

**Script:** `scripts/analyse_inv/gaze/analyze_gaze.py` + `analyze_gaze_directional.py`

Processes eye-tracking data (VR only for directional gaze; PC has no gaze capture). Computes group-level and dyadic metrics over a sliding window framework.

Key features produced:
- `gaze_convergence_ratio` — proportion of time with directional gaze convergence (canonical)
- `gaze_mutual_gaze_ratio` — proportion of time with mutual gaze between members
- `gaze_entropy_dir_mean` — directional gaze entropy (canonical; low = focused attention)
- `gaze_joint_attention_idx_raw` — composite JVA score (convergence + mutual gaze)
- `gaze_attention_coordination_idx` — composite coordination index

### Stage 3 — Face (action units & synchrony)

**Script:** `scripts/analyse_inv/face/analyze_aus_group.py`

Processes OpenFace FACS outputs (Action Units intensity + presence).

Key features produced:
- `face_smile_ratio` — Duchenne smile ratio (AU6+AU12 co-active)
- `face_negative_affect_ratio` — sadness marker ratio (AU15+AU17 co-active)
- `face_facial_synchrony` — inter-member facial synchrony (Pearson, per-AU)
- `affect_alignment_idx` — composite positive/negative affect alignment

### Stage 4 — High-Level Feature Fusion

**Script:** `scripts/analyse_inv/hlf/compute_high_level_features.py`

Takes the three modality-level CSVs as input. For each high-level index, a `first_valid_series` resolution strategy selects the best available upstream source (handles missing gaze in PC condition). Produces three outputs:

| Output | Description |
|--------|-------------|
| `high_level_features.csv` | Compact dataset — one row per group × condition × scenario × timepoint |
| `high_level_features_audit.csv` | Full audit dataset with all intermediate values and `*_source` columns |
| `hlf_availability_summary.csv` | Per-feature missingness and coverage statistics |

### Stage 5 — Structure Analysis (PCA + Redundancy)

**Script:** `scripts/analyse_inv/analyze_inv_structure.py`

Runs hierarchical clustering, correlation matrix, and PCA on the full HLF feature space. Produces two parallel outputs:

- `results_inv_structure/with_pruning/` — PCA after removing redundant features (|r| > 0.80)
- `results_inv_structure/without_pruning/` — PCA on all valid features (for comparison)

---

## Repository Structure

```
<project_root>/
├── scripts/
│   ├── run_inv.py                        # Entry point: full INV pipeline
│   ├── run_performance.py                # Entry point: task performance scoring
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
│   │
│   ├── analyse_inv/
│   │   ├── speech/
│   │   │   ├── analyze_audio.py          # IPU → CA turn aggregation, floor exchanges
│   │   │   └── compute_audio_features.py # Feature derivation from turn-level metrics
│   │   ├── gaze/
│   │   │   ├── analyze_gaze.py           # Group-level gaze: JVA, shared attention
│   │   │   ├── analyze_gaze_directional.py  # Directional gaze toward partners/objects
│   │   │   ├── diagnose_eyetracking.py   # Completeness diagnostic
│   │   │   ├── reconstruct_eyetracking.py
│   │   │   └── refine_yaw_eyetracking.py
│   │   ├── face/
│   │   │   ├── analyze_aus_group.py      # AU/emotion aggregation at group level
│   │   │   └── openface/
│   │   │       ├── preprocess_pc_videos.py
│   │   │       └── run_openface_and_export_facs.py
│   │   ├── hlf/
│   │   │   ├── compute_high_level_features.py      # Multimodal fusion + provenance
│   │   │   └── compute_high_level_features_old_gaze.py
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
│   │   ├── corrections/                  # Reference solutions per scenario/modality
│   │   └── script_revit/                 # Dynamo scripts for Revit data export
│   │
│   ├── analyse_TCI/                      # Collective intelligence measures
│   │   ├── TCI.py                        # C-factor computation (Woolley et al., 2010)
│   │   └── compute_team_indicators_rields.py  # Riedl indicators (skill, strategy, effort)
│   │
│   ├── analyse_questionnaire/            # Questionnaire reliability & descriptive analysis
│   │   ├── main.py
│   │   └── py/                           # config, io_read, transform, reliability,
│   │                                     # descriptives, role_tests, plots, item_pruning,
│   │                                     # g3_context, scenario_modalite
│   │
│   ├── rapport/v2/                       # PDF report generation (publication pipeline)
│   │   ├── main.py                       # Orchestrator (vr_only, pc_vr, inv_vr, pca_vr)
│   │   └── py/                           # 30+ report modules
│   │
│   ├── sem/
│   │   └── pls_sem_vr.py                 # PLS-SEM path analysis
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

Set your paths as environment variables or pass them directly to each script. No hardcoded paths — all scripts accept `--data-dir`, `--out-dir`, and similar arguments.

```bash
DATA_DIR=/path/to/raw/data
RESULTS_DIR=/path/to/results
SCRIPTS_DIR=/path/to/scripts
```

### Full INV pipeline

```bash
cd $SCRIPTS_DIR

# All modalities
python run_inv.py --data-dir $DATA_DIR --out-dir $RESULTS_DIR/INV

# Speech only
python run_inv.py --data-dir $DATA_DIR --inv speech --out-dir $RESULTS_DIR/INV

# Recompute HLF from existing modality CSVs
python run_inv.py --hlf-only \
    --speech-csv $RESULTS_DIR/INV/audio_features.csv \
    --gaze-group-csv $RESULTS_DIR/INV/gaze/ALL_metrics_overall.csv \
    --gaze-pair-csv $RESULTS_DIR/INV/gaze/ALL_metrics_pairs.csv \
    --face-csv $RESULTS_DIR/INV/face_emotion_metrics_all.csv \
    --out-dir $RESULTS_DIR/INV
```

### Task performance scoring

```bash
python run_performance.py --results-dir $RESULTS_DIR
```

**Scoring weights:** M1 (rules followed) × 0.5 + M3 (spatial precision) × 0.5 → `Score_perf_tsk`

### C-factor & team indicators

```bash
# C-factor (Woolley et al., 2010)
python analyse_TCI/TCI.py $DATA_DIR/data_TCI \
    --out $RESULTS_DIR/TCI/c_scores.csv \
    --missing mean --profile --scatter --heatmap

# Riedl team indicators
python analyse_TCI/compute_team_indicators_rields.py \
    --groups $DATA_DIR/data_TCI \
    --c-scores $RESULTS_DIR/TCI/c_scores.csv \
    --out-dir $RESULTS_DIR/indices_collab
```

### PCA structure analysis

```bash
cd $SCRIPTS_DIR/analyse_inv
python analyze_inv_structure.py \
    --data $RESULTS_DIR/INV/high_level_features_audit.csv \
    --out $RESULTS_DIR/results_inv_structure
# Outputs: with_pruning/ and without_pruning/
```

### PDF report

```bash
python $SCRIPTS_DIR/rapport/v2/main.py \
    --results-dir $RESULTS_DIR \
    --out-dir $RESULTS_DIR/rapport_v2
```

| Argument | Description |
|----------|-------------|
| `--mode` | `vr_only` · `pc_vr` · `inv_vr` · `pca_vr` · bundle (default = all) |
| `--inv-analysis-mode` | `pruning` (default) or `no-pruning` |
| `--fdr` | Apply FDR correction per table |
| `--bayes` | Bayesian ICC via MCMC (slow) |

### Multimodal sociogram

```bash
python $SCRIPTS_DIR/visualisation_sociale/mirage_sociogram.py \
    --group-id <group_id> --modality VR --scenario S2 --timepoint T1 \
    --data-dir $DATA_DIR \
    --out-dir $RESULTS_DIR/visualisation_sociale/<group_id> \
    --snapshot-at 120
```

---

## Feature Configuration — `inv_features_config.py`

**`scripts/config/inv_features_config.py`** is the single source of truth governing which INV features enter each analysis stage. All report sections, PCA runs, and statistical regressions read their feature lists from this file — no feature selection happens ad hoc elsewhere.

### What it controls

| Constant / flag | Role |
|-----------------|------|
| `INV_FEATURES` | Master registry — one entry per feature with family, priority, flags, description |
| `_AU_INDIVIDUAL_SUBSTRINGS` | AU-level features excluded from PCA and regression (too granular) |
| `_EXCLUDED_EXACT_NAMES` | Variables explicitly excluded by name (redundant, non-analytical, or legacy) |
| `EXCLUDE_SUFFIXES / EXCLUDE_PREFIXES` | Pattern-based exclusion (`_old`, `_source`, `z_`, `log_`, …) |
| `PRUNING_PROTECTED_PAIRS` | Pairs kept together even when |r| > pruning threshold |
| `REDUNDANCY_CORR_THRESHOLD` | Correlation threshold for hard pruning (default 0.85) |
| `CORE_RIEDL_COLS` | Riedl team indicators used as Y columns in INV↔CI matrices |

### How feature selection works per context

Feature selection is **intentionally dissociated** across three analytical contexts — a feature can be in the report without being in the regression, or vice versa:

| Context | Mechanism | Where defined |
|---------|-----------|---------------|
| **Report (descriptive blocks)** | `core` / `core_hl` flags + `report_preferred` — defines which features appear in each modality section | `INV_FEATURES` entries |
| **PCA** | `FEATURE_PRIORITY` rank — lower rank = kept when a redundant pair must be pruned | `INV_FEATURES` entries |
| **Stepwise regression** | `REGRESSION_RETAINED_INV_FEATURES` ∩ `inv_pruned_features.csv` (`kept=1`) + `REGRESSION_FORCE_INCLUDE` override | Top-level lists |

The pruning pipeline (`analyze_inv_structure.py`) writes `inv_pruned_features.csv` with a `kept` column. The report's regression stage reads this file and intersects it with `REGRESSION_RETAINED_INV_FEATURES` to get the final candidate set. `REGRESSION_FORCE_INCLUDE` can re-inject specific variables regardless of pruning outcome.

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

## Data Quality

Groups with incomplete data (missing performance files, corrupted audio, absent markers) are excluded automatically. Exclusion criteria are defined in `common/constants.py` (`EXCLUDED_GROUPS`) and applied across all pipeline stages without manual intervention.
