# TRACE — Scripts Reference

All commands run from this directory (`scripts/`).

## Module Structure

```
scripts/
├── run_inv.py                        # Entry point: full INV pipeline (speech → gaze → face → HLF)
├── run_performance.py                # Entry point: task performance scoring (M1–M4)
├── build_schema.py                   # Pipeline diagram generator
│
├── common/                           # Shared package — imported by all modules
│   ├── constants.py                  # ROLES, EXCLUDED_GROUPS, PERF_WEIGHTS, MAX_TASK_DURATION_S
│   ├── metadata.py                   # extract_condition(), extract_scenario(), extract_timepoint()
│   ├── io_utils.py                   # read_csv_eu(), find_groups(), find_gaze_groups()
│   ├── stats.py                      # safe_div(), zscore_df(), shannon_entropy(), coef_var()
│   └── temporal.py                   # sliding_windows(), episodes_from_bool(), pairwise_overlap()
│
├── config/
│   └── inv_features_config.py        # Single source of truth for INV features
│                                     # (families, priorities, redundancy, core/report flags)
│
├── analyse_inv/
│   ├── analyze_inv_structure.py      # PCA + correlation matrix + hierarchical clustering
│   ├── speech/
│   │   ├── analyze_audio.py          # IPU → CA turn aggregation, floor exchanges, overlaps
│   │   ├── compute_audio_features.py # Feature derivation from turn-level metrics
│   │   ├── test_turn_aggregation.py  # Unit tests for turn aggregation logic
│   │   ├── viewer.py                 # Turn timeline viewer
│   │   └── viewer_annotated.py       # Annotated turn timeline viewer
│   ├── gaze/
│   │   ├── analyze_gaze.py           # Group-level gaze: JVA, shared attention, mutual gaze
│   │   ├── analyze_gaze_directional.py  # Directional gaze toward partners and objects
│   │   ├── diagnose_eyetracking.py   # Eye-tracking completeness diagnostic
│   │   ├── reconstruct_eyetracking.py   # Missing gaze data reconstruction
│   │   └── refine_yaw_eyetracking.py    # Yaw correction (MOD-16)
│   ├── face/
│   │   ├── analyze_aus_group.py      # AU/emotion aggregation at group level
│   │   └── openface/
│   │       ├── preprocess_pc_videos.py
│   │       └── run_openface_and_export_facs.py
│   └── hlf/
│       ├── compute_high_level_features.py       # Multimodal fusion + provenance tracing
│       └── compute_high_level_features_old_gaze.py  # Variant for legacy directional gaze
│
├── analyse_performance/
│   ├── analyze_performance_effects.py   # 2×2 ANOVA, t-tests, Welch, ANCOVA
│   ├── correlation_metrics_final.py
│   ├── cronbach_alpha_performance.py
│   └── multiple_regression_perf.py
│
├── evaluation_performance_scenario/
│   ├── score_groupes.py              # Aggregates Score_perf_tsk across all groups/conditions
│   ├── score_v2.py                   # Score aggregation variant
│   ├── S1/                           # Scenario 1 evaluators
│   │   ├── performance_eval_unified.py
│   │   ├── PC/eval_consignes_align.py
│   │   └── VR/eval_consignes_align.py
│   ├── S2/                           # Scenario 2 evaluators
│   │   ├── PC/{eval_consignes_s2,performance_eval}.py
│   │   └── VR/{eval_consignes_s2,performance_eval}.py
│   ├── corrections/                  # Reference solutions (PC_s1, VR_S1, PC_s2, VR_S2)
│   └── script_revit/                 # Dynamo scripts for Revit data export
│
├── analyse_TCI/
│   ├── TCI.py                        # C-factor computation (Woolley et al., 2010)
│   └── compute_team_indicators_rields.py  # Riedl indicators (skill, strategy, effort)
│
├── analyse_questionnaire/
│   ├── main.py                       # Orchestrator
│   └── py/
│       ├── config.py                 # Dimensions, Likert scales, reverse-scoring, thresholds
│       ├── io_read.py                # Raw data reader (xlsx/csv)
│       ├── transform.py              # Recoding, dimension scores
│       ├── reliability.py            # Cronbach alpha
│       ├── descriptives.py           # Descriptive statistics
│       ├── role_tests.py             # Role-based ANOVA / Kruskal-Wallis
│       ├── plots.py                  # PDF distributions
│       ├── item_pruning.py           # Greedy item pruning to maximize alpha
│       ├── g3_context.py             # Participant profile + free comments (G3Q00001–G3Q00007)
│       └── scenario_modalite.py      # Scenario × modality analysis
│
├── rapport/v2/                       # PDF report generation (canonical publication pipeline)
│   ├── main.py                       # Orchestrator (modes: vr_only, pc_vr, inv_vr, pca_vr)
│   ├── main_old_gaze.py              # Variant for legacy gaze pipeline
│   └── py/                           # 30+ report modules (data, corr, regression, pca, network…)
│
├── sem/
│   ├── pls_sem_vr.py                 # PLS-SEM path analysis (VR condition)
│   └── pls_sem_vr_old_gaze.py
│
├── visualisation_sociale/
│   ├── mirage_sociogram.py           # Offline multimodal sociogram (MIRAGE-inspired)
│   └── run_all_vr_mirage_sociograms.py  # Batch all VR groups
│
└── utilitaires/
    ├── adapter_revit_to_pc.py
    ├── build_inv_variable_inventory.py
    ├── build_inv_variable_inventory_old_gaze.py
    ├── compute_lecteur_durations.py
    └── merge_csv_by_suffix.py
```

---

## 1. INV Pipeline

`run_inv.py` orchestrates the full pipeline: speech → gaze → face → HLF → structure analysis.

```bash
# Full pipeline
python run_inv.py --data-dir ../data_e2 --out-dir ../results/INV

# Speech only
python run_inv.py --data-dir ../data_e2 --inv speech --out-dir ../results/INV

# Speech + gaze
python run_inv.py --data-dir ../data_e2 --inv speech gaze --out-dir ../results/INV

# Recompute HLF from existing modality CSVs (no re-extraction needed)
python run_inv.py --hlf-only \
    --speech-csv ../results/INV/audio_features.csv \
    --gaze-group-csv ../results/INV/gaze/ALL_metrics_overall.csv \
    --gaze-pair-csv ../results/INV/gaze/ALL_metrics_pairs.csv \
    --face-csv ../results/INV/face_emotion_metrics_all.csv \
    --out-dir ../results/INV
```

Individual module execution:

```bash
# Speech
python analyse_inv/speech/analyze_audio.py ../data_e2 \
    --out ../results/INV/audio_features.csv

# Gaze (group-level)
python analyse_inv/gaze/analyze_gaze.py \
    --data-dir ../data_e2 --out-dir ../results/INV/gaze

# Face
python analyse_inv/face/analyze_aus_group.py ../data_e2 \
    --out ../results/INV/face_emotion_metrics_all.csv

# HLF fusion
python analyse_inv/hlf/compute_high_level_features.py \
    --speech ../results/INV/audio_features.csv \
    --gaze-group ../results/INV/gaze/ALL_metrics_overall.csv \
    --gaze-pair ../results/INV/gaze/ALL_metrics_pairs.csv \
    --face ../results/INV/face_emotion_metrics_all.csv \
    --out ../results/INV/high_level_features.csv
```

---

## 2. Task Performance Scoring

```bash
# All modalities and scenarios
python run_performance.py --results-dir ../results

# VR only
python run_performance.py --results-dir ../results --modality VR

# PC, scenario S2
python run_performance.py --results-dir ../results --modality PC --scenario S2
```

**Performance metrics:**

| Metric | Weight | Definition |
|--------|--------|------------|
| M1 | 0.50 | Rules followed: `n_rules_passed / n_rules_total` |
| M2 | 0.00 | Objects placed: `n_placed / n_expected` |
| M3 | 0.50 | Spatial precision: % objects within tolerance |
| M4 | 0.00 | Time: `1 - duration / tref` |
| **Score_perf_tsk** | — | `0.5×M1 + 0.5×M3` |
| **Score_perf_tsk_z** | — | z-score per scenario (computed in report) |

M1–M4 are all produced for audit; only `Score_perf_tsk` and its z-score are used in v2 analyses.

Per-scenario low-level evaluation (all groups via PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File evaluation_performance_scenario/S1/VR/run_perf_s1_vr.ps1
```

Score aggregation across all groups:

```bash
cd evaluation_performance_scenario
python score_groupes.py \
    --roots ../results/performance_task/performance_VR/S1 \
            ../results/performance_task/performance_PC/S1 \
            ../results/performance_task/performance_VR/S2 \
            ../results/performance_task/performance_PC/S2 \
    --out ../results/performance_task/recap_scores_all.csv
```

---

## 3. PCA Structure Analysis

```bash
cd analyse_inv

# Default: produces both pruning modes
python analyze_inv_structure.py

# Custom options
python analyze_inv_structure.py \
    --data ../results/INV/high_level_features_audit.csv \
    --out ../results/results_inv_structure \
    --max-missing 0.20 \
    --min-cumvar 0.70 \
    --prune-threshold 0.80

# Single mode
python analyze_inv_structure.py --only-pruning-mode with      # results/with_pruning/
python analyze_inv_structure.py --only-pruning-mode without   # results/without_pruning/
```

**Outputs per mode:**

| File | Description |
|------|-------------|
| `inv_features_used.csv` | Features retained for PCA |
| `inv_pruned_features.csv` | Redundancy report (pairs removed + kept) |
| `pca_loadings.csv` | Component loadings |
| `pca_scree_plot.png` | Scree plot |
| `analysis_summary.csv` | Run parameters and diagnostics |

---

## 4. TCI (Team Collective Intelligence)

```bash
# C-factor (Woolley et al., 2010)
python analyse_TCI/TCI.py ../data_e2/data_TCI \
    --out ../results/TCI/c_scores.csv \
    --out-wide ../results/TCI/c_scores_with_tasks.csv \
    --missing mean --profile --scatter --heatmap

# Riedl indicators (skill, strategy, effort, congruence)
python analyse_TCI/compute_team_indicators_rields.py \
    --groups ../data_e2/data_TCI \
    --c-scores ../results/TCI/c_scores.csv \
    --out-dir ../results/indices_collab \
    --effort-mode event_count \
    --congruence-mode clip_zero
```

`--effort-mode`: `event_count` (default) or `text_length` (keystroke proxy)  
`--congruence-mode`: `clip_zero` (default), `raw`, or `rescale01`

---

## 5. Questionnaire Analysis

```bash
cd analyse_questionnaire

# Full analysis (reliability, descriptives, roles, plots, exploratory pruning)
python main.py \
    --data ../data_e2/results-survey.xlsx \
    --out ../../results/questionnaire \
    --mode all

# With greedy item pruning applied (recalculates alpha on pruned set)
python main.py ... --apply-pruning

# Custom alpha threshold (default 0.70 — only prunes dimensions below threshold)
python main.py ... --apply-pruning --alpha-threshold 0.80
```

**Outputs in `results/questionnaire/`:**
- `global/` — reliability (`cronbach_alpha_questionnaire.csv`), descriptives, role tests, PDFs
- `global/pruned/` — recalculated results after item pruning + `alpha_comparison.csv`
- `analyse/` — dimension scores, scenario × modality models
- `participant_profile_*`, `free_comments_*` — G3 profile and open-ended responses

---

## 6. Performance Statistical Analysis

```bash
cd analyse_performance

python analyze_performance_effects.py \
    --perf ../results/performance_task/recap_scores_all.csv \
    --merged ../results/merged_dataset/with_pruning/merged_dataset_complete_all.csv \
    --out ../results/analyse_performance

# Force a specific ANCOVA covariate
python analyze_performance_effects.py --covariate skill_mean
```

Runs: Shapiro-Wilk normality, Levene homogeneity, Welch t-test, Mann-Whitney, Cohen's d, 2×2 ANOVA (type II, partial η²), ANCOVA.

---

## 7. PDF Report

```bash
# Full bundle (VR + PC_VR reports)
python rapport/v2/main.py \
    --results-dir ../results \
    --out-dir ../results/rapport_v2

# Single mode
python rapport/v2/main.py ... --mode vr_only    # VR main report
python rapport/v2/main.py ... --mode inv_vr     # Detailed INV report (VR)
python rapport/v2/main.py ... --mode pca_vr     # PCA report (VR)
python rapport/v2/main.py ... --mode pc_vr      # Comparative PC+VR report
```

| Argument | Description |
|----------|-------------|
| `--inv-analysis-mode` | `pruning` (default) or `no-pruning` |
| `--fdr` | FDR correction per table |
| `--bayes` | Bayesian ICC via MCMC (slow) |
| `--no-sem` | Skip PLS-SEM sections |

**Bundle outputs:**
```
results/rapport_v2/
  PC_VR/rapport_principal_PC_VR.pdf
  VR_only/
    rapport_principal_VR.pdf
    rapport_INV_VR.pdf
    rapport_PCA_VR.pdf
    behavioral_indices_v2/rapport_indices_v2.pdf
    data_questionnaire/   ← questionnaire CSVs
    data_riedl_tci/       ← Riedl + TCI CSVs
    data_h2b/             ← H2b analysis CSVs
results/merged_dataset/
  with_pruning/merged_dataset_complete_all.csv
  without_pruning/merged_dataset_complete_all.csv
```

---

## 8. Multimodal Sociogram (MIRAGE-inspired)

```bash
# Static snapshot at t=120s
python visualisation_sociale/mirage_sociogram.py \
    --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
    --data-dir ../data_e2 \
    --out-dir ../results/visualisation_sociale/bim073 \
    --snapshot-at 120

# Lightweight GIF animation (1 frame per 10)
python visualisation_sociale/mirage_sociogram.py \
    --group-id bim073 --modality VR --scenario S2 --timepoint T1 \
    --data-dir ../data_e2 \
    --out-dir ../results/visualisation_sociale/bim073 \
    --export-gif --frame-stride 10 --frame-dpi 90

# Batch all VR groups
python visualisation_sociale/run_all_vr_mirage_sociograms.py \
    --data-dir ../data_e2 \
    --out-dir ../results/visualisation_sociale_all_vr
```

Gaze, JVA, and object positions are available in VR only. On PC, only speech and facial synchrony layers are active.

**Outputs per group:** `mirage_snapshot.png`, `window_summary.csv`, `node_metrics.csv`, `edge_metrics.csv`, `run_metadata.json`

---

## Common Package

| Module | Key exports |
|--------|-------------|
| `constants.py` | `ROLES`, `EXCLUDED_GROUPS`, `PERF_WEIGHTS`, `MAX_TASK_DURATION_S` |
| `metadata.py` | `extract_condition()`, `extract_scenario()`, `extract_timepoint()` |
| `io_utils.py` | `read_csv_eu()`, `read_csv_smart()`, `find_groups()`, `find_gaze_groups()` |
| `stats.py` | `safe_div()`, `zscore_df()`, `shannon_entropy()`, `coef_var()` |
| `temporal.py` | `sliding_windows()`, `episodes_from_bool()`, `pairwise_overlap()` |

CSV format: separator `;`, decimal `,`, encoding UTF-8. `read_csv_eu()` handles this automatically.

---

## Known Data Issues

| Group | Issue |
|-------|-------|
| bim002 | No performance file, corrupted data |
| bim032 | No performance file, missing Mod/Calc audio, no Lecteur marker |
| bim065_2 | No performance file |
| bim075 | No audio track, no markers, no performance file |

Excluded automatically via `common.constants.EXCLUDED_GROUPS`.
