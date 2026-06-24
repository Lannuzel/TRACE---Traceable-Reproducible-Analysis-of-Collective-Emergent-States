#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script : main.py

Description :
-------------
Ce script génère un rapport PDF détaillé pour l'analyse des groupes
en utilisant des données de performance, des questionnaires, et des métriques
d'interaction (gaze, speech, face, etc.). Il inclut des visualisations, des 
statistiques descriptives, et des corrélations entre différentes modalités.

Fonctionnalités principales :
- Chargement et nettoyage des données (performances, questionnaires, etc.).
- Génération de graphiques (disponibilité des données, performances par modalité, etc.).
- Calculs statistiques (corrélations de Spearman, ajustements FDR).
- Export des résultats sous forme de tableaux et graphiques dans un rapport PDF.

Structure du script :
---------------------
1. Constantes et paramètres généraux.
2. Fonctions utilitaires pour le nettoyage et la normalisation des données.
3. Chargement des données depuis des fichiers CSV.
4. Calculs statistiques et agrégations.
5. Génération de graphiques et tableaux.
6. Création du rapport PDF.

Dépendances :
-------------
- Python >= 3.8
- pandas
- numpy
- matplotlib
- scipy
- reportlab
- markdown (optionnel, pour les tableaux Markdown)

Fichiers requis :
-----------------
- Données de performance : `performance_task/recap_scores_all.csv` ou
  sous-dossiers `performance_task/performance_{PC,VR}/`.
- Données de questionnaires : fichiers CSV dans le répertoire `results/`.
- Données d'interaction (gaze, speech, face) : fichiers CSV dans les sous-dossiers.

Instructions d'exécution :
--------------------------
1. Assurez-vous que toutes les dépendances sont installées.
2. Placez les fichiers de données dans les répertoires appropriés.
3. Exécutez le script avec Python :
   ```bash
   python main.py --results-dir <chemin_vers_les_données> --out-dir <chemin_sortie>

"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MPLCONFIG_DIR = _PROJECT_ROOT / ".mplconfig"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
_VENDOR_DIR = _PROJECT_ROOT / "vendor"
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    # Garder vendor en fallback : le placer en tête masque Pillow installé par
    # un vendor/PIL incomplet et fait échouer matplotlib sous Python 3.12.
    sys.path.append(str(_VENDOR_DIR))
_SEM_DIR = _PROJECT_ROOT / "scripts" / "sem"
if _SEM_DIR.exists() and str(_SEM_DIR) not in sys.path:
    sys.path.insert(0, str(_SEM_DIR))
try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    _orig_md5 = hashlib.md5

    def _compat_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _orig_md5(*args, **kwargs)

    hashlib.md5 = _compat_md5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr, t as student_t

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import PageBreak

# HTML optionnel
try:
    import markdown as md_lib
    HAS_MD = True
except Exception:
    HAS_MD = False

try:
    import pingouin as pg
except Exception:
    pg = None

from py.theory_diagrams import generate_computational_diagram, generate_theory_diagrams
from py.cfactor_population_report import generate_cfactor_population_report
from py.corr import (
    add_two_plots_row,
    fmt2,
    md_table_highlight,
    n_units,
    pdf_table_from_df,
    render_corr_block,
    render_performance_stats_section,
    safe_filename,
    spearman_table,
    top_corr_df,
)
from py.network import (
    plot_global_correlation_network,
    prepare_global_correlation_network_data,
    render_network_metrics_section,
)
from py.regression import (
    COHESION_SCORE_COL,
    QUESTIONNAIRE_ALPHA_THRESHOLD,
    TMS_DIMENSIONS,
    compute_aggregated_scores,
    render_regression_graphs_section,
    render_regression_section,
    render_inv_stepwise_regression_section,
    render_tms_reliability_analysis,
)
from py.questionnaire import (
    compute_questionnaire_icc_by_dimension_from_survey,
    compute_questionnaire_rwg_construct_matrix_from_survey,
    render_questionnaire_comments_section,
    render_questionnaire_icc_section,
    render_questionnaire_cronbach_table,
    render_questionnaire_descriptif_table,
    render_questionnaire_modality_effect_section,
    render_questionnaire_profile_section,
    render_questionnaire_pruning_section,
    summarize_questionnaire_rwg_construct_matrix,
)
from py.h2b import render_h2b_section
from py.plots import (
    n_groups,
    plot_modalities_availability,
    plot_performance_by_modalite,
)
from py.pca_inv import render_inv_pca_section
from py.complementary_blocks import generate_complementary_reports
from py.behavioral_indices_report import generate_behavioral_indices_report
from py.behavioral_indices_v2_report import generate_behavioral_indices_v2_report
from py.data import (
    EXCLUDED_GROUPS,
    EXCLUSION_REASONS,
    ID_LIKE_COLS,
    add_group_base_id,
    aggregate_numeric_by_unit,
    analysis_keys_for_df,
    apply_modality_filter,
    available_unit_cols,
    build_group_master_csv,
    coerce_numeric_columns,
    common_unit_cols,
    exclude_bad_groups,
    extract_timepoint_from_group_like,
    filter_df_by_group_ids,
    has_real_timepoint,
    harmonize_timepoint,
    load_alpha_comparison,
    load_desc_dim_pruned,
    load_desc_dim_questionnaire,
    load_exploratory_summary,
    load_high_level_features,
    load_inv_face,
    load_inv_gaze_all,
    load_inv_pruned_features,
    load_inv_pruned_features_full,
    load_inv_speech,
    load_performance,
    load_questionnaire_cronbach,
    load_questionnaire_cronbach_pruned,
    load_questionnaire_descriptifs,
    load_questionnaire_free_comments,
    load_questionnaire_modeles,
    load_questionnaire_participant_profile,
    load_questionnaire_scores,
    load_riedl,
    load_tci,
    merge_on_unit,
    normalize_group,
    normalize_timepoint,
    perf_group_mean,
    performance_score_col,
    questionnaire_group_wide,
    read_csv_auto,
    enrich_inv_face_with_high_level,
    export_merged_dataset_bundle,
)

from sem.pls_sem_vr import run_pls_sem_vr, run_refined_path_analysis_vr

# Import configuration centralisée
from pathlib import Path as _Path
_scripts_dir = _Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    CORE_RIEDL_COLS,
    COHESION_COMPONENTS,
    CORE_FACE_V2 as CORE_FACE,
    CORE_GAZE_V2 as CORE_GAZE,
    CORE_HL_V2 as CORE_HL,
    CORE_MAP_V2 as CORE_MAP,
    CORE_SPEECH_V2 as CORE_SPEECH,
    CORE_FACE_REPORT,
    CORE_GAZE_REPORT,
    CORE_HL_REPORT,
    CORE_MAP_REPORT,
    CORE_SPEECH_REPORT,
    PERFORMANCE_ANALYSIS_COLS,
    QUESTIONNAIRE_ANALYSIS_COLS,
    QUESTIONNAIRE_RELIABILITY_DIMENSIONS,
    SCENARIO_COVARIATE_ONLY,
    filter_inv_dataframe,
    infer_family_from_name,
    is_excluded_inv_feature,
)
from config.inv_features_config import (
    REGRESSION_FORCE_INCLUDE,
    REGRESSION_RETAINED_INV_FEATURES,
)


# ===============================
# PARAMÈTRES GÉNÉRAUX
# ===============================
all_corr_results: list[dict] = []

# étoiles seulement si n >= N_MIN_SIG
N_MIN_SIG = 6

# tableaux : limites pour éviter des PDF illisibles
MAX_X_COLS = 30
MAX_Y_COLS = 25
MAX_ROWS_MD = 120
MAX_ROWS_PDF = 40

# couleurs p-values (on colore sur p_fdr si présent, sinon p)
P_COLOR_THRESH = [(0.01, colors.orange), (0.05, colors.yellow)]

COHESION_SUBDIM_COLS = ["SOC", "TSK", "COM"]
SUPPLEMENTAL_BOOTSTRAP_B = 5000

DIMENSION_LABELS_FULL = {
    "COR": "Coordination (Michinov, 2007)",
    "CRE": "Credibility (Michinov, 2007)",
    "SPE": "Specialization (Michinov, 2007)",
    "SOC": "Social Cohesion (Sassier-Roublin et al., 2025)",
    "TSK": "Task Cohesion (Sassier-Roublin et al., 2025)",
    "COM": "Communication (Sassier-Roublin et al., 2025)",
}

TCI_ANALYSIS_COL_ORDER = [
    "c_score",
    "c_factor_pop",
    "c_factor_sample",
    "rme_mean",
    "rme_max",
    "rme_min",
]


def _select_tci_analysis_cols(tci_df: pd.DataFrame | None) -> list[str]:
    """Garde uniquement les scores TCI analytiques, pas les scores bruts par tâche."""
    if tci_df is None or tci_df.empty:
        return []
    return [
        col
        for col in TCI_ANALYSIS_COL_ORDER
        if col in tci_df.columns and pd.api.types.is_numeric_dtype(tci_df[col])
    ]


def _norm_group_values(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _parse_exclude_groups(raw_list: Iterable[str] | None) -> tuple[set[str], set[str]]:
    if not raw_list:
        return set(), set()
    tokens: list[str] = []
    for raw in raw_list:
        tokens.extend(re.split(r"[;,]", str(raw)))
    tokens = [t.strip().lower() for t in tokens if t and str(t).strip()]
    exact: set[str] = set()
    base: set[str] = set()
    for t in tokens:
        if re.search(r"_\d+$", t):
            exact.add(t)
        else:
            base.add(t)
    return exact, base


def _is_excluded_group_id(group_id: str, extra_exact: set[str], extra_base: set[str]) -> bool:
    if group_id is None:
        return False
    gid = str(group_id).strip().lower()
    if gid in EXCLUDED_GROUPS or gid in extra_exact:
        return True
    gid_base = re.sub(r"_\d+$", "", gid)
    return gid_base in extra_base


def _build_tci_passed_scope_table(
    tci_all_passed: pd.DataFrame | None,
    riedl_available: pd.DataFrame | None,
    base_tci_riedl: pd.DataFrame | None,
    rcols: list[str],
    tci_cols: list[str],
    *,
    extra_exclude_exact: set[str] | None = None,
    extra_exclude_base: set[str] | None = None,
) -> pd.DataFrame:
    """Table d'audit : groupes avec TCI et statut d'utilisation dans 2.3."""
    if tci_all_passed is None or tci_all_passed.empty or "group_id" not in tci_all_passed.columns:
        return pd.DataFrame()

    tci = tci_all_passed.copy()
    tci["group_id"] = _norm_group_values(tci["group_id"])
    show_tci_cols = [c for c in _select_tci_analysis_cols(tci) if c in tci.columns]
    keep_cols = ["group_id"] + show_tci_cols
    out = tci[keep_cols].drop_duplicates(subset=["group_id"], keep="first").copy()

    riedl_groups: set[str] = set()
    if riedl_available is not None and not riedl_available.empty and "group_id" in riedl_available.columns:
        riedl_groups = set(_norm_group_values(riedl_available["group_id"]).dropna().unique())

    used_groups: set[str] = set()
    if base_tci_riedl is not None and not base_tci_riedl.empty and "group_id" in base_tci_riedl.columns:
        tmp = base_tci_riedl.copy()
        tmp["group_id"] = _norm_group_values(tmp["group_id"])
        riedl_present = tmp[[c for c in rcols if c in tmp.columns]].notna().any(axis=1) if rcols else False
        tci_present = tmp[[c for c in tci_cols if c in tmp.columns]].notna().any(axis=1) if tci_cols else False
        if not isinstance(riedl_present, bool) and not isinstance(tci_present, bool):
            used_groups = set(tmp.loc[riedl_present & tci_present, "group_id"].dropna().unique())

    extra_exclude_exact = extra_exclude_exact or set()
    extra_exclude_base = extra_exclude_base or set()
    out["statut_rapport"] = np.where(
        out["group_id"].apply(lambda gid: _is_excluded_group_id(gid, extra_exclude_exact, extra_exclude_base)),
        "exclu",
        "retenu",
    )
    out["indicateurs_riedl"] = np.where(out["group_id"].isin(riedl_groups), "oui", "non")
    out["utilise_2_3"] = np.where(out["group_id"].isin(used_groups), "oui", "non")

    def _comment(row: pd.Series) -> str:
        if row["statut_rapport"] == "exclu":
            return "groupe exclu du rapport"
        if row["indicateurs_riedl"] == "non":
            return "TCI disponible sans indicateurs Riedl"
        if row["utilise_2_3"] == "non":
            return "hors périmètre analytique 2.3"
        return "utilisé dans 2.3"

    out["commentaire"] = out.apply(_comment, axis=1)
    for col in show_tci_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)

    return out.sort_values("group_id").reset_index(drop=True)


def _spearman_rho_pair(df: pd.DataFrame, x: str, y: str) -> float:
    """Spearman rho robuste aux colonnes absentes/constantes."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return np.nan
    sub = df[[x, y]].dropna()
    if len(sub) < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        return np.nan
    rho, _ = spearmanr(sub[x], sub[y])
    return float(rho) if np.isfinite(rho) else np.nan


def _select_bootstrap_corr_pairs(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    min_abs_rho: float = 0.50,
) -> list[tuple[str, str]]:
    """Sélectionne les corrélations fortes du bloc 2.3 à diagnostiquer par bootstrap."""
    pairs: list[tuple[str, str]] = []
    for x in x_cols:
        for y in y_cols:
            rho = _spearman_rho_pair(df, x, y)
            if np.isfinite(rho) and abs(rho) >= min_abs_rho:
                pairs.append((x, y))
    if ("skill_mean", "c_score") not in pairs and {"skill_mean", "c_score"}.issubset(df.columns):
        pairs.insert(0, ("skill_mean", "c_score"))
    return pairs


def _bootstrap_spearman_ci_table(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    *,
    n_boot: int = 5000,
    random_state: int = 20260424,
) -> pd.DataFrame:
    """IC bootstrap percentile 95 % pour rho de Spearman."""
    if df is None or df.empty or not pairs:
        return pd.DataFrame()

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for x, y in pairs:
        cols = ["group_id", x, y] if "group_id" in df.columns else [x, y]
        sub = df[cols].dropna().copy()
        if len(sub) < 4 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
            continue

        values = sub[[x, y]].astype(float).to_numpy()
        n = len(values)
        rho = _spearman_rho_pair(sub, x, y)
        boot: list[float] = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = values[idx]
            if len(np.unique(sample[:, 0])) < 2 or len(np.unique(sample[:, 1])) < 2:
                continue
            r, _ = spearmanr(sample[:, 0], sample[:, 1])
            if np.isfinite(r):
                boot.append(float(r))

        if not boot:
            continue
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        includes_050 = bool(ci_low <= 0.50 <= ci_high) if rho > 0 else False
        rows.append({
            "x": x,
            "y": y,
            "rho": round(rho, 3),
            "n": n,
            "IC95_low": round(float(ci_low), 3),
            "IC95_high": round(float(ci_high), 3),
            "IC_inclut_0.50": "oui" if includes_050 else "non",
            "n_boot_valides": len(boot),
        })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    out = out.sort_values(["abs_rho", "y", "x"], ascending=[False, True, True])
    return out.drop(columns=["abs_rho"]).reset_index(drop=True)


def _compare_tci_scope_descriptives(
    analytic_df: pd.DataFrame,
    all_tci_df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Compare les descriptifs des variables clés entre n=12 et n=20."""
    rows: list[dict[str, Any]] = []
    scopes = [
        ("2.3 analytique", analytic_df),
        ("TCI all", all_tci_df),
    ]
    for var in variables:
        for scope_label, df in scopes:
            if df is None or df.empty or var not in df.columns:
                continue
            vals = pd.to_numeric(df[var], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "variable": var,
                "echantillon": scope_label,
                "n": int(vals.shape[0]),
                "mean": round(float(vals.mean()), 3),
                "sd": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan,
                "median": round(float(vals.median()), 3),
                "min": round(float(vals.min()), 3),
                "max": round(float(vals.max()), 3),
            })
    return pd.DataFrame(rows)


def _spearman_pair_stats(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    """Retourne rho, p et n pour une paire Spearman."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return {"rho": np.nan, "p": np.nan, "n": 0}
    sub = df[[x, y]].dropna()
    n = int(len(sub))
    if n < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        return {"rho": np.nan, "p": np.nan, "n": n}
    rho, p = spearmanr(sub[x], sub[y])
    return {
        "rho": float(rho) if np.isfinite(rho) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
        "n": n,
    }


def _build_rtd_tci_comparison_table(
    analytic_df: pd.DataFrame,
    extended_df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    bootstrap_ci_df: pd.DataFrame,
) -> pd.DataFrame:
    """Table comparant les corrélations Riedl-TCI n=12 vs n=20 avec IC bootstrap n=12."""
    if analytic_df is None or analytic_df.empty or not x_cols or not y_cols:
        return pd.DataFrame()

    ci_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if bootstrap_ci_df is not None and not bootstrap_ci_df.empty:
        for _, row in bootstrap_ci_df.iterrows():
            ci_lookup[(str(row.get("x")), str(row.get("y")))] = (
                float(row.get("IC95_low")) if pd.notna(row.get("IC95_low")) else np.nan,
                float(row.get("IC95_high")) if pd.notna(row.get("IC95_high")) else np.nan,
            )

    rows: list[dict[str, Any]] = []
    for x in x_cols:
        for y in y_cols:
            s12 = _spearman_pair_stats(analytic_df, x, y)
            s20 = _spearman_pair_stats(extended_df, x, y)
            if not (np.isfinite(s12["rho"]) or np.isfinite(s20["rho"])):
                continue
            ci_low, ci_high = ci_lookup.get((x, y), (np.nan, np.nan))
            rows.append({
                "x": x,
                "y": y,
                "rho_n12": round(float(s12["rho"]), 3) if np.isfinite(s12["rho"]) else np.nan,
                "p_n12": round(float(s12["p"]), 4) if np.isfinite(s12["p"]) else np.nan,
                "n_n12": int(s12["n"]),
                "IC95_n12": (
                    f"[{ci_low:.3f}; {ci_high:.3f}]"
                    if np.isfinite(ci_low) and np.isfinite(ci_high)
                    else ""
                ),
                "rho_n20": round(float(s20["rho"]), 3) if np.isfinite(s20["rho"]) else np.nan,
                "p_n20": round(float(s20["p"]), 4) if np.isfinite(s20["p"]) else np.nan,
                "n_n20": int(s20["n"]),
            })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["sort_abs"] = out["rho_n12"].abs().fillna(0)
    out = out.sort_values(["sort_abs", "p_n12"], ascending=[False, True]).drop(columns=["sort_abs"])
    return out.reset_index(drop=True)


def _leave_one_out_spearman_table(
    df: pd.DataFrame,
    x: str,
    y: str,
    threshold: float = 0.70,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Diagnostic d'influence par retrait d'une observation."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return pd.DataFrame(), {}
    cols = ["group_id", x, y] if "group_id" in df.columns else [x, y]
    sub = df[cols].dropna().copy()
    if len(sub) < 4:
        return pd.DataFrame(), {}

    rho_full = _spearman_rho_pair(sub, x, y)
    rows: list[dict[str, Any]] = []
    for idx, row in sub.iterrows():
        loo = sub.drop(index=idx)
        rho_loo = _spearman_rho_pair(loo, x, y)
        rows.append({
            "groupe_retiré": row.get("group_id", str(idx)),
            "rho_leave_one_out": round(float(rho_loo), 3) if np.isfinite(rho_loo) else np.nan,
            "delta_vs_rho_complet": round(float(rho_loo - rho_full), 3) if np.isfinite(rho_loo) and np.isfinite(rho_full) else np.nan,
            "sous_0.70": "oui" if np.isfinite(rho_loo) and rho_loo < threshold else "non",
        })

    out = pd.DataFrame(rows).sort_values("rho_leave_one_out", ascending=True).reset_index(drop=True)
    below = out[out["sous_0.70"] == "oui"]
    summary = {
        "rho_full": round(float(rho_full), 3) if np.isfinite(rho_full) else np.nan,
        "n": int(len(sub)),
        "rho_min_leave_one_out": round(float(out["rho_leave_one_out"].min()), 3) if not out.empty else np.nan,
        "n_retraits_sous_0.70": int(len(below)),
        "un_retrait_suffit": "oui" if len(below) > 0 else "non",
    }
    return out, summary


def _minimal_removal_spearman_search(
    df: pd.DataFrame,
    x: str,
    y: str,
    threshold: float = 0.70,
    max_remove: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cherche le plus petit nombre de groupes dont le retrait fait passer rho sous un seuil."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return pd.DataFrame(), {}
    cols = ["group_id", x, y] if "group_id" in df.columns else [x, y]
    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) < 4:
        return pd.DataFrame(), {}

    rho_full = _spearman_rho_pair(sub, x, y)
    rows: list[dict[str, Any]] = []
    for k in range(1, min(max_remove, len(sub) - 3) + 1):
        k_rows: list[dict[str, Any]] = []
        for idxs in combinations(range(len(sub)), k):
            keep = sub.drop(index=list(idxs))
            rho_after = _spearman_rho_pair(keep, x, y)
            if not np.isfinite(rho_after) or rho_after >= threshold:
                continue
            removed = sub.loc[list(idxs), "group_id"].astype(str).tolist() if "group_id" in sub.columns else [str(i) for i in idxs]
            k_rows.append({
                "n_groupes_retirés": k,
                "groupes_retirés": "; ".join(removed),
                "rho_après_retrait": round(float(rho_after), 3),
                "delta_vs_rho_complet": round(float(rho_after - rho_full), 3) if np.isfinite(rho_full) else np.nan,
            })
        if k_rows:
            rows = k_rows
            break

    if not rows:
        return pd.DataFrame(), {
            "rho_full": round(float(rho_full), 3) if np.isfinite(rho_full) else np.nan,
            "seuil": threshold,
            "max_remove": max_remove,
            "n_min_retraits": f">{max_remove}",
        }

    out = pd.DataFrame(rows).sort_values("rho_après_retrait", ascending=True).reset_index(drop=True)
    summary = {
        "rho_full": round(float(rho_full), 3) if np.isfinite(rho_full) else np.nan,
        "seuil": threshold,
        "n_min_retraits": int(out["n_groupes_retirés"].iloc[0]),
        "n_combinaisons_sous_seuil": int(len(out)),
    }
    return out, summary


def _build_vr_familiarity_performance_dataset(
    q_profile: pd.DataFrame | None,
    perf: pd.DataFrame | None,
) -> pd.DataFrame:
    """Construit le dataset groupe VR : familiarité VR moyenne x performance."""
    if q_profile is None or q_profile.empty or perf is None or perf.empty:
        return pd.DataFrame()
    if "vr_familiarity_score" not in q_profile.columns or "Score_perf_tsk" not in perf.columns:
        return pd.DataFrame()

    q = q_profile.copy()
    p = perf.copy()
    q = normalize_group(q)
    p = normalize_group(p)
    q = normalize_timepoint(q)
    p = normalize_timepoint(p)
    if "modalite" in q.columns:
        q = q[q["modalite"].astype(str).str.upper() == "VR"].copy()
    if "modalite" in p.columns:
        p = p[p["modalite"].astype(str).str.upper() == "VR"].copy()
    if q.empty or p.empty:
        return pd.DataFrame()

    q["vr_familiarity_score"] = pd.to_numeric(q["vr_familiarity_score"], errors="coerce")
    q["team_familiarity_mean_score"] = pd.to_numeric(
        q.get("team_familiarity_mean_score", pd.Series(index=q.index, dtype=float)),
        errors="coerce",
    )
    group_cols = [c for c in ["group_id", "timepoint", "modalite", "scenario"] if c in q.columns]
    fam = (
        q.groupby(group_cols, dropna=False)
        .agg(
            vr_familiarity_mean=("vr_familiarity_score", "mean"),
            vr_familiarity_sd=("vr_familiarity_score", "std"),
            vr_familiarity_min=("vr_familiarity_score", "min"),
            vr_familiarity_max=("vr_familiarity_score", "max"),
            team_familiarity_mean=("team_familiarity_mean_score", "mean"),
            n_members=("vr_familiarity_score", "count"),
        )
        .reset_index()
    )

    keep_perf = [
        c for c in [
            "group_id", "timepoint", "modalite", "scenario",
            "Score_perf_tsk", "M1_consignes_%", "M2_nombre_%",
            "M3_precision_%", "M4_temps_%",
        ]
        if c in p.columns
    ]
    out = merge_on_unit(fam, p[keep_perf], how="inner")
    for col in ["modalite", "scenario"]:
        if col not in out.columns:
            left = out.get(f"{col}_x", pd.Series([np.nan] * len(out), index=out.index))
            right = out.get(f"{col}_y", pd.Series([np.nan] * len(out), index=out.index))
            out[col] = left.where(left.notna() & (left.astype(str).str.strip() != ""), right)
    out = coerce_numeric_columns(out, exclude={"group_id", "timepoint", "modalite", "scenario"})
    return out.sort_values([c for c in ["scenario", "group_id", "timepoint"] if c in out.columns]).reset_index(drop=True)


def _fit_manual_ols(y: np.ndarray, predictors: dict[str, np.ndarray]) -> dict[str, Any]:
    """OLS minimal avec erreurs standards et p-values, sans dépendance statsmodels."""
    y = np.asarray(y, dtype=float)
    names = ["Intercept"] + list(predictors.keys())
    x_cols = [np.ones(len(y), dtype=float)] + [np.asarray(v, dtype=float) for v in predictors.values()]
    X = np.column_stack(x_cols)
    n, p = X.shape
    if n <= p:
        return {"ok": False}
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
        resid = y - X @ beta
        df_resid = n - p
        rss = float(np.sum(resid ** 2))
        tss = float(np.sum((y - y.mean()) ** 2))
        mse = rss / df_resid if df_resid > 0 else np.nan
        se = np.sqrt(np.diag(xtx_inv) * mse)
        t_vals = beta / se
        p_vals = 2 * student_t.sf(np.abs(t_vals), df=df_resid)
        r2 = np.nan if tss <= 0 else 1.0 - rss / tss
    except Exception:
        return {"ok": False}
    return {
        "ok": True,
        "names": names,
        "beta": dict(zip(names, beta)),
        "se": dict(zip(names, se)),
        "p": dict(zip(names, p_vals)),
        "r2": float(r2),
        "n": int(n),
        "df_resid": int(df_resid),
    }


def _fit_vr_familiarity_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corrélations et régressions familiarité VR -> performance."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = df[["vr_familiarity_mean", "Score_perf_tsk"] + ([ "scenario" ] if "scenario" in df.columns else [])].dropna(
        subset=["vr_familiarity_mean", "Score_perf_tsk"]
    ).copy()
    if len(sub) < 3:
        return pd.DataFrame(), pd.DataFrame()

    x = sub["vr_familiarity_mean"].astype(float)
    y = sub["Score_perf_tsk"].astype(float)
    corr_rows: list[dict[str, Any]] = []
    if x.nunique() >= 2 and y.nunique() >= 2:
        rho, p_s = spearmanr(x, y)
        r_p, p_p = pearsonr(x, y)
        corr_rows.extend([
            {"test": "Spearman", "x": "vr_familiarity_mean", "y": "Score_perf_tsk", "stat": "rho", "estimate": rho, "p": p_s, "n": len(sub)},
            {"test": "Pearson", "x": "vr_familiarity_mean", "y": "Score_perf_tsk", "stat": "r", "estimate": r_p, "p": p_p, "n": len(sub)},
        ])

    reg_rows: list[dict[str, Any]] = []
    if x.nunique() >= 2 and y.nunique() >= 2:
        reg = linregress(x, y)
        x_sd = float(x.std(ddof=1))
        y_sd = float(y.std(ddof=1))
        beta_std = float(reg.slope * x_sd / y_sd) if x_sd > 0 and y_sd > 0 else np.nan
        reg_rows.append({
            "modele": "simple",
            "formule": "Score_perf_tsk ~ vr_familiarity_mean",
            "terme": "vr_familiarity_mean",
            "beta_raw": reg.slope,
            "beta_std": beta_std,
            "p": reg.pvalue,
            "r2": reg.rvalue ** 2,
            "n": len(sub),
        })

    if "scenario" in sub.columns and sub["scenario"].nunique(dropna=True) >= 2:
        scenario = sub["scenario"].astype(str).str.strip()
        baseline = sorted(scenario.dropna().unique())[0]
        predictors = {"vr_familiarity_mean": x.to_numpy(dtype=float)}
        for level in sorted(scenario.dropna().unique()):
            if level == baseline:
                continue
            predictors[f"scenario_{level}_vs_{baseline}"] = (scenario == level).astype(float).to_numpy()
        fit = _fit_manual_ols(y.to_numpy(dtype=float), predictors)
        if fit.get("ok"):
            x_sd = float(x.std(ddof=1))
            y_sd = float(y.std(ddof=1))
            beta_raw = float(fit["beta"].get("vr_familiarity_mean", np.nan))
            beta_std = beta_raw * x_sd / y_sd if x_sd > 0 and y_sd > 0 else np.nan
            reg_rows.append({
                "modele": "covariable_scenario",
                "formule": f"Score_perf_tsk ~ vr_familiarity_mean + scenario (réf. {baseline})",
                "terme": "vr_familiarity_mean",
                "beta_raw": beta_raw,
                "beta_std": beta_std,
                "p": float(fit["p"].get("vr_familiarity_mean", np.nan)),
                "r2": float(fit["r2"]),
                "n": int(fit["n"]),
            })

    corr_df = pd.DataFrame(corr_rows)
    reg_df = pd.DataFrame(reg_rows)
    for tab in [corr_df, reg_df]:
        for col in ["estimate", "beta_raw", "beta_std", "p", "r2"]:
            if col in tab.columns:
                tab[col] = pd.to_numeric(tab[col], errors="coerce").round(4)
    return corr_df, reg_df


def _plot_vr_familiarity_performance(df: pd.DataFrame, outpath: Path) -> bool:
    """Nuage de points familiarité VR x performance avec droite globale."""
    if df is None or df.empty:
        return False
    cols = ["vr_familiarity_mean", "Score_perf_tsk"]
    if not set(cols).issubset(df.columns):
        return False
    sub = df[cols].dropna(subset=cols).copy()
    if len(sub) < 3 or sub["vr_familiarity_mean"].nunique() < 2:
        return False

    plt.figure(figsize=(5.4, 3.8))
    plt.scatter(
        sub["vr_familiarity_mean"],
        sub["Score_perf_tsk"],
        s=64,
        alpha=0.84,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.7,
        label=f"Groupes VR (n={len(sub)})",
    )

    reg = linregress(sub["vr_familiarity_mean"], sub["Score_perf_tsk"])
    xx = np.linspace(sub["vr_familiarity_mean"].min(), sub["vr_familiarity_mean"].max(), 100)
    yy = reg.slope * xx + reg.intercept
    plt.plot(xx, yy, color="#222222", linewidth=1.8, linestyle="--", label="Régression linéaire")
    plt.xlabel("Familiarité VR moyenne du groupe (1-5)")
    plt.ylabel("Score_perf_tsk")
    plt.title("Familiarité VR et performance de tâche")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return True


def render_vr_familiarity_performance_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    q_profile: pd.DataFrame | None,
    perf: pd.DataFrame | None,
    fig_dir: Path,
    out_dir: Path,
    section_num: str = "1.3c",
) -> None:
    """Ajoute l'analyse familiarité VR -> performance après 1.3b."""
    title = f"{section_num} Familiarité VR comme covariable de performance"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))

    data = _build_vr_familiarity_performance_dataset(q_profile, perf)
    if data.empty:
        msg = "Analyse indisponible : familiarité VR ou performance absente pour le sous-échantillon VR."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    _vfdir = out_dir / "data_vr_familiarity"
    _vfdir.mkdir(parents=True, exist_ok=True)
    data.to_csv(_vfdir / "vr_familiarity_performance_dataset.csv", index=False, encoding="utf-8-sig")
    corr_df, reg_df = _fit_vr_familiarity_models(data)
    if not corr_df.empty:
        corr_df.to_csv(_vfdir / "vr_familiarity_performance_correlations.csv", index=False, encoding="utf-8-sig")
    if not reg_df.empty:
        reg_df.to_csv(_vfdir / "vr_familiarity_performance_regressions.csv", index=False, encoding="utf-8-sig")

    intro = (
        "Analyse VR uniquement : le score de familiarité VR est la moyenne intra-groupe de `vr_familiarity_score` "
        "issue du questionnaire participant. Il est testé comme prédicteur/covariable de `Score_perf_tsk` au niveau groupe."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro.replace("`", ""), styles["Normal"]))

    desc = data[["vr_familiarity_mean", "Score_perf_tsk"]].describe().T.reset_index().rename(columns={"index": "variable"})
    desc = desc[["variable", "count", "mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median", "count": "n"})
    for col in ["n", "mean", "std", "min", "median", "max"]:
        desc[col] = pd.to_numeric(desc[col], errors="coerce").round(3)
    lines.append("**Descriptifs groupe-level**\n\n")
    lines.append(md_table_highlight(desc, max_rows=8))
    pdf_elems.append(Paragraph("Descriptifs groupe-level", styles["Heading4"]))
    pdf_elems.append(pdf_table_from_df(desc, max_rows=8))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    if not corr_df.empty:
        lines.append("**Corrélations familiarité VR ↔ performance**\n\n")
        lines.append(md_table_highlight(corr_df, max_rows=8))
        pdf_elems.append(Paragraph("Corrélations familiarité VR ↔ performance", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(corr_df, max_rows=8))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    if not reg_df.empty:
        lines.append("**Régressions performance ~ familiarité VR**\n\n")
        lines.append(md_table_highlight(reg_df, max_rows=8))
        pdf_elems.append(Paragraph("Régressions performance ~ familiarité VR", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(reg_df, max_rows=8))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    fig_path = fig_dir / "vr_familiarity_vs_performance.png"
    if _plot_vr_familiarity_performance(data, fig_path):
        lines.append(f"![]({fig_path.name})\n\n")
        pdf_elems.append(Image(str(fig_path), width=5.5 * inch, height=3.8 * inch))
        pdf_elems.append(Spacer(1, 0.12 * inch))


def render_pls_sem_vr_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    merged_master: pd.DataFrame | None,
    results_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    section_num: str = "1.4",
    markdown_heading: str = "##",
    pdf_heading_style: str = "Heading2",
    add_page_break: bool = True,
) -> None:
    """Rend une section PLS-SEM exploratoire pour le sous-echantillon VR."""
    title = f"{section_num} PLS-SEM exploratoire sur le sous-échantillon VR"
    lines.append(f"{markdown_heading} {title}\n")
    if add_page_break:
        pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))

    intro = (
        "Les régressions INV déjà rapportées sont complétées ici par une approche de type PLS-SEM. "
        "L'objectif n'est pas de produire un test confirmatoire d'ajustement global, mais de représenter "
        "simultanément un modèle de mesure et un modèle structurel. Le modèle de mesure regroupe les indices "
        "INV retenus comme composite formatif, conserve les dimensions TMS séparées (`COR`, `CRE`, `SPE`) "
        "et traite la cohésion comme score observé agrégé. Le modèle structurel relie les causes racines "
        "(`c_score`, indicateurs Riedl et RME), les médiateurs INV/TMS/cohésion et la performance finale "
        "`Score_perf_tsk`."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro.replace("`", ""), styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.10 * inch))

    sem_out_dir = results_dir / "sem" / "pls_sem_vr"
    try:
        result = run_pls_sem_vr(
            merged_df=merged_master,
            output_dir=sem_out_dir,
        )
    except Exception as exc:
        msg = f"Analyse PLS-SEM non estimée : erreur lors de l'exécution ({exc})."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    method_note = (
        "Implémentation : composite INV par moyenne de z-scores à poids égaux, puis estimation des chemins "
        "par OLS standardisé. Cette procédure est utilisée comme approximation PLS-SEM prédictive et exploratoire, "
        "compatible avec les contraintes de faible effectif, mais elle ne doit pas être interprétée comme une "
        "validation causale ou confirmatoire."
    )
    lines.append(method_note + "\n\n")
    pdf_elems.append(Paragraph(method_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    availability = result.get("availability")
    if isinstance(availability, pd.DataFrame) and not availability.empty:
        lines.append("**Disponibilité des variables du modèle**\n\n")
        lines.append(md_table_highlight(availability, max_rows=40))
        pdf_elems.append(Paragraph("Disponibilité des variables du modèle", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(availability, max_rows=40))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    if not result.get("estimated", False):
        msg = result.get("note", "Analyse PLS-SEM non estimable faute de variables disponibles.")
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    n_model = len(result.get("dataset", pd.DataFrame()))
    inv_features = ", ".join(result.get("inv_features", []))
    exog = ", ".join(result.get("exogenous", []))
    summary = (
        f"Le modèle est estimé sur le sous-échantillon VR disponible (N = {n_model}). "
        f"Composite INV formatif : {inv_features}. Variables exogènes incluses : {exog}."
    )
    lines.append(summary + "\n\n")
    pdf_elems.append(Paragraph(summary, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    measurement = result.get("measurement")
    if isinstance(measurement, pd.DataFrame) and not measurement.empty:
        show = measurement[[
            c for c in ["indicator", "mode", "weight", "loading_corr_with_composite", "n"]
            if c in measurement.columns
        ]].copy()
        lines.append("**Modèle de mesure INV formatif**\n\n")
        lines.append(md_table_highlight(show, max_rows=20))
        pdf_elems.append(Paragraph("Modèle de mesure INV formatif", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(show, max_rows=20))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    paths = result.get("paths")
    if isinstance(paths, pd.DataFrame) and not paths.empty:
        show_cols = [c for c in ["source", "target", "path_coef_std", "se", "t", "p", "n", "df_resid"] if c in paths.columns]
        path_show = paths[show_cols].copy()
        lines.append("**Coefficients de chemin standardisés**\n\n")
        lines.append(md_table_highlight(path_show, max_rows=60))
        pdf_elems.append(Paragraph("Coefficients de chemin standardisés", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(path_show, max_rows=60))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    r2_df = result.get("r2")
    if isinstance(r2_df, pd.DataFrame) and not r2_df.empty:
        show_cols = [c for c in ["target", "n", "r2", "n_predictors", "df_resid"] if c in r2_df.columns]
        r2_show = r2_df[show_cols].copy()
        lines.append("**R² des variables endogènes**\n\n")
        lines.append(md_table_highlight(r2_show, max_rows=20))
        pdf_elems.append(Paragraph("R² des variables endogènes", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(r2_show, max_rows=20))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    indirect = result.get("indirect")
    if isinstance(indirect, pd.DataFrame) and not indirect.empty:
        indirect_show = indirect.drop(columns=["warning"], errors="ignore").copy()
        lines.append("**Effets indirects exploratoires**\n\n")
        lines.append(md_table_highlight(indirect_show, max_rows=30))
        pdf_elems.append(Paragraph("Effets indirects exploratoires", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(indirect_show, max_rows=30))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    diagram_path = result.get("diagram_path")
    if diagram_path is not None and Path(diagram_path).exists():
        local_diagram = fig_dir / "pls_sem_vr_model.png"
        try:
            shutil.copyfile(Path(diagram_path), local_diagram)
            lines.append(f"![]({fig_dir.name}/{local_diagram.name})\n\n")
            pdf_elems.append(Image(str(local_diagram), width=6.4 * inch, height=3.6 * inch))
            pdf_elems.append(Spacer(1, 0.10 * inch))
        except Exception:
            pass

    limits = (
        "Limites spécifiques du PLS-SEM : l'effectif VR est très faible, les coefficients peuvent être instables "
        "et sensibles aux observations individuelles, et les effets indirects ne constituent pas une preuve de "
        "médiation causale. Cette analyse doit donc être lue comme une carte exploratoire des chemins plausibles, "
        "à valider sur un échantillon plus large avant toute conclusion forte."
    )
    lines.append("**Limites spécifiques du PLS-SEM**\n\n")
    lines.append(limits + "\n\n")
    pdf_elems.append(Paragraph("Limites spécifiques du PLS-SEM", styles["Heading4"]))
    pdf_elems.append(Paragraph(limits, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))


def render_refined_path_analysis_vr_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    merged_master: pd.DataFrame | None,
    results_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    section_num: str = "3.1.5",
    markdown_heading: str = "####",
    pdf_heading_style: str = "Heading4",
    add_page_break: bool = True,
    cronbach_df: "pd.DataFrame | None" = None,
) -> None:
    """Rend la path analysis VR alignée sur les sous-systèmes transactifs."""
    title = f"{section_num} Path analysis exploratoire sur composites formatifs alignés sur les sous-systèmes transactifs"
    lines.append(f"{markdown_heading} {title}\n")
    if add_page_break:
        pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))

    # Les tableaux de cette section sont très larges avec les noms de colonnes
    # bruts. On garde les CSV complets en sortie, mais on compacte l'affichage
    # Markdown/PDF pour préserver la lisibilité.
    short_names = {
        "INPUT_composite": "INPUT",
        "Score_perf_tsk": "Performance",
        "Cohesion_questionnaire_score": "Cohésion",
        "face_negative_affect_ratio": "Affect négatif",
        "gaze_attention_coordination_idx": "gaze_attention",
        "gaze_shared_visual_attention_ratio": "gaze_sva_ratio",
        "gaze_entropy_mean_participants": "gaze_entropy",
        "shared_obj_ratio": "shared_obj",
        "audio_distrib_speech": "distrib_speech",
        "audio_participation_entropy": "particip_entropy",
        "audio_successful_interruption_ratio": "interruptions",
        "audio_floor_exchange_pause_mean_s": "floor_pause",
        "face_facial_synchrony": "face_sync",
        "audio_avg_speaking_turn_duration_s": "turn_duration",
        "audio_backchannel_rate_per_min": "backchannel_rate",
        "strategy_ratio_mean": "strategy_ratio",
        "strategy_norm": "strategy_norm",
        "effort_task_mean": "effort_task",
        "effort_task_norm": "effort_task_norm",
        "skill_congruence_mean": "skill_congruence",
        "rme_mean": "RME_mean",
        "c_score": "c_score",
    }

    def _short_name(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        return short_names.get(text, text)

    def _short_expr(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value)
        # Remplacer d'abord les noms les plus longs pour éviter les collisions.
        for raw, short in sorted(short_names.items(), key=lambda kv: len(kv[0]), reverse=True):
            text = text.replace(raw, short)
        return text

    def _add_ci95(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"ci95_low", "ci95_high"}.issubset(out.columns):
            out["IC95"] = out.apply(
                lambda r: f"[{float(r['ci95_low']):.3g}; {float(r['ci95_high']):.3g}]"
                if pd.notna(r["ci95_low"]) and pd.notna(r["ci95_high"])
                else "",
                axis=1,
            )
            out = out.drop(columns=["ci95_low", "ci95_high"], errors="ignore")
        return out

    def _effect_magnitude(beta: float) -> str:
        if not np.isfinite(beta):
            return "NA"
        abs_beta = abs(float(beta))
        if abs_beta < 0.10:
            return "négligeable"
        if abs_beta < 0.30:
            return "faible"
        if abs_beta < 0.50:
            return "modérée"
        return "forte"

    path_out_dir = results_dir / "sem" / "path_analysis_vr"
    try:
        result = run_refined_path_analysis_vr(
            merged_df=merged_master,
            output_dir=path_out_dir,
            n_boot=5000,
        )
    except Exception as exc:
        msg = f"Path analysis refactorisée non estimée : erreur lors de l'exécution ({exc})."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    availability = result.get("availability")
    if isinstance(availability, pd.DataFrame) and not availability.empty:
        availability_show = availability.copy()
        if "variable" in availability_show.columns:
            availability_show["variable"] = availability_show["variable"].map(_short_name)
        keep_cols = [
            c for c in ["variable", "available", "n_non_missing"]
            if c in availability_show.columns
        ]
        availability_show = availability_show[keep_cols].copy()
        lines.append("**Disponibilité des variables du modèle refactorisé**\n\n")
        lines.append(md_table_highlight(availability_show, max_rows=60))
        pdf_elems.append(Paragraph("Disponibilité des variables du modèle refactorisé", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(availability_show, max_rows=60))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    if not result.get("estimated", False):
        msg = result.get("note", "Path analysis refactorisée non estimable faute de variables disponibles.")
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    spec_df = result.get("composite_justification")
    if not isinstance(spec_df, pd.DataFrame) or spec_df.empty:
        specs = result.get("composite_specs", {})
        if isinstance(specs, dict) and specs:
            spec_df = pd.DataFrame([
                {"construct": construct, "indicators": ", ".join(indicators)}
                for construct, indicators in specs.items()
            ])
    if isinstance(spec_df, pd.DataFrame) and not spec_df.empty:
        spec_show = spec_df.copy()
        if "indicators" in spec_show.columns:
            spec_show["indicators"] = spec_show["indicators"].apply(_short_expr)
        if "inverted_indicators" in spec_show.columns:
            spec_show["inverted_indicators"] = spec_show["inverted_indicators"].apply(_short_expr)
        if "theoretical_anchor" in spec_show.columns:
            spec_show["theoretical_anchor"] = spec_show["theoretical_anchor"].replace({
                "Input CI/Riedl : potentiel collectif et processus de base": "Input CI/Riedl",
                "TAS comportemental : attention partagée visuelle": "TAS : attention visuelle",
                "TMS comportemental : régulation des tours et coordination conversationnelle": "TMS : coord. conversationnelle",
                "TRS comportemental : synchronie cognitive et fluidité conversationnelle": "TRS : raisonnement coordonné",
            })
        # Pour le PDF : garder seulement les colonnes concises (exclure reason/evidence_key qui sont très longs)
        pdf_spec_cols = [c for c in ["construct", "indicators", "inverted_indicators", "theoretical_anchor"]
                         if c in spec_show.columns]
        lines.append("**Justification des composites formatifs**\n\n")
        lines.append(md_table_highlight(spec_show, max_rows=10))
        pdf_elems.append(Paragraph("Justification des composites formatifs", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(spec_show[pdf_spec_cols], max_rows=10))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    measurement = result.get("measurement")
    if isinstance(measurement, pd.DataFrame) and not measurement.empty:
        show = measurement[[
            c for c in ["construct", "indicator", "orientation", "weight", "loading_corr_with_composite", "n"]
            if c in measurement.columns
        ]].copy()
        if "indicator" in show.columns:
            show["indicator"] = show["indicator"].map(_short_name)
        if "loading_corr_with_composite" in show.columns:
            show = show.rename(columns={"loading_corr_with_composite": "loading"})
        lines.append("**Modèle de mesure : composites formatifs**\n\n")
        lines.append(md_table_highlight(show, max_rows=30))
        pdf_elems.append(Paragraph("Modèle de mesure : composites formatifs", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(show, max_rows=30))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    # Détection auto des composites réflexifs via VIF :
    # un composite dont au moins un indicateur a VIF > seuil est considéré réflexif
    # (multicolinéarité interne incompatible avec l'hypothèse formative).
    _VIF_REFLEXIVE_THRESHOLD = 5.0
    _reflexive_composites: set[str] = set()
    vif_df = result.get("vif")
    if isinstance(vif_df, pd.DataFrame) and not vif_df.empty:
        _vif_comp_col = next((c for c in ["composite", "construct"] if c in vif_df.columns), None)
        _vif_val_col = next((c for c in ["vif", "VIF"] if c in vif_df.columns), None)
        if _vif_comp_col and _vif_val_col:
            _over = vif_df[pd.to_numeric(vif_df[_vif_val_col], errors="coerce") > _VIF_REFLEXIVE_THRESHOLD]
            _reflexive_composites = set(_over[_vif_comp_col].dropna().unique())

    # Tableau alpha de Cronbach pour les composites réflexifs détectés via VIF.
    # L'alpha est calculé à la volée depuis les indicateurs du modèle de mesure
    # et les données merged_master (indicateurs INV comportementaux).
    if _reflexive_composites and measurement is not None and not measurement.empty and merged_master is not None:
        _meas_comp_col = next((c for c in ["construct", "composite"] if c in measurement.columns), None)
        _meas_ind_col = next((c for c in ["indicator"] if c in measurement.columns), None)
        if _meas_comp_col and _meas_ind_col:
            _alpha_rows: list[dict] = []
            for _comp in sorted(_reflexive_composites):
                _inds_raw = measurement[measurement[_meas_comp_col] == _comp][_meas_ind_col].tolist()
                # Les indicateurs peuvent être des noms courts (via _short_name) — chercher le nom long dans merged_master
                _inds = [_i for _i in _inds_raw if _i in merged_master.columns]
                if not _inds:
                    # Essayer les noms via reverse short_names
                    _rev = {v: k for k, v in short_names.items()}
                    _inds = [_rev.get(_i, _i) for _i in _inds_raw]
                    _inds = [_i for _i in _inds if _i in merged_master.columns]
                if len(_inds) < 2:
                    continue
                # Calcul alpha de Cronbach : α = (k/(k-1)) * (1 - Σvar_i / var_total)
                _mat = merged_master[_inds].apply(pd.to_numeric, errors="coerce").dropna()
                _n_obs = len(_mat)
                if _n_obs < 3:
                    continue
                _k = len(_inds)
                _var_items = _mat.var(ddof=1).sum()
                _var_total = _mat.sum(axis=1).var(ddof=1)
                _alpha_val = (_k / (_k - 1)) * (1 - _var_items / _var_total) if _var_total > 0 else np.nan
                _alpha_rows.append({
                    "Composite": _comp,
                    "Indicateurs": ", ".join(_inds_raw),
                    "n_items": _k,
                    "n": _n_obs,
                    "α de Cronbach": round(float(_alpha_val), 3) if np.isfinite(_alpha_val) else "NA",
                    "Interprétation": (
                        "acceptable" if np.isfinite(_alpha_val) and _alpha_val >= QUESTIONNAIRE_ALPHA_THRESHOLD
                        else "insuffisant" if np.isfinite(_alpha_val)
                        else "NA"
                    ),
                })
            if _alpha_rows:
                _alpha_df = pd.DataFrame(_alpha_rows)
                _refl_names = ", ".join(sorted(_reflexive_composites))
                _cronbach_note = (
                    f"Composites réflexifs détectés (VIF > {_VIF_REFLEXIVE_THRESHOLD:.0f}) : {_refl_names}. "
                    "L'alpha de Cronbach est calculé sur les indicateurs INV du composite. "
                    f"Seuil acceptable : α ≥ {QUESTIONNAIRE_ALPHA_THRESHOLD:.2f}."
                )
                lines.append(f"**Fiabilité interne des composites réflexifs — α de Cronbach ({_refl_names})**\n\n")
                lines.append(md_table_highlight(_alpha_df, max_rows=10))
                lines.append(f"\n_{_cronbach_note}_\n\n")
                pdf_elems.append(Paragraph(
                    f"Fiabilité interne des composites réflexifs — α de Cronbach ({_refl_names})",
                    styles["Heading4"]
                ))
                pdf_elems.append(pdf_table_from_df(_alpha_df, max_rows=10))
                pdf_elems.append(Paragraph(_cronbach_note, styles["Normal"]))
                pdf_elems.append(Spacer(1, 0.10 * inch))

    if isinstance(vif_df, pd.DataFrame) and not vif_df.empty:
        vif_show = vif_df.drop(columns=["threshold"], errors="ignore").copy()
        if "indicator" in vif_show.columns:
            vif_show["indicator"] = vif_show["indicator"].map(_short_name)
        lines.append("**VIF intra-composite (seuil < 5)**\n\n")
        lines.append(md_table_highlight(vif_show, max_rows=30))
        pdf_elems.append(Paragraph("VIF intra-composite (seuil < 5)", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(vif_show, max_rows=30))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    paths = result.get("paths")
    if isinstance(paths, pd.DataFrame) and not paths.empty:
        show_cols = [
            c for c in [
                "source", "target", "equation", "path_type", "path_coef_std",
                "ci95_low", "ci95_high", "n",
            ]
            if c in paths.columns
        ]
        path_show = _add_ci95(paths[show_cols].copy())
        if {"source", "target"}.issubset(path_show.columns):
            if "equation" in path_show.columns:
                path_show["path"] = path_show.apply(
                    lambda r: _short_expr(r["equation"])
                    if str(r.get("path_type", "")) == "adjusted_direct"
                    else f"{_short_name(r['source'])} -> {_short_name(r['target'])}",
                    axis=1,
                )
            else:
                path_show["path"] = path_show.apply(
                    lambda r: f"{_short_name(r['source'])} -> {_short_name(r['target'])}",
                    axis=1,
                )
            path_show = path_show.drop(columns=["source", "target", "equation"], errors="ignore")
        path_show = path_show.rename(columns={"path_coef_std": "beta"})
        if "beta" in path_show.columns:
            path_show["signe"] = path_show["beta"].apply(lambda v: "+" if pd.notna(v) and float(v) >= 0 else "-" if pd.notna(v) else "")
            path_show["magnitude"] = path_show["beta"].apply(lambda v: _effect_magnitude(float(v)) if pd.notna(v) else "NA")
        ordered_cols = [c for c in ["path", "path_type", "signe", "magnitude", "beta", "n", "IC95"] if c in path_show.columns]
        path_show = path_show[ordered_cols].copy()
        if "path_type" in path_show.columns:
            path_show = path_show.rename(columns={"path_type": "type"})
            path_show["type"] = path_show["type"].replace({"adjusted_direct": "adj.", "structure": "struct."})
        lines.append("**Coefficients de chemin standardisés avec IC bootstrap 95%**\n\n")
        lines.append(md_table_highlight(path_show, max_rows=90))
        pdf_elems.append(Paragraph("Coefficients de chemin standardisés avec IC bootstrap 95%", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(path_show, max_rows=90))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    r2_df = result.get("r2")
    if isinstance(r2_df, pd.DataFrame) and not r2_df.empty:
        show_cols = [c for c in ["target", "equation", "n", "r2", "n_predictors"] if c in r2_df.columns]
        r2_show = r2_df[show_cols].copy()
        if "target" in r2_show.columns:
            r2_show["target"] = r2_show["target"].map(_short_name)
        if "equation" in r2_show.columns:
            r2_show["model"] = r2_show["equation"].apply(_short_expr)
            r2_show = r2_show.drop(columns=["equation"], errors="ignore")
        lines.append("**R² des équations structurelles**\n\n")
        lines.append(md_table_highlight(r2_show, max_rows=60))
        pdf_elems.append(Paragraph("R² des équations structurelles", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(r2_show, max_rows=60))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    indirect = result.get("indirect")
    if isinstance(indirect, pd.DataFrame) and not indirect.empty:
        # Ne garder que les chemins à 1 médiateur (INPUT → INV_* → Performance).
        # Les chemins à 2 médiateurs (INPUT → INV_* → dim → Performance) ne sont pas
        # additifs en OLS séquentiel : la somme double-compte les β partagés.
        from sem.pls_sem_vr import PERFORMANCE, INPUT_COMPOSITE
        _perf_short = _short_name(PERFORMANCE)
        _input_short = _short_name(INPUT_COMPOSITE)
        indirect_1med = indirect[
            indirect["indirect_path"].str.count("->") == 2
        ].copy() if "indirect_path" in indirect.columns else indirect.copy()
        indirect_show = _add_ci95(indirect_1med.drop(columns=["warning"], errors="ignore").copy())
        if "indirect_path" in indirect_show.columns:
            indirect_show["indirect_path"] = indirect_show["indirect_path"].apply(_short_expr)
        lines.append("**Effets indirects (INPUT → INV_* → Performance) avec IC bootstrap 95%**\n\n")
        lines.append(md_table_highlight(indirect_show, max_rows=20))
        pdf_elems.append(Paragraph("Effets indirects (INPUT → INV_* → Performance) avec IC bootstrap 95%", styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(indirect_show, max_rows=20))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    # Décomposition effet total supprimée : la somme de tous les effets indirects
    # OLS séquentiels n'est pas additive (double-comptabilisation des β partagés).
    # Seuls les effets indirects par composite (INPUT→INV_*→Performance) sont interprétables.

    bivariate_corr = result.get("bivariate_corr")
    if isinstance(bivariate_corr, pd.DataFrame) and not bivariate_corr.empty:
        bivariate_show = bivariate_corr.copy()
        if "root" in bivariate_show.columns:
            bivariate_show["root"] = bivariate_show["root"].map(_short_name)
        if "target" in bivariate_show.columns:
            bivariate_show["target"] = bivariate_show["target"].map(_short_name)
        bivariate_show = _add_ci95(bivariate_show)
        keep_cols = [c for c in ["root", "target", "n", "rho_spearman", "IC95", "warning"] if c in bivariate_show.columns]
        bivariate_show = bivariate_show[keep_cols].copy()
        bivariate_show = bivariate_show.rename(columns={"root": "cause_racine", "rho_spearman": "rho (Spearman)"})
        bivariate_title = "Corrélations bivariées Spearman : causes racines individuelles → gaze_attention_coordination_idx"
        bivariate_note = (
            "Note : ces corrélations bivariées documentent la dilution des associations individuelles très fortes "
            "(RME : ρ ≈ −0.93 ; c_score : ρ ≈ −0.88) dans l'agrégation INPUT_composite à poids égaux. "
            "Limite inhérente à l'approche par composite formatif."
        )
        lines.append(f"**{bivariate_title}**\n\n")
        lines.append(md_table_highlight(bivariate_show, max_rows=20))
        lines.append(bivariate_note + "\n\n")
        pdf_elems.append(Paragraph(bivariate_title, styles["Heading4"]))
        pdf_elems.append(pdf_table_from_df(bivariate_show, max_rows=20))
        pdf_elems.append(Paragraph(bivariate_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.10 * inch))

    diagram_path = result.get("diagram_path")
    if diagram_path is not None and Path(diagram_path).exists():
        local_diagram = fig_dir / "path_analysis_vr_model.png"
        try:
            shutil.copyfile(Path(diagram_path), local_diagram)
            lines.append(f"![]({fig_dir.name}/{local_diagram.name})\n\n")
            pdf_elems.append(Image(str(local_diagram), width=6.4 * inch, height=3.7 * inch))
            pdf_elems.append(Spacer(1, 0.10 * inch))
        except Exception:
            pass

    limits = (
        "Note exploratoire : l'effectif VR reste très faible et certains chemins reposent sur n réduit "
        "lorsqu'ils impliquent le C-factor ou les indicateurs Riedl. Les IC bootstrap décrivent "
        "l'instabilité empirique des coefficients, mais ne transforment pas cette analyse en test "
        "confirmatoire. Aucune causalité forte ne doit être inférée ; les résultats servent à repérer "
        "des patrons d'association à répliquer."
    )
    lines.append("**Note d'interprétation**\n\n")
    lines.append(limits + "\n\n")
    pdf_elems.append(Paragraph("Note d'interprétation", styles["Heading4"]))
    pdf_elems.append(Paragraph(limits, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))


def render_sd_dispersion_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    merged_master: pd.DataFrame | None,
    results_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    section_num: str = "3.1.6",
    markdown_heading: str = "####",
    pdf_heading_style: str = "Heading4",
    add_page_break: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (df_paths, df_desc) pour la matrice de synthèse 3.1.8."""
    from sem.pls_sem_vr import (
        _load_individual_scores, _run_sd_augmented_regressions,
        INDIVIDUAL_DIMS, BOOTSTRAP_B,
    )
    title = f"{section_num} Dispersion intra-groupe comme prédicteur complémentaire"
    lines.append(f"{markdown_heading} {title}\n\n")
    if add_page_break:
        pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))

    intro = (
        "Suivant Aguinis, Gottfredson & Culpepper (2013) et la logique de décomposition variance "
        "between/within (Enders & Tofighi 2007), la dispersion intra-groupe (écart-type SD) des "
        "réponses individuelles aux états émergents capture le degré de désaccord perceptuel. "
        "Un désaccord élevé peut signaler un défaut d'émergence du construit collectif, "
        "indépendamment du niveau moyen. Cette approche permet d'exploiter la richesse multiniveau "
        "sans MLM formel, préférable compte tenu de K=8 groupes VR S2 disponibles. "
        "Limite principale : N_groupes = 8 après restriction aux groupes ayant des données "
        "individuelles S2 (groupes S1 bim006, bim010, bim066, bim073_2 exclus)."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    q_path = results_dir / "questionnaire" / "analyse" / "scores_dimension_par_participant.csv"
    group_ids = list(merged_master["group_id"].astype(str)) if merged_master is not None else None
    df_ind, df_stats, note_load = _load_individual_scores(q_path, group_ids_filter=group_ids)

    lines.append(f"*{note_load}*\n\n")
    pdf_elems.append(Paragraph(note_load, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.06 * inch))

    if df_stats.empty:
        msg = "Données individuelles insuffisantes pour cette analyse."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return pd.DataFrame(), pd.DataFrame()

    # Tableau descriptif SD intra-groupe
    desc_title = "Descriptifs de la dispersion intra-groupe (SD) par dimension"
    lines.append(f"**{desc_title}**\n\n")
    pdf_elems.append(Paragraph(desc_title, styles["Heading4"]))
    lines.append(md_table_highlight(df_stats, max_rows=40))
    pdf_elems.append(pdf_table_from_df(df_stats, max_rows=40))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # Régressions augmentées mean+sd → performance + diagnostic ρ(mean, SD)
    if merged_master is not None and not df_stats.empty:
        df_paths, df_desc = _run_sd_augmented_regressions(df_stats, merged_master, n_boot=BOOTSTRAP_B)

        # --- Diagnostic ρ(mean, SD) — confound floor/ceiling (Aguinis et al. 2013) ---
        if not df_desc.empty:
            corr_title = "Diagnostic colinéarité mean ↔ SD (effet plafond/plancher)"
            lines.append(f"**{corr_title}**\n\n")
            pdf_elems.append(Paragraph(corr_title, styles["Heading4"]))

            corr_intro = (
                "Si ρ(mean_X, SD_X) est fort (|ρ| > 0.70), la dispersion n'est pas indépendante du "
                "niveau moyen (typique sur Likert borné 1–5 : effet plancher comprime la SD aux scores "
                "bas, effet plafond aux scores élevés). Dans ce cas, le coefficient de SD en régression "
                "multiple absorbe en partie l'effet de la moyenne et l'interprétation de l'effet propre "
                "de la dispersion est fragile. Seuils : |ρ| ≤ 0.50 = ok ; 0.50 < |ρ| ≤ 0.70 = "
                "CONFOUND_MODERE (à mentionner) ; |ρ| > 0.70 = CONFOUND_FORT (résultats non interprétables)."
            )
            lines.append(corr_intro + "\n\n")
            pdf_elems.append(Paragraph(corr_intro, styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.06 * inch))

            corr_cols = [c for c in ["dimension", "n_groupes", "rho_mean_vs_sd", "p_rho_mean_vs_sd",
                                      "confound_flag"] if c in df_desc.columns]
            corr_show = df_desc[corr_cols].copy()
            lines.append(md_table_highlight(corr_show, max_rows=10))
            pdf_elems.append(pdf_table_from_df(corr_show, max_rows=10))
            pdf_elems.append(Spacer(1, 0.08 * inch))

            # Avertissement si au moins une dimension en confound fort
            forte_dims = df_desc[df_desc.get("confound_flag", pd.Series(dtype=str)) == "CONFOUND_FORT"]["dimension"].tolist() if "confound_flag" in df_desc.columns else []
            if forte_dims:
                warn_txt = (
                    f"⚠ Confound fort détecté pour : {', '.join(forte_dims)}. "
                    "Les coefficients sd_X correspondants dans le tableau ci-dessous ne sont pas "
                    "interprétables comme effets propres de la dispersion ; ils reflètent en partie "
                    "la colinéarité avec la moyenne."
                )
                lines.append(f"*{warn_txt}*\n\n")
                pdf_elems.append(Paragraph(warn_txt, styles["Normal"]))
                pdf_elems.append(Spacer(1, 0.06 * inch))

        # --- Tableau des coefficients sd_X → performance ---
        if not df_paths.empty:
            reg_title = "Effet propre de la SD intra-groupe sur la performance (modèle mean + sd)"
            lines.append(f"**{reg_title}**\n\n")
            pdf_elems.append(Paragraph(reg_title, styles["Heading4"]))

            show_cols = [c for c in ["dimension", "source", "path_coef_std", "ci95_low", "ci95_high",
                                      "r2_model1", "r2_model2", "delta_r2", "n", "warning"] if c in df_paths.columns]
            show = df_paths[show_cols].copy()
            for col in ["path_coef_std", "ci95_low", "ci95_high", "r2_model1", "r2_model2", "delta_r2"]:
                if col in show.columns:
                    show[col] = pd.to_numeric(show[col], errors="coerce").round(4)

            lines.append(md_table_highlight(show, max_rows=20))
            pdf_elems.append(pdf_table_from_df(show, max_rows=20))
            pdf_elems.append(Spacer(1, 0.08 * inch))

            # Sauvegarder CSV
            out_csv = results_dir / "sem" / "path_analysis_vr" / "section316_sd_dispersion_paths.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df_paths.to_csv(out_csv, index=False, encoding="utf-8-sig")
            if not df_desc.empty:
                df_desc.to_csv(
                    results_dir / "sem" / "path_analysis_vr" / "section316_mean_sd_collinearity.csv",
                    index=False, encoding="utf-8-sig",
                )

        interp_note = (
            "Note d'interprétation : un coefficient sd_X robuste (IC bootstrap excluant zéro) indique "
            "que le désaccord perceptuel intra-groupe sur la dimension X prédit la performance "
            "au-delà du niveau moyen. delta_R² mesure l'apport marginal de la SD. "
            "Avec K=8 groupes, la puissance est très faible pour les effets modérés (d < 0.50). "
            "Ces résultats sont exploratoires et ne constituent pas un test confirmatoire."
        )
        lines.append(interp_note + "\n\n")
        pdf_elems.append(Paragraph(interp_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))
        return df_paths, df_desc
    return pd.DataFrame(), pd.DataFrame()


def render_mlm_icc_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    merged_master: pd.DataFrame | None,
    results_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    section_num: str = "3.1.7",
    markdown_heading: str = "####",
    pdf_heading_style: str = "Heading4",
    add_page_break: bool = True,
    bayes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (df_icc, df_ctx) pour la matrice de synthèse 3.1.6 — Voie A."""
    from sem.pls_sem_vr import (
        _load_individual_scores, _run_mlm_icc_analysis,
        INDIVIDUAL_DIMS, BOOTSTRAP_B,
    )
    title = (
        f"{section_num} Structure multiniveau des perceptions individuelles — "
        "ICC et effets contextuels des composites transactifs"
    )
    lines.append(f"{markdown_heading} {title}\n\n")
    if add_page_break:
        pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))

    intro = (
        "Les états émergents collectifs (TMS : COR, CRE, SPE ; cohésion : SOC, TSK, COM) sont mesurés au "
        "niveau individuel puis agrégés au niveau groupe dans les analyses précédentes. Cette section "
        "exploite la structure multiniveau authentique (réponses individuelles nichées dans les groupes) "
        "pour deux objectifs indépendants de toute spécification structurelle : (1) quantifier la part de "
        "variance des perceptions individuelles attribuable au groupe via l'ICC (modèle nul), justifiant "
        "ou non l'agrégation employée ailleurs (Bliese 2000) ; (2) estimer les effets contextuels des "
        "trois composites comportementaux transactifs (INV_TAS, INV_TMS, INV_TRS) sur les perceptions "
        "individuelles, c'est-à-dire la transmission groupe → individu des dynamiques comportementales. "
        "L'affect négatif est traité comme indicateur du composite INV_TRS (cf. 3.1.3) et n'est donc pas "
        "isolé ici. Avec K = 12 groupes VR (11 pour INV_TMS) et n = 3 individus/groupe, l'inférence "
        "bayésienne régularise les estimations mais ne compense pas un K aussi faible (Maas & Hox 2005 : "
        "K ≥ 30 recommandé ; McNeish & Stapleton 2016). Tous les résultats sont explicitement exploratoires."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    interp_note = (
        "Note d'interprétation : les composites INV étant des prédicteurs purement de niveau 2 "
        "(valeur constante au sein d'un groupe), beta_std estime l'effet entre-groupes des dynamiques "
        "comportementales sur la perception individuelle moyenne (modèle « means-as-outcomes ») ; "
        "il n'existe pas de variance intra-groupe sur le prédicteur. L'apport vis-à-vis de la path "
        "analysis 3.1.3 (niveau groupe agrégé) est la partition explicite de la variance intra/inter-groupe "
        "et des intervalles de crédibilité valides à petit K."
    )
    lines.append(interp_note + "\n\n")
    pdf_elems.append(Paragraph(interp_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    q_path = results_dir / "questionnaire" / "analyse" / "scores_dimension_par_participant.csv"
    group_ids = list(merged_master["group_id"].astype(str)) if merged_master is not None else None
    df_ind, _, note_load = _load_individual_scores(q_path, group_ids_filter=group_ids)

    lines.append(f"*{note_load}*\n\n")
    pdf_elems.append(Paragraph(note_load, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.06 * inch))

    if df_ind.empty or merged_master is None:
        msg = "Données individuelles insuffisantes pour l'analyse ICC."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return pd.DataFrame(), pd.DataFrame()

    _bayes_icc_path = results_dir / "sem" / "path_analysis_vr" / "section315_icc_bayes.csv"
    _bayes_ctx_path = results_dir / "sem" / "path_analysis_vr" / "section315_contextual_bayes.csv"
    _bayes_sens_path = results_dir / "sem" / "path_analysis_vr" / "section315_sensitivity_bayes.csv"

    df_icc, df_ctx, note_impl = _run_mlm_icc_analysis(df_ind, merged_master, n_boot=BOOTSTRAP_B, bayes=bayes)

    def _is_bayes_degraded(df: pd.DataFrame, ctx: bool = False) -> bool:
        if df.empty:
            return True
        if "method" in df.columns:
            methods = df["method"].astype(str)
            if methods.str.contains("ANOVA_point_only|insuffisant", regex=True).all():
                return True
        if ctx:
            bayes_cols = [c for c in ["beta_std", "hdi95_low"] if c in df.columns]
        else:
            bayes_cols = [c for c in ["ICC_bayes_sd", "ICC_hdi95_low"] if c in df.columns]
        if bayes_cols and df[bayes_cols].isna().all().all():
            return True
        return False

    _used_cache = False
    _cache_date: str | None = None

    def _load_cache(path: Path) -> tuple[pd.DataFrame, str | None]:
        try:
            cached = _read_csv(path)
            if "compiled_at" in cached.columns:
                date = str(cached["compiled_at"].iloc[0])
            else:
                import datetime as _dt
                date = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            return cached, date
        except Exception:
            return pd.DataFrame(), None

    if _is_bayes_degraded(df_icc) and _bayes_icc_path.exists():
        cached_icc, _cache_date = _load_cache(_bayes_icc_path)
        if not _is_bayes_degraded(cached_icc):
            df_icc = cached_icc
            _used_cache = True

    if _is_bayes_degraded(df_ctx, ctx=True) and _bayes_ctx_path.exists():
        cached_ctx, _cache_date_ctx = _load_cache(_bayes_ctx_path)
        if not _is_bayes_degraded(cached_ctx, ctx=True):
            df_ctx = cached_ctx
            if _cache_date is None:
                _cache_date = _cache_date_ctx
            _used_cache = True

    if _used_cache and _cache_date:
        note_impl = note_impl + f" ⚠ Résultats bayésiens issus du cache (dernière compilation : {_cache_date})."

    lines.append(f"*{note_impl}*\n\n")
    pdf_elems.append(Paragraph(note_impl, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.06 * inch))

    from sem.pls_sem_vr import MLM_MCMC_CONFIG
    rhat_thresh = MLM_MCMC_CONFIG.get("rhat_threshold", 1.01)
    ess_thresh = MLM_MCMC_CONFIG.get("ess_threshold", 400)

    # --- Table A : ICC modèle nul ---
    if not df_icc.empty:
        icc_title = "Table A — ICC par dimension (modèle nul bayésien, intercepts aléatoires, PyMC 5)"
        lines.append(f"**{icc_title}**\n\n")
        pdf_elems.append(Paragraph(icc_title, styles["Heading4"]))

        icc_cols = [c for c in [
            "dimension", "K_groupes", "N_individus",
            "ICC_ANOVA", "ICC_bayes_mean", "ICC_bayes_sd",
            "ICC_hdi95_low", "ICC_hdi95_high", "Rhat_null", "ESS_null",
        ] if c in df_icc.columns]
        icc_show = df_icc[icc_cols].copy()
        for col in ["ICC_ANOVA", "ICC_bayes_mean", "ICC_bayes_sd", "ICC_hdi95_low", "ICC_hdi95_high", "Rhat_null"]:
            if col in icc_show.columns:
                icc_show[col] = pd.to_numeric(icc_show[col], errors="coerce").round(4)

        lines.append(md_table_highlight(icc_show, max_rows=10))
        pdf_elems.append(pdf_table_from_df(icc_show, max_rows=10))

        if "Rhat_null" in df_icc.columns:
            bad_rhat = df_icc[pd.to_numeric(df_icc["Rhat_null"], errors="coerce") > rhat_thresh]["dimension"].tolist()
            if bad_rhat:
                w = f"⚠ Convergence MCMC insuffisante (Rhat > {rhat_thresh}) pour : {', '.join(bad_rhat)}."
                lines.append(f"*{w}*\n\n")
                pdf_elems.append(Paragraph(w, styles["Normal"]))
        if "ESS_null" in df_icc.columns:
            low_ess = df_icc[pd.to_numeric(df_icc["ESS_null"], errors="coerce") < ess_thresh]["dimension"].tolist()
            if low_ess:
                w2 = f"⚠ ESS faible (< {ess_thresh}) pour : {', '.join(low_ess)}."
                lines.append(f"*{w2}*\n\n")
                pdf_elems.append(Paragraph(w2, styles["Normal"]))

        icc_note = (
            "Lecture : ICC_ANOVA = valeur point analytique (référence). ICC_bayes_mean = moyenne "
            "postérieure. ICC_hdi95 = HDI 95%. Seuil agrégation : ICC ≥ 0.30 (Bliese 2000). "
            "Rhat < 1.01 et ESS > 400 indiquent une bonne convergence."
        )
        lines.append(icc_note + "\n\n")
        pdf_elems.append(Paragraph(icc_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

        if not _used_cache and not _is_bayes_degraded(df_icc):
            import datetime as _dt
            _now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
            df_icc_save = df_icc.copy()
            df_icc_save["compiled_at"] = _now_str
            (_bayes_icc_path.parent).mkdir(parents=True, exist_ok=True)
            df_icc_save.to_csv(_bayes_icc_path, index=False, encoding="utf-8-sig")

    # --- Table B : effets contextuels des 3 composites ---
    if not df_ctx.empty:
        ctx_title = (
            "Table B — Effets contextuels des composites transactifs (18 modèles : "
            "3 composites × 6 dimensions)"
        )
        lines.append(f"**{ctx_title}**\n\n")
        pdf_elems.append(Paragraph(ctx_title, styles["Heading4"]))

        ctx_b_cols = [c for c in [
            "composite", "dimension",
            "beta_std", "beta_sd", "hdi95_low", "hdi95_high",
            "robust_hdi", "Rhat", "ESS", "K_groupes",
        ] if c in df_ctx.columns]
        ctx_show = df_ctx[ctx_b_cols].copy()
        for col in ["beta_std", "beta_sd", "hdi95_low", "hdi95_high", "Rhat"]:
            if col in ctx_show.columns:
                ctx_show[col] = pd.to_numeric(ctx_show[col], errors="coerce").round(4)

        lines.append(md_table_highlight(ctx_show, max_rows=20))
        pdf_elems.append(pdf_table_from_df(ctx_show, max_rows=20))

        # Avertissements convergence
        if "Rhat" in df_ctx.columns:
            bad = df_ctx[pd.to_numeric(df_ctx["Rhat"], errors="coerce") > rhat_thresh][["composite", "dimension"]].apply(
                lambda r: f"{r['composite']} × {r['dimension']}", axis=1
            ).tolist()
            if bad:
                w = f"⚠ Rhat > {rhat_thresh} pour : {', '.join(bad)}."
                lines.append(f"*{w}*\n\n")
                pdf_elems.append(Paragraph(w, styles["Normal"]))
        if "ESS" in df_ctx.columns:
            low = df_ctx[pd.to_numeric(df_ctx["ESS"], errors="coerce") < ess_thresh][["composite", "dimension"]].apply(
                lambda r: f"{r['composite']} × {r['dimension']}", axis=1
            ).tolist()
            if low:
                w2 = f"⚠ ESS < {ess_thresh} pour : {', '.join(low)}."
                lines.append(f"*{w2}*\n\n")
                pdf_elems.append(Paragraph(w2, styles["Normal"]))

        ctx_note = (
            "beta_std = coefficient standardisé du composite L2 sur les perceptions individuelles, "
            "après contrôle de la structure aléatoire intra-groupe. robust_hdi = True si le HDI 95% "
            "exclut zéro. Paramétrisation non-centrée obligatoire à petit K (Bürkner 2017). "
            "Variables standardisées avant échantillonnage ; beta_std interprétable comme corrélation partielle."
        )
        lines.append(ctx_note + "\n\n")
        pdf_elems.append(Paragraph(ctx_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    # --- Table C : sensibilité aux priors ---
    if not df_ctx.empty and "beta_N01" in df_ctx.columns:
        sens_title = "Table C — Sensibilité aux priors : N(0,1) vs N(0,0.5)"
        lines.append(f"**{sens_title}**\n\n")
        pdf_elems.append(Paragraph(sens_title, styles["Heading4"]))

        sens_cols = [c for c in [
            "composite", "dimension",
            "beta_N01", "robust_N01",
            "beta_N005", "robust_N005",
            "prior_stable",
        ] if c in df_ctx.columns]
        sens_show = df_ctx[sens_cols].copy()
        if "beta_N01" in sens_show.columns:
            sens_show["beta_N01"] = pd.to_numeric(sens_show["beta_N01"], errors="coerce").round(4)
        if "beta_N005" in sens_show.columns:
            sens_show["beta_N005"] = pd.to_numeric(sens_show["beta_N005"], errors="coerce").round(4)

        lines.append(md_table_highlight(sens_show, max_rows=20))
        pdf_elems.append(pdf_table_from_df(sens_show, max_rows=20))

        sens_note = (
            "prior_stable = True si le signe et la robustesse (robust_hdi) sont identiques sous les deux "
            "priors. Les effets prior_stable = True offrent une plus grande confiance dans les conclusions "
            "malgré le faible K. Les effets prior_stable = False sont à considérer uniquement comme pistes."
        )
        lines.append(sens_note + "\n\n")
        pdf_elems.append(Paragraph(sens_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    # Sauvegarde cache
    if not _used_cache and not _is_bayes_degraded(df_ctx, ctx=True):
        import datetime as _dt
        _now_str = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        (_bayes_ctx_path.parent).mkdir(parents=True, exist_ok=True)
        df_ctx_save = df_ctx.copy()
        df_ctx_save["compiled_at"] = _now_str
        df_ctx_save.to_csv(_bayes_ctx_path, index=False, encoding="utf-8-sig")
        # Séparé : table de sensibilité
        if "beta_N01" in df_ctx.columns:
            df_ctx_save.to_csv(_bayes_sens_path, index=False, encoding="utf-8-sig")

    limits = (
        "Limites : K = 12 groupes VR (11 pour INV_TMS), n = 3 individus/groupe — l'inférence "
        "bayésienne régularise mais ne compense pas un K aussi faible (Maas & Hox 2005 : K ≥ 30 "
        "recommandé). Les composites INV sont des prédicteurs purement de niveau 2 ; la stabilité "
        "aux priors est rapportée en Table C. Sans g++ (PyTensor mode Python pur), l'échantillonnage "
        "est lent. Résultats exploratoires — non confirmatoires."
    )
    lines.append(limits + "\n\n")
    pdf_elems.append(Paragraph(limits, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))
    return df_icc, df_ctx


def _build_multilevel_synthesis_matrix(
    df_icc: pd.DataFrame,
    df_ctx: pd.DataFrame,
    df_sd_paths: pd.DataFrame,
    df_sd_desc: pd.DataFrame,
) -> pd.DataFrame:
    """Construit la matrice de synthèse multiniveau — Voie A.

    Colonnes :
      Dimension | ICC (IC95%) | SD → perf (β, IC95%, ΔR²)
      | Contextuel INV_TAS (β, IC95) | Contextuel INV_TMS (β, IC95)
      | Contextuel INV_TRS (β, IC95) | Convergence
    """
    from sem.pls_sem_vr import INDIVIDUAL_DIMS

    composite_names = ["INV_TAS", "INV_TMS", "INV_TRS"]

    def _ctx_str_for(dim: str, cname: str) -> tuple[str, bool | None]:
        if df_ctx.empty or "composite" not in df_ctx.columns:
            return "n/d", None
        mask = (df_ctx["composite"] == cname) & (df_ctx["dimension"] == dim)
        if not mask.any():
            return "n/d", None
        row = df_ctx[mask].iloc[0]
        beta = row.get("beta_std", np.nan)
        hdi_lo = row.get("hdi95_low", np.nan)
        hdi_hi = row.get("hdi95_high", np.nan)
        rob = row.get("robust_hdi", None)
        stable = row.get("prior_stable", None)
        if all(np.isfinite(v) for v in [float(beta) if beta is not None else np.nan,
                                         float(hdi_lo) if hdi_lo is not None else np.nan,
                                         float(hdi_hi) if hdi_hi is not None else np.nan]):
            rob_flag = bool(rob) if rob is not None else ((float(hdi_lo) > 0) or (float(hdi_hi) < 0))
            stable_flag = bool(stable) if stable is not None else False
            suffix = "✓" if (rob_flag and stable_flag) else ("~" if rob_flag else "")
            s = f"β={float(beta):+.3f} [{float(hdi_lo):+.3f}, {float(hdi_hi):+.3f}]{suffix}"
            return s, rob_flag
        return "n/d", None

    rows = []
    for dim in INDIVIDUAL_DIMS:
        # --- ICC ---
        icc_row = df_icc[df_icc["dimension"] == dim].iloc[0] if not df_icc.empty and (df_icc["dimension"] == dim).any() else None
        if icc_row is not None:
            icc_val = icc_row.get("ICC_bayes_mean", icc_row.get("ICC_ANOVA", np.nan))
            icc_lo = icc_row.get("ICC_hdi95_low", np.nan)
            icc_hi = icc_row.get("ICC_hdi95_high", np.nan)
            icc_anova = icc_row.get("ICC_ANOVA", np.nan)
            if all(np.isfinite(v) for v in [icc_val, icc_lo, icc_hi]):
                icc_ok = float(icc_lo) >= 0.10
                icc_str = f"{icc_val:.3f} [{icc_lo:.3f}, {icc_hi:.3f}]"
                if np.isfinite(icc_anova):
                    icc_str += f" (ANOVA={icc_anova:.3f})"
            elif np.isfinite(icc_anova):
                icc_str = f"{icc_anova:.3f} (ANOVA)"
                icc_ok = icc_anova >= 0.30
            else:
                icc_str = "n/d"
                icc_ok = None
        else:
            icc_str = "n/d"
            icc_ok = None

        # --- SD → perf ---
        sd_row = df_sd_paths[df_sd_paths["dimension"] == dim].iloc[0] if not df_sd_paths.empty and (df_sd_paths["dimension"] == dim).any() else None
        sd_desc_row = df_sd_desc[df_sd_desc["dimension"] == dim].iloc[0] if not df_sd_desc.empty and (df_sd_desc["dimension"] == dim).any() else None
        if sd_row is not None:
            beta_sd = sd_row.get("path_coef_std", np.nan)
            ci_lo = sd_row.get("ci95_low", np.nan)
            ci_hi = sd_row.get("ci95_high", np.nan)
            dr2 = sd_row.get("delta_r2", np.nan)
            if all(np.isfinite(v) for v in [beta_sd, ci_lo, ci_hi]):
                robust = (ci_lo > 0) or (ci_hi < 0)
                dr2_str = f"ΔR²={dr2:.3f}" if np.isfinite(dr2) else ""
                sd_str = f"β={beta_sd:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}] {dr2_str}"
                sd_ok = robust
            else:
                sd_str = "n/d"
                sd_ok = None
        else:
            sd_str = "n/d"
            sd_ok = None

        confound = ""
        if sd_desc_row is not None and "confound_flag" in df_sd_desc.columns:
            flag = sd_desc_row.get("confound_flag", "ok")
            if flag != "ok":
                confound = f" [{flag}]"
        sd_str += confound

        # --- Effets contextuels 3 composites ---
        ctx_strings = {}
        ctx_robs = {}
        for cname in composite_names:
            s, r = _ctx_str_for(dim, cname)
            ctx_strings[cname] = s
            ctx_robs[cname] = r

        # --- Convergence : ICC ≥ 0.30, SD robuste, ≥ 1 composite contextuel robuste ---
        ctx_any_robust = any(v is True for v in ctx_robs.values())
        signals = [x for x in [icc_ok, sd_ok, ctx_any_robust] if x is not None]
        n_robust = sum(1 for x in signals if x)
        if len(signals) == 0:
            convergence = "données insuffisantes"
        elif n_robust >= 2:
            convergence = "convergent"
        elif n_robust == 1:
            convergence = "partiel"
        else:
            convergence = "divergent"

        rows.append({
            "Dimension": dim,
            "ICC (IC 95%)": icc_str,
            "SD → perf (β, IC 95%, ΔR²)": sd_str,
            "Contextuel INV_TAS (β, IC95)": ctx_strings["INV_TAS"],
            "Contextuel INV_TMS (β, IC95)": ctx_strings["INV_TMS"],
            "Contextuel INV_TRS (β, IC95)": ctx_strings["INV_TRS"],
            "Convergence": convergence,
        })

    return pd.DataFrame(rows)


def render_multilevel_synthesis_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    df_icc: pd.DataFrame | None = None,
    df_ctx: pd.DataFrame | None = None,
    df_sd_paths: pd.DataFrame | None = None,
    df_sd_desc: pd.DataFrame | None = None,
    section_num: str = "3.1.8",
    markdown_heading: str = "####",
    pdf_heading_style: str = "Heading4",
    add_page_break: bool = True,
) -> None:
    title = f"{section_num} Synthèse multiniveau — articulation des trois niveaux d'analyse"
    lines.append(f"{markdown_heading} {title}\n\n")
    if add_page_break:
        pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))

    synth = (
        "Les trois analyses complémentaires (3.1.3, 3.1.4, 3.1.5) éclairent des aspects distincts "
        "de la structure hiérarchique des données VR. "
        "Section 3.1.3 (path analysis, niveau groupe) : chemins standardisés entre composites "
        "formatifs transactifs, états émergents collectifs et performance (K=12 VR). "
        "Section 3.1.4 (SD intra-groupe) : effet propre du désaccord perceptuel intra-groupe "
        "au-delà du niveau moyen sur la performance (Enders & Tofighi 2007). "
        "Section 3.1.5 (ICC et effets contextuels) : proportion de variance des perceptions "
        "individuelles attribuable aux groupes, et transmission groupe→individu via les composites "
        "comportementaux transactifs (Maas & Hox 2005). "
        "La matrice ci-dessous résume les indicateurs pour chaque dimension TMS/cohésion. "
        "Un ICC ≥ 0.30 valide l'agrégation utilisée en 3.1.3 (Bliese 2000). "
        "Un effet SD robuste (IC excluant zéro) signale une hétérogénéité perceptuelle prédictive. "
        "Un effet contextuel robuste (✓ = robuste et prior_stable, ~ = robuste sans stabilité prior) "
        "signale une transmission groupe→individu de la dynamique comportementale correspondante. "
        "La convergence compte les indicateurs robustes parmi {ICC ≥ 0.30, SD → perf robuste, "
        "≥ 1 composite contextuel robuste} (≥ 2/3 = convergent)."
    )
    lines.append(synth + "\n\n")
    pdf_elems.append(Paragraph(synth, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # Matrice de synthèse
    df_icc_ = df_icc if df_icc is not None else pd.DataFrame()
    df_ctx_ = df_ctx if df_ctx is not None else pd.DataFrame()
    df_sd_paths_ = df_sd_paths if df_sd_paths is not None else pd.DataFrame()
    df_sd_desc_ = df_sd_desc if df_sd_desc is not None else pd.DataFrame()

    df_matrix = _build_multilevel_synthesis_matrix(df_icc_, df_ctx_, df_sd_paths_, df_sd_desc_)

    mat_title = "Matrice de synthèse multiniveau par dimension (6 dimensions × 6 indicateurs)"
    lines.append(f"**{mat_title}**\n\n")
    pdf_elems.append(Paragraph(mat_title, styles["Heading4"]))
    lines.append(md_table_highlight(df_matrix, max_rows=10))
    pdf_elems.append(pdf_table_from_df(df_matrix, max_rows=10))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # Lecture automatique des convergences
    if not df_matrix.empty and "Convergence" in df_matrix.columns:
        conv_dims = df_matrix[df_matrix["Convergence"] == "convergent"]["Dimension"].tolist()
        part_dims = df_matrix[df_matrix["Convergence"] == "partiel"]["Dimension"].tolist()
        div_dims = df_matrix[df_matrix["Convergence"] == "divergent"]["Dimension"].tolist()
        reading_parts = []
        if conv_dims:
            reading_parts.append(f"Convergence multi-indicateurs (≥ 2/3 robustes) : {', '.join(conv_dims)}.")
        if part_dims:
            reading_parts.append(f"Signal partiel (1/3 robuste) : {', '.join(part_dims)}.")
        if div_dims:
            reading_parts.append(f"Absence de signal robuste : {', '.join(div_dims)}.")
        if reading_parts:
            reading = " ".join(reading_parts)
            lines.append(reading + "\n\n")
            pdf_elems.append(Paragraph(reading, styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.06 * inch))

    footer = (
        "Limites communes : K = 12 groupes VR (11 pour INV_TMS), n = 3 individus/groupe. "
        "Les effets contextuels reposent sur des prédicteurs purement de niveau 2 ; "
        "la stabilité aux priors (N(0,1) vs N(0,0.5)) est rapportée dans la Table C de la section 3.1.5. "
        "L'inférence bayésienne régularise mais ne compense pas un K aussi faible. "
        "Tous résultats exploratoires — non confirmatoires. "
        "Références : Aguinis et al. (2013), Enders & Tofighi (2007), Maas & Hox (2005), "
        "McNeish & Stapleton (2016), Bürkner (2017), Gelman et al. (2013), Bliese (2000)."
    )
    lines.append(footer + "\n\n")
    pdf_elems.append(Paragraph(footer, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.10 * inch))


_MD_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def _rewrite_markdown_image_paths(markdown_text: str, *, fig_dir: Path, out_dir: Path) -> str:
    """
    Remappe les images Markdown vers le sous-dossier de figures du rapport.

    Plusieurs sections injectent seulement le nom du fichier (`image.png`) alors
    que les figures sont ecrites dans `figs_<report>/`. Ce post-traitement garde
    les references existantes mais les rend resolubles depuis le Markdown/HTML.
    """
    if not markdown_text:
        return markdown_text

    try:
        rel_fig_dir = fig_dir.relative_to(out_dir).as_posix()
    except ValueError:
        rel_fig_dir = fig_dir.name

    def _replace(match: re.Match[str]) -> str:
        alt = match.group("alt")
        src = match.group("src").strip()
        if not src:
            return match.group(0)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        normalized_src = src.replace("\\", "/")
        if "/" in normalized_src:
            return match.group(0)

        candidate = fig_dir / src
        if not candidate.exists():
            return match.group(0)

        return f"![{alt}]({rel_fig_dir}/{src})"

    return _MD_IMAGE_PATTERN.sub(_replace, markdown_text)


def _detect_questionnaire_meta_cols(df: pd.DataFrame) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for target, keywords in [
        ("Session", ["Session"]),
        ("Modalite", ["Modalité", "Modalite"]),
        ("Scenario", ["Scénario", "Scenario"]),
        ("Groupe", ["Groupe"]),
    ]:
        found = [c for c in df.columns if any(k in str(c) for k in keywords)]
        mapping[target] = found[0] if found else None
    return mapping


def _normalize_questionnaire_timepoint(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m:
        return f"T{int(m.group(1))}"
    return s


def _normalize_questionnaire_meta(value: Any, kind: str) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if kind == "group":
        return s.lower()
    if kind == "timepoint":
        return _normalize_questionnaire_timepoint(s)
    if kind == "modalite":
        s_up = s.upper().replace(" ", "")
        if "VR" in s_up:
            return "VR"
        if "PC" in s_up:
            return "PC"
        return s_up
    if kind == "scenario":
        s_up = s.upper().replace(" ", "")
        if "S1" in s_up:
            return "S1"
        if "S2" in s_up:
            return "S2"
        return s_up
    return s


def _build_questionnaire_raw_keys(df: pd.DataFrame) -> pd.Series:
    meta = _detect_questionnaire_meta_cols(df)
    group_col = meta.get("Groupe")
    session_col = meta.get("Session")
    modalite_col = meta.get("Modalite")
    scenario_col = meta.get("Scenario")

    if group_col is None:
        return pd.Series([""] * len(df), index=df.index, dtype=str)

    group_vals = df[group_col].apply(lambda x: _normalize_questionnaire_meta(x, "group"))
    time_vals = (
        df[session_col].apply(lambda x: _normalize_questionnaire_meta(x, "timepoint"))
        if session_col in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=str)
    )
    modalite_vals = (
        df[modalite_col].apply(lambda x: _normalize_questionnaire_meta(x, "modalite"))
        if modalite_col in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=str)
    )
    scenario_vals = (
        df[scenario_col].apply(lambda x: _normalize_questionnaire_meta(x, "scenario"))
        if scenario_col in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=str)
    )
    return group_vals + "|" + time_vals + "|" + modalite_vals + "|" + scenario_vals


def _build_questionnaire_excluded_keys(excluded_groups_df: pd.DataFrame) -> set[str]:
    if excluded_groups_df is None or excluded_groups_df.empty:
        return set()

    keys: set[str] = set()
    for _, row in excluded_groups_df.iterrows():
        group_id = _normalize_questionnaire_meta(row.get("group_id"), "group")
        timepoint = _normalize_questionnaire_meta(row.get("timepoint"), "timepoint")
        modalite = _normalize_questionnaire_meta(row.get("modalite"), "modalite")
        scenario = _normalize_questionnaire_meta(row.get("scenario"), "scenario")
        keys.add(f"{group_id}|{timepoint}|{modalite}|{scenario}")
    return keys


def _build_context_keys(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=str)
    cols = []
    for col in ["group_id", "timepoint", "modalite", "scenario"]:
        if col in df.columns:
            vals = df[col].where(pd.notna(df[col]), "").astype(str).str.strip().str.lower()
        else:
            vals = pd.Series([""] * len(df), index=df.index, dtype=str)
        cols.append(vals)
    out = cols[0]
    for col in cols[1:]:
        out = out + "|" + col
    return out


def _filter_context_df_by_questionnaire_exclusions(
    df: pd.DataFrame,
    excluded_groups_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if df is None or df.empty or excluded_groups_df is None or excluded_groups_df.empty:
        return pd.DataFrame() if df is None else df.copy()
    excluded_keys = _build_questionnaire_excluded_keys(excluded_groups_df)
    if not excluded_keys:
        return df.copy()
    work = df.copy()
    work["_ctx_key"] = _build_context_keys(work)
    out = work.loc[~work["_ctx_key"].isin(excluded_keys)].drop(columns=["_ctx_key"])
    return out.reset_index(drop=True)


def _find_questionnaire_source_file(results_dir: Path) -> Path | None:
    candidates = [
        Path(r"D:\data_e2\results-survey.xlsx"),
        results_dir.parent.parent / "data_e2" / "results-survey.xlsx",
        results_dir.parent / "data_e2" / "results-survey.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _filter_raw_questionnaire_scope(
    raw: pd.DataFrame,
    modality_filter: str | None = None,
    excluded_groups_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Filtre le questionnaire brut selon le périmètre du rapport.

    - `modality_filter` restreint PC/VR avant tout recalcul questionnaire.
    - `excluded_groups_df` retire ensuite les groupes jugés `poor` par l'ICC.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = raw.copy()
    meta = _detect_questionnaire_meta_cols(out)

    if modality_filter is not None:
        modalite_col = meta.get("Modalite")
        if modalite_col and modalite_col in out.columns:
            modality_norm = str(modality_filter).strip().upper()
            modalite_vals = out[modalite_col].apply(
                lambda x: _normalize_questionnaire_meta(x, "modalite")
            )
            out = out.loc[modalite_vals == modality_norm].copy()

    if excluded_groups_df is not None and not excluded_groups_df.empty:
        raw_keys = _build_questionnaire_raw_keys(out)
        excluded_keys = _build_questionnaire_excluded_keys(excluded_groups_df)
        if excluded_keys:
            out = out.loc[~raw_keys.isin(excluded_keys)].copy()

    return out.reset_index(drop=True)


def recompute_questionnaire_outputs_for_scope(
    results_dir: Path,
    apply_pruning: bool,
    modality_filter: str | None = None,
    excluded_groups_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """
    Recalcule les sorties questionnaire sur un périmètre explicite du rapport.

    Ce helper sert à obtenir :
    - les alpha / pruning sur le sous-échantillon courant avant exclusion ICC ;
    - les descriptifs questionnaire aval après exclusion des groupes `poor`.
    """
    survey_path = _find_questionnaire_source_file(results_dir)
    if survey_path is None:
        print("[WARN] Fichier source questionnaire introuvable : impossible de recalculer les tables questionnaire du rapport.")
        return None

    raw = pd.read_excel(survey_path)
    filtered_raw = _filter_raw_questionnaire_scope(
        raw,
        modality_filter=modality_filter,
        excluded_groups_df=excluded_groups_df,
    )
    if filtered_raw.empty:
        print("[WARN] Filtrage questionnaire produirait un dataset vide ; recalcul annulé.")
        return None

    scope = "all" if modality_filter is None else str(modality_filter).lower()
    icc_tag = "post_icc" if excluded_groups_df is not None and not excluded_groups_df.empty else "pre_icc"
    tmp_root = results_dir / "_tmp_questionnaire_scope" / scope / icc_tag
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Utiliser un CSV intermediaire evite de dependre d'openpyxl dans le
    # sous-processus questionnaire, notamment avec les environnements Python 3.9.
    filtered_raw_path = tmp_root / f"results-survey_{scope}_{icc_tag}.csv"
    filtered_raw.to_csv(filtered_raw_path, sep=";", index=False, encoding="utf-8-sig")

    questionnaire_out_root = tmp_root / "questionnaire"
    cmd = [
        sys.executable,
        str(_Path(__file__).resolve().parents[2] / "analyse_questionnaire" / "main.py"),
        "--data", str(filtered_raw_path),
        "--out", str(questionnaire_out_root),
        "--mode", "all",
    ]
    if apply_pruning:
        cmd.append("--apply-pruning")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        print("[WARN] Recalcul questionnaire scoped échoué ; fallback sur les sorties globales existantes.")
        if verbose and proc.stderr.strip():
            print(proc.stderr.strip())
        return None

    if verbose and proc.stdout.strip():
        print(proc.stdout.strip())

    filtered_results_root = tmp_root
    q_scores_filtered = load_questionnaire_scores(filtered_results_root, use_pruned=apply_pruning)
    q_cronbach_filtered = load_questionnaire_cronbach(filtered_results_root)
    q_alpha_comp_filtered = load_alpha_comparison(filtered_results_root)
    q_explo_summary_filtered = load_exploratory_summary(filtered_results_root)
    q_desc_dim_filtered = load_desc_dim_questionnaire(filtered_results_root)
    q_desc_dim_pruned_filtered = load_desc_dim_pruned(filtered_results_root)
    q_cronbach_pruned_filtered = load_questionnaire_cronbach_pruned(filtered_results_root)

    return {
        "q_scores": q_scores_filtered,
        "q_cronbach": q_cronbach_filtered,
        "q_alpha_comp": q_alpha_comp_filtered,
        "q_explo_summary": q_explo_summary_filtered,
        "q_desc_dim": q_desc_dim_filtered,
        "q_desc_dim_pruned": q_desc_dim_pruned_filtered,
        "q_cronbach_pruned": q_cronbach_pruned_filtered,
    }


def recompute_questionnaire_outputs_with_icc_exclusion(
    results_dir: Path,
    excluded_groups_df: pd.DataFrame,
    apply_pruning: bool,
    modality_filter: str | None,
    verbose: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """
    Recalcule les sorties questionnaire globales sur le sous-échantillon
    conservé après exclusion des groupes à ICC "poor".
    """
    if excluded_groups_df is None or excluded_groups_df.empty:
        return None
    return recompute_questionnaire_outputs_for_scope(
        results_dir=results_dir,
        apply_pruning=apply_pruning,
        modality_filter=modality_filter,
        excluded_groups_df=excluded_groups_df,
        verbose=verbose,
    )


def _supplemental_sig_stars(p: float, n: int) -> str:
    if not np.isfinite(p) or n < N_MIN_SIG:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _rho_magnitude_label(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    abs_rho = abs(float(rho))
    if abs_rho < 0.10:
        return "négligeable"
    if abs_rho < 0.30:
        return "faible"
    if abs_rho < 0.50:
        return "modérée"
    return "forte"


def _stable_seed(*parts: Any) -> int:
    raw = "||".join(str(p) for p in parts)
    return int(hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:8], 16)


def _spearman_pingouin_or_scipy(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if pg is not None:
        try:
            res = pg.corr(x=x, y=y, method="spearman")
            return float(res["r"].iloc[0]), float(res["p-val"].iloc[0])
        except Exception:
            pass
    rho, p_value = spearmanr(x, y)
    return float(rho), float(p_value)


def _bootstrap_spearman_ci(
    sub: pd.DataFrame,
    x: str,
    y: str,
    *,
    n_boot: int = SUPPLEMENTAL_BOOTSTRAP_B,
) -> tuple[float, float]:
    data = sub[[x, y]].to_numpy(dtype=float)
    n = len(data)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(_stable_seed("spearman_boot", x, y, n))
    boots = np.full(n_boot, np.nan, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = data[idx, :]
        if len(np.unique(sample[:, 0])) < 2 or len(np.unique(sample[:, 1])) < 2:
            continue
        rho, _ = spearmanr(sample[:, 0], sample[:, 1])
        if np.isfinite(rho):
            boots[i] = float(rho)
    if not np.isfinite(boots).any():
        return np.nan, np.nan
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    return float(low), float(high)


def supplemental_spearman_table(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    *,
    block: str,
    n_boot: int = SUPPLEMENTAL_BOOTSTRAP_B,
    sort_by_abs: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for x in dict.fromkeys(x_cols):
        if x not in df.columns:
            continue
        for y in dict.fromkeys(y_cols):
            if y not in df.columns:
                continue
            sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            n = int(len(sub))
            if n < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
                continue
            rho, p_value = _spearman_pingouin_or_scipy(sub[x], sub[y])
            if not (np.isfinite(rho) and np.isfinite(p_value)):
                continue
            ci_low, ci_high = _bootstrap_spearman_ci(sub, x, y, n_boot=n_boot)
            rows.append({
                "block": block,
                "x": x,
                "y": y,
                "rho": float(rho),
                "p": float(p_value),
                "n": n,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sig": _supplemental_sig_stars(float(p_value), n),
                "magnitude": _rho_magnitude_label(float(rho)),
            })

    if not rows:
        return pd.DataFrame(columns=["x", "y", "rho", "p", "n", "sig"]), []

    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    if sort_by_abs:
        out = out.sort_values(["abs_rho", "p"], ascending=[False, True])
    else:
        out = out.sort_values(["p", "abs_rho"], ascending=[True, False])
    out = out.reset_index(drop=True)

    full_rows = out.drop(columns=["abs_rho"], errors="ignore").to_dict("records")
    display = out[["x", "y", "rho", "p", "n", "sig"]].copy()
    display["rho"] = display["rho"].round(2)
    display["p"] = display["p"].round(4)
    return display, full_rows


def _format_corr_item(row: pd.Series) -> str:
    return f"{row['x']} ↔ {row['y']} (ρ={float(row['rho']):.2f}, p={float(row['p']):.3g})"


def _supplemental_corr_note(table: pd.DataFrame, *, context: str, n8_warning: bool = False) -> str:
    if table is None or table.empty:
        return f"Aucune paire estimable pour {context}."

    sig = table[table["sig"].astype(str).str.strip() != ""].copy()
    moderate = table[(table["rho"].abs() >= 0.30) & (table["sig"].astype(str).str.strip() == "")].copy()
    if not sig.empty:
        first_sentence = (
            "Associations significatives : "
            + "; ".join(_format_corr_item(row) for _, row in sig.head(3).iterrows())
            + "."
        )
    elif not moderate.empty:
        first_sentence = (
            "Aucune association significative ; les tendances |ρ|≥0.30 les plus fortes sont "
            + "; ".join(_format_corr_item(row) for _, row in moderate.head(3).iterrows())
            + "."
        )
    else:
        first_sentence = "Aucune association significative ni tendance au moins modérée (|ρ|≥0.30)."

    if n8_warning:
        caution = "Les paires impliquant Riedl/TCI restent exploratoires (n=8) et les IC95 bootstrap percentile B=5000 sont larges."
    else:
        caution = "Ces estimations restent exploratoires avec n faible ; les IC95 bootstrap percentile B=5000 sont à lire comme un diagnostic d'instabilité."
    return f"{first_sentence} {caution}"


def _append_supplemental_corr_block(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    *,
    title: str,
    table: pd.DataFrame,
    note: str,
    heading_md: str = "####",
    pdf_heading_style: str = "Heading4",
    max_rows: int = 120,
) -> None:
    lines.append(f"{heading_md} {title}\n\n")
    lines.append(md_table_highlight(table, max_rows=max_rows))
    lines.append(note + "\n\n")
    pdf_elems.append(Paragraph(title, styles[pdf_heading_style]))
    pdf_elems.append(pdf_table_from_df(table, max_rows=max_rows))
    pdf_elems.append(Spacer(1, 0.06 * inch))
    pdf_elems.append(Paragraph(note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.10 * inch))


# NOTE : Les définitions de CORE_RIEDL_COLS, CORE_SPEECH, CORE_GAZE, CORE_FACE,
# CORE_HL et CORE_MAP sont maintenant importées depuis config.inv_features_config.
# Les commentaires ci-dessous sont conservés pour référence historique.

# ===============================
# OUTILS GÉNÉRAUX
# ===============================

def infer_bim(val: str) -> str | None:
    if not isinstance(val, str):
        return None
    m = re.search(r"\b(bim\d{3})\b", val.lower())
    return m.group(1) if m else None

def debug_timepoint(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        print(f"[DEBUG] {name}: empty")
        return
    cols = list(df.columns)
    print(f"[DEBUG] {name}: shape={df.shape}")
    print(f"[DEBUG] {name}: has_timepoint={'timepoint' in cols}")
    if "timepoint" in cols:
        print(f"[DEBUG] {name}: timepoint values={df['timepoint'].astype(str).value_counts(dropna=False).to_dict()}")

# ===============================
# INV RÉCAP DISPONIBILITÉ
# ===============================

def inv_recap(inv_name: str, inv_df: pd.DataFrame) -> pd.DataFrame:
    if inv_df is None or inv_df.empty:
        return pd.DataFrame(columns=["INV", "var", "n_non_na", "%NA"])

    inv_df = normalize_group(inv_df)
    inv_df = normalize_timepoint(inv_df)
    if inv_df is None or inv_df.empty or "group_id" not in inv_df.columns:
        return pd.DataFrame(columns=["INV", "var", "n_non_na", "%NA"])

    num_cols = [c for c in inv_df.columns if pd.api.types.is_numeric_dtype(inv_df[c]) and c not in ID_LIKE_COLS]
    if not num_cols:
        return pd.DataFrame(columns=["INV", "var", "n_non_na", "%NA"])

    group_cols = available_unit_cols(inv_df)
    g = inv_df.groupby(group_cols, dropna=False)[num_cols].mean()

    rows = []
    n_obs = len(g)
    for c in num_cols:
        n_non_na = int(g[c].notna().sum())
        pct_na = float(100.0 * (1.0 - (n_non_na / n_obs))) if n_obs > 0 else np.nan
        rows.append({"INV": inv_name, "var": c, "n_non_na": n_non_na, "%NA": round(pct_na, 2)})

    return pd.DataFrame(rows).sort_values(["%NA", "var"], ascending=[True, True]).reset_index(drop=True)


def summarize_source_diagnostics(inv_df: pd.DataFrame, feature_cols: list[str], max_items: int = 8) -> list[str]:
    """Résume les colonnes *_source disponibles pour documenter provenance et fallbacks."""
    messages: list[str] = []
    seen: set[str] = set()
    for feat in feature_cols:
        src_col = f"{feat}_source"
        if src_col in seen or src_col not in inv_df.columns:
            continue
        seen.add(src_col)
        src = inv_df[src_col].dropna().astype(str).str.strip()
        src = src[src != ""]
        if src.empty:
            continue
        counts = src.value_counts().head(3)
        preview = ", ".join(f"{label} ({count})" for label, count in counts.items())
        messages.append(f"{src_col}: {preview}")
        if len(messages) >= max_items:
            break
    return messages


def render_inv_section(
    lines: list[str],
    pdf_elems: list,
    styles,
    section_num: str,
    inv_label: str,
    inv_df: pd.DataFrame,
    *,
    fig_dir: Path,
    apply_fdr: bool,
    inv_pruned_features: list[str] | None,
    riedl: pd.DataFrame | None,
    tci: pd.DataFrame | None,
    perf_g: pd.DataFrame | None,
    q_group: pd.DataFrame | None,
    rcols: list[str],
    tci_cols: list[str],
    q_questionnaire_cols: list[str],
    all_corr_results: list[dict],
):
    """
    Rend une sous-section INV (Face / Speech / Gaze / High-level) hors de build_report.
    """
    lines.append(f"### {section_num}. {inv_label}\n")
    pdf_elems.append(Paragraph(f"{section_num}. {inv_label}", styles["Heading3"]))

    if inv_label == "Speech":
        _speech_method_note = (
            "**Note opérationnelle (pipeline v2, mai 2026)** — "
            "Les tours de parole sont définis selon la convention CA (Conversation Analysis) en 3 passes "
            "depuis les IPU VAD (seuil absolu −30 dBFS) : fusion des IPU consécutifs du même locuteur, "
            "filtre ≥ 1,0 s, re-fusion des tours adjacents de même rôle. "
            "Les backchannels sont détectés par 4 filtres en cascade : durée IPU ∈ [0,10–0,70 s], "
            "chevauchement ≥ 100 ms avec un autre locuteur actif, non-continuation (aucun IPU de même rôle "
            "se terminant dans les 200 ms précédents), non-tour-CA (aucun tour CA de même rôle démarrant "
            "dans les 500 ms suivants). Ces définitions strictes réduisent significativement les faux positifs "
            "par rapport à un seuil audio seul."
        )
        lines.append(f"{_speech_method_note}\n\n")
        pdf_elems.append(Paragraph(_speech_method_note.replace("**", ""), styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))

    if inv_df is None or inv_df.empty:
        lines.append("_(vide)_\n\n")
        pdf_elems.append(Paragraph("(vide)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.2 * inch))
        return

    inv_df = normalize_group(inv_df)
    inv_df = normalize_timepoint(inv_df)
    if inv_df is None or inv_df.empty or "group_id" not in inv_df.columns:
        lines.append("group_id introuvable -> skip.\n\n")
        pdf_elems.append(Paragraph("group_id introuvable -> skip.", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.2 * inch))
        return

    inv_df = coerce_numeric_columns(
        inv_df,
        exclude={"group_id", "timepoint", "group_base_id", "condition", "modalite", "scenario", "session"},
    )

    if inv_label == "Speech":
        core_candidates = CORE_SPEECH
    elif inv_label == "Face":
        core_candidates = CORE_FACE
    elif inv_label == "Gaze":
        core_candidates = CORE_GAZE
    elif inv_label == "High-level":
        core_candidates = CORE_HL
    else:
        core_candidates = CORE_MAP.get(inv_label, [])

    missing_core = [c for c in core_candidates if c not in inv_df.columns]
    source_diagnostics = summarize_source_diagnostics(inv_df, list(dict.fromkeys(core_candidates)))

    icols = [c for c in core_candidates if c in inv_df.columns]
    if not icols:
        icols = [
            c for c in inv_df.columns
            if pd.api.types.is_numeric_dtype(inv_df[c]) and c not in ID_LIKE_COLS
        ][:MAX_X_COLS]

    if inv_pruned_features is not None and inv_label in ("High-level", "Face"):
        icols_filtered = [c for c in icols if c in inv_pruned_features]
        if icols_filtered:
            n_dropped = len(icols) - len(icols_filtered)
            if n_dropped > 0:
                print(f"  [PRUNING] {inv_label}: {n_dropped} features redondantes supprimées")
            icols = icols_filtered

    if not icols:
        lines.append("Aucune colonne numérique exploitable.\n\n")
        pdf_elems.append(Paragraph("Aucune colonne numérique exploitable.", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.2 * inch))
        return

    if missing_core or source_diagnostics:
        print(
            f"  [DIAG] {inv_label}: {len(missing_core)} core absente(s), "
            f"{len(source_diagnostics)} provenance(s) *_source documentée(s)"
        )
        lines.append("#### Diagnostic disponibilité et provenance\n")
        pdf_elems.append(Paragraph("Diagnostic disponibilité et provenance", styles["Heading4"]))
        if missing_core:
            missing_txt = ", ".join(missing_core[:12])
            if len(missing_core) > 12:
                missing_txt += ", ..."
            lines.append(f"- Features core absentes: {missing_txt}\n")
            pdf_elems.append(Paragraph(f"Features core absentes: {missing_txt}", styles["Normal"]))
        if source_diagnostics:
            lines.append("- Provenance / fallbacks observés via *_source :\n")
            pdf_elems.append(Paragraph("Provenance / fallbacks observés via *_source :", styles["Normal"]))
            for msg in source_diagnostics:
                lines.append(f"- {msg}\n")
                pdf_elems.append(Paragraph(msg, styles["Normal"]))
        lines.append("\n")
        pdf_elems.append(Spacer(1, 0.12 * inch))

    inv_g = aggregate_numeric_by_unit(inv_df, icols)

    merged = inv_g.copy()
    if inv_label == "Speech" and "participation_entropy" in inv_df.columns and "participation_entropy" not in merged.columns:
        participation_g = aggregate_numeric_by_unit(inv_df, ["participation_entropy"])
        if participation_g is not None and not participation_g.empty:
            merged = merge_on_unit(merged, participation_g, how="left")
    retained_detail_features = {
        "Speech": REGRESSION_RETAINED_INV_FEATURES.get("speech", []),
        "Face": REGRESSION_RETAINED_INV_FEATURES.get("face", []),
        "Gaze": REGRESSION_RETAINED_INV_FEATURES.get("gaze", []),
    }.get(inv_label, [])
    missing_detail_features = [
        c for c in retained_detail_features
        if c in inv_df.columns and c not in merged.columns
    ]
    if missing_detail_features:
        detail_g = aggregate_numeric_by_unit(inv_df, missing_detail_features)
        if detail_g is not None and not detail_g.empty:
            merged = merge_on_unit(merged, detail_g, how="left")
    if riedl is not None and not riedl.empty:
        merged = merge_on_unit(merged, riedl, how="left")
    if tci is not None and not tci.empty and tci_cols:
        merged = merge_on_unit(
            merged,
            tci[[c for c in tci.columns if c in set(common_unit_cols(tci, merged) + tci_cols)]],
            how="left",
        )
    if perf_g is not None and not perf_g.empty:
        merged = merge_on_unit(merged, perf_g, how="left")
    if q_group is not None and not q_group.empty:
        merged = merge_on_unit(merged, q_group, how="left")

    cohesion_subdim_cols = [
        c for c in COHESION_SUBDIM_COLS
        if c in merged.columns and pd.to_numeric(merged[c], errors="coerce").notna().any()
    ]

    if inv_label == "Speech" and "participation_entropy" in merged.columns:
        participation_targets = [
            c for c in (
                list(rcols)
                + list(tci_cols)
                + [c for c in ["Score_perf_tsk"] if c in merged.columns]
                + list(q_questionnaire_cols)
            )
            if c in merged.columns and pd.to_numeric(merged[c], errors="coerce").notna().any()
        ]
        participation_table, participation_rows = supplemental_spearman_table(
            merged,
            ["participation_entropy"],
            participation_targets,
            block="Speech participation_entropy ↔ cibles complètes",
            sort_by_abs=True,
        )
        if not participation_table.empty:
            all_corr_results.extend(participation_rows)
            participation_note = (
                "`participation_entropy` remplace `turn_balance_cv` dans la sélection v2 et reçoit donc "
                "le même traitement bivarié que les autres indicateurs Speech. "
                + _supplemental_corr_note(
                    participation_table,
                    context="participation_entropy",
                    n8_warning=True,
                )
            )
            _append_supplemental_corr_block(
                lines,
                pdf_elems,
                styles,
                title="Participation entropy ↔ cibles principales (complément v2)",
                table=participation_table,
                note=participation_note,
                heading_md="####",
                pdf_heading_style="Heading4",
                max_rows=25,
            )

    if rcols:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md=f"{section_num}.1 {inv_label} ↔ Riedl",
            title_pdf=f"{section_num}.1 {inv_label} ↔ Riedl",
            df=merged,
            x_cols=icols,
            y_cols=rcols,
            fig_dir=fig_dir,
            fig_prefix=f"{inv_label}_vs_Riedl",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    if tci_cols:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md=f"{section_num}.2 {inv_label} ↔ TCI",
            title_pdf=f"{section_num}.2 {inv_label} ↔ TCI",
            df=merged,
            x_cols=icols,
            y_cols=tci_cols,
            fig_dir=fig_dir,
            fig_prefix=f"{inv_label}_vs_TCI",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    inv_perf_y = [c for c in ["Score_perf_tsk"] if c in merged.columns and merged[c].dropna().any()]
    if inv_perf_y:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md=f"{section_num}.3 {inv_label} ↔ Performance (Score final)",
            title_pdf=f"{section_num}.3 {inv_label} ↔ Performance (Score final)",
            df=merged,
            x_cols=icols,
            y_cols=inv_perf_y,
            fig_dir=fig_dir,
            fig_prefix=f"{inv_label}_vs_ScoreFinal",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    if q_questionnaire_cols:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md=f"{section_num}.4 {inv_label} ↔ Questionnaire",
            title_pdf=f"{section_num}.4 {inv_label} ↔ Questionnaire",
            df=merged,
            x_cols=icols,
            y_cols=q_questionnaire_cols[:MAX_Y_COLS],
            fig_dir=fig_dir,
            fig_prefix=f"{inv_label}_vs_Questionnaire",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    if cohesion_subdim_cols and inv_label != "High-level":
        retained_by_label = {
            "Speech": REGRESSION_RETAINED_INV_FEATURES.get("speech", []),
            "Face": REGRESSION_RETAINED_INV_FEATURES.get("face", []),
            "Gaze": REGRESSION_RETAINED_INV_FEATURES.get("gaze", []),
        }
        retained_all = (
            REGRESSION_RETAINED_INV_FEATURES.get("speech", [])
            + REGRESSION_RETAINED_INV_FEATURES.get("face", [])
            + REGRESSION_RETAINED_INV_FEATURES.get("gaze", [])
        )
        cohesion_x = retained_by_label.get(inv_label, retained_all if inv_label == "High-level" else icols)
        cohesion_x = [c for c in dict.fromkeys(cohesion_x) if c in merged.columns and pd.to_numeric(merged[c], errors="coerce").notna().any()]
        if cohesion_x:
            cohesion_table, cohesion_rows = supplemental_spearman_table(
                merged,
                cohesion_x,
                cohesion_subdim_cols,
                block=f"{inv_label} ↔ sous-dimensions Cohésion",
                sort_by_abs=True,
            )
            if not cohesion_table.empty:
                all_corr_results.extend(cohesion_rows)
                cohesion_note = _supplemental_corr_note(
                    cohesion_table,
                    context=f"{inv_label} ↔ sous-dimensions Cohésion",
                    n8_warning=False,
                )
                _append_supplemental_corr_block(
                    lines,
                    pdf_elems,
                    styles,
                    title=f"{section_num}.5 {inv_label} ↔ sous-dimensions Cohésion",
                    table=cohesion_table,
                    note=cohesion_note,
                    heading_md="####",
                    pdf_heading_style="Heading4",
                    max_rows=80,
                )

    pdf_elems.append(Spacer(1, 0.2 * inch))


def collect_inv_corr_results(
    inv_label: str,
    inv_df: pd.DataFrame,
    *,
    apply_fdr: bool,
    inv_pruned_features: list[str] | None,
    riedl: pd.DataFrame | None,
    tci: pd.DataFrame | None,
    perf_g: pd.DataFrame | None,
    q_group: pd.DataFrame | None,
    rcols: list[str],
    tci_cols: list[str],
    q_questionnaire_cols: list[str],
) -> list[dict[str, Any]]:
    """Calcule les corrélations INV du bloc sans rendre les sections détaillées."""
    if inv_df is None or inv_df.empty:
        return []

    inv_df = normalize_group(inv_df)
    inv_df = normalize_timepoint(inv_df)
    if inv_df is None or inv_df.empty or "group_id" not in inv_df.columns:
        return []

    inv_df = coerce_numeric_columns(
        inv_df,
        exclude={"group_id", "timepoint", "group_base_id", "condition", "modalite", "scenario", "session"},
    )

    if inv_label == "Speech":
        core_candidates = CORE_SPEECH_REPORT
    elif inv_label == "Face":
        core_candidates = CORE_FACE_REPORT
    elif inv_label == "Gaze":
        core_candidates = CORE_GAZE_REPORT
    elif inv_label == "High-level":
        core_candidates = CORE_HL_REPORT
    else:
        core_candidates = CORE_MAP_REPORT.get(inv_label, [])

    icols = [c for c in core_candidates if c in inv_df.columns]
    if not icols:
        icols = [
            c for c in inv_df.columns
            if pd.api.types.is_numeric_dtype(inv_df[c]) and c not in ID_LIKE_COLS
        ][:MAX_X_COLS]

    if inv_pruned_features is not None and inv_label in ("High-level", "Face"):
        icols_filtered = [c for c in icols if c in inv_pruned_features]
        if icols_filtered:
            icols = icols_filtered

    if not icols:
        return []

    inv_g = aggregate_numeric_by_unit(inv_df, icols)
    merged = inv_g.copy()
    if riedl is not None and not riedl.empty:
        merged = merge_on_unit(merged, riedl, how="left")
    if tci is not None and not tci.empty and tci_cols:
        merged = merge_on_unit(
            merged,
            tci[[c for c in tci.columns if c in set(common_unit_cols(tci, merged) + tci_cols)]],
            how="left",
        )
    if perf_g is not None and not perf_g.empty:
        merged = merge_on_unit(merged, perf_g, how="left")
    if q_group is not None and not q_group.empty:
        merged = merge_on_unit(merged, q_group, how="left")

    results: list[dict[str, Any]] = []
    corr_specs = [
        ("Riedl", rcols),
        ("TCI", tci_cols),
        ("Performance", [c for c in ["Score_perf_tsk"] if c in merged.columns and merged[c].dropna().any()]),
        ("Questionnaire", q_questionnaire_cols[:MAX_Y_COLS]),
    ]
    for target_label, y_cols in corr_specs:
        if not y_cols:
            continue
        _, rows_all = spearman_table(
            merged,
            x_cols=icols,
            y_cols=y_cols,
            apply_fdr=apply_fdr,
            block=f"{inv_label} ↔ {target_label}",
            n_min_sig=N_MIN_SIG,
        )
        results.extend(rows_all)
    return results


def _supplement_inv_block_with_high_level(
    base_df: pd.DataFrame | None,
    hl_df: pd.DataFrame | None,
    candidate_features: list[str],
) -> pd.DataFrame:
    """
    Complète un bloc modalité (Face / Gaze) avec des features prunées qui
    existent seulement dans le dataset high-level.
    """
    if hl_df is None or hl_df.empty or "group_id" not in hl_df.columns:
        return base_df if base_df is not None else pd.DataFrame()

    base_df = pd.DataFrame() if base_df is None else base_df
    missing = [c for c in candidate_features if c in hl_df.columns and c not in base_df.columns]
    if not missing:
        return base_df

    hl_block = aggregate_numeric_by_unit(hl_df, missing)
    if hl_block.empty:
        return base_df

    if base_df.empty:
        return hl_block
    return filter_inv_dataframe(merge_on_unit(base_df, hl_block, how="left"))


def build_inv_stepwise_dataset(
    inv_face: pd.DataFrame | None,
    inv_speech: pd.DataFrame | None,
    inv_gaze_all: pd.DataFrame | None,
    hl: pd.DataFrame | None,
    perf_g: pd.DataFrame | None,
    q_group: pd.DataFrame | None,
    inv_pruned_features: list[str] | None,
    use_regression_whitelist: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    blocks: list[pd.DataFrame] = []
    predictor_groups: dict[str, list[str]] = {}

    pruned_kept: set[str] = set(inv_pruned_features or [])
    if pruned_kept:
        print(f"[INV stepwise] Features prunées kept=1 : {len(pruned_kept)}")
    else:
        print("[INV stepwise] inv_pruned_features.csv absent — fallback whitelist")

    # Candidats pour _supplement : toutes les features prunées (ou whitelist si demandé)
    face_candidates = list(REGRESSION_RETAINED_INV_FEATURES["face"]) if use_regression_whitelist else [
        f for f in pruned_kept if infer_family_from_name(f) in ("face", "affect")
    ] or list(REGRESSION_RETAINED_INV_FEATURES["face"])
    gaze_candidates = list(REGRESSION_RETAINED_INV_FEATURES["gaze"]) if use_regression_whitelist else [
        f for f in pruned_kept if infer_family_from_name(f) == "gaze"
    ] or list(REGRESSION_RETAINED_INV_FEATURES["gaze"])

    inv_face_aug = _supplement_inv_block_with_high_level(inv_face, hl, face_candidates)
    inv_gaze_aug = _supplement_inv_block_with_high_level(inv_gaze_all, hl, gaze_candidates)

    source_specs = [
        ("Audio/Speech", inv_speech, REGRESSION_RETAINED_INV_FEATURES["speech"]),
        ("Face", inv_face_aug, REGRESSION_RETAINED_INV_FEATURES["face"]),
        ("Gaze", inv_gaze_aug, REGRESSION_RETAINED_INV_FEATURES["gaze"]),
    ]

    for label, df, preferred_cols in source_specs:
        if df is None or df.empty:
            continue
        clean = filter_inv_dataframe(df)

        if use_regression_whitelist:
            # Mode explicite : intersection pruning PCA × whitelist métier
            whitelist = set(preferred_cols)
            if pruned_kept:
                selected = pruned_kept & whitelist
                print(f"[INV stepwise][{label}] Whitelist activée — après filtre regression_preferred : {len(selected)}")
            else:
                selected = whitelist.copy()
                print(f"[INV stepwise][{label}] Fallback whitelist : {len(selected)}")
        else:
            # Mode défaut : features prunées kept=1 filtrées par famille du bloc courant
            _family_map = {
                "Audio/Speech": ("audio",),
                "Face": ("face", "affect"),
                "Gaze": ("gaze",),
            }
            _allowed_families = _family_map.get(label, ())
            if pruned_kept:
                selected = {
                    f for f in pruned_kept
                    if infer_family_from_name(f) in _allowed_families
                }
                print(f"[INV stepwise][{label}] Features prunées kept=1 (famille {_allowed_families}) : {len(selected)}")
            else:
                selected = set(preferred_cols)
                print(f"[INV stepwise][{label}] Fallback whitelist : {len(selected)}")

        # Réinjection forcée de variables métier critiques
        reinjected: list[str] = []
        for feat in REGRESSION_FORCE_INCLUDE:
            if feat in clean.columns and feat not in selected:
                selected.add(feat)
                reinjected.append(feat)
        if reinjected:
            print(f"[INV stepwise][{label}] Réinjectées via REGRESSION_FORCE_INCLUDE : {reinjected}")

        # Conserver seulement les colonnes réellement présentes, dans l'ordre preferred_cols
        # puis features prunées hors whitelist
        missing = sorted(f for f in selected if f not in clean.columns)
        if missing:
            print(f"[INV stepwise][{label}] Absentes du dataframe (ignorées) : {missing}")

        if use_regression_whitelist:
            avail = [f for f in preferred_cols if f in selected and f in clean.columns]
        else:
            avail = [f for f in preferred_cols if f in selected and f in clean.columns]
            avail += [f for f in sorted(selected) if f in clean.columns and f not in avail]
        for feat in REGRESSION_FORCE_INCLUDE:
            if feat in selected and feat in clean.columns and feat not in avail:
                avail.append(feat)
        print(f"[INV stepwise][{label}] Features finales pour régression : {avail}")

        if not avail:
            continue
        block = aggregate_numeric_by_unit(clean, avail)
        if block.empty:
            continue
        blocks.append(block)
        predictor_groups[label] = avail

    if not blocks:
        return pd.DataFrame(), {}

    merged = blocks[0].copy()
    for block in blocks[1:]:
        merged = merge_on_unit(merged, block, how="outer")

    if perf_g is not None and not perf_g.empty:
        merged = merge_on_unit(merged, perf_g, how="left")
    if q_group is not None and not q_group.empty:
        regression_questionnaire_cols = (
            QUESTIONNAIRE_ANALYSIS_COLS
            + [c for c in COHESION_COMPONENTS if c not in QUESTIONNAIRE_ANALYSIS_COLS]
        )
        keep_q = [c for c in ["group_id", "timepoint"] + regression_questionnaire_cols if c in q_group.columns]
        merged = merge_on_unit(merged, q_group[keep_q].drop_duplicates(), how="left")

    predictor_groups_combo: dict[str, list[str]] = dict(predictor_groups)
    labels = list(predictor_groups.keys())
    if len(labels) >= 2:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                name = f"{labels[i]} + {labels[j]}"
                predictor_groups_combo[name] = predictor_groups[labels[i]] + predictor_groups[labels[j]]
    if len(labels) == 3:
        predictor_groups_combo["Audio/Speech + Face + Gaze"] = (
            predictor_groups.get("Audio/Speech", [])
            + predictor_groups.get("Face", [])
            + predictor_groups.get("Gaze", [])
        )

    return merged, predictor_groups_combo


def _dedupe_keep_order_local(names: list[str]) -> list[str]:
    """Déduplique une liste en conservant l'ordre d'apparition."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _collect_regression_pca_features(
    inv_reg_df: pd.DataFrame,
    predictor_groups: dict[str, list[str]],
) -> tuple[list[str], pd.DataFrame]:
    """
    Extrait l'union des features atomiques réellement utilisées par la régression.

    On ignore volontairement les groupes combinés ("A + B", etc.) ajoutés pour
    les régressions multi-blocs afin de ne garder qu'une seule occurrence par
    feature dans l'espace PCA restreint.
    """
    feature_rows: list[dict[str, str]] = []
    feature_order: list[str] = []

    for block_name, block_features in predictor_groups.items():
        if " + " in str(block_name):
            continue
        for feat in block_features:
            if feat not in inv_reg_df.columns:
                continue
            feature_order.append(feat)
            feature_rows.append({"feature": feat, "source_block": block_name})

    feature_order = _dedupe_keep_order_local(feature_order)
    feature_manifest = (
        pd.DataFrame(feature_rows).drop_duplicates(subset=["feature"]).reset_index(drop=True)
        if feature_rows else pd.DataFrame(columns=["feature", "source_block"])
    )
    return feature_order, feature_manifest


def _build_regression_pca_frame(
    inv_reg_df: pd.DataFrame,
    predictor_groups: dict[str, list[str]],
    merged_master: pd.DataFrame | None,
    modality_filter: str | None,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Construit le dataset PCA restreint au sous-espace des prédicteurs INV de
    régression, en réinjectant uniquement les métadonnées nécessaires.
    """
    if inv_reg_df is None or inv_reg_df.empty:
        return pd.DataFrame(), [], pd.DataFrame(columns=["feature", "source_block"])

    feature_cols, feature_manifest = _collect_regression_pca_features(inv_reg_df, predictor_groups)
    if not feature_cols:
        return pd.DataFrame(), [], feature_manifest

    unit_cols = [c for c in ["group_id", "timepoint"] if c in inv_reg_df.columns]
    pca_frame = inv_reg_df[unit_cols + feature_cols].copy()

    if merged_master is not None and not merged_master.empty:
        meta_cols = [c for c in ["group_id", "timepoint", "scenario", "modalite", "condition"] if c in merged_master.columns]
        if "group_id" in meta_cols:
            meta = merged_master[meta_cols].drop_duplicates().copy()
            pca_frame = merge_on_unit(meta, pca_frame, how="right")

    if modality_filter is not None:
        modality_value = str(modality_filter).upper().strip()
        if "modalite" not in pca_frame.columns:
            pca_frame["modalite"] = modality_value
        else:
            pca_frame["modalite"] = pca_frame["modalite"].fillna(modality_value)
        if "condition" not in pca_frame.columns:
            pca_frame["condition"] = modality_value
        else:
            pca_frame["condition"] = pca_frame["condition"].fillna(modality_value)

    sort_cols = [c for c in ["group_id", "scenario", "modalite", "timepoint"] if c in pca_frame.columns]
    if sort_cols:
        pca_frame = pca_frame.sort_values(sort_cols).reset_index(drop=True)

    return pca_frame, feature_cols, feature_manifest


def _ensure_regression_pca_outputs(
    *,
    results_dir: Path,
    modality_filter: str | None,
    inv_analysis_mode: str,
    inv_reg_df: pd.DataFrame,
    predictor_groups: dict[str, list[str]],
    merged_master: pd.DataFrame | None,
    pca_rotation: str = "none",
) -> str:
    """
    Génère les sorties PCA dédiées au sous-espace de régression INV.

    Cette passe conserve le moteur PCA existant (`run_inv_analysis_pipeline`) et
    ne change que le dataset injecté : seules les features effectivement retenues
    par la régression stepwise INV sont analysées.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
    from analyse_inv.analyze_inv_structure import run_inv_analysis_pipeline

    pca_frame, feature_cols, feature_manifest = _build_regression_pca_frame(
        inv_reg_df=inv_reg_df,
        predictor_groups=predictor_groups,
        merged_master=merged_master,
        modality_filter=modality_filter,
    )
    if pca_frame.empty or len(feature_cols) < 3:
        raise ValueError(
            "Impossible de construire la PCA restreinte à la régression : "
            f"{len(feature_cols)} feature(s) disponible(s)."
        )

    inv_root = (
        "results_inv_structure_vr_only_regression"
        if modality_filter is not None and str(modality_filter).upper() == "VR"
        else "results_inv_structure_regression"
    )
    inv_mode_subdir = "with_pruning" if inv_analysis_mode == "pruning" else "without_pruning"
    inv_subdir = f"{inv_root}/{inv_mode_subdir}"
    out_dir = results_dir / inv_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [c for c in ["group_id", "condition", "scenario", "timepoint"] if c in pca_frame.columns]
    df_meta = pca_frame[meta_cols].copy() if meta_cols else pd.DataFrame(index=pca_frame.index)

    feature_manifest_path = out_dir / "regression_feature_pool.csv"
    feature_manifest.to_csv(feature_manifest_path, index=False)
    print(
        "[PCA regression] Sous-espace régression : "
        f"{len(feature_cols)} feature(s) exportée(s) vers {feature_manifest_path}"
    )

    run_inv_analysis_pipeline(
        df=pca_frame,
        df_meta=df_meta,
        out_dir=out_dir,
        mode_name=inv_mode_subdir,
        apply_pruning=(inv_analysis_mode == "pruning"),
        max_missing=0.20,
        min_cumvar=0.70,
        prune_threshold=0.80,
        rotation=str(pca_rotation).lower().strip(),
    )
    return inv_subdir


class MyDocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heading_counter = 0

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return

        style_name = getattr(flowable.style, "name", "")
        text = flowable.getPlainText()

        # Ne pas indexer la page de titre ni le sommaire lui-même.
        if text == "Sommaire" or text.lower().startswith("rapport "):
            return

        if style_name == "Heading1":
            level = 0
        elif style_name == "Heading2":
            level = 1
        elif style_name == "Heading3":
            level = 2
        else:
            return

        # Ajoute un bookmark PDF stable pour rendre le sommaire cliquable.
        # La clé doit rester identique entre les différentes passes de multiBuild,
        # sinon le TOC ne se stabilise jamais.
        bookmark_key = getattr(flowable, "_bookmark_key", None)
        if bookmark_key is None:
            self._heading_counter += 1
            bookmark_key = f"heading_{level}_{self._heading_counter}"
            setattr(flowable, "_bookmark_key", bookmark_key)

        self.canv.bookmarkPage(bookmark_key)
        # L'outline PDF ne peut pas commencer directement au niveau 1.
        # Comme le titre et le sommaire sont exclus, on rebase l'outline pour
        # faire des Heading2 le niveau racine cliquable du panneau PDF.
        outline_level = max(level - 1, 0)
        self.canv.addOutlineEntry(text, bookmark_key, level=outline_level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, bookmark_key))


def build_principal_sections_2_4_report(
    results_dir: Path,
    out_dir: Path,
    *,
    apply_fdr: bool = False,
    verbose: bool = False,
    apply_pruning: bool = True,
    inv_analysis_mode: str = "pruning",
    pca_rotation: str = "none",
    file_stem: str = "rapport_principal_PC_VR_cfactor_pop_2_4",
    report_title: str = "Rapport principal v2 — PC + VR — sections 2 a 4",
    tci_allowed_file: Path | None = None,
    tci_all_file: Path | None = None,
    exclude_groups: Iterable[str] | None = None,
) -> dict[str, Path | None]:
    """
    Variante compacte du rapport principal limitee aux sections 2-4.

    Cette sortie sert a comparer directement les resultats historiques du
    rapport principal avec une autre source TCI (par exemple `c_scores_*_pop.csv`)
    sans reproduire les sections 1, 5, 6 et 7.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    global all_corr_results
    all_corr_results.clear()

    fig_dir = out_dir / f"figs_{file_stem}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    pdf_elems: list = []
    lines: list[str] = []

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCHeading1",
            parent=styles["Normal"],
            fontSize=11,
            leading=13,
            leftIndent=10,
            firstLineIndent=-2,
            spaceBefore=6,
        ),
        ParagraphStyle(
            name="TOCHeading2",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            leftIndent=24,
            firstLineIndent=-2,
            spaceBefore=2,
        ),
        ParagraphStyle(
            name="TOCHeading3",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            leftIndent=38,
            firstLineIndent=-2,
            spaceBefore=1,
        ),
    ]

    def _exclude_groups_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        return exclude_bad_groups(df, extra_excluded=exclude_groups)

    perf = _exclude_groups_df(load_performance(results_dir))
    riedl_raw = _exclude_groups_df(load_riedl(results_dir))
    tci_allowed = _exclude_groups_df(load_tci(results_dir, scope="allowed", explicit_path=tci_allowed_file))
    tci_all_global = _exclude_groups_df(load_tci(results_dir, scope="all", explicit_path=tci_all_file))
    inv_face = _exclude_groups_df(filter_inv_dataframe(load_inv_face(results_dir)))
    inv_speech = _exclude_groups_df(filter_inv_dataframe(load_inv_speech(results_dir)))
    inv_gaze_all = _exclude_groups_df(filter_inv_dataframe(load_inv_gaze_all(results_dir)))
    hl, _ = load_high_level_features(results_dir)
    hl = _exclude_groups_df(filter_inv_dataframe(hl))
    q_scores = _exclude_groups_df(load_questionnaire_scores(results_dir, use_pruned=apply_pruning))

    if verbose and tci_allowed.attrs.get("source_path"):
        print(f"[INFO] Source TCI allowed (sections 2-4) : {tci_allowed.attrs['source_path']}")
    if verbose and tci_all_global.attrs.get("source_path"):
        print(f"[INFO] Source TCI all (sections 2-4)     : {tci_all_global.attrs['source_path']}")

    analysis_scope_groups: set[str] = set()
    for scope_df in [perf, q_scores, inv_face, inv_speech, inv_gaze_all, hl]:
        if scope_df is None or scope_df.empty or "group_id" not in scope_df.columns:
            continue
        analysis_scope_groups |= set(
            scope_df["group_id"].astype(str).str.strip().str.lower().dropna().unique()
        )

    def _restrict_to_scope(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "group_id" not in df.columns:
            return pd.DataFrame()
        out = df.copy()
        if analysis_scope_groups:
            out = out[
                out["group_id"].astype(str).str.strip().str.lower().isin(analysis_scope_groups)
            ].copy()
        return out

    riedl = _restrict_to_scope(riedl_raw)
    tci_allowed = _restrict_to_scope(tci_allowed)
    tci_all_global = _restrict_to_scope(tci_all_global)

    if not riedl.empty and not tci_allowed.empty:
        allowed_common = (
            set(riedl["group_id"].astype(str).str.strip().str.lower())
            & set(tci_allowed["group_id"].astype(str).str.strip().str.lower())
        )
        riedl_allowed = riedl[
            riedl["group_id"].astype(str).str.strip().str.lower().isin(allowed_common)
        ].copy()
        tci_allowed = tci_allowed[
            tci_allowed["group_id"].astype(str).str.strip().str.lower().isin(allowed_common)
        ].copy()
    else:
        riedl_allowed = riedl.copy()

    if not riedl_raw.empty and not tci_all_global.empty:
        global_common = (
            set(riedl_raw["group_id"].astype(str).str.strip().str.lower())
            & set(tci_all_global["group_id"].astype(str).str.strip().str.lower())
        )
        if analysis_scope_groups:
            global_common &= analysis_scope_groups
        riedl_all = riedl_raw[
            riedl_raw["group_id"].astype(str).str.strip().str.lower().isin(global_common)
        ].copy()
        tci_all = tci_all_global[
            tci_all_global["group_id"].astype(str).str.strip().str.lower().isin(global_common)
        ].copy()
    else:
        riedl_all = riedl_raw.copy()
        tci_all = tci_all_global.copy()

    perf_g = perf_group_mean(perf)
    q_group, _ = questionnaire_group_wide(q_scores)
    q_group, _ = compute_aggregated_scores(
        q_group,
        alpha_threshold=QUESTIONNAIRE_ALPHA_THRESHOLD,
        include_tms_score=False,
        include_cohesion_score=True,
    )
    q_questionnaire_corr_cols = [
        c for c in QUESTIONNAIRE_ANALYSIS_COLS
        if q_group is not None and not q_group.empty and c in q_group.columns and q_group[c].notna().any()
    ]

    rcols = [c for c in CORE_RIEDL_COLS if c in (riedl_allowed.columns if riedl_allowed is not None else [])]
    if not rcols and riedl_allowed is not None and not riedl_allowed.empty:
        rcols = [c for c in riedl_allowed.columns if pd.api.types.is_numeric_dtype(riedl_allowed[c]) and c != "group_id"][:MAX_Y_COLS]

    tci_cols = _select_tci_analysis_cols(tci_allowed)

    lines.append(f"# {report_title}\n")
    pdf_elems.append(Paragraph(report_title, styles["Heading1"]))
    pdf_elems.append(Spacer(1, 0.2 * inch))

    pdf_elems.append(Paragraph("Sommaire", styles["Heading1"]))
    pdf_elems.append(Spacer(1, 0.15 * inch))
    pdf_elems.append(toc)
    pdf_elems.append(PageBreak())

    intro = (
        "Cette variante reproduit uniquement les sections 2 a 4 du rapport principal PC+VR. "
        "Elle sert a evaluer l'impact du remplacement du c-factor historique par la variante "
        "calculee a partir de la population parente. Les sections 1, 5, 6 et 7 sont volontairement "
        "omises pour concentrer la comparaison sur les correlations, les variables INV externalisees "
        "dans le rapport principal, puis les corrélations les plus fortes."
    )
    source_note = (
        f"Source TCI allowed : {tci_allowed.attrs.get('source_path', tci_allowed_file)}. "
        f"Source TCI all : {tci_all.attrs.get('source_path', tci_all_file)}."
    )
    lines.append(intro + "\n\n")
    lines.append(source_note + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))
    pdf_elems.append(Paragraph(source_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))

    pdf_elems.append(PageBreak())
    lines.append("## 2. Corrélations globales (group-level)\n")
    pdf_elems.append(Paragraph("2. Corrélations globales (group-level)", styles["Heading2"]))

    base = riedl_allowed.copy() if riedl_allowed is not None and not riedl_allowed.empty else pd.DataFrame()
    if tci_allowed is not None and not tci_allowed.empty:
        base = merge_on_unit(base, tci_allowed, how="outer") if not base.empty else tci_allowed.copy()
    if perf_g is not None and not perf_g.empty:
        base = merge_on_unit(base, perf_g, how="outer") if not base.empty else perf_g.copy()
    if q_group is not None and not q_group.empty:
        base = merge_on_unit(base, q_group, how="outer") if not base.empty else q_group.copy()
    base = normalize_group(base)
    base = coerce_numeric_columns(base, exclude={"group_id"})

    perf_y_cols = [c for c in ["Score_perf_tsk"] if c in base.columns]
    if perf_y_cols and rcols:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.1 Riedl ↔ Performance (Score final)",
            title_pdf="2.1 Riedl ↔ Performance (Score final)",
            df=base,
            x_cols=rcols,
            y_cols=perf_y_cols,
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_ScoreFinal",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    if perf_y_cols and tci_cols:
        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.2 TCI ↔ Performance (Score final)",
            title_pdf="2.2 TCI ↔ Performance (Score final)",
            df=base,
            x_cols=tci_cols,
            y_cols=perf_y_cols,
            fig_dir=fig_dir,
            fig_prefix="TCI_vs_ScoreFinal",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    if rcols and tci_cols:
        base_tci_riedl = riedl_all.copy() if not riedl_all.empty else pd.DataFrame()
        if not tci_all.empty:
            base_tci_riedl = merge_on_unit(base_tci_riedl, tci_all, how="outer") if not base_tci_riedl.empty else tci_all.copy()
        base_tci_riedl = normalize_group(base_tci_riedl)
        base_tci_riedl = coerce_numeric_columns(base_tci_riedl, exclude={"group_id"})
        tci_passed_scope_table = _build_tci_passed_scope_table(
            tci_all_passed=tci_all_global,
            riedl_available=riedl_all,
            base_tci_riedl=base_tci_riedl,
            rcols=rcols,
            tci_cols=tci_cols,
            extra_exclude_exact=extra_exclude_exact,
            extra_exclude_base=extra_exclude_base,
        )

        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.3 Riedl ↔ TCI (périmètre analytique)",
            title_pdf="2.3 Riedl ↔ TCI (périmètre analytique)",
            df=base_tci_riedl,
            x_cols=rcols,
            y_cols=tci_cols,
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_TCI",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

        tci_all_for_corr = _exclude_groups_df(tci_all_global)
        tci_all_corr_cols = _select_tci_analysis_cols(tci_all_for_corr)
        base_all_tci_riedl = pd.DataFrame()
        if (
            tci_all_for_corr is not None and not tci_all_for_corr.empty and
            riedl_all is not None and not riedl_all.empty and
            tci_all_corr_cols
        ):
            base_all_tci_riedl = merge_on_unit(
                riedl_all.copy(),
                tci_all_for_corr.copy(),
                how="inner",
            )
            base_all_tci_riedl = normalize_group(base_all_tci_riedl)
            base_all_tci_riedl = coerce_numeric_columns(base_all_tci_riedl, exclude={"group_id"})

        diagnostic_title = "Diagnostics de stabilité des corrélations Riedl-TCI"
        lines.append(f"**{diagnostic_title}**\n\n")
        pdf_elems.append(Spacer(1, 0.10 * inch))
        pdf_elems.append(Paragraph(diagnostic_title, styles["Heading4"]))

        bootstrap_note = (
            "Les IC ci-dessous sont des intervalles bootstrap percentile à 95 % "
            "(5000 rééchantillonnages avec remise) calculés sur le périmètre analytique n=12. "
            "La colonne `IC_inclut_0.50` indique si l'intervalle reste compatible avec une corrélation forte mais moins spectaculaire."
        )
        lines.append(bootstrap_note + "\n\n")
        pdf_elems.append(Paragraph(bootstrap_note.replace("`", ""), styles["Normal"]))
        # IC pour toutes les paires calculables : les lignes plus faibles peuvent
        # être nécessaires dans le tableau comparatif n=12 vs n=20.
        bootstrap_pairs = _select_bootstrap_corr_pairs(base_tci_riedl, rcols, tci_cols, min_abs_rho=0.00)
        bootstrap_df = _bootstrap_spearman_ci_table(base_tci_riedl, bootstrap_pairs)
        if not bootstrap_df.empty:
            _rtdir = out_dir / "data_riedl_tci"
            _rtdir.mkdir(parents=True, exist_ok=True)
            bootstrap_df.to_csv(_rtdir / "riedl_tci_bootstrap_spearman_n12.csv", index=False, encoding="utf-8-sig")
            lines.append(md_table_highlight(bootstrap_df, max_rows=40))
            pdf_elems.append(pdf_table_from_df(bootstrap_df, max_rows=40))
            pdf_elems.append(Spacer(1, 0.12 * inch))

            comparison_corr_df = _build_rtd_tci_comparison_table(
                analytic_df=base_tci_riedl,
                extended_df=base_all_tci_riedl,
                x_cols=rcols,
                y_cols=tci_all_corr_cols or tci_cols,
                bootstrap_ci_df=bootstrap_df,
            )
            if not comparison_corr_df.empty:
                comparison_corr_df.to_csv(
                    _rtdir / "riedl_tci_comparison_n12_n20_with_ic.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                comparison_note = (
                    "Tableau comparatif complet : mêmes paires Riedl-TCI sur le périmètre analytique n=12 "
                    "et sur le périmètre TCI étendu n=20, avec IC95 bootstrap pour n=12."
                )
                lines.append(comparison_note + "\n\n")
                lines.append(md_table_highlight(comparison_corr_df, max_rows=40))
                pdf_elems.append(Paragraph(comparison_note, styles["Normal"]))
                pdf_elems.append(pdf_table_from_df(comparison_corr_df, max_rows=40))
                pdf_elems.append(Spacer(1, 0.12 * inch))

        desc_vars = [
            c for c in ["c_score", "skill_mean", "rme_mean", "rme_max", "rme_min"]
            if c in base_tci_riedl.columns or c in base_all_tci_riedl.columns
        ]
        desc_compare_df = _compare_tci_scope_descriptives(base_tci_riedl, base_all_tci_riedl, desc_vars)
        if not desc_compare_df.empty:
            desc_compare_df.to_csv(_rtdir / "riedl_tci_descriptives_n12_vs_n20.csv", index=False, encoding="utf-8-sig")
            desc_note = (
                "Descriptifs comparés entre le périmètre analytique du 2.3 et tous les groupes TCI disponibles, "
                "afin d'identifier si la différence de corrélations vient d'un déplacement de distribution."
            )
            lines.append(desc_note + "\n\n")
            lines.append(md_table_highlight(desc_compare_df, max_rows=30))
            pdf_elems.append(Paragraph(desc_note, styles["Normal"]))
            pdf_elems.append(pdf_table_from_df(desc_compare_df, max_rows=30))
            pdf_elems.append(Spacer(1, 0.12 * inch))

        loo_df, loo_summary = _leave_one_out_spearman_table(
            base_tci_riedl,
            x="skill_mean",
            y="c_score",
            threshold=0.70,
        )
        if not loo_df.empty:
            loo_df.to_csv(_rtdir / "riedl_tci_leave_one_out_skill_mean_c_score.csv", index=False, encoding="utf-8-sig")
            loo_text = (
                "Leave-one-out sur `skill_mean ↔ c_score` : "
                f"rho complet = {loo_summary.get('rho_full')}, "
                f"rho minimal après retrait d'un groupe = {loo_summary.get('rho_min_leave_one_out')}, "
                f"retraits uniques faisant passer rho sous 0.70 = {loo_summary.get('n_retraits_sous_0.70')} "
                f"(un retrait suffit : {loo_summary.get('un_retrait_suffit')})."
            )
            lines.append(loo_text + "\n\n")
            lines.append(md_table_highlight(loo_df, max_rows=20))
            pdf_elems.append(Paragraph(loo_text.replace("`", ""), styles["Normal"]))
            pdf_elems.append(pdf_table_from_df(loo_df, max_rows=20))
            pdf_elems.append(Spacer(1, 0.12 * inch))

            combo_df, combo_summary = _minimal_removal_spearman_search(
                base_tci_riedl,
                x="skill_mean",
                y="c_score",
                threshold=0.70,
                max_remove=4,
            )
            if not combo_df.empty:
                combo_df.to_csv(out_dir / "riedl_tci_minimal_removal_skill_mean_c_score.csv", index=False, encoding="utf-8-sig")
                combo_text = (
                    "Recherche combinatoire complémentaire : "
                    f"il faut retirer au minimum {combo_summary.get('n_min_retraits')} groupe(s) "
                    f"pour faire passer rho sous {combo_summary.get('seuil')}. "
                    f"{combo_summary.get('n_combinaisons_sous_seuil')} combinaison(s) atteignent ce seuil au premier rang de retrait."
                )
                lines.append(combo_text + "\n\n")
                lines.append(md_table_highlight(combo_df, max_rows=10))
                pdf_elems.append(Paragraph(combo_text, styles["Normal"]))
                pdf_elems.append(pdf_table_from_df(combo_df, max_rows=10))
                pdf_elems.append(Spacer(1, 0.12 * inch))
            elif combo_summary:
                combo_text = (
                    "Recherche combinatoire complémentaire : aucune combinaison de retrait jusqu'à "
                    f"{combo_summary.get('max_remove')} groupe(s) ne fait passer rho sous {combo_summary.get('seuil')}."
                )
                lines.append(combo_text + "\n\n")
                pdf_elems.append(Paragraph(combo_text, styles["Normal"]))
                pdf_elems.append(Spacer(1, 0.12 * inch))

        if not tci_passed_scope_table.empty:
            scope_note = (
                "Le tableau ci-dessous liste tous les groupes présents dans le fichier TCI `all` après exclusions documentées du rapport. "
                "La colonne `utilise_2_3` indique les groupes ayant à la fois un score TCI et des "
                "indicateurs Riedl disponibles dans le périmètre analytique de cette section."
            )
            lines.append("**Périmètre des groupes ayant passé le TCI**\n\n")
            lines.append(scope_note + "\n\n")
            lines.append(md_table_highlight(tci_passed_scope_table, max_rows=60))
            pdf_elems.append(Spacer(1, 0.12 * inch))
            pdf_elems.append(Paragraph("Périmètre des groupes ayant passé le TCI", styles["Heading4"]))
            pdf_elems.append(Paragraph(scope_note.replace("`", ""), styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.06 * inch))
            pdf_elems.append(pdf_table_from_df(tci_passed_scope_table, max_rows=60))
            pdf_elems.append(Spacer(1, 0.12 * inch))

    if rcols and q_questionnaire_corr_cols:
        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.4 Riedl ↔ Questionnaire",
            title_pdf="2.4 Riedl ↔ Questionnaire",
            df=base,
            x_cols=rcols,
            y_cols=q_questionnaire_corr_cols[:MAX_Y_COLS],
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_Questionnaire",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    pdf_elems.append(PageBreak())
    lines.append("## 3. Indices non verbaux (INV) et features haut niveau\n")
    pdf_elems.append(Paragraph("3. Indices non verbaux (INV) et features haut niveau", styles["Heading2"]))
    note_inv = (
        "Dans la version v2, les analyses INV detaillees sont externalisees dans un rapport separe pour le sous-ensemble VR (`rapport_INV_VR.pdf`). "
        "La presente variante 2-4 conserve donc exactement la meme logique que le rapport principal et ne reproduit pas ces details ici."
    )
    lines.append(note_inv + "\n\n")
    pdf_elems.append(Paragraph(note_inv, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))

    pdf_elems.append(PageBreak())
    lines.append("## 4. Corrélations les plus fortes\n")
    pdf_elems.append(Paragraph("4. Corrélations les plus fortes", styles["Heading2"]))

    has_p_fdr = any(
        isinstance(row, dict) and pd.notna(row.get("p_fdr"))
        for row in all_corr_results
    )
    sig_label = "p_fdr" if has_p_fdr else "p"
    section4_note = (
        "Cette section distingue les associations les plus fortes "
        "(`|rho| >= 0.55`) et des associations significatives supplémentaires, "
        "plus modestes en taille d'effet (`0.40 <= |rho| < 0.55`, "
        f"`{sig_label} <= 0.05`)."
    )
    lines.append(section4_note + "\n\n")
    pdf_elems.append(Paragraph(section4_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    top_df = top_corr_df(all_corr_results, top_n=50, min_abs_rho=0.55, max_p=0.2)
    lines.append("Corrélations les plus fortes retenues (`|rho| >= 0.55`, `p < 0.20`) :\n\n")
    pdf_elems.append(Paragraph("Corrélations les plus fortes retenues (|rho| >= 0.55, p < 0.20) :", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.05 * inch))
    if top_df.empty:
        lines.append("_(aucune corrélation |rho| ≥ 0.55 avec p < 0.20)_\n\n")
        pdf_elems.append(Paragraph("(aucune corrélation |rho| ≥ 0.55 avec p < 0.20)", styles["Normal"]))
    else:
        lines.append(top_df.to_markdown(index=False) + "\n\n")
        pdf_elems.append(pdf_table_from_df(top_df, max_rows=40))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    pdf_elems.append(Paragraph("4.1 Corrélations significatives supplémentaires", styles["Heading3"]))
    lines.append("### 4.1 Corrélations significatives supplémentaires\n")
    supp_note = (
        "Ce sous-ensemble met en évidence les corrélations avec p < 0.20 qui ne franchissent pas le seuil des effets les plus forts. "
        "Il permet de ne pas perdre des résultats potentiellement interprétables tout en les distinguant des relations les plus robustes."
    )
    lines.append(supp_note + "\n\n")
    pdf_elems.append(Paragraph(supp_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    supp_df = top_corr_df(
        all_corr_results,
        top_n=50,
        min_abs_rho=0.4,
        max_abs_rho=0.55,
        max_p=0.2,
    )
    if supp_df.empty:
        lines.append("_(aucune corrélation avec p < 0.20 pour `0.40 <= |rho| < 0.55`)_\n\n")
        pdf_elems.append(Paragraph(
            "(aucune corrélation avec p < 0.20 pour 0.40 <= |rho| < 0.55)",
            styles["Normal"],
        ))
    else:
        lines.append(supp_df.to_markdown(index=False) + "\n\n")
        pdf_elems.append(pdf_table_from_df(supp_df, max_rows=40))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    pdf_elems.append(Paragraph("4.2 Réseau des corrélations fortes", styles["Heading3"]))
    lines.append("### 4.2 Réseau des corrélations fortes\n")
    net_note = (
        "Le schéma ci-dessous représente les associations les plus robustes entre variables de différentes familles théoriques "
        "(performance, TCI, Riedl, questionnaires). Chaque noeud est une variable, colorié selon sa famille. "
        "Les arêtes rouges indiquent une corrélation positive, les arêtes bleues pointillées une corrélation négative. "
        "L'épaisseur est proportionnelle à |rho| et la taille des noeuds reflète leur degré pondéré."
    )
    lines.append(net_note + "\n\n")
    pdf_elems.append(Paragraph(net_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    network_data = prepare_global_correlation_network_data(
        all_corr_results,
        rho_threshold=0.55,
        p_threshold=0.1,
        min_n=N_MIN_SIG,
    )
    net_png = plot_global_correlation_network(
        all_corr_results,
        out_dir=fig_dir,
        rho_threshold=0.55,
        p_threshold=0.1,
        min_n=N_MIN_SIG,
        network_data=network_data,
    )
    if net_png is not None and net_png.exists():
        lines.append(f"![]({net_png.name})\n_Réseau des corrélations globales fortes (|rho|>=0.55, p<=0.1)_\n\n")
        pdf_elems.append(Image(str(net_png), width=6.5 * inch, height=5.0 * inch))
        pdf_elems.append(Paragraph(
            "<i>Réseau des corrélations globales fortes (|rho| >= 0.55, p <= 0.1). "
            "Couleur = famille théorique. Rouge = corrélation positive. Bleu pointillé = corrélation négative. "
            "Taille du noeud = degré pondéré.</i>",
            styles["Normal"]
        ))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        render_network_metrics_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            network_data=network_data,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            fmt2_fn=fmt2,
            inch_unit=inch,
            section_num="4.3",
            top_n=10,
        )
        fam_png = fig_dir / "global_correlation_family_network.png"
        if fam_png.exists():
            lines.append(f"![]({fam_png.name})\n_Réseau agrégé par famille de variables_\n\n")
            pdf_elems.append(Paragraph("Réseau agrégé par famille de variables :", styles["Heading4"]))
            pdf_elems.append(Image(str(fam_png), width=5.5 * inch, height=4.2 * inch))
            pdf_elems.append(Paragraph(
                "<i>Vue agrégée : chaque noeud est une famille théorique. "
                "L'épaisseur des arêtes = nombre d'associations fortes entre les deux familles. "
                "Utile pour identifier les blocs conceptuellement liés.</i>",
                styles["Normal"]
            ))
            pdf_elems.append(Spacer(1, 0.15 * inch))
    else:
        msg = "Réseau non généré (pas assez de corrélations robustes ou networkx absent)."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    markdown_text = _rewrite_markdown_image_paths(
        "\n".join(lines),
        fig_dir=fig_dir,
        out_dir=out_dir,
    )
    md_path = out_dir / f"{file_stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    html_path = None
    if HAS_MD:
        html_path = out_dir / f"{file_stem}.html"
        html_path.write_text(
            md_lib.markdown(markdown_text, extensions=["tables"]),
            encoding="utf-8"
        )

    pdf_path = out_dir / f"{file_stem}.pdf"
    doc = MyDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.multiBuild(pdf_elems)

    print(f"[OK] MD  : {md_path}")
    if HAS_MD and html_path is not None:
        print(f"[OK] HTML: {html_path}")
    print(f"[OK] PDF : {pdf_path}")
    return {"md_path": md_path, "html_path": html_path, "pdf_path": pdf_path}

def build_report(
    results_dir: Path,
    out_dir: Path,
    apply_fdr: bool = False,
    verbose: bool = False,
    modality_filter: str | None = None,
    apply_pruning: bool = True,
    inv_analysis_mode: str = "pruning",
    pca_rotation: str = "none",
    file_stem: str | None = None,
    report_scope: str = "principal",
    report_title: str = "Rapport CI v2",
    tci_scope: str = "allowed",
    tci_file: Path | None = None,
    tci_all_file: Path | None = None,
    regression_pca_only: bool = False,
    skip_sem: bool = False,
    bayes: bool = True,
    exclude_groups: Iterable[str] | None = None,
    questionnaire_pruning_scope: str = "scoped",
):
    import time as _time
    _t0 = _time.time()
    _step_t = [_t0]

    def _step(label: str) -> None:
        now = _time.time()
        elapsed_total = now - _t0
        elapsed_step = now - _step_t[0]
        _step_t[0] = now
        scope = report_scope if "report_scope" in dir() else "?"
        stem = file_stem if file_stem else f"report_{modality_filter or 'all'}"
        print(f"  [{stem}] {label:<55} +{elapsed_step:5.1f}s  total={elapsed_total:6.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    global all_corr_results
    all_corr_results.clear()

    suffix = "all" if modality_filter is None else str(modality_filter).lower()
    report_scope = str(report_scope).lower().strip()
    file_stem = file_stem or f"report_{suffix}"
    fig_dir = out_dir / f"figs_{file_stem}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    pdf_elems: list = []
    lines: list[str] = []

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            name="TOCHeading1",
            parent=styles["Normal"],
            fontSize=11,
            leading=13,
            leftIndent=10,
            firstLineIndent=-2,
            spaceBefore=6,
        ),
        ParagraphStyle(
            name="TOCHeading2",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            leftIndent=24,
            firstLineIndent=-2,
            spaceBefore=2,
        ),
        ParagraphStyle(
            name="TOCHeading3",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            leftIndent=38,
            firstLineIndent=-2,
            spaceBefore=1,
        ),
    ]

    extra_exclude_exact, extra_exclude_base = _parse_exclude_groups(exclude_groups)

    def _exclude_groups_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
        return exclude_bad_groups(df, extra_excluded=exclude_groups)

    print(f"\n{'='*70}")
    print(f"  BUILD REPORT : {file_stem}  (scope={report_scope}, modality={modality_filter or 'ALL'})")
    print(f"{'='*70}")
    _step("init")

    # -----------------------------------
    # 0) Chargements
    # -----------------------------------
    perf = load_performance(results_dir)
    riedl = load_riedl(results_dir)
    tci = load_tci(results_dir, scope=tci_scope, explicit_path=tci_file)
    tci_all_explicit_file = tci_all_file
    if tci_all_explicit_file is None and tci_file is not None and str(tci_file).endswith("_pop.csv"):
        candidate_all_pop = results_dir / "TCI" / "c_scores_all_pop.csv"
        if candidate_all_pop.exists():
            tci_all_explicit_file = candidate_all_pop
    tci_all_passed = load_tci(results_dir, scope="all", explicit_path=tci_all_explicit_file)
    if verbose and tci.attrs.get("source_path"):
        print(f"[INFO] Source TCI ({tci_scope}) : {tci.attrs['source_path']}")

    inv_face = load_inv_face(results_dir)
    inv_speech = load_inv_speech(results_dir)
    inv_gaze_all = load_inv_gaze_all(results_dir)

    hl, hl_path = load_high_level_features(results_dir)
    q_scores = load_questionnaire_scores(results_dir, use_pruned=apply_pruning)
    q_cronbach = load_questionnaire_cronbach(results_dir)
    q_cronbach_pruned = load_questionnaire_cronbach_pruned(results_dir)
    q_alpha_comp = load_alpha_comparison(results_dir)
    q_explo_summary = load_exploratory_summary(results_dir)
    q_desc_dim = load_desc_dim_questionnaire(results_dir)
    q_desc_dim_pruned = load_desc_dim_pruned(results_dir)
    q_profile = load_questionnaire_participant_profile(results_dir)
    q_comments = load_questionnaire_free_comments(results_dir)

    # Déterminer le dossier INV approprié selon le mode d'analyse
    # Structure : results_inv_structure[_vr_only]/with_pruning ou without_pruning
    inv_base_subdir = "results_inv_structure_vr_only" if modality_filter is not None and str(modality_filter).upper() == "VR" else "results_inv_structure"
    inv_pruning_subdir = "with_pruning" if inv_analysis_mode == "pruning" else "without_pruning"
    inv_subdir = f"{inv_base_subdir}/{inv_pruning_subdir}"

    # Vérifier l'existence du dossier et gérer la rétrocompatibilité
    inv_dir_full = results_dir / inv_subdir
    if not inv_dir_full.exists():
        # Fallback : essayer l'ancien format (sans sous-dossier pruning)
        inv_subdir_fallback = inv_base_subdir
        inv_dir_fallback = results_dir / inv_subdir_fallback
        if inv_dir_fallback.exists():
            print(f"[WARNING] Nouveau format de dossier INV non trouvé ({inv_dir_full}).")
            print(f"          Utilisation du format ancien ({inv_dir_fallback}).")
            print(f"          Relancez analyze_inv_structure.py pour générer les deux modes.")
            inv_subdir = inv_subdir_fallback
        else:
            print(f"[ERROR] Dossier INV introuvable : {inv_dir_full}")
            print(f"        Ni l'ancien format : {inv_dir_fallback}")
            raise FileNotFoundError(f"Dossier INV manquant. Mode demandé : {inv_analysis_mode}")

    print(f"[INFO] Mode d'analyse INV : {inv_analysis_mode}")
    print(f"[INFO] Chargement depuis : {results_dir / inv_subdir}")

    # Features INV non-redondantes (hard pruning |r| > 0.80).
    # En mode pruning, on charge la liste des features prunées.
    # En mode no-pruning, on utilise toutes les features (inv_pruned_features = None).
    inv_pruned_features = load_inv_pruned_features(results_dir, inv_subdir=inv_subdir) if apply_pruning else None
    # Charger la liste complète des prunées pour l'info dans le rapport, même si pruning désactivé
    _all_pruned_info = load_inv_pruned_features_full(results_dir, inv_subdir=inv_subdir)
    inv_pruned_features_base = inv_pruned_features
    _all_pruned_info_base = _all_pruned_info

    # Force cast numérique INV et HL
    inv_face = normalize_timepoint(inv_face)
    inv_speech = normalize_timepoint(inv_speech)
    inv_gaze_all = normalize_timepoint(inv_gaze_all)
    hl = normalize_timepoint(hl)
    perf = normalize_timepoint(perf)
    riedl = normalize_timepoint(riedl)
    tci = normalize_timepoint(tci)
    q_scores = normalize_timepoint(q_scores)

    inv_face = coerce_numeric_columns(inv_face, exclude={"group_id", "group_base_id", "timepoint", "condition", "modalite", "scenario", "session"})
    inv_speech = coerce_numeric_columns(inv_speech, exclude={"group_id", "group_base_id", "timepoint", "condition", "modalite", "scenario", "session"})
    inv_gaze_all = coerce_numeric_columns(inv_gaze_all, exclude={"group_id", "group_base_id", "timepoint", "condition", "modalite", "scenario", "session"})
    hl = coerce_numeric_columns(hl, exclude={"group_id", "group_base_id", "timepoint", "condition", "modalite", "scenario", "session"})

    # Certaines features core Face sont calculées dans le dataset high-level.
    # On les complète ici pour éviter que la section Face ne tombe sur un
    # fallback dépendant de l'ordre des colonnes du CSV brut.
    inv_face = enrich_inv_face_with_high_level(inv_face, hl)
    inv_face = filter_inv_dataframe(inv_face)
    inv_speech = filter_inv_dataframe(inv_speech)
    inv_gaze_all = filter_inv_dataframe(inv_gaze_all)
    hl = filter_inv_dataframe(hl)

    perf = _exclude_groups_df(perf)
    riedl = _exclude_groups_df(riedl)
    riedl_available_for_tci_table = riedl.copy() if riedl is not None else pd.DataFrame()
    tci = _exclude_groups_df(tci)
    q_scores = _exclude_groups_df(q_scores)
    q_profile = _exclude_groups_df(q_profile)
    q_comments = _exclude_groups_df(q_comments)
    inv_face = _exclude_groups_df(inv_face)
    inv_speech = _exclude_groups_df(inv_speech)
    inv_gaze_all = _exclude_groups_df(inv_gaze_all)
    hl = _exclude_groups_df(hl)

    # Riedl/TCI sont des mesures group-level ; on les restreint au même
    # périmètre analytique effectif que le reste du rapport. Cela retire les
    # groupes présents uniquement dans Riedl/TCI mais absents des autres blocs
    # réellement analysés en aval (performance, questionnaire, INV, HLF).
    if (
        riedl is not None and not riedl.empty and
        tci is not None and not tci.empty and
        "group_id" in riedl.columns and "group_id" in tci.columns
    ):
        analysis_scope_groups: set[str] = set()
        for scope_df in [perf, q_scores, inv_face, inv_speech, inv_gaze_all, hl]:
            if scope_df is None or scope_df.empty or "group_id" not in scope_df.columns:
                continue
            analysis_scope_groups |= set(
                scope_df["group_id"].astype(str).str.strip().str.lower().dropna().unique()
            )

        riedl_groups = set(riedl["group_id"].astype(str).str.strip().str.lower().dropna().unique())
        tci_groups = set(tci["group_id"].astype(str).str.strip().str.lower().dropna().unique())
        target_groups = (riedl_groups & tci_groups & analysis_scope_groups) if analysis_scope_groups else (riedl_groups & tci_groups)

        riedl_before_n = len(riedl_groups)
        tci_before_n = len(tci_groups)
        riedl = riedl[
            riedl["group_id"].astype(str).str.strip().str.lower().isin(target_groups)
        ].copy()
        tci = tci[
            tci["group_id"].astype(str).str.strip().str.lower().isin(target_groups)
        ].copy()
        riedl_after_n = riedl["group_id"].astype(str).str.strip().str.lower().nunique()
        tci_after_n = tci["group_id"].astype(str).str.strip().str.lower().nunique()
        if riedl_after_n != riedl_before_n or tci_after_n != tci_before_n:
            print(
                "[Riedl/TCI] Filtrage aligne sur le perimetre analytique final : "
                f"Riedl {riedl_before_n} -> {riedl_after_n}, "
                f"TCI {tci_before_n} -> {tci_after_n} groupes"
            )

    # Conserver les versions non-filtrées par modalité de TCI et Riedl pour les
    # corrélations TCI ↔ Riedl (mesures group-level indépendantes de la modalité).
    riedl_all = riedl.copy() if riedl is not None else pd.DataFrame()
    tci_all = tci.copy() if tci is not None else pd.DataFrame()

    perf, riedl, tci, inv_face, inv_speech, inv_gaze_all, hl, q_scores = apply_modality_filter(
        modality_filter,
        perf,
        riedl,
        tci,
        inv_face,
        inv_speech,
        inv_gaze_all,
        hl,
        q_scores,
    )
    if q_profile is not None and not q_profile.empty and "modalite" in q_profile.columns and modality_filter is not None:
        q_profile = q_profile[q_profile["modalite"].astype(str).str.upper() == str(modality_filter).upper()].copy()
    if q_comments is not None and not q_comments.empty and "modalite" in q_comments.columns and modality_filter is not None:
        q_comments = q_comments[q_comments["modalite"].astype(str).str.upper() == str(modality_filter).upper()].copy()

    # Recalcule les sorties questionnaire sur le périmètre courant du rapport
    # avant toute exclusion ICC : ces tables pilotent 1.4.1 et 1.4.2.
    # questionnaire_pruning_scope="global" : utilise les résultats pruned PC+VR globaux,
    #   filtre uniquement les groupes par modalité — pas de recalcul alpha/pruning.
    # questionnaire_pruning_scope="scoped" (défaut) : recalcule alpha et pruning sur le
    #   sous-échantillon courant (comportement historique).
    q_scoped_outputs = None
    if modality_filter is not None and questionnaire_pruning_scope != "global":
        q_scoped_outputs = recompute_questionnaire_outputs_for_scope(
            results_dir=results_dir,
            apply_pruning=apply_pruning,
            modality_filter=modality_filter,
            excluded_groups_df=None,
            verbose=verbose,
        )
    if q_scoped_outputs is not None:
        q_cronbach = q_scoped_outputs.get("q_cronbach", q_cronbach)
        q_cronbach_pruned = q_scoped_outputs.get("q_cronbach_pruned", q_cronbach_pruned)
        q_alpha_comp = q_scoped_outputs.get("q_alpha_comp", q_alpha_comp)
        q_explo_summary = q_scoped_outputs.get("q_explo_summary", q_explo_summary)
        q_desc_dim = q_scoped_outputs.get("q_desc_dim", q_desc_dim)
        q_desc_dim_pruned = q_scoped_outputs.get("q_desc_dim_pruned", q_desc_dim_pruned)
        q_scores_scoped = q_scoped_outputs.get("q_scores")
        if q_scores_scoped is not None and not q_scores_scoped.empty:
            q_scores = q_scores_scoped.copy()

    # Si un filtre de modalité est actif, recalcule les descriptifs par dimension
    # depuis les scores participant déjà filtrés — indépendamment de questionnaire_pruning_scope.
    # Cela garantit que 1.4.1/1.4.2 reflètent les n_participants VR (ou PC) réels.
    def _desc_from_scores(df: pd.DataFrame) -> pd.DataFrame:
        _score_col = next((c for c in ["score", "Score", "value"] if c in df.columns), None)
        _dim_col = next((c for c in ["dimension", "Dimension"] if c in df.columns), None)
        _label_col = next((c for c in ["dimension_label", "label"] if c in df.columns), None)
        _part_col = next((c for c in ["Participant", "participant", "participant_id"] if c in df.columns), None)
        _items_col = next((c for c in ["n_items", "n_item"] if c in df.columns), None)
        if not _score_col or not _dim_col:
            return pd.DataFrame()
        _grp = df.groupby(_dim_col)
        _agg = _grp[_score_col].agg(
            n_responses="count",
            mean="mean",
            sd="std",
            median="median",
            min="min",
            max="max",
        ).reset_index().rename(columns={_dim_col: "dimension"})
        if _part_col:
            _agg["n_participants"] = _grp[_part_col].nunique().values
        if _items_col:
            _agg["n_items"] = _grp[_items_col].first().values
        if _label_col:
            _agg["label"] = _grp[_label_col].first().values
        for _c in ["mean", "sd", "median", "min", "max"]:
            if _c in _agg.columns:
                _agg[_c] = _agg[_c].round(2)
        return _agg

    if modality_filter is not None and q_scores is not None and not q_scores.empty:
        # q_scores est le fichier non-pruned filtré VR → descriptifs tous items
        _q_scores_orig = load_questionnaire_scores(results_dir, use_pruned=False)
        if _q_scores_orig is not None and not _q_scores_orig.empty and "modalite" in _q_scores_orig.columns:
            _q_scores_orig = _q_scores_orig[
                _q_scores_orig["modalite"].astype(str).str.upper() == str(modality_filter).upper()
            ].copy()
        _desc_orig = _desc_from_scores(_q_scores_orig) if _q_scores_orig is not None and not _q_scores_orig.empty else pd.DataFrame()
        if not _desc_orig.empty:
            q_desc_dim = _desc_orig
        # q_scores (déjà filtré VR, pruned si apply_pruning) → descriptifs après pruning
        _desc_pruned = _desc_from_scores(q_scores)
        if not _desc_pruned.empty:
            q_desc_dim_pruned = _desc_pruned

    pruned_questionnaire_items: set[str] = set()
    if apply_pruning and q_explo_summary is not None and not q_explo_summary.empty and "items_retires" in q_explo_summary.columns:
        for items_str in q_explo_summary["items_retires"].dropna():
            items_str = str(items_str).strip()
            if not items_str:
                continue
            for code in items_str.split(";"):
                code = code.strip()
                if code:
                    pruned_questionnaire_items.add(code)

    # L'analyse ICC intervient après le pruning exploratoire des items.
    # Les items sont d'abord agrégés en score dimensionnel participant ;
    # l'ICC est ensuite calculé sur la matrice groupes x participants.
    q_scores_for_icc_report = q_scores.copy() if q_scores is not None else pd.DataFrame()
    survey_path = _find_questionnaire_source_file(results_dir)
    q_icc_by_dimension = compute_questionnaire_icc_by_dimension_from_survey(
        q_scores_reference=q_scores_for_icc_report,
        survey_path=survey_path,
        removed_item_cols=pruned_questionnaire_items if apply_pruning else None,
    )
    if q_icc_by_dimension is not None and not q_icc_by_dimension.empty:
        _qdir = out_dir / "data_questionnaire"
        _qdir.mkdir(parents=True, exist_ok=True)
        q_icc_by_dimension.to_csv(_qdir / "questionnaire_icc_by_dimension.csv", index=False, encoding="utf-8-sig")
    q_rwg_matrix = compute_questionnaire_rwg_construct_matrix_from_survey(
        q_scores_reference=q_scores_for_icc_report,
        survey_path=survey_path,
        removed_item_cols=pruned_questionnaire_items if apply_pruning else None,
    )
    q_rwg_summary = summarize_questionnaire_rwg_construct_matrix(q_rwg_matrix)
    if q_rwg_matrix is not None and not q_rwg_matrix.empty:
        _qdir = out_dir / "data_questionnaire"
        _qdir.mkdir(parents=True, exist_ok=True)
        q_rwg_matrix.to_csv(_qdir / "questionnaire_rwg_by_group.csv", index=False, encoding="utf-8-sig")
    if q_rwg_summary is not None and not q_rwg_summary.empty:
        _qdir = out_dir / "data_questionnaire"
        _qdir.mkdir(parents=True, exist_ok=True)
        q_rwg_summary.to_csv(_qdir / "questionnaire_rwg_summary.csv", index=False, encoding="utf-8-sig")
    if q_icc_by_dimension is not None and not q_icc_by_dimension.empty:
        print("[Questionnaire ICC] ICC inter-membres calculé par dimension (groupes x participants).")
    else:
        print("[Questionnaire ICC] ICC inter-membres indisponible.")
    _step("chargements + filtrage + ICC questionnaire")

    # -----------------------------------
    # Export CSV fusionné final
    # -----------------------------------
    merged_master = build_group_master_csv(
        perf=perf,
        riedl=riedl,
        tci=tci,
        inv_face=inv_face,
        inv_speech=inv_speech,
        inv_gaze_all=inv_gaze_all,
        hl=hl,
        q_scores=q_scores,
    )

    suffix = "all" if modality_filter is None else str(modality_filter).lower()
    merged_mode_subdir = "with_pruning" if inv_analysis_mode == "pruning" else "without_pruning"

    # Export dataset fusionné et son rapport de structure dans results/merged_dataset/<mode>/
    if merged_master is not None and not merged_master.empty:
        export_paths = export_merged_dataset_bundle(
            merged_master,
            results_dir / "merged_dataset" / merged_mode_subdir,
            suffix=suffix,
            inv_analysis_mode=inv_analysis_mode,
            modality_filter=modality_filter,
            inv_subdir=inv_subdir,
            source_tables={
                "performance": perf,
                "riedl": riedl,
                "tci": tci,
                "questionnaire_scores": q_scores,
                "inv_face": inv_face,
                "inv_speech": inv_speech,
                "inv_gaze": inv_gaze_all,
                "high_level_features": hl,
            },
        )
        print(f"[OK] Dataset complet : {export_paths['csv']}")
        print(f"[OK] Rapport merged dataset : {export_paths['report']}")
    else:
        print("[WARN] CSV fusionné non généré : table vide")
    _step("merged_master + export CSV")

    fig_perf = fig_dir / "performance_modalite.png"
    ok = plot_performance_by_modalite(perf, fig_perf)

    # ==========================================
    # A) Performance agrégée au niveau groupe
    # ==========================================

    perf_g_group = pd.DataFrame()

    if perf is not None and not perf.empty and "group_id" in perf.columns:

        # détecte colonne score
        score_col = "Score_perf_tsk" if "Score_perf_tsk" in perf.columns else None

        if score_col is None:
            cand = [c for c in perf.columns if isinstance(c, str) and "score" in c.lower()]
            score_col = cand[0] if cand else None

        if score_col is not None:
            perf[score_col] = pd.to_numeric(perf[score_col], errors="coerce")

            perf_g_group = (
                perf.groupby("group_id", dropna=False)[score_col]
                .mean()
                .reset_index()
                .rename(columns={score_col: "Score_perf_tsk"})
            )

    perf_g = perf_group_mean(perf)
    q_group, q_dim_cols = questionnaire_group_wide(q_scores)
    q_group, questionnaire_construct_alphas = compute_aggregated_scores(
        q_group,
        alpha_threshold=QUESTIONNAIRE_ALPHA_THRESHOLD,
        include_tms_score=False,
        include_cohesion_score=True,
    )
    inv_face = filter_inv_dataframe(inv_face)
    inv_speech = filter_inv_dataframe(inv_speech)
    inv_gaze_all = filter_inv_dataframe(inv_gaze_all)
    hl = filter_inv_dataframe(hl)
    q_questionnaire_corr_cols = [
        c for c in QUESTIONNAIRE_ANALYSIS_COLS
        if q_group is not None and not q_group.empty and c in q_group.columns and q_group[c].notna().any()
    ]
    if (
        q_group is not None
        and not q_group.empty
        and COHESION_SCORE_COL in q_group.columns
        and q_group[COHESION_SCORE_COL].notna().any()
    ):
        if merged_master is not None and not merged_master.empty and "group_id" in merged_master.columns:
            merged_master = merge_on_unit(
                merged_master,
                q_group[[c for c in ["group_id", "timepoint", COHESION_SCORE_COL] if c in q_group.columns]].drop_duplicates(),
                how="left",
            )

    if q_group is not None and not q_group.empty:
        q_group = q_group[[c for c in q_group.columns if c in {"group_id", "timepoint"} | set(QUESTIONNAIRE_RELIABILITY_DIMENSIONS) | {COHESION_SCORE_COL}]].copy()

    # Variante PCA v2 : espace factoriel restreint aux features retenues par la stepwise INV.
    # Si stepwise_retained_features.json existe (produit par inv_vr/bundle), on l'utilise.
    # Sinon fallback sur la whitelist regression_preferred (première exécution ou mode pca_vr_regression seul).
    if regression_pca_only:
        import json as _json
        _retained_json = out_dir / "stepwise_retained_features.json"
        _loaded_retained: set[str] = set()
        if _retained_json.exists():
            try:
                _loaded_retained = set(_json.loads(_retained_json.read_text(encoding="utf-8")))
                print(f"[INFO] {len(_loaded_retained)} features stepwise retenues chargées depuis {_retained_json}")
            except Exception as _e:
                print(f"[WARN] Impossible de lire stepwise_retained_features.json : {_e}")

        if _loaded_retained:
            # Mode principal : PCA sur les features réellement retenues par la stepwise
            _family_label = {"audio": "Audio/Speech", "face": "Face", "gaze": "Gaze"}
            predictor_groups_for_pca = {}
            for feat in sorted(_loaded_retained):
                fam = infer_family_from_name(feat)
                label = _family_label.get(str(fam).lower(), "Other") if fam else "Other"
                predictor_groups_for_pca.setdefault(label, []).append(feat)
            inv_reg_df_for_pca, _ = build_inv_stepwise_dataset(
                inv_face=inv_face,
                inv_speech=inv_speech,
                inv_gaze_all=inv_gaze_all,
                hl=hl,
                perf_g=perf_g,
                q_group=q_group,
                inv_pruned_features=inv_pruned_features_base,
            )
            # Filtrer inv_reg_df_for_pca aux seules features retenues
            _meta_cols = {"group_id", "timepoint", "scenario", "condition", "modalite"}
            _keep = [c for c in inv_reg_df_for_pca.columns if c in _loaded_retained or c in _meta_cols]
            inv_reg_df_for_pca = inv_reg_df_for_pca[_keep].copy()
        else:
            # Fallback : whitelist regression_preferred (première exécution)
            inv_reg_df_for_pca, predictor_groups_for_pca = build_inv_stepwise_dataset(
                inv_face=inv_face,
                inv_speech=inv_speech,
                inv_gaze_all=inv_gaze_all,
                hl=hl,
                perf_g=perf_g,
                q_group=q_group,
                inv_pruned_features=inv_pruned_features_base,
            )
            print("[INFO] stepwise_retained_features.json absent — fallback whitelist regression_preferred")

        # Pas de pruning sur la PCA régression : on veut exactement les features
        # retenues par la stepwise, sans suppression de redondance supplémentaire.
        _reg_pca_mode = "no-pruning" if _loaded_retained else inv_analysis_mode
        inv_subdir = _ensure_regression_pca_outputs(
            results_dir=results_dir,
            modality_filter=modality_filter,
            inv_analysis_mode=_reg_pca_mode,
            inv_reg_df=inv_reg_df_for_pca,
            predictor_groups=predictor_groups_for_pca,
            merged_master=merged_master,
            pca_rotation=pca_rotation,
        )
        print(f"[INFO] PCA restreinte à la régression chargée depuis : {results_dir / inv_subdir}")
        inv_pruned_features = None  # Pas de pruning sur la PCA régression
        _all_pruned_info = load_inv_pruned_features_full(results_dir, inv_subdir=inv_subdir)

    # colonnes riedl
    rcols = [c for c in CORE_RIEDL_COLS if c in (riedl.columns if riedl is not None else [])]
    if not rcols and riedl is not None and not riedl.empty:
        rcols = [c for c in riedl.columns if pd.api.types.is_numeric_dtype(riedl[c]) and c != "group_id"][:MAX_Y_COLS]

    # colonnes tci
    tci_cols = _select_tci_analysis_cols(tci)

    if report_scope == "inv_only":
        lines.append(f"# {report_title}\n")
        pdf_elems.append(Paragraph(report_title, styles["Heading1"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))

        # Tableau récapitulatif des scores questionnaire (participant-level)
        render_questionnaire_descriptif_table(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            desc_df=q_desc_dim,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            fmt2_fn=fmt2,
            max_rows_md=MAX_ROWS_MD,
            max_rows_pdf=MAX_ROWS_PDF,
            title_suffix=" (participant-level, tous items)",
            section_num="1.4.1",
        )
        render_questionnaire_descriptif_table(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            desc_df=q_desc_dim_pruned,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            fmt2_fn=fmt2,
            max_rows_md=MAX_ROWS_MD,
            max_rows_pdf=MAX_ROWS_PDF,
            title_suffix=" (participant-level, après pruning)",
            section_num="1.4.2",
        )
        pdf_elems.append(PageBreak())

        lines.append("## 3. Indices non verbaux (INV) et features haut niveau\n")
        pdf_elems.append(Paragraph("3. Indices non verbaux (INV) et features haut niveau", styles["Heading2"]))

        inv_gaze_for_sections = _supplement_inv_block_with_high_level(
            inv_gaze_all,
            hl,
            REGRESSION_RETAINED_INV_FEATURES["gaze"],
        )
        inv_reg_df, predictor_groups = build_inv_stepwise_dataset(
            inv_face=inv_face,
            inv_speech=inv_speech,
            inv_gaze_all=inv_gaze_all,
            hl=hl,
            perf_g=perf_g,
            q_group=q_group,
            inv_pruned_features=inv_pruned_features_base,
        )
        if inv_reg_df is not None and not inv_reg_df.empty:
            base_group_order = ["Audio/Speech", "Face", "Gaze"]
            retained_feats: list[str] = []
            for group_name in base_group_order:
                for feat in predictor_groups.get(group_name, []):
                    if feat not in retained_feats:
                        retained_feats.append(feat)
            unit_cols = [c for c in ["group_id", "timepoint", "scenario", "condition", "modalite"] if c in inv_reg_df.columns]
            export_cols = unit_cols + [f for f in retained_feats if f in inv_reg_df.columns]
            if export_cols:
                export_df = inv_reg_df[export_cols].copy()
                if "group_id" in export_df.columns:
                    subset_cols = [c for c in ["group_id", "timepoint", "scenario"] if c in export_df.columns]
                    if subset_cols:
                        export_df = export_df.drop_duplicates(subset=subset_cols)
                        export_df = export_df.sort_values(subset_cols)
                export_path = out_dir / "inv_vr_retained_inv_variables.csv"
                export_df.to_csv(export_path, index=False)
                print(f"[OK] INV VR variables exportées : {export_path}")
        _step("§3.1.1 régression stepwise INV")
        _stepwise_retained = render_inv_stepwise_regression_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            df=inv_reg_df,
            predictor_groups=predictor_groups,
            section_num="3.1.1",
            figs_dir=fig_dir,
        )
        # Sauvegarder les features retenues pour que pca_vr_regression puisse les relire
        if _stepwise_retained:
            import json as _json
            _retained_path = out_dir / "stepwise_retained_features.json"
            _retained_path.write_text(
                _json.dumps(sorted(_stepwise_retained), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[INFO] {len(_stepwise_retained)} features stepwise retenues sauvegardées : {_retained_path}")

        # Reconstruire la PCA régression sur les features réellement retenues par la stepwise
        if regression_pca_only and _stepwise_retained:
            _family_label = {"audio": "Audio/Speech", "face": "Face", "gaze": "Gaze"}
            _pca_groups: dict[str, list[str]] = {}
            for feat in sorted(_stepwise_retained):
                fam = infer_family_from_name(feat)
                label = _family_label.get(str(fam).lower(), "Other") if fam else "Other"
                _pca_groups.setdefault(label, []).append(feat)
            if _pca_groups:
                _pca_frame_ret = inv_reg_df[[
                    c for c in inv_reg_df.columns
                    if c in _stepwise_retained or c in {"group_id", "timepoint", "scenario", "condition", "modalite"}
                ]].copy()
                inv_subdir = _ensure_regression_pca_outputs(
                    results_dir=results_dir,
                    modality_filter=modality_filter,
                    inv_analysis_mode="no-pruning",
                    inv_reg_df=_pca_frame_ret,
                    predictor_groups=_pca_groups,
                    merged_master=merged_master,
                    pca_rotation=pca_rotation,
                )
                print(f"[INFO] PCA régression recalculée sur {len(_stepwise_retained)} features stepwise retenues (no-pruning).")
                inv_pruned_features = None
                _all_pruned_info = load_inv_pruned_features_full(results_dir, inv_subdir=inv_subdir)
        if skip_sem:
            print("[INFO] --no-sem : sections 3.1.3-3.1.6 (PLS-SEM, path analysis, ICC bayesien) sautees.")
            _icc_317, _ctx_317 = pd.DataFrame(), pd.DataFrame()
            _sd_paths_316, _sd_desc_316 = pd.DataFrame(), pd.DataFrame()
        else:
            _step("§3.1.3 path analysis VR (bootstrap)")
            render_refined_path_analysis_vr_section(
                lines=lines,
                pdf_elems=pdf_elems,
                styles=styles,
                merged_master=merged_master,
                results_dir=results_dir,
                out_dir=out_dir,
                fig_dir=fig_dir,
                section_num="3.1.3",
                markdown_heading="####",
                pdf_heading_style="Heading4",
                add_page_break=True,
                cronbach_df=q_cronbach,
            )
            _step("§3.1.4 dispersion SD intra-groupe")
            _sd_paths_316, _sd_desc_316 = render_sd_dispersion_section(
                lines=lines,
                pdf_elems=pdf_elems,
                styles=styles,
                merged_master=merged_master,
                results_dir=results_dir,
                out_dir=out_dir,
                fig_dir=fig_dir,
                section_num="3.1.4",
                markdown_heading="####",
                pdf_heading_style="Heading4",
                add_page_break=True,
            )
            _step("§3.1.5 MLM bayésien ICC + 18 modèles contextuels (MCMC — lent)")
            _icc_317, _ctx_317 = render_mlm_icc_section(
                lines=lines,
                pdf_elems=pdf_elems,
                styles=styles,
                merged_master=merged_master,
                results_dir=results_dir,
                out_dir=out_dir,
                fig_dir=fig_dir,
                section_num="3.1.5",
                markdown_heading="####",
                pdf_heading_style="Heading4",
                add_page_break=True,
                bayes=bayes,
            )
            _step("§3.1.6 synthèse multiniveau")
            render_multilevel_synthesis_section(
                lines=lines,
                pdf_elems=pdf_elems,
                styles=styles,
                df_icc=_icc_317,
                df_ctx=_ctx_317,
                df_sd_paths=_sd_paths_316,
                df_sd_desc=_sd_desc_316,
                section_num="3.1.6",
                markdown_heading="####",
                pdf_heading_style="Heading4",
                add_page_break=True,
            )
        _step("§3.2-3.5 collecte corrélations INV")
        inv_top_corr_results: list[dict[str, Any]] = []
        inv_top_corr_results.extend(
            collect_inv_corr_results(
                "Face", inv_face,
                apply_fdr=apply_fdr,
                inv_pruned_features=inv_pruned_features,
                riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
                rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            )
        )
        inv_top_corr_results.extend(
            collect_inv_corr_results(
                "Speech", inv_speech,
                apply_fdr=apply_fdr,
                inv_pruned_features=inv_pruned_features,
                riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
                rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            )
        )
        inv_top_corr_results.extend(
            collect_inv_corr_results(
                "Gaze", inv_gaze_for_sections,
                apply_fdr=apply_fdr,
                inv_pruned_features=inv_pruned_features,
                riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
                rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            )
        )
        inv_top_corr_results.extend(
            collect_inv_corr_results(
                "High-level", hl,
                apply_fdr=apply_fdr,
                inv_pruned_features=inv_pruned_features,
                riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
                rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            )
        )
        q_cohesion_subdims = [
            c for c in COHESION_SUBDIM_COLS
            if q_group is not None and not q_group.empty and c in q_group.columns and q_group[c].notna().any()
        ]
        if "participation_entropy" in (inv_speech.columns if inv_speech is not None else []):
            participation_block = aggregate_numeric_by_unit(inv_speech, ["participation_entropy"])
            participation_merged = participation_block.copy()
            for aux in [riedl, tci, perf_g, q_group]:
                if aux is not None and not aux.empty:
                    participation_merged = merge_on_unit(participation_merged, aux, how="left")
            participation_targets = [
                c for c in (
                    list(rcols)
                    + list(tci_cols)
                    + [c for c in ["Score_perf_tsk"] if c in participation_merged.columns]
                    + list(q_questionnaire_corr_cols)
                )
                if c in participation_merged.columns and participation_merged[c].notna().any()
            ]
            _, participation_rows = supplemental_spearman_table(
                participation_merged,
                ["participation_entropy"],
                participation_targets,
                block="Speech participation_entropy ↔ cibles complètes",
                sort_by_abs=True,
            )
            inv_top_corr_results.extend(participation_rows)

        # Dans le rapport inv_only, la régression utilise les scores PCA (pas les INV bruts)
        # => predictor_groups / inv_reg_df ne sont pas définis ici ; on passe une liste vide.
        retained_inv_cols = []
        if retained_inv_cols and q_cohesion_subdims:
            _, cohesion_rows = supplemental_spearman_table(
                inv_reg_df,
                retained_inv_cols,
                q_cohesion_subdims,
                block="INV retenus ↔ sous-dimensions Cohésion",
                sort_by_abs=True,
            )
            inv_top_corr_results.extend(cohesion_rows)
        top_inv_df = top_corr_df(
            inv_top_corr_results,
            top_n=20,
            min_abs_rho=0.50,
            max_p=0.2,
        )
        if top_inv_df.empty:
            top_inv_df = top_corr_df(
                inv_top_corr_results,
                top_n=20,
                min_abs_rho=0.40,
                max_p=0.2,
            )
        if not top_inv_df.empty and {"x", "y"}.issubset(top_inv_df.columns):
            top_inv_df = top_inv_df.drop_duplicates(subset=["x", "y"], keep="first").reset_index(drop=True)
        if not top_inv_df.empty:
            lines.append(f"#### 3.1.9 Top corrélations INV\n\n")
            lines.append(
                "Les associations ci-dessous résument les corrélations les plus fortes observées "
                "sur les blocs Face, Speech, Gaze et High-level avant le détail bloc par bloc. "
                "Le classement privilégie d'abord la significativité, puis la taille d'effet absolue.\n\n"
            )
            lines.append(md_table_highlight(top_inv_df, max_rows=20) + "\n")
            pdf_elems.append(Paragraph("3.1.9 Top corrélations INV", styles["Heading4"]))
            pdf_elems.append(
                Paragraph(
                    "Les associations ci-dessous résument les corrélations les plus fortes observées "
                    "sur les blocs Face, Speech, Gaze et High-level avant le détail bloc par bloc. "
                    "Le classement privilégie d'abord la significativité, puis la taille d'effet absolue.",
                    styles["Normal"],
                )
            )
            pdf_elems.append(Spacer(1, 0.06 * inch))
            pdf_elems.append(pdf_table_from_df(top_inv_df, max_rows=20))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        pdf_elems.append(PageBreak())

        _step("§3.2 corrélations INV Face")
        render_inv_section(
            lines, pdf_elems, styles, "3.2", "Face", inv_face,
            fig_dir=fig_dir, apply_fdr=apply_fdr, inv_pruned_features=inv_pruned_features,
            riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
            rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        _step("§3.3 corrélations INV Speech")
        render_inv_section(
            lines, pdf_elems, styles, "3.3", "Speech", inv_speech,
            fig_dir=fig_dir, apply_fdr=apply_fdr, inv_pruned_features=inv_pruned_features,
            riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
            rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        _step("§3.4 corrélations INV Gaze")
        render_inv_section(
            lines, pdf_elems, styles, "3.4", "Gaze", inv_gaze_for_sections,
            fig_dir=fig_dir, apply_fdr=apply_fdr, inv_pruned_features=inv_pruned_features,
            riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
            rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        _step("§3.5 corrélations INV High-level")
        render_inv_section(
            lines, pdf_elems, styles, "3.5", "High-level", hl,
            fig_dir=fig_dir, apply_fdr=apply_fdr, inv_pruned_features=inv_pruned_features,
            riedl=riedl, tci=tci, perf_g=perf_g, q_group=q_group,
            rcols=rcols, tci_cols=tci_cols, q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        if retained_inv_cols and q_cohesion_subdims:
            retained_cohesion_table, retained_cohesion_rows = supplemental_spearman_table(
                inv_reg_df,
                retained_inv_cols,
                q_cohesion_subdims,
                block="INV retenus ↔ sous-dimensions Cohésion",
                sort_by_abs=True,
            )
            if not retained_cohesion_table.empty:
                all_corr_results.extend(retained_cohesion_rows)
                retained_note = (
                    "Synthèse des 14 indicateurs INV retenus en v2 "
                    "(6 Speech, 4 Face, 4 Gaze) croisés avec SOC, TSK et COM. "
                    + _supplemental_corr_note(
                        retained_cohesion_table,
                        context="INV retenus ↔ sous-dimensions Cohésion",
                        n8_warning=False,
                    )
                )
                _append_supplemental_corr_block(
                    lines,
                    pdf_elems,
                    styles,
                    title="3.5.5 Synthèse INV retenus ↔ sous-dimensions Cohésion",
                    table=retained_cohesion_table,
                    note=retained_note,
                    heading_md="####",
                    pdf_heading_style="Heading4",
                    max_rows=60,
                )
        _step("rendu PDF (multiBuild) — inv_only")
        md_path = out_dir / f"{file_stem}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        pdf_path = out_dir / f"{file_stem}.pdf"
        doc = MyDocTemplate(
            str(pdf_path),
            pagesize=landscape(A4),
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.multiBuild(pdf_elems)
        _step("TERMINÉ")
        print(f"[OK] MD  : {md_path}")
        print(f"[OK] PDF : {pdf_path}")
        return

    if report_scope == "pca_only":
        lines.append(f"# {report_title}\n")
        pdf_elems.append(Paragraph(report_title, styles["Heading1"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        if regression_pca_only:
            pca_note = (
                "Cette variante du rapport PCA est restreinte aux features "
                "effectivement utilisées par la régression INV."
            )
            lines.append(f"_{pca_note}_\n\n")
            pdf_elems.append(Paragraph(f"<i>{pca_note}</i>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.08 * inch))
        render_inv_pca_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            results_dir=results_dir,
            fig_dir=fig_dir,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            section_num="5",
            inv_pruned_features=inv_pruned_features,
            all_pruned_info=_all_pruned_info,
            apply_pruning=apply_pruning,
            inv_subdir=inv_subdir,
            merged_data=merged_master,
            inv_face=inv_face,
            inv_speech=inv_speech,
            inv_gaze=inv_gaze_all,
            pca_rotation=pca_rotation,
            max_rows_md=MAX_ROWS_MD,
            max_rows_pdf=MAX_ROWS_PDF,
            show_pruning_audit=not regression_pca_only,
        )
        md_path = out_dir / f"{file_stem}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        pdf_path = out_dir / f"{file_stem}.pdf"
        doc = MyDocTemplate(
            str(pdf_path),
            pagesize=landscape(A4),
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.multiBuild(pdf_elems)
        print(f"[OK] MD  : {md_path}")
        print(f"[OK] PDF : {pdf_path}")
        return

    # -----------------------------------
    # Page titre + Infos générales
    # -----------------------------------
    lines.append(f"# {report_title}\n")
    pdf_elems.append(Paragraph(report_title, styles["Heading1"]))
    pdf_elems.append(Spacer(1, 0.2 * inch))

    pdf_elems.append(Paragraph("Sommaire", styles["Heading1"]))
    pdf_elems.append(Spacer(1, 0.15 * inch))
    pdf_elems.append(toc)
    pdf_elems.append(PageBreak())

    info_lines = [
        "## 1. Infos générales\n",
        "Le rapport v2 repose sur les données agrégées au niveau groupe après exclusion des groupes documentés comme incomplets ou corrompus.\n",
        f"Les comparaisons descriptives sont ensuite construites sur {n_units(perf)} groupe(s) avec performance, {n_units(q_scores)} groupe(s) avec questionnaire, et {n_units(hl)} groupe(s) avec indicateurs high-level.\n\n",
    ]
    if extra_exclude_exact or extra_exclude_base:
        extra_list = ", ".join(sorted(extra_exclude_base) + sorted(extra_exclude_exact))
        info_lines.append(f"Exclusions manuelles appliquées (--exclude-groups) : {extra_list}.\n\n")
    lines.extend(info_lines)
    pdf_elems.append(Paragraph("1. Infos générales", styles["Heading2"]))
    base_info = (
        "Le rapport v2 repose sur les données agrégées au niveau groupe après exclusion des groupes documentés comme incomplets ou corrompus. "
        f"Les comparaisons descriptives sont ensuite construites sur {n_units(perf)} groupe(s) avec performance, "
        f"{n_units(q_scores)} groupe(s) avec questionnaire, et {n_units(hl)} groupe(s) avec indicateurs high-level."
    )
    if extra_exclude_exact or extra_exclude_base:
        extra_list = ", ".join(sorted(extra_exclude_base) + sorted(extra_exclude_exact))
        base_info += f" Exclusions manuelles appliquées (--exclude-groups) : {extra_list}."
    pdf_elems.append(Paragraph(base_info, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.2 * inch))

    lines.append("- groupes exclus de l'analyse:\n")
    for gid in sorted(EXCLUDED_GROUPS):
        reason = EXCLUSION_REASONS.get(gid, "")
        lines.append(f"  - {gid}: {reason}\n")
    if extra_exclude_exact or extra_exclude_base:
        extra_list = ", ".join(sorted(extra_exclude_base) + sorted(extra_exclude_exact))
        lines.append(f"  - exclusions manuelles (--exclude-groups): {extra_list}\n")

    pdf_elems.append(Paragraph("Groupes exclus de l'analyse :", styles["Heading4"]))
    for gid in sorted(EXCLUDED_GROUPS):
        reason = EXCLUSION_REASONS.get(gid, "")
        pdf_elems.append(Paragraph(f"- {gid}: {reason}", styles["Normal"]))
    if extra_exclude_exact or extra_exclude_base:
        extra_list = ", ".join(sorted(extra_exclude_base) + sorted(extra_exclude_exact))
        pdf_elems.append(Paragraph(f"- exclusions manuelles (--exclude-groups): {extra_list}", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.15 * inch))

    avail_counts = {
        "Performance": n_groups(perf),
        "Riedl": n_groups(riedl),
        "TCI": n_groups(tci),
        "Face": n_groups(inv_face),
        "Speech": n_groups(inv_speech),
        "Gaze": n_groups(inv_gaze_all),
        "High-level": n_groups(hl),
        "Questionnaire": n_groups(q_scores),
    }

    avail_fig = fig_dir / "availability_by_modality.png"
    plot_modalities_availability(
        avail_fig,
        avail_counts,
        title="Disponibilité des unités par modalité",
    )

    lines.append("### 1.1 Disponibilité des unités par modalité\n")
    pdf_elems.append(Paragraph("1.1 Disponibilité des unités par modalité", styles["Heading3"]))
    lines.append("Figure ajoutée dans le PDF : nombre d'unités disponibles par source.\n\n")
    pdf_elems.append(Image(str(avail_fig), width=6.5 * inch, height=3.6 * inch))
    pdf_elems.append(Spacer(1, 0.2 * inch))

    render_questionnaire_profile_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        profile_df=q_profile,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        section_num="1.2",
    )

    render_questionnaire_comments_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        comments_df=q_comments,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        section_num="1.2b",
    )

    if verbose :
        print("-----------------------------------------------------------------------------")
        print(modality_filter)
        debug_timepoint(perf, "perf")
        debug_timepoint(riedl, "riedl")
        debug_timepoint(tci, "tci")
        debug_timepoint(q_scores, "q_scores")
        debug_timepoint(inv_face, "inv_face")
        debug_timepoint(inv_speech, "inv_speech")
        debug_timepoint(inv_gaze_all, "inv_gaze_all")
        debug_timepoint(hl, "hl")
        print("-----------------------------------------------------------------------------")
        # print("Speech numeric:", numeric_cols(inv_speech))
        # print("Face numeric:", numeric_cols(inv_face))
        # print("Gaze numeric:", numeric_cols(inv_gaze_all))
        # print("High-level features numeric:", numeric_cols(hl))
        print("Groups Face:", sorted(inv_face["group_id"].unique()))
        print("Groups Gaze:", sorted(inv_gaze_all["group_id"].unique()))
        print("Groups Speech:", sorted(inv_speech["group_id"].unique()))
        print("Groups TCI:", sorted(tci["group_id"].unique()))
        print("Groups Riedl:", sorted(riedl["group_id"].unique()))
        print("Groups Perf:", sorted(perf_g_group["group_id"].unique()))
        print("Groups Questionnaire:", sorted(q_scores["group_id"].unique()))
        print("Groups HLF:", sorted(hl["group_id"].unique()))
        print("-----------------------------------------------------------------------------")
        # for name, df_ in [
        #     ("inv_face", inv_face),
        #     ("inv_speech", inv_speech),
        #     ("inv_gaze_all", inv_gaze_all),
        #     ("hl", hl),
        # ]:
        #     if df_ is not None and not df_.empty:
        #         print(f"[DEBUG] {name} columns = {list(df_.columns)}")
        #         if "modalite" in df_.columns:
        #             print(f"[DEBUG] {name} modalites = {sorted(df_['modalite'].astype(str).str.upper().unique())}")
        #         if "condition" in df_.columns:
        #             print(f"[DEBUG] {name} conditions = {sorted(df_['condition'].astype(str).str.upper().unique())}")
        # print("-----------------------------------------------------------------------------")

    lines.append("### 1.3 Performance finale par modalité\n")
    if ok:
        lines.append(f"![]({fig_perf.name})\n\n")
        pdf_elems.append(Paragraph("1.3 Performance finale par modalité", styles["Heading3"]))
        pdf_elems.append(Image(str(fig_perf), width=5.5*inch, height=3.2*inch))
        pdf_elems.append(Spacer(1, 0.2 * inch))

    # L'ANCOVA additive modalité + scénario devient l'analyse de référence.
    pdf_elems.append(PageBreak())
    render_performance_stats_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        results_dir=results_dir,
        fig_dir=fig_dir,
        section_num="1.3b",
    )
    if str(modality_filter or "").upper() == "VR":
        render_vr_familiarity_performance_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            q_profile=q_profile,
            perf=perf,
            fig_dir=fig_dir,
            out_dir=out_dir,
            section_num="1.3c",
        )

    questionnaire_section = "1.4"

    # -----------------------------------
    # 1) Questionnaire
    # -----------------------------------
    pdf_elems.append(PageBreak())
    lines.append(f"## {questionnaire_section}. Questionnaire\n")
    pdf_elems.append(Paragraph(f"{questionnaire_section} Questionnaire", styles["Heading3"]))
    questionnaire_icc_note = (
        f"L'ordre méthodologique de cette section est le suivant : {questionnaire_section}.1 présente les "
        f"descriptifs des dimensions, {questionnaire_section}.2 présente la "
        f"fiabilité interne initiale des dimensions, {questionnaire_section}.3 applique si nécessaire le "
        f"pruning exploratoire des items pour améliorer l'alpha, puis {questionnaire_section}.6 calcule "
        "l'accord inter-membres (ICC) après moyenne des items par participant et par "
        "dimension. Cette étape produit un ICC2k unique par dimension sur une matrice "
        "`groupes x participants`, sans utiliser les items comme cibles ICC. Un "
        "diagnostic complémentaire `rwg(j)` (James et al., 1984) est aussi reporté pour "
        "les construits `TMS` et `Cohesion`."
    )
    lines.append(questionnaire_icc_note + "\n\n")
    pdf_elems.append(Paragraph(questionnaire_icc_note.replace("`", ""), styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # 1.4.1 Descriptifs par dimension (participant-level, tous items)
    render_questionnaire_descriptif_table(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        desc_df=q_desc_dim,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        fmt2_fn=fmt2,
        max_rows_md=MAX_ROWS_MD,
        max_rows_pdf=MAX_ROWS_PDF,
        title_suffix=" (participant-level, tous items)",
        section_num=f"{questionnaire_section}.1",
    )

    # 1.4.2 Descriptifs par dimension (participant-level, après pruning)
    render_questionnaire_descriptif_table(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        desc_df=q_desc_dim_pruned,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        fmt2_fn=fmt2,
        max_rows_md=MAX_ROWS_MD,
        max_rows_pdf=MAX_ROWS_PDF,
        title_suffix=" (participant-level, après pruning)",
        section_num=f"{questionnaire_section}.2",
    )

    # 1.4.3 Fiabilité interne (alpha de Cronbach)
    render_questionnaire_cronbach_table(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        cronbach_df=q_cronbach,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        max_rows_md=MAX_ROWS_MD,
        max_rows_pdf=MAX_ROWS_PDF,
        section_num=f"{questionnaire_section}.3",
    )

    # 1.4.4 Optimisation exploratoire (item pruning)
    render_questionnaire_pruning_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        explo_summary=q_explo_summary,
        alpha_comp=q_alpha_comp,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        max_rows_md=MAX_ROWS_MD,
        max_rows_pdf=MAX_ROWS_PDF,
        section_num=f"{questionnaire_section}.4",
    )

    # 1.4.5 Accord inter-membres avant agrégation
    render_questionnaire_icc_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        q_scores=q_scores_for_icc_report,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        section_num=f"{questionnaire_section}.5",
        icc_by_dimension_df=q_icc_by_dimension,
        n_groups_retained=n_units(q_scores),
        icc_after_pruning=bool(apply_pruning),
        rwg_matrix_df=q_rwg_matrix,
        rwg_summary_df=q_rwg_summary,
    )

    # 1.4.6 Moyennes group-level des variables questionnaire retenues
    if q_group is not None and not q_group.empty:
        lines.append(f"### {questionnaire_section}.6 Moyennes questionnaire retenues (group-level)\n")
        pdf_elems.append(Paragraph(f"{questionnaire_section}.6 Moyennes questionnaire retenues (group-level)", styles["Heading4"]))

        desc_cols = [c for c in QUESTIONNAIRE_ANALYSIS_COLS if c in q_group.columns]
        desc = q_group[desc_cols].describe().T.reset_index().rename(columns={"index": "dimension"})
        desc["dimension_label"] = desc["dimension"].map(DIMENSION_LABELS_FULL).fillna(desc["dimension"])
        desc = desc[["dimension", "dimension_label", "mean", "std", "min", "max"]]
        for c in ["mean", "std", "min", "max"]:
            desc[c] = desc[c].apply(fmt2)

        if "dimension" in desc.columns:
            order = QUESTIONNAIRE_ANALYSIS_COLS
            rank = {k: i for i, k in enumerate(order)}
            desc = desc.sort_values(by=["dimension"], key=lambda s: s.map(rank).fillna(999)).reset_index(drop=True)

        lines.append(md_table_highlight(desc, max_rows=MAX_ROWS_MD))
        pdf_elems.append(pdf_table_from_df(desc, max_rows=MAX_ROWS_PDF))
        pdf_elems.append(Spacer(1, 0.15 * inch))
    else:
        lines.append(f"### {questionnaire_section}.6 Moyennes questionnaire retenues (group-level)\n")
        pdf_elems.append(Paragraph(f"{questionnaire_section}.6 Moyennes questionnaire retenues (group-level)", styles["Heading4"]))
        lines.append("_(Questionnaire indisponible ou pivot vide)_\n\n")
        pdf_elems.append(Paragraph("(Questionnaire indisponible ou pivot vide)", styles["Normal"]))

    # -----------------------------------
    # 1.4.7) Score agrégé de cohésion
    # -----------------------------------
    lines.append(f"### {questionnaire_section}.7 Score agrégé de cohésion\n")
    pdf_elems.append(Paragraph(f"{questionnaire_section}.7 Score agrégé de cohésion", styles["Heading4"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))
    
    if q_group is not None and not q_group.empty:
        cohesion_alpha = questionnaire_construct_alphas.get("COHESION", np.nan)
        intro = (
            "Les dimensions **SOC**, **TSK** et **COM** sont agrégées en un score "
            "de cohésion uniquement si l'alpha de Cronbach atteint le seuil "
            f"acceptable de {QUESTIONNAIRE_ALPHA_THRESHOLD:.2f}. "
            "Les dimensions TMS (**COR**, **CRE**, **SPE**) restent analysées "
            "séparément dans le rapport ; aucun score TMS moyen n'est injecté "
            "dans les corrélations aval."
        )
        lines.append(intro + "\n\n")
        intro_pdf = (
            intro.replace("**SOC**", "<b>SOC</b>")
            .replace("**TSK**", "<b>TSK</b>")
            .replace("**COM**", "<b>COM</b>")
            .replace("**COR**", "<b>COR</b>")
            .replace("**CRE**", "<b>CRE</b>")
            .replace("**SPE**", "<b>SPE</b>")
        )
        pdf_elems.append(Paragraph(intro_pdf, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

        if np.isnan(cohesion_alpha):
            msg = "Alpha de cohésion non calculable : score latent non créé."
            lines.append(f"_{msg}_\n\n")
            pdf_elems.append(Paragraph(msg, styles["Normal"]))
        elif COHESION_SCORE_COL in q_group.columns:
            msg = (
                f"Alpha de cohésion = {cohesion_alpha:.3f} : score "
                f"`{COHESION_SCORE_COL}` calculé et utilisé dans les corrélations questionnaire."
            )
            lines.append(f"{msg}\n\n")
            pdf_elems.append(Paragraph(msg, styles["Normal"]))

            desc_agg = q_group[[COHESION_SCORE_COL]].describe().T.reset_index().rename(columns={"index": "score"})
            for c in desc_agg.columns:
                if c != "score":
                    desc_agg[c] = pd.to_numeric(desc_agg[c], errors="coerce").apply(fmt2)
            lines.append(md_table_highlight(desc_agg, max_rows=10))
            pdf_elems.append(pdf_table_from_df(desc_agg, max_rows=10))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        else:
            msg = (
                f"Alpha de cohésion = {cohesion_alpha:.3f} (< {QUESTIONNAIRE_ALPHA_THRESHOLD:.2f}) : "
                "score latent non retenu dans les analyses aval."
            )
            lines.append(f"_{msg}_\n\n")
            pdf_elems.append(Paragraph(msg, styles["Normal"]))

        # --- 1.4.7b Analyse approfondie de la fiabilité TMS ---
        render_tms_reliability_analysis(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            q_group=q_group,
            section_num=f"{questionnaire_section}.7b",
        )
    else:
        lines.append("_(Score agrégé de cohésion non disponible — questionnaire vide)_\n\n")
        pdf_elems.append(Paragraph("(Score agrégé de cohésion non disponible)", styles["Normal"]))
    
    pdf_elems.append(Spacer(1, 0.15 * inch))

    # 1.4.8 Effet de la modalité sur les questionnaires
    render_questionnaire_modality_effect_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        q_scores=q_scores,
        fig_dir=fig_dir,
        add_two_plots_row_fn=add_two_plots_row,
        safe_filename_fn=safe_filename,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        section_num=f"{questionnaire_section}.8",
    )

    # -----------------------------------
    # 2) Corrélations “classiques”
    # -----------------------------------
    pdf_elems.append(PageBreak())
    lines.append("## 2. Corrélations globales (group-level)\n")
    pdf_elems.append(Paragraph("2. Corrélations globales (group-level)", styles["Heading2"]))

    # merge base : riedl + tci + perf
    base = riedl.copy() if riedl is not None and not riedl.empty else pd.DataFrame()

    if tci is not None and not tci.empty:
        base = merge_on_unit(base, tci, how="outer") if not base.empty else tci.copy()

    if perf_g is not None and not perf_g.empty:
        base = merge_on_unit(base, perf_g, how="outer") if not base.empty else perf_g.copy()

    if q_group is not None and not q_group.empty:
        base = merge_on_unit(base, q_group, how="outer") if not base.empty else q_group.copy()

    base = normalize_group(base)
    base = coerce_numeric_columns(base, exclude={"group_id"})

    # 2.1 Riedl ↔ Score final
    perf_y_cols = [c for c in ["Score_perf_tsk"] if c in base.columns]
    if perf_y_cols and rcols:
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.1 Riedl ↔ Performance (Score final)",
            title_pdf="2.1 Riedl ↔ Performance (Score final)",
            df=base,
            x_cols=rcols,
            y_cols=perf_y_cols,
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_ScoreFinal",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    # 2.2 TCI ↔ Score final
    if perf_y_cols and tci_cols:
        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.2 TCI ↔ Performance (Score final)",
            title_pdf="2.2 TCI ↔ Performance (Score final)",
            df=base,
            x_cols=tci_cols,
            y_cols=perf_y_cols,
            fig_dir=fig_dir,
            fig_prefix="TCI_vs_ScoreFinal",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    # 2.3 Riedl ↔ TCI (tous groupes, indépendant de la modalité)
    # Les mesures TCI et Riedl sont au niveau groupe (pas par session/modalité),
    # donc on utilise l'ensemble des groupes pour maximiser la puissance statistique.
    if rcols and tci_cols:
        base_tci_riedl = riedl_all.copy() if not riedl_all.empty else pd.DataFrame()
        if not tci_all.empty:
            base_tci_riedl = merge_on_unit(base_tci_riedl, tci_all, how="outer") if not base_tci_riedl.empty else tci_all.copy()
        base_tci_riedl = normalize_group(base_tci_riedl)
        base_tci_riedl = coerce_numeric_columns(base_tci_riedl, exclude={"group_id"})
        tci_passed_scope_table = _build_tci_passed_scope_table(
            tci_all_passed=tci_all_passed,
            riedl_available=riedl_available_for_tci_table,
            base_tci_riedl=base_tci_riedl,
            rcols=rcols,
            tci_cols=tci_cols,
            extra_exclude_exact=extra_exclude_exact,
            extra_exclude_base=extra_exclude_base,
        )

        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.3 Riedl ↔ TCI (périmètre analytique)",
            title_pdf="2.3 Riedl ↔ TCI (périmètre analytique)",
            df=base_tci_riedl,
            x_cols=rcols,
            y_cols=tci_cols,
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_TCI",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

        tci_all_for_corr = _exclude_groups_df(tci_all_passed)
        tci_all_corr_cols = _select_tci_analysis_cols(tci_all_for_corr)
        base_all_tci_riedl = pd.DataFrame()
        if (
            tci_all_for_corr is not None and not tci_all_for_corr.empty and
            riedl_available_for_tci_table is not None and not riedl_available_for_tci_table.empty and
            tci_all_corr_cols
        ):
            base_all_tci_riedl = merge_on_unit(
                riedl_available_for_tci_table.copy(),
                tci_all_for_corr.copy(),
                how="inner",
            )
            base_all_tci_riedl = normalize_group(base_all_tci_riedl)
            base_all_tci_riedl = coerce_numeric_columns(base_all_tci_riedl, exclude={"group_id"})

        diagnostic_title = "Diagnostics de stabilité des corrélations Riedl-TCI"
        lines.append(f"**{diagnostic_title}**\n\n")
        pdf_elems.append(Spacer(1, 0.10 * inch))
        pdf_elems.append(Paragraph(diagnostic_title, styles["Heading4"]))

        bootstrap_note = (
            "Les IC ci-dessous sont des intervalles bootstrap percentile à 95 % "
            "(5000 rééchantillonnages avec remise) calculés sur le périmètre analytique n=12. "
            "La colonne `IC_inclut_0.50` indique si l'intervalle reste compatible avec une corrélation forte mais moins spectaculaire."
        )
        lines.append(bootstrap_note + "\n\n")
        pdf_elems.append(Paragraph(bootstrap_note.replace("`", ""), styles["Normal"]))
        # IC pour toutes les paires calculables : les lignes plus faibles peuvent
        # être nécessaires dans le tableau comparatif n=12 vs n=20.
        bootstrap_pairs = _select_bootstrap_corr_pairs(base_tci_riedl, rcols, tci_cols, min_abs_rho=0.00)
        bootstrap_df = _bootstrap_spearman_ci_table(base_tci_riedl, bootstrap_pairs)
        if not bootstrap_df.empty:
            _rtdir = out_dir / "data_riedl_tci"
            _rtdir.mkdir(parents=True, exist_ok=True)
            bootstrap_df.to_csv(_rtdir / "riedl_tci_bootstrap_spearman_n12.csv", index=False, encoding="utf-8-sig")
            lines.append(md_table_highlight(bootstrap_df, max_rows=40))
            pdf_elems.append(pdf_table_from_df(bootstrap_df, max_rows=40))
            pdf_elems.append(Spacer(1, 0.12 * inch))

            comparison_corr_df = _build_rtd_tci_comparison_table(
                analytic_df=base_tci_riedl,
                extended_df=base_all_tci_riedl,
                x_cols=rcols,
                y_cols=tci_all_corr_cols or tci_cols,
                bootstrap_ci_df=bootstrap_df,
            )
            if not comparison_corr_df.empty:
                comparison_corr_df.to_csv(
                    _rtdir / "riedl_tci_comparison_n12_n20_with_ic.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                comparison_note = (
                    "Tableau comparatif complet : mêmes paires Riedl-TCI sur le périmètre analytique n=12 "
                    "et sur le périmètre TCI étendu n=20, avec IC95 bootstrap pour n=12."
                )
                lines.append(comparison_note + "\n\n")
                lines.append(md_table_highlight(comparison_corr_df, max_rows=40))
                pdf_elems.append(Paragraph(comparison_note, styles["Normal"]))
                pdf_elems.append(pdf_table_from_df(comparison_corr_df, max_rows=40))
                pdf_elems.append(Spacer(1, 0.12 * inch))

        desc_vars = [
            c for c in ["c_score", "skill_mean", "rme_mean", "rme_max", "rme_min"]
            if c in base_tci_riedl.columns or c in base_all_tci_riedl.columns
        ]
        desc_compare_df = _compare_tci_scope_descriptives(base_tci_riedl, base_all_tci_riedl, desc_vars)
        if not desc_compare_df.empty:
            desc_compare_df.to_csv(_rtdir / "riedl_tci_descriptives_n12_vs_n20.csv", index=False, encoding="utf-8-sig")
            desc_note = (
                "Descriptifs comparés entre le périmètre analytique du 2.3 et tous les groupes TCI disponibles, "
                "afin d'identifier si la différence de corrélations vient d'un déplacement de distribution."
            )
            lines.append(desc_note + "\n\n")
            lines.append(md_table_highlight(desc_compare_df, max_rows=30))
            pdf_elems.append(Paragraph(desc_note, styles["Normal"]))
            pdf_elems.append(pdf_table_from_df(desc_compare_df, max_rows=30))
            pdf_elems.append(Spacer(1, 0.12 * inch))

        loo_df, loo_summary = _leave_one_out_spearman_table(
            base_tci_riedl,
            x="skill_mean",
            y="c_score",
            threshold=0.70,
        )
        if not loo_df.empty:
            loo_df.to_csv(_rtdir / "riedl_tci_leave_one_out_skill_mean_c_score.csv", index=False, encoding="utf-8-sig")
            loo_text = (
                "Leave-one-out sur `skill_mean ↔ c_score` : "
                f"rho complet = {loo_summary.get('rho_full')}, "
                f"rho minimal après retrait d'un groupe = {loo_summary.get('rho_min_leave_one_out')}, "
                f"retraits uniques faisant passer rho sous 0.70 = {loo_summary.get('n_retraits_sous_0.70')} "
                f"(un retrait suffit : {loo_summary.get('un_retrait_suffit')})."
            )
            lines.append(loo_text + "\n\n")
            lines.append(md_table_highlight(loo_df, max_rows=20))
            pdf_elems.append(Paragraph(loo_text.replace("`", ""), styles["Normal"]))
            pdf_elems.append(pdf_table_from_df(loo_df, max_rows=20))
            pdf_elems.append(Spacer(1, 0.12 * inch))

        if not tci_passed_scope_table.empty:
            scope_note = (
                "Le tableau ci-dessous liste tous les groupes présents dans le fichier TCI `all` après exclusions documentées du rapport. "
                "La colonne `utilise_2_3` indique les groupes ayant à la fois un score TCI et des "
                "indicateurs Riedl disponibles dans le périmètre analytique de cette section."
            )
            lines.append("**Périmètre des groupes ayant passé le TCI**\n\n")
            lines.append(scope_note + "\n\n")
            lines.append(md_table_highlight(tci_passed_scope_table, max_rows=60))
            pdf_elems.append(Spacer(1, 0.12 * inch))
            pdf_elems.append(Paragraph("Périmètre des groupes ayant passé le TCI", styles["Heading4"]))
            pdf_elems.append(Paragraph(scope_note.replace("`", ""), styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.06 * inch))
            pdf_elems.append(pdf_table_from_df(tci_passed_scope_table, max_rows=60))
            pdf_elems.append(Spacer(1, 0.12 * inch))

        if (
            base_all_tci_riedl is not None and not base_all_tci_riedl.empty and
            tci_all_corr_cols
        ):
            all_tci_note = (
                "Analyse complémentaire : ce tableau utilise tous les groupes présents dans le fichier TCI `all` "
                "après exclusions documentées du rapport, puis conserve les lignes disposant aussi des indicateurs Riedl. "
                "Il permet de comparer les associations Riedl-TCI au tableau 2.3, qui reste limité au périmètre analytique principal."
            )
            lines.append(all_tci_note + "\n\n")
            pdf_elems.append(Paragraph(all_tci_note.replace("`", ""), styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.08 * inch))
            # Table informative uniquement : elle ne modifie pas le réseau global des corrélations fortes.
            render_corr_block(
                lines, pdf_elems, styles,
                title_md="2.3b Riedl ↔ TCI (tous les groupes TCI disponibles)",
                title_pdf="2.3b Riedl ↔ TCI (tous les groupes TCI disponibles)",
                df=base_all_tci_riedl,
                x_cols=rcols,
                y_cols=tci_all_corr_cols,
                fig_dir=fig_dir,
                fig_prefix="Riedl_vs_TCI_all_available",
                apply_fdr=apply_fdr,
                all_corr_results=[],
                top_k_plots=3,
            )

    # 2.4 Riedl ↔ Questionnaire
    if rcols and q_questionnaire_corr_cols:
        pdf_elems.append(PageBreak())
        render_corr_block(
            lines, pdf_elems, styles,
            title_md="2.4 Riedl ↔ Questionnaire",
            title_pdf="2.4 Riedl ↔ Questionnaire",
            df=base,
            x_cols=rcols,
            y_cols=q_questionnaire_corr_cols[:MAX_Y_COLS],
            fig_dir=fig_dir,
            fig_prefix="Riedl_vs_Questionnaire",
            apply_fdr=apply_fdr,
            all_corr_results=all_corr_results,
            top_k_plots=3,
        )

    cohesion_subdim_corr_cols = [
        c for c in COHESION_SUBDIM_COLS
        if c in base.columns and pd.to_numeric(base[c], errors="coerce").notna().any()
    ]
    input_subdim_cols = [
        c for c in [
            "skill_mean",
            "skill_max",
            "skill_congruence_mean",
            "contribution_mean",
            "strategy_norm",
            "effort_task_norm",
            "c_score",
            "rme_max",
            "rme_mean",
            "rme_min",
        ]
        if c in base.columns and pd.to_numeric(base[c], errors="coerce").notna().any()
    ]
    if input_subdim_cols and cohesion_subdim_corr_cols:
        riedl_cohesion_table, riedl_cohesion_rows = supplemental_spearman_table(
            base,
            input_subdim_cols,
            cohesion_subdim_corr_cols,
            block="2.4a Inputs Riedl/TCI ↔ sous-dimensions Cohésion",
            sort_by_abs=True,
        )
        if not riedl_cohesion_table.empty:
            all_corr_results.extend(riedl_cohesion_rows)
            riedl_note = _supplemental_corr_note(
                riedl_cohesion_table,
                context="Inputs Riedl/TCI ↔ sous-dimensions Cohésion",
                n8_warning=True,
            )
            _append_supplemental_corr_block(
                lines,
                pdf_elems,
                styles,
                title="2.4a Inputs Riedl/TCI ↔ sous-dimensions Cohésion",
                table=riedl_cohesion_table,
                note=riedl_note,
                heading_md="####",
                pdf_heading_style="Heading4",
                max_rows=40,
            )

    tci_questionnaire_cols = [
        c for c in ["COR", "CRE", "SPE", COHESION_SCORE_COL]
        if c in base.columns and pd.to_numeric(base[c], errors="coerce").notna().any()
    ]
    tci_direct_cols = [
        c for c in ["c_score", "rme_max", "rme_mean", "rme_min"]
        if c in base.columns and pd.to_numeric(base[c], errors="coerce").notna().any()
    ]
    if tci_direct_cols and tci_questionnaire_cols:
        tci_questionnaire_table, tci_questionnaire_rows = supplemental_spearman_table(
            base,
            tci_direct_cols,
            tci_questionnaire_cols,
            block="2.4b TCI ↔ Questionnaire",
            sort_by_abs=True,
        )
        if not tci_questionnaire_table.empty:
            all_corr_results.extend(tci_questionnaire_rows)
            tci_note = (
                "Cette section comble l'asymétrie avec le bloc Riedl et teste directement "
                "le bras TCI → médiateurs transactifs du modèle IMOI. "
                + _supplemental_corr_note(
                    tci_questionnaire_table,
                    context="TCI ↔ Questionnaire",
                    n8_warning=True,
                )
            )
            _append_supplemental_corr_block(
                lines,
                pdf_elems,
                styles,
                title="2.4b TCI ↔ Questionnaire",
                table=tci_questionnaire_table,
                note=tci_note,
                heading_md="####",
                pdf_heading_style="Heading4",
                max_rows=40,
            )

    pdf_elems.append(PageBreak())
    render_h2b_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        merged_master=merged_master,
        out_dir=out_dir,
        fig_dir=fig_dir,
        md_table_fn=md_table_highlight,
        pdf_table_fn=pdf_table_from_df,
        section_num="2.5",
        parent_tci_path=(results_dir / "TCI" / "c_scores_allowed_pop.csv") if tci_file is not None and str(tci_file).endswith("_pop.csv") else None,
    )

    pdf_elems.append(Spacer(1, 0.2 * inch))

    # -----------------------------------
    # 3) INV + High-level
    # -----------------------------------
    if report_scope == "principal":
        pdf_elems.append(PageBreak())
        lines.append("## 3. Indices non verbaux (INV) et features haut niveau\n")
        pdf_elems.append(Paragraph("3. Indices non verbaux (INV) et features haut niveau", styles["Heading2"]))
        note_inv = (
            "Dans la version v2, les analyses INV détaillées sont externalisées "
            "dans un rapport séparé pour le sous-ensemble VR (`rapport_INV_VR.pdf`). "
            "Le rapport principal ne reproduit donc pas cette section."
        )
        lines.append(note_inv + "\n\n")
        pdf_elems.append(Paragraph(note_inv, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
    else:
        pdf_elems.append(PageBreak())
        lines.append("## 3. Indices non verbaux (INV) et features haut niveau\n")
        pdf_elems.append(Paragraph("3. Indices non verbaux (INV) et features haut niveau", styles["Heading2"]))
        render_inv_section(
            lines, pdf_elems, styles,
            "3.1", "Face", inv_face,
            fig_dir=fig_dir,
            apply_fdr=apply_fdr,
            inv_pruned_features=inv_pruned_features,
            riedl=riedl,
            tci=tci,
            perf_g=perf_g,
            q_group=q_group,
            rcols=rcols,
            tci_cols=tci_cols,
            q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        render_inv_section(
            lines, pdf_elems, styles,
            "3.2", "Speech", inv_speech,
            fig_dir=fig_dir,
            apply_fdr=apply_fdr,
            inv_pruned_features=inv_pruned_features,
            riedl=riedl,
            tci=tci,
            perf_g=perf_g,
            q_group=q_group,
            rcols=rcols,
            tci_cols=tci_cols,
            q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        render_inv_section(
            lines, pdf_elems, styles,
            "3.3", "Gaze", inv_gaze_all,
            fig_dir=fig_dir,
            apply_fdr=apply_fdr,
            inv_pruned_features=inv_pruned_features,
            riedl=riedl,
            tci=tci,
            perf_g=perf_g,
            q_group=q_group,
            rcols=rcols,
            tci_cols=tci_cols,
            q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )
        pdf_elems.append(PageBreak())
        render_inv_section(
            lines, pdf_elems, styles,
            "3.4", "High-level", hl,
            fig_dir=fig_dir,
            apply_fdr=apply_fdr,
            inv_pruned_features=inv_pruned_features,
            riedl=riedl,
            tci=tci,
            perf_g=perf_g,
            q_group=q_group,
            rcols=rcols,
            tci_cols=tci_cols,
            q_questionnaire_cols=q_questionnaire_corr_cols,
            all_corr_results=all_corr_results,
        )

        inv_reg_df, predictor_groups = build_inv_stepwise_dataset(
            inv_face=inv_face,
            inv_speech=inv_speech,
            inv_gaze_all=inv_gaze_all,
            hl=hl,
            perf_g=perf_g,
            q_group=q_group,
            inv_pruned_features=inv_pruned_features,
        )
        pdf_elems.append(PageBreak())
        render_inv_stepwise_regression_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            df=inv_reg_df,
            predictor_groups=predictor_groups,
            section_num="3.5",
            figs_dir=fig_dir,
        )

    # ==========================================
    # 4) TOP CORRELATIONS
    # ==========================================

    pdf_elems.append(PageBreak())
    lines.append("## 4. Corrélations les plus fortes\n")
    pdf_elems.append(Paragraph("4. Corrélations les plus fortes", styles["Heading2"]))

    has_p_fdr = any(
        isinstance(row, dict) and pd.notna(row.get("p_fdr"))
        for row in all_corr_results
    )
    sig_label = "p_fdr" if has_p_fdr else "p"
    section4_note = (
        "Cette section distingue les associations les plus fortes "
        "(`|rho| >= 0.55`) et des associations significatives supplémentaires, "
        "plus modestes en taille d'effet (`0.40 <= |rho| < 0.55`, "
        f"`{sig_label} <= 0.05`)."
    )
    lines.append(section4_note + "\n\n")
    pdf_elems.append(Paragraph(section4_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    top_df = top_corr_df(all_corr_results, top_n=50, min_abs_rho=0.55, max_p=0.2)

    lines.append("Corrélations les plus fortes retenues (`|rho| >= 0.55`, `p < 0.20`) :\n\n")
    pdf_elems.append(Paragraph("Corrélations les plus fortes retenues (|rho| >= 0.55, p < 0.20) :", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.05 * inch))

    if top_df.empty:
        lines.append("_(aucune corrélation |rho| ≥ 0.55 avec p < 0.20)_\n\n")
        pdf_elems.append(Paragraph("(aucune corrélation |rho| ≥ 0.55 avec p < 0.20)", styles["Normal"]))
    else:
        lines.append(top_df.to_markdown(index=False) + "\n\n")
        pdf_elems.append(pdf_table_from_df(top_df, max_rows=40))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    # ---- 4.1 Corrélations significatives supplémentaires ----
    pdf_elems.append(Paragraph("4.1 Corrélations significatives supplémentaires", styles["Heading3"]))
    lines.append("### 4.1 Corrélations significatives supplémentaires\n")

    supp_note = (
        "Ce sous-ensemble met en évidence les corrélations avec p < 0.20 qui ne franchissent pas le seuil des effets les plus "
        "forts. Il permet de ne pas perdre des résultats potentiellement "
        "interprétables, comme certaines associations face ou questionnaire, "
        "tout en les distinguant des relations les plus robustes."
    )
    lines.append(supp_note + "\n\n")
    pdf_elems.append(Paragraph(supp_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    supp_df = top_corr_df(
        all_corr_results,
        top_n=50,
        min_abs_rho=0.4,
        max_abs_rho=0.55,
        max_p=0.2,
    )

    if supp_df.empty:
        lines.append("_(aucune corrélation avec p < 0.20 pour `0.40 <= |rho| < 0.55`)_\n\n")
        pdf_elems.append(Paragraph(
            "(aucune corrélation avec p < 0.20 pour 0.40 <= |rho| < 0.55)",
            styles["Normal"],
        ))
    else:
        lines.append(supp_df.to_markdown(index=False) + "\n\n")
        pdf_elems.append(pdf_table_from_df(supp_df, max_rows=40))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    # ---- 4.2 Réseau des corrélations globales fortes ----
    pdf_elems.append(Paragraph("4.2 Réseau des corrélations fortes", styles["Heading3"]))
    lines.append("### 4.2 Réseau des corrélations fortes\n")

    net_note = (
        "Le schéma ci-dessous représente les associations les plus robustes entre variables "
        "de différentes familles théoriques (performance, TCI, Riedl, questionnaires, INV audio/gaze/face). "
        "Chaque noeud est une variable, colorié selon sa famille. "
        "Les arêtes rouges indiquent une corrélation positive, les arêtes bleues (pointillés) une corrélation négative. "
        "L'épaisseur est proportionnelle à |rho|. La taille des noeuds reflète leur degré pondéré (nombre de liens forts). "
        "Cette visualisation permet d'identifier les variables centrales et les ponts entre blocs théoriques."
    )
    lines.append(net_note + "\n\n")
    pdf_elems.append(Paragraph(net_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    print("\n[RÉSEAU CORRÉLATIONS] Construction du réseau global...")
    network_data = prepare_global_correlation_network_data(
        all_corr_results,
        rho_threshold=0.55,
        p_threshold=0.1,
        min_n=N_MIN_SIG,
    )
    net_png = plot_global_correlation_network(
        all_corr_results,
        out_dir=fig_dir,
        rho_threshold=0.55,
        p_threshold=0.1,
        min_n=N_MIN_SIG,
        network_data=network_data,
    )

    if net_png is not None and net_png.exists():
        lines.append(f"![]({net_png.name})\n_Réseau des corrélations globales fortes (|rho|>=0.55, p<=0.1)_\n\n")
        pdf_elems.append(Image(str(net_png), width=6.5 * inch, height=5.0 * inch))
        pdf_elems.append(Paragraph(
            "<i>Réseau des corrélations globales fortes (|rho| >= 0.55, p <= 0.1). "
            "Couleur = famille théorique. Rouge = corrélation positive. Bleu pointillé = corrélation négative. "
            "Taille du noeud = degré pondéré.</i>",
            styles["Normal"]
        ))
        pdf_elems.append(Spacer(1, 0.15 * inch))

        render_network_metrics_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            network_data=network_data,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            fmt2_fn=fmt2,
            inch_unit=inch,
            section_num="4.3",
            top_n=10,
        )

        # Graphe agrégé par famille
        fam_png = fig_dir / "global_correlation_family_network.png"
        if fam_png.exists():
            lines.append(f"![]({fam_png.name})\n_Réseau agrégé par famille de variables_\n\n")
            pdf_elems.append(Paragraph("Réseau agrégé par famille de variables :", styles["Heading4"]))
            pdf_elems.append(Image(str(fam_png), width=5.5 * inch, height=4.2 * inch))
            pdf_elems.append(Paragraph(
                "<i>Vue agrégée : chaque noeud est une famille théorique. "
                "L'épaisseur des arêtes = nombre d'associations fortes entre les deux familles. "
                "Utile pour identifier les blocs conceptuellement liés.</i>",
                styles["Normal"]
            ))
            pdf_elems.append(Spacer(1, 0.15 * inch))

        # Tableau top noeuds
        nodes_csv = fig_dir / "global_correlation_network_nodes.csv"
        if nodes_csv.exists():
            try:
                nodes_top = pd.read_csv(nodes_csv).head(15)
                pdf_elems.append(Paragraph(
                    "Top 15 variables les plus connectées (degree pondéré) :",
                    styles["Heading4"]
                ))
                pdf_elems.append(pdf_table_from_df(nodes_top))
                pdf_elems.append(Spacer(1, 0.1 * inch))
            except Exception:
                pass
    else:
        msg = "Réseau non généré (pas assez de corrélations robustes ou networkx absent)."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))


    # ==========================================
    # 5 PCA / Analyse factorielle des INV
    # ==========================================
    pdf_elems.append(PageBreak())
    if report_scope == "principal":
        lines.append("## 5 Analyse factorielle des INV (PCA + clustering)\n")
        pdf_elems.append(Paragraph("5 Analyse factorielle des INV (PCA + clustering)", styles["Heading2"]))
        note_pca = (
            "Dans la version v2, l'analyse PCA des INV est diffusée séparément "
            "dans `rapport_PCA_VR.pdf` pour le sous-ensemble VR. Le rapport principal "
            "ne reproduit donc pas cette section."
        )
        lines.append(note_pca + "\n\n")
        pdf_elems.append(Paragraph(note_pca, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
    else:
        render_inv_pca_section(
            lines=lines,
            pdf_elems=pdf_elems,
            styles=styles,
            results_dir=results_dir,
            fig_dir=fig_dir,
            md_table_fn=md_table_highlight,
            pdf_table_fn=pdf_table_from_df,
            section_num="5",
            inv_pruned_features=inv_pruned_features,
            all_pruned_info=_all_pruned_info,
            apply_pruning=apply_pruning,
            inv_subdir=inv_subdir,
            merged_data=merged_master,
            inv_face=inv_face,
            inv_speech=inv_speech,
            inv_gaze=inv_gaze_all,
            pca_rotation=pca_rotation,
            max_rows_md=MAX_ROWS_MD,
            max_rows_pdf=MAX_ROWS_PDF,
        )

    # -----------------------------------
    # 6) Diagramme théorique + soutien empirique
    # -----------------------------------
    pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph("6 Modèle théorique et soutien empirique", styles["Heading2"]))
    lines.append("\n## 6 Modèle théorique et soutien empirique\n")

    theory_intro = (
        "Cette section synthétise le cadre conceptuel du projet sous la forme d'une "
        "structure hiérarchique `Input -> Mediators -> Outputs`. Le nœud "
        "`CI / C-factor` représente le potentiel collectif initial, à la fois comme "
        "capital collectif d'entrée et comme formalisation empirique via les scores "
        "`c_score` et `rme_*`. Il alimente théoriquement les dynamiques médiatrices "
        "(TMS, TAS, TRS), tout en conservant un lien direct attendu avec la performance. "
        "Le panneau A présente la structure théorique seule ; le panneau B superpose "
        "les liens empiriques soutenus par les corrélations agrégées au niveau des "
        "construits. Les relations latent-dimension questionnaire ne sont affichées "
        "dans la vue empirique que lorsque la cohérence interne du construit est "
        f"acceptable (alpha de Cronbach ≥ {QUESTIONNAIRE_ALPHA_THRESHOLD:.2f})."
    )
    lines.append(theory_intro + "\n\n")
    pdf_elems.append(Paragraph(theory_intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.15 * inch))

    theory_reading = (
        "Les liens théoriques sont tracés en gris pointillé. Les liens empiriques "
        "positifs apparaissent en rouge, les liens négatifs en bleu, et les liens "
        "exploratoires dans une couleur dédiée. Les modalités non verbales "
        "(audio, face, regard) sont explicitement représentées comme une couche "
        "de mesure multimodale servant à opérationnaliser les processus médiateurs, "
        "et non comme des construits théoriques centraux de même niveau."
    )
    lines.append(theory_reading + "\n\n")
    pdf_elems.append(Paragraph(theory_reading, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))

    print("\n[DIAGRAMME THÉORIQUE] Génération des diagrammes...")
    theory_pngs = generate_theory_diagrams(
        all_corr_results,
        out_dir=fig_dir,
        latent_construct_alphas=questionnaire_construct_alphas,
        latent_alpha_threshold=QUESTIONNAIRE_ALPHA_THRESHOLD,
    )

    if theory_pngs.get("comparison") and theory_pngs["comparison"].exists():
        lines.append("### 6.1 Vue comparée (théorie vs empirique)\n")
        pdf_elems.append(Paragraph("6.1 Vue comparée (théorie vs empirique)", styles["Heading3"]))
        lines.append(f"![]({theory_pngs['comparison'].name})\n\n")
        pdf_elems.append(Image(str(theory_pngs["comparison"]), width=6.5 * inch, height=6.5 * inch))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    # -----------------------------------
    # 7) Articulation computationnelle
    # -----------------------------------
    pdf_elems.append(PageBreak())
    pdf_elems.append(Paragraph("7 Articulation entre modèle computationnel et mesures empiriques", styles["Heading2"]))
    lines.append("\n## 7 Articulation entre modèle computationnel et mesures empiriques\n")

    comp_intro = (
        "Cette section explicite la chaîne d'opérationnalisation reliant le cadre "
        "théorique, les signaux multimodaux effectivement capturés dans le projet, "
        "les indicateurs calculés à partir de ces signaux, puis les construits "
        "médiateurs mobilisés dans les analyses corrélationnelles, factorielles et "
        "interprétatives du rapport."
    )
    lines.append(comp_intro + "\n\n")
    pdf_elems.append(Paragraph(comp_intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))

    comp_pngs = generate_computational_diagram(out_dir=fig_dir)
    if comp_pngs.get("computation") and comp_pngs["computation"].exists():
        lines.append(f"![]({comp_pngs['computation'].name})\n\n")
        pdf_elems.append(Image(str(comp_pngs["computation"]), width=7.15 * inch, height=4.45 * inch))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    lines.append("### 7.1 Input layer\n")
    pdf_elems.append(Paragraph("7.1 Input layer", styles["Heading3"]))
    txt_81 = (
        "Le niveau d'entrée rassemble la composition de l'équipe, le contexte "
        "expérimental et les caractéristiques de la tâche. Dans le pipeline actuel, "
        "ces éléments ne sont pas tous modélisés comme variables explicites dans la "
        "même table, mais ils structurent la variabilité inter-groupe via la "
        "modalité (`PC` / `VR`), le timepoint, et les contraintes "
        "propres à l'exécution BIM collaborative."
    )
    lines.append(txt_81 + "\n\n")
    pdf_elems.append(Paragraph(txt_81, styles["Normal"]))

    lines.append("### 7.2 C-factor / collective potential\n")
    pdf_elems.append(Paragraph("7.2 C-factor / collective potential", styles["Heading3"]))
    txt_82 = (
        "Le `C-factor` est traité ici comme une estimation du potentiel collectif "
        "initial, opérationnalisée par `c_score` ainsi que par les indicateurs "
        "`rme_mean`, `rme_min` et `rme_max`. Dans le diagramme refactoré, ces "
        "mesures sont fusionnées dans le nœud `CI / C-factor`, afin d'éviter de "
        "séparer artificiellement la formalisation empirique du capital collectif "
        "qu'elle cherche précisément à quantifier."
    )
    lines.append(txt_82 + "\n\n")
    pdf_elems.append(Paragraph(txt_82, styles["Normal"]))

    lines.append("### 7.3 Signal layer\n")
    pdf_elems.append(Paragraph("7.3 Signal layer", styles["Heading3"]))
    txt_83 = (
        "Le niveau des signaux regroupe les dynamiques de regard, l'activité vocale "
        "et paraverbale, ainsi que les expressions faciales. Concrètement, le projet "
        "mobilise des familles de mesures telles que `shared_obj_ratio`, "
        "`shared_obj_dur_mean_s`, `gaze_entropy_mean_participants`, "
        "`mean_turn_s`, `overlap_ratio`, `audio_backchannel_rate_per_min`, "
        "`audio_floor_exchange_pause_mean_s`, `audio_participation_entropy`, "
        "`audio_distrib_speech`, `face_smile_ratio`, "
        "`face_negative_affect_ratio` et `face_facial_synchrony`."
    )
    lines.append(txt_83 + "\n\n")
    pdf_elems.append(Paragraph(txt_83, styles["Normal"]))

    lines.append("### 7.4 Feature & indicator layer\n")
    pdf_elems.append(Paragraph("7.4 Feature & indicator layer", styles["Heading3"]))
    txt_84 = (
        "Ces signaux sont ensuite transformés en indicateurs calculables et "
        "comparables entre groupes. Pour l'audio, cela inclut notamment la structure "
        "de participation (entropie de Shannon, variance de Woolley), le chevauchement "
        "de parole, les transitions de floor et les interruptions overlap-based. "
        "Pour le regard, le pipeline dérive `gaze_shared_visual_attention_ratio` "
        "(objet partagé, chaîne de fallback) et `gaze_attention_coordination_idx` "
        "(composite : objet partagé − entropie du regard, 2 termes depuis MOD-10). "
        "Pour la face, il calcule des marqueurs de valence, "
        "d'alignement affectif et de synchronie faciale. Les high-level features "
        "condensent ensuite ces familles de mesures pour les rendre exploitables "
        "dans les analyses multivariées."
    )
    lines.append(txt_84 + "\n\n")
    pdf_elems.append(Paragraph(txt_84, styles["Normal"]))

    lines.append("### 7.5 Mediator dynamics\n")
    pdf_elems.append(Paragraph("7.5 Mediator dynamics", styles["Heading3"]))
    txt_85 = (
        "Le niveau médiateur organise les construits interprétatifs utilisés dans le "
        "rapport. Les dimensions `COR`, `CRE` et `SPE` alimentent la lecture de type "
        "TMS sans être agrégées en score global. Les mesures de stratégie "
        "(`strategy_ratio_mean`, `strategy_norm`) et certains "
        "indices de coordination attentionnelle alimentent la composante TAS. "
        "`effort_task_sum`, `effort_task_norm` et "
        f"`{COHESION_SCORE_COL}` soutiennent la lecture TRS centrée sur l'effort "
        "collectif et la cohésion. Les composantes `SOC`, `TSK` et `COM` servent "
        "en v2 à construire le score agrégé de cohésion, mais ne sont plus reportées "
        "séparément dans les tableaux corrélationnels et régressifs. "
        "`skill_mean`, `skill_max` et "
        "`skill_congruence_mean` sont positionnés comme indicateurs transversaux de "
        "fonctionnement collectif."
    )
    lines.append(txt_85 + "\n\n")
    pdf_elems.append(Paragraph(txt_85, styles["Normal"]))

    lines.append("### 7.6 Outputs\n")
    pdf_elems.append(Paragraph("7.6 Outputs", styles["Heading3"]))
    txt_86 = (
        "Le niveau de sortie correspond principalement à la performance objective "
        "du groupe, capturée ici par `Score_perf_tsk`. Les évaluations questionnaire et les "
        "indices de type CI / c-factor constituent également des sorties "
        "interprétatives ou intermédiaires mobilisées pour qualifier la qualité du "
        "fonctionnement collectif."
    )
    lines.append(txt_86 + "\n\n")
    pdf_elems.append(Paragraph(txt_86, styles["Normal"]))

    lines.append("### 7.7 Lecture intégrative\n")
    pdf_elems.append(Paragraph("7.7 Lecture intégrative", styles["Heading3"]))
    txt_87 = (
        "Le modèle computationnel fournit ainsi la chaîne d'opérationnalisation qui "
        "va des signaux multimodaux bruts vers des indicateurs interprétables, puis "
        "vers des dynamiques médiatrices théoriques. Le diagramme théorie / empirie "
        "de la section 7 montre ensuite quels liens attendus sont ou non soutenus "
        "par les corrélations observées. Ensemble, ces deux représentations forment "
        "un pont entre cadre conceptuel, instrumentation multimodale et validation "
        "empirique des processus d'intelligence collective."
    )
    lines.append(txt_87 + "\n\n")
    pdf_elems.append(Paragraph(txt_87, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.15 * inch))

    # -----------------------------------
    # Export MD / HTML / PDF
    # -----------------------------------
    markdown_text = _rewrite_markdown_image_paths(
        "\n".join(lines),
        fig_dir=fig_dir,
        out_dir=out_dir,
    )
    md_path = out_dir / f"{file_stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    if HAS_MD:
        html_path = out_dir / f"{file_stem}.html"
        html_path.write_text(
            md_lib.markdown(markdown_text, extensions=["tables"]),
            encoding="utf-8"
        )

    pdf_path = out_dir / f"{file_stem}.pdf"
    doc = MyDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.multiBuild(pdf_elems)

    behavioral_outputs = None
    behavioral_v2_outputs = None
    if str(modality_filter or "").upper() == "VR":
        behavioral_outputs = generate_behavioral_indices_report(
            out_dir / "behavioral_indices_vr"
        )
        behavioral_v2_outputs = generate_behavioral_indices_v2_report(
            out_dir / "behavioral_indices_v2"
        )

    print(f"[OK] MD  : {md_path}")
    if HAS_MD:
        print(f"[OK] HTML: {html_path}")
    print(f"[OK] PDF : {pdf_path}")
    if behavioral_outputs is not None:
        print(f"[OK] PDF indices comportementaux : {behavioral_outputs['pdf_path']}")
        print(f"[OK] Sorties indices            : {behavioral_outputs['output_dir']}")
    if behavioral_v2_outputs is not None:
        print(f"[OK] PDF indices comportementaux v2 : {behavioral_v2_outputs['pdf_path']}")
        print(f"[OK] Sorties indices v2            : {behavioral_v2_outputs['output_dir']}")

    return {
        "md_path": md_path,
        "html_path": html_path if HAS_MD else None,
        "pdf_path": pdf_path,
        "behavioral_indices": behavioral_outputs,
        "behavioral_indices_v2": behavioral_v2_outputs,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fdr", action="store_true", help="Applique correction FDR (Benjamini–Hochberg) par tableau")
    ap.add_argument(
        "--inv-analysis-mode",
        type=str,
        choices=["pruning", "no-pruning"],
        default="pruning",
        help="Mode d'analyse INV : 'pruning' (défaut) utilise les résultats avec hard pruning, "
             "'no-pruning' utilise les résultats sans pruning (toutes les features conservées).",
    )
    ap.add_argument(
        "--pca-rotation",
        type=str,
        choices=["none", "varimax"],
        default="none",
        help="Rotation des loadings PCA utilisée pour la génération du rapport et la PCA régression "
             "('none' par défaut, 'varimax' pour utiliser les loadings varimax).",
    )
    # Rétrocompatibilité : --no-pruning est maintenu comme alias pour --inv-analysis-mode no-pruning
    ap.add_argument(
        "--no-pruning",
        action="store_true",
        help="[DÉPRÉCIÉ] Alias pour --inv-analysis-mode no-pruning. "
             "Préférer --inv-analysis-mode no-pruning.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["bundle", "pc_vr", "vr_only", "inv_vr", "pca_vr", "pca_vr_regression", "complementary_blocks", "cfactor_population", "pc_vr_sections_2_4"],
        default="bundle",
        help="bundle = génère automatiquement PC_VR/ et VR_only/ ; les autres modes génèrent une seule sortie v2.",
    )
    ap.add_argument(
        "--tci-allowed-file",
        type=str,
        default=None,
        help="Chemin explicite vers le fichier TCI 'allowed' (ex: c_scores_allowed.csv).",
    )
    ap.add_argument(
        "--tci-all-file",
        type=str,
        default=None,
        help="Chemin explicite vers le fichier TCI 'all' (ex: c_scores_all.csv).",
    )
    ap.add_argument(
        "--use-parent-population",
        action="store_true",
        help="Utilise par défaut `c_scores_allowed_pop.csv` / `c_scores_all_pop.csv` lorsqu'ils sont attendus par le rapport.",
    )
    ap.add_argument(
        "--no-sem",
        action="store_true",
        help="Saute toutes les sections SEM/MLM (3.1.5-3.1.8 : PLS-SEM, path analysis, ICC).",
    )
    ap.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Active l'echantillonnage MCMC PyMC (section 3.1.7 — ICC bayesien, HDI, Rhat). "
             "Desactive par defaut : ajoute plusieurs minutes de calcul.",
    )
    ap.add_argument(
        "--exclude-groups",
        action="append",
        default=[],
        help="Groupes à exclure du rapport (ex: bim015,bim023). Peut être répété.",
    )
    ap.add_argument(
        "--questionnaire-pruning-scope",
        type=str,
        choices=["scoped", "global"],
        default="global",
        help=(
            "Scope du pruning alpha Cronbach des questionnaires. "
            "'scoped'  : recalcule alpha et pruning sur le sous-échantillon courant (ex: VR seul). "
            "'global' (défaut) : utilise les résultats pruned PC+VR globaux déjà calculés — "
            "filtre uniquement les groupes par modalité sans recalcul. "
            "Recommandé pour les modes vr_only/inv_vr/pca_vr afin d'appliquer "
            "le pruning PC+VR aux groupes VR."
        ),
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    verbose = args.verbose

    # Détermine le mode d'analyse INV (nouveau système avec sous-dossiers)
    # Si --no-pruning est utilisé (rétrocompatibilité), forcer no-pruning
    if args.no_pruning:
        print("[WARNING] --no-pruning est déprécié. Utilisez --inv-analysis-mode no-pruning.")
        inv_analysis_mode = "no-pruning"
    else:
        inv_analysis_mode = args.inv_analysis_mode
    pca_rotation = args.pca_rotation

    # Pour la logique interne du rapport
    apply_pruning = (inv_analysis_mode == "pruning")
    skip_sem = args.no_sem
    run_bayes = args.bayes
    questionnaire_pruning_scope = args.questionnaire_pruning_scope

    tci_allowed_file = Path(args.tci_allowed_file) if args.tci_allowed_file else None
    tci_all_file = Path(args.tci_all_file) if args.tci_all_file else None
    if args.use_parent_population:
        if tci_allowed_file is None:
            candidate_allowed = results_dir / "TCI" / "c_scores_allowed_pop.csv"
            if not candidate_allowed.exists():
                raise FileNotFoundError(
                    f"--use-parent-population activé mais fichier absent : {candidate_allowed}"
                )
            tci_allowed_file = candidate_allowed
        if tci_all_file is None:
            candidate_all = results_dir / "TCI" / "c_scores_all_pop.csv"
            if not candidate_all.exists():
                raise FileNotFoundError(
                    f"--use-parent-population activé mais fichier absent : {candidate_all}"
                )
            tci_all_file = candidate_all

    if args.mode == "pc_vr":
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "PC_VR",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter=None,
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_principal_PC_VR",
            report_scope="principal",
            report_title="Rapport principal v2 — PC + VR",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
        )
    elif args.mode == "vr_only":
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_principal_VR",
            report_scope="principal",
            report_title="Rapport principal v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
    elif args.mode == "inv_vr":
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_INV_VR",
            report_scope="inv_only",
            report_title="Rapport INV v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            regression_pca_only=True,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
    elif args.mode == "pca_vr":
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_PCA_VR",
            report_scope="pca_only",
            report_title="Rapport PCA v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
    elif args.mode == "pca_vr_regression":
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_PCA_VR_regression",
            report_scope="pca_only",
            report_title="Rapport PCA v2 — VR uniquement — espace régression",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            regression_pca_only=True,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
    elif args.mode == "complementary_blocks":
        outputs = generate_complementary_reports(
            results_dir=results_dir,
            out_dir=out_dir / "complementary_blocks",
            verbose=verbose,
            tci_allowed_file=tci_allowed_file,
            tci_all_file=tci_all_file,
        )
        print(f"[OK] PDF blocs 1+3 : {outputs['rapport_vr_pdf']}")
        print(f"[OK] PDF blocs 2+4 : {outputs['rapport_global_pdf']}")
        print(f"[OK] Tables       : {outputs['tables_dir']}")
        print(f"[OK] Figures      : {outputs['figures_dir']}")
    elif args.mode == "cfactor_population":
        outputs = generate_cfactor_population_report(
            results_dir=results_dir,
            output_dir=out_dir / "cfactor_population",
        )
        print(f"[OK] PDF c-factor population : {outputs['pdf_path']}")
        print(f"[OK] Tables                  : {outputs['tables_dir']}")
        print(f"[OK] Figures                 : {outputs['figures_dir']}")
    elif args.mode == "pc_vr_sections_2_4":
        stem = "rapport_principal_PC_VR_sections_2_4"
        title = "Rapport principal v2 — PC + VR — sections 2 à 4"
        if args.use_parent_population:
            stem = "rapport_principal_PC_VR_cfactor_pop_sections_2_4"
            title = "Rapport principal v2 — PC + VR — sections 2 à 4 — c-factor population parente"
        outputs = build_principal_sections_2_4_report(
            results_dir=results_dir,
            out_dir=out_dir / "PC_VR",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem=stem,
            report_title=title,
            tci_allowed_file=tci_allowed_file,
            tci_all_file=tci_all_file,
            exclude_groups=args.exclude_groups,
        )
        print(f"[OK] Rapport sections 2-4 MD  : {outputs['md_path']}")
        if outputs.get("html_path") is not None:
            print(f"[OK] Rapport sections 2-4 HTML: {outputs['html_path']}")
        print(f"[OK] Rapport sections 2-4 PDF : {outputs['pdf_path']}")
    else:
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "PC_VR",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter=None,
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_principal_PC_VR",
            report_scope="principal",
            report_title="Rapport principal v2 — PC + VR",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
        )
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_principal_VR",
            report_scope="principal",
            report_title="Rapport principal v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_INV_VR",
            report_scope="inv_only",
            report_title="Rapport INV v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            regression_pca_only=True,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_PCA_VR",
            report_scope="pca_only",
            report_title="Rapport PCA v2 — VR uniquement",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            skip_sem=skip_sem,
            bayes=run_bayes,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )
        build_report(
            results_dir=results_dir,
            out_dir=out_dir / "VR_only",
            apply_fdr=bool(args.fdr),
            verbose=verbose,
            modality_filter="VR",
            apply_pruning=apply_pruning,
            inv_analysis_mode=inv_analysis_mode,
            pca_rotation=pca_rotation,
            file_stem="rapport_PCA_VR_regression",
            report_scope="pca_only",
            report_title="Rapport PCA v2 — VR uniquement — espace régression",
            tci_scope="allowed",
            tci_file=tci_allowed_file,
            regression_pca_only=True,
            exclude_groups=args.exclude_groups,
            questionnaire_pruning_scope=questionnaire_pruning_scope,
        )

if __name__ == "__main__":
    main()
