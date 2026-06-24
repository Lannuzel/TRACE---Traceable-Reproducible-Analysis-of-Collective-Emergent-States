# -*- coding: utf-8 -*-
"""
Helpers de rendu pour les sections questionnaire du rapport.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from reportlab.platypus import Image, Paragraph, Spacer
from reportlab.lib.units import inch
from scipy import stats

from pathlib import Path as _Path

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))
_scripts_dir = _Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    COHESION_COMPONENTS,
    COHESION_SCORE_COL,
    QUESTIONNAIRE_ANALYSIS_COLS,
    QUESTIONNAIRE_RELIABILITY_DIMENSIONS,
    TMS_DIMENSIONS,
)

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except Exception:
    HAS_PINGOUIN = False


PROFILE_FAMILIARITY_ORDER = [
    "Pas du tout familier(e)",
    "Peu familier(e)",
    "Modérément familier(e)",
    "Assez familier(e)",
    "Très familier(e)",
]

QUESTIONNAIRE_ICC_LIKERT_LEVELS = [
    "Fortement en désaccord",
    "En désaccord",
    "Plutôt en désaccord",
    "Légèrement en désaccord",
    "Neutre",
    "Légèrement en accord",
    "Plutôt en accord",
    "En accord",
    "Fortement en accord",
]
QUESTIONNAIRE_ICC_CODE_MAP = {
    label: i + 1 for i, label in enumerate(QUESTIONNAIRE_ICC_LIKERT_LEVELS)
}
QUESTIONNAIRE_AGREEMENT_N_CATEGORIES = len(QUESTIONNAIRE_ICC_LIKERT_LEVELS)
QUESTIONNAIRE_RWG_THRESHOLD = 0.70
QUESTIONNAIRE_RWG_MAX_RANGE_OK = 2.0
QUESTIONNAIRE_MODALITY_FORCE_WELCH_DIMS = {"COM"}
QUESTIONNAIRE_ICC_INVERT_SHORT = {
    "G1Q00001.COR03",
    "G1Q00001.COR05",
    "G1Q00001.CRE04",
    "G1Q00001.CRE02",
}


def plot_questionnaire_by_condition(q_long: pd.DataFrame, dim: str, outpng: Path, title: str):
    """
    q_long attendu: colonnes au moins ['dimension','score'] + modalite/scenario.
    """
    if q_long is None or q_long.empty:
        return False

    df = q_long.copy()
    ren = {}
    if "Modalite" in df.columns and "modalite" not in df.columns:
        ren["Modalite"] = "modalite"
    if "Scenario" in df.columns and "scenario" not in df.columns:
        ren["Scenario"] = "scenario"
    if ren:
        df = df.rename(columns=ren)

    needed = {"modalite", "scenario", "dimension", "score"}
    if not needed.issubset(df.columns):
        return False

    df = df.dropna(subset=["modalite", "scenario", "dimension", "score"]).copy()
    df["dimension"] = df["dimension"].astype(str).str.strip()
    df = df[df["dimension"] == str(dim)]
    if df.empty:
        return False

    df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()
    df["cond"] = df["modalite"] + " " + df["scenario"]

    def cond_key(c):
        parts = c.split()
        mod = parts[0] if len(parts) >= 1 else ""
        scen = parts[1] if len(parts) >= 2 else ""
        mod_k = 0 if mod == "PC" else 1
        scen_k = int(scen.replace("S", "")) if scen.startswith("S") and scen[1:].isdigit() else 99
        return (mod_k, scen_k, c)

    conds = sorted(df["cond"].unique(), key=cond_key)
    x = np.arange(len(conds))

    data = []
    ns = []
    for c in conds:
        vals = pd.to_numeric(df.loc[df["cond"] == c, "score"], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        data.append(vals)
        ns.append(len(vals))

    if all(len(v) == 0 for v in data):
        return False

    plt.figure(figsize=(8.5, 4.2))
    plt.boxplot(
        data,
        positions=x,
        widths=0.55,
        patch_artist=False,
        showfliers=False,
        medianprops=dict(linewidth=1.5),
    )

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.06, size=len(vals))
        plt.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.7, s=18)

    plt.xticks(x, [f"{c}\n(n={n})" for c, n in zip(conds, ns)], rotation=0)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, 10)
    plt.yticks(range(0, 11))
    plt.savefig(outpng, dpi=180)
    plt.close()
    return True


def _prepare_questionnaire_long_v2(q_scores: pd.DataFrame) -> pd.DataFrame:
    if q_scores is None or q_scores.empty:
        return pd.DataFrame()

    df = q_scores.copy()
    ren = {}
    if "Modalite" in df.columns and "modalite" not in df.columns:
        ren["Modalite"] = "modalite"
    if "Scenario" in df.columns and "scenario" not in df.columns:
        ren["Scenario"] = "scenario"
    if "Participant" in df.columns and "participant" not in df.columns:
        ren["Participant"] = "participant"
    if "Role" in df.columns and "role" not in df.columns:
        ren["Role"] = "role"
    if ren:
        df = df.rename(columns=ren)

    needed = {"group_id", "dimension", "score", "modalite"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    df["dimension"] = df["dimension"].astype(str).str.upper().str.strip()
    df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["group_id", "dimension", "score", "modalite"])
    return df.reset_index(drop=True)


def build_questionnaire_group_modality_long(q_scores: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_questionnaire_long_v2(q_scores)
    if df.empty:
        return df

    group_cols = ["group_id", "modalite", "dimension"]
    out = df.groupby(group_cols, dropna=False)["score"].mean().reset_index()

    cohesion_source = out[out["dimension"].isin(COHESION_COMPONENTS)].copy()
    if not cohesion_source.empty:
        coh = (
            cohesion_source.groupby(["group_id", "modalite"], dropna=False)["score"]
            .mean()
            .reset_index()
            .assign(dimension=COHESION_SCORE_COL)
        )
        out = pd.concat([out, coh], ignore_index=True)

    out = out[out["dimension"].isin(QUESTIONNAIRE_ANALYSIS_COLS + QUESTIONNAIRE_RELIABILITY_DIMENSIONS)].copy()
    return out.reset_index(drop=True)


def _questionnaire_modality_effect_dimensions() -> list[str]:
    """
    Ordre d'affichage pour la section 1.4.8.

    On conserve les dimensions TMS, on ajoute explicitement les dimensions de
    cohésion (SOC/TSK/COM), puis on garde le score agrégé de cohésion en
    complément quand il est disponible.
    """
    ordered = list(TMS_DIMENSIONS) + list(COHESION_COMPONENTS)
    if COHESION_SCORE_COL in QUESTIONNAIRE_ANALYSIS_COLS:
        ordered.append(COHESION_SCORE_COL)
    return ordered


def _compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled


def _normality_ok(vals: np.ndarray) -> bool:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return False
    try:
        return bool(stats.shapiro(vals).pvalue > 0.05)
    except Exception:
        return False


def compute_questionnaire_modality_effects(q_scores: pd.DataFrame) -> pd.DataFrame:
    group_long = build_questionnaire_group_modality_long(q_scores)
    if group_long.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for dim in _questionnaire_modality_effect_dimensions():
        sub = group_long[group_long["dimension"] == dim].copy()
        if sub.empty:
            continue
        pc = sub.loc[sub["modalite"] == "PC", "score"].dropna().to_numpy(dtype=float)
        vr = sub.loc[sub["modalite"] == "VR", "score"].dropna().to_numpy(dtype=float)
        if len(pc) < 2 or len(vr) < 2:
            continue

        use_welch = dim in QUESTIONNAIRE_MODALITY_FORCE_WELCH_DIMS or (_normality_ok(pc) and _normality_ok(vr))
        if use_welch:
            test_name = "Welch t-test"
            stat, pval = stats.ttest_ind(pc, vr, equal_var=False, nan_policy="omit")
        else:
            test_name = "Mann-Whitney"
            stat, pval = stats.mannwhitneyu(pc, vr, alternative="two-sided")

        rows.append({
            "dimension": dim,
            "test": test_name,
            "n_pc": int(len(pc)),
            "mean_pc": float(np.mean(pc)),
            "sd_pc": float(np.std(pc, ddof=1)) if len(pc) >= 2 else np.nan,
            "n_vr": int(len(vr)),
            "mean_vr": float(np.mean(vr)),
            "sd_vr": float(np.std(vr, ddof=1)) if len(vr) >= 2 else np.nan,
            "stat": float(stat) if np.isfinite(stat) else np.nan,
            "p_value": float(pval) if np.isfinite(pval) else np.nan,
            "cohens_d": _compute_cohens_d(pc, vr),
        })

    return pd.DataFrame(rows)


def plot_questionnaire_by_modalite(q_long: pd.DataFrame, dim: str, outpng: Path, title: str) -> bool:
    if q_long is None or q_long.empty:
        return False
    df = q_long.copy()
    needed = {"modalite", "dimension", "score"}
    if not needed.issubset(df.columns):
        return False
    df = df[(df["dimension"].astype(str) == str(dim)) & df["modalite"].astype(str).isin(["PC", "VR"])].copy()
    if df.empty:
        return False

    base_order = ["PC", "VR"]
    palette = {"PC": "#4e79a7", "VR": "#e15759"}
    plot_order = []
    data = []
    ns = []
    for mod in base_order:
        vals = pd.to_numeric(df.loc[df["modalite"] == mod, "score"], errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        plot_order.append(mod)
        data.append(vals)
        ns.append(len(vals))

    if not data:
        return False

    plt.figure(figsize=(6.2, 4.0))
    positions = np.arange(len(plot_order))
    use_violin = all(len(vals) >= 2 and np.nanstd(vals) > 0 for vals in data)
    if use_violin:
        violin = plt.violinplot(data, positions=positions, widths=0.7, showmeans=False, showmedians=True)
        for body, mod in zip(violin["bodies"], plot_order):
            body.set_facecolor(palette.get(mod, "#4e79a7"))
            body.set_alpha(0.25)
    else:
        box = plt.boxplot(
            data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.2},
        )
        for patch, mod in zip(box["boxes"], plot_order):
            patch.set_facecolor(palette.get(mod, "#4e79a7"))
            patch.set_alpha(0.22)
            patch.set_edgecolor(palette.get(mod, "#4e79a7"))
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.05, size=len(vals))
        mod = plot_order[i]
        plt.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.75, s=18, color=palette.get(mod, "#4e79a7"))

    plt.xticks(positions, [f"{mod}\n(n={n})" for mod, n in zip(plot_order, ns)])
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, 10)
    plt.yticks(range(0, 11))
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()
    return True


def plot_questionnaire_dimensions_by_modalite(q_long: pd.DataFrame, outpng: Path) -> bool:
    """Trace toutes les dimensions questionnaire dans un seul boxplot PC vs VR."""
    if q_long is None or q_long.empty:
        return False
    needed = {"modalite", "dimension", "score"}
    if not needed.issubset(q_long.columns):
        return False

    dims = [dim for dim in list(TMS_DIMENSIONS) + list(COHESION_COMPONENTS) if dim in set(q_long["dimension"].astype(str))]
    if not dims:
        return False

    df = q_long.copy()
    df["dimension"] = df["dimension"].astype(str).str.upper().str.strip()
    df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["dimension"].isin(dims) & df["modalite"].isin(["PC", "VR"])].dropna(subset=["score"])
    if df.empty:
        return False

    palette = {"PC": "#4e79a7", "VR": "#e15759"}
    offsets = {"PC": -0.18, "VR": 0.18}
    width = 0.30

    data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    rng = np.random.default_rng(0)

    plt.figure(figsize=(9.8, 4.8))
    ax = plt.gca()
    for i, dim in enumerate(dims):
        for mod in ["PC", "VR"]:
            vals = df.loc[(df["dimension"] == dim) & (df["modalite"] == mod), "score"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            pos = i + offsets[mod]
            data.append(vals)
            positions.append(pos)
            colors.append(palette[mod])
            jitter = rng.normal(0, 0.025, size=len(vals))
            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                s=18,
                alpha=0.72,
                color=palette[mod],
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

    if not data:
        plt.close()
        return False

    boxes = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.3},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels(dims)
    ax.set_ylabel("Score moyen groupe")
    ax.set_title("Dimensions questionnaire selon la modalite")
    ax.set_ylim(0, 10)
    ax.set_yticks(range(0, 11))
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(
        handles=[
            Patch(facecolor=palette["PC"], edgecolor=palette["PC"], alpha=0.3, label="PC"),
            Patch(facecolor=palette["VR"], edgecolor=palette["VR"], alpha=0.3, label="VR"),
        ],
        frameon=False,
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(outpng, dpi=180)
    plt.close()
    return True


def _icc2k_from_matrix(matrix: np.ndarray) -> float:
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        return np.nan
    valid_rows = ~np.any(np.isnan(data), axis=1)
    data = data[valid_rows]
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    grand = data.mean()
    mean_targets = data.mean(axis=1)
    mean_raters = data.mean(axis=0)

    ss_targets = k * np.sum((mean_targets - grand) ** 2)
    ss_raters = n * np.sum((mean_raters - grand) ** 2)
    resid = data - mean_targets[:, None] - mean_raters[None, :] + grand
    ss_error = np.sum(resid ** 2)

    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_targets + (ms_raters - ms_error) / n
    if not np.isfinite(denom) or abs(float(denom)) < 1e-12:
        return np.nan
    icc_value = float((ms_targets - ms_error) / denom)
    if not np.isfinite(icc_value):
        return np.nan
    if abs(icc_value) < 1e-12:
        return 0.0
    if abs(icc_value) > 1e6:
        return np.nan
    return icc_value


def _stabilize_icc_value(value: Any) -> float:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return np.nan
    val = float(val)
    if not np.isfinite(val):
        return np.nan
    if abs(val) < 1e-12:
        return 0.0
    if abs(val) > 1e6:
        return np.nan
    return val


def _interpret_icc_koo_li(value: Any) -> str:
    """
    Interprétation qualitative de l'ICC selon les seuils usuels de
    Koo & Li (2016).
    """
    val = _stabilize_icc_value(value)
    if pd.isna(val):
        return "non_estimable"
    if val < 0.50:
        return "poor"
    if val < 0.75:
        return "moderate"
    if val < 0.90:
        return "good"
    return "excellent"


def _prepare_icc_matrix(
    matrix: pd.DataFrame,
    missing: str = "drop",
) -> pd.DataFrame:
    """
    Prépare une matrice targets x raters pour l'ICC.

    Paramètres
    ----------
    matrix:
        Matrice indexée par unités évaluées, colonnes = participants/raters.
    missing:
        - "drop" : supprime les unités incomplètes.
        - "target_mean" : impute les valeurs manquantes par la moyenne de l'unité.
    """
    mat = matrix.apply(pd.to_numeric, errors="coerce")
    if missing == "drop":
        mat = mat.dropna(axis=0, how="any")
    elif missing in {"target_mean", "item_mean"}:
        mat = mat.apply(lambda row: row.fillna(row.mean()), axis=1)
        mat = mat.dropna(axis=0, how="any")
    else:
        raise ValueError("missing doit être 'drop' ou 'target_mean'")
    return mat


def _icc21_from_matrix(matrix: np.ndarray) -> float:
    """
    ICC(2,1) : modèle two-way random effects, absolute agreement,
    mesure simple.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        return np.nan
    valid_rows = ~np.any(np.isnan(data), axis=1)
    data = data[valid_rows]
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    grand = data.mean()
    mean_targets = data.mean(axis=1)
    mean_raters = data.mean(axis=0)

    ss_targets = k * np.sum((mean_targets - grand) ** 2)
    ss_raters = n * np.sum((mean_raters - grand) ** 2)
    resid = data - mean_targets[:, None] - mean_raters[None, :] + grand
    ss_error = np.sum(resid ** 2)

    ms_targets = ss_targets / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_targets + (k - 1) * ms_error + (k * (ms_raters - ms_error) / n)
    if not np.isfinite(denom) or abs(float(denom)) < 1e-12:
        return np.nan
    return _stabilize_icc_value((ms_targets - ms_error) / denom)


def _icc1_from_matrix(matrix: np.ndarray) -> float:
    """
    ICC(1,1) indicatif : modèle one-way random effects, mesure simple.
    Conservé comme diagnostic optionnel, l'analyse principale restant ICC(2,k).
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        return np.nan
    valid_rows = ~np.any(np.isnan(data), axis=1)
    data = data[valid_rows]
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    grand = data.mean()
    mean_targets = data.mean(axis=1)
    ss_between = k * np.sum((mean_targets - grand) ** 2)
    ss_within = np.sum((data - mean_targets[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    if not np.isfinite(denom) or abs(float(denom)) < 1e-12:
        return np.nan
    return _stabilize_icc_value((ms_between - ms_within) / denom)


def compute_icc_by_dimension(
    df: pd.DataFrame,
    *,
    group_col: str = "group_id",
    participant_col: str = "participant_id",
    item_col: str = "item_id",
    score_col: str = "score",
    dimension_col: str = "dimension",
    missing: str = "drop",
    include_optional_icc: bool = True,
) -> pd.DataFrame:
    """
    Calcule l'ICC(2,k) inter-membres par dimension questionnaire.

    Format attendu
    --------------
    Une ligne = réponse item-level :
    `group_id | participant_id | item_id | dimension | score`

    Méthode
    -------
    1. Agrège d'abord les items en score moyen participant × dimension.
    2. Construit, pour chaque dimension, une matrice `groupes x participants`.
    3. Calcule un seul ICC(2,k) par dimension.

    Les items ne sont jamais utilisés comme cibles ICC : ils servent uniquement
    à former le score dimensionnel individuel.
    """
    required = {group_col, participant_col, item_col, score_col, dimension_col}
    missing_cols = sorted(required - set(df.columns if df is not None else []))
    if missing_cols:
        raise ValueError(f"Colonnes manquantes pour le calcul ICC : {missing_cols}")

    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[group_col, participant_col, item_col, dimension_col, score_col])
    if work.empty:
        return pd.DataFrame(columns=["dimension", "n_groups_used", "ICC2k", "interpretation"])

    # Items -> score moyen par participant dans chaque dimension.
    participant_scores = (
        work.groupby([dimension_col, group_col, participant_col], dropna=False)[score_col]
        .mean()
        .reset_index(name="score_dim_participant")
    )

    rows: list[dict[str, Any]] = []
    for dimension, sub in participant_scores.groupby(dimension_col, dropna=False):
        n_groups_total = int(sub[group_col].nunique(dropna=False))
        n_participants_by_group = sub.groupby(group_col, dropna=False)[participant_col].nunique()
        valid_groups = n_participants_by_group[n_participants_by_group >= 2].index
        sub_valid = sub[sub[group_col].isin(valid_groups)].copy()

        if sub_valid.empty:
            rows.append({
                "dimension": dimension,
                "n_groups_total": n_groups_total,
                "n_groups_used": 0,
                "n_groups_excluded_lt2_participants": int(n_groups_total),
                "n_groups_dropped_missing": 0,
                "n_raters": 0,
                "ICC2k": np.nan,
                "ICC21": np.nan,
                "ICC1": np.nan,
                "interpretation": "non_estimable",
            })
            continue

        # Matrice correcte pour l'agrégation : cibles = groupes, raters = membres.
        pivot = sub_valid.pivot(
            index=group_col,
            columns=participant_col,
            values="score_dim_participant",
        )
        n_groups_before_missing = int(pivot.shape[0])
        mat = _prepare_icc_matrix(pivot, missing=missing)
        n_groups_used, n_raters = mat.shape
        n_dropped_missing = n_groups_before_missing - n_groups_used

        if n_groups_used < 2 or n_raters < 2:
            icc2k = np.nan
            icc21 = np.nan
            icc1 = np.nan
        else:
            arr = mat.to_numpy(dtype=float)
            icc2k = _stabilize_icc_value(_icc2k_from_matrix(arr))
            icc21 = _icc21_from_matrix(arr) if include_optional_icc else np.nan
            icc1 = _icc1_from_matrix(arr) if include_optional_icc else np.nan

        rows.append({
            "dimension": dimension,
            "n_groups_total": n_groups_total,
            "n_groups_used": int(n_groups_used),
            "n_groups_excluded_lt2_participants": int((n_participants_by_group < 2).sum()),
            "n_groups_dropped_missing": int(n_dropped_missing),
            "n_raters": int(n_raters),
            "ICC2k": icc2k,
            "ICC21": icc21,
            "ICC1": icc1,
            "interpretation": _interpret_icc_koo_li(icc2k),
        })

    return pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)


def _detect_questionnaire_meta_cols_raw(df: pd.DataFrame) -> dict[str, str | None]:
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


def _normalize_questionnaire_timepoint_local(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m:
        return f"T{int(m.group(1))}"
    return s


def _normalize_questionnaire_meta_local(value: Any, kind: str) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if kind == "group":
        return s.lower()
    if kind == "timepoint":
        return _normalize_questionnaire_timepoint_local(s)
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


def _build_reference_keys_from_qscores(q_scores: pd.DataFrame) -> set[str]:
    if q_scores is None or q_scores.empty:
        return set()

    key_sources = {
        "group": next((c for c in ["group_id", "Groupe"] if c in q_scores.columns), None),
        "timepoint": next((c for c in ["timepoint", "Session"] if c in q_scores.columns), None),
        "modalite": next((c for c in ["modalite", "Modalite", "condition", "Condition"] if c in q_scores.columns), None),
        "scenario": next((c for c in ["scenario", "Scenario"] if c in q_scores.columns), None),
    }
    if not any(key_sources.values()):
        return set()

    keys: set[str] = set()
    subset_cols = [c for c in key_sources.values() if c is not None]
    for _, row in q_scores[subset_cols].drop_duplicates().iterrows():
        group_id = _normalize_questionnaire_meta_local(
            row.get(key_sources["group"]) if key_sources["group"] else "",
            "group",
        )
        timepoint = _normalize_questionnaire_meta_local(
            row.get(key_sources["timepoint"]) if key_sources["timepoint"] else "",
            "timepoint",
        )
        modalite = _normalize_questionnaire_meta_local(
            row.get(key_sources["modalite"]) if key_sources["modalite"] else "",
            "modalite",
        )
        scenario = _normalize_questionnaire_meta_local(
            row.get(key_sources["scenario"]) if key_sources["scenario"] else "",
            "scenario",
        )
        keys.add(f"{group_id}|{timepoint}|{modalite}|{scenario}")
    return keys


def _extract_dimension_from_item_col(colname: str) -> str | None:
    m = re.search(r"(G1Q00001|G2Q00001)(?:\.|\[)([A-Z]{3})", str(colname))
    return m.group(2) if m else None


def _extract_item_short_from_col(colname: str) -> str | None:
    m = re.search(r"(G1Q00001|G2Q00001)\.([A-Z]{3}\d{2})", str(colname))
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    m = re.search(r"(G1Q00001|G2Q00001)\[([A-Z]{3}\d{2})\]", str(colname))
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


def _recode_questionnaire_response(val: Any) -> float | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if re.fullmatch(r"[1-9]", s):
        return float(s)
    if s in QUESTIONNAIRE_ICC_CODE_MAP:
        return float(QUESTIONNAIRE_ICC_CODE_MAP[s])
    return None


def _load_raw_questionnaire_for_icc(survey_path: Path) -> pd.DataFrame:
    if survey_path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(survey_path)
    else:
        raw = pd.DataFrame()
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                raw = pd.read_csv(survey_path, sep=";", encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if raw.empty:
            raise ValueError(f"Impossible de lire {survey_path} avec les encodages connus.")

    id_col = [c for c in raw.columns if "ID de la réponse" in str(c)]
    if id_col:
        raw = raw.rename(columns={id_col[0]: "Participant"})
    raw["Participant"] = raw["Participant"].astype(str)
    return raw


def _prepare_questionnaire_item_long_for_agreement(
    q_scores_reference: pd.DataFrame,
    survey_path: Path | None,
    removed_item_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Prépare les réponses questionnaire item-level pour les analyses
    d'accord intra-groupe (ICC, rwg).
    """
    if survey_path is None or not survey_path.exists():
        return pd.DataFrame()
    if q_scores_reference is None or q_scores_reference.empty:
        return pd.DataFrame()

    reference_keys = _build_reference_keys_from_qscores(q_scores_reference)
    if not reference_keys:
        return pd.DataFrame()

    raw = _load_raw_questionnaire_for_icc(survey_path)
    meta = _detect_questionnaire_meta_cols_raw(raw)
    role_col = next(
        (
            c for c in raw.columns
            if ("activité en VR" in str(c) and "étiez" in str(c)) or str(c).startswith("G3Q00008")
        ),
        None,
    )

    id_cols = ["Participant"]
    for meta_col in [meta.get("Groupe"), meta.get("Session"), meta.get("Modalite"), meta.get("Scenario"), role_col]:
        if meta_col and meta_col not in id_cols:
            id_cols.append(meta_col)

    item_cols = [c for c in raw.columns if re.match(r"^G[12]Q00001(?:\[|\.)", str(c))]
    if not item_cols:
        return pd.DataFrame()

    long = raw[id_cols + item_cols].copy().melt(
        id_vars=id_cols,
        var_name="item_col",
        value_name="reponse",
    )
    if removed_item_cols:
        long = long[~long["item_col"].isin(removed_item_cols)].copy()
    long["dimension"] = long["item_col"].apply(_extract_dimension_from_item_col)
    long["item_code"] = long["item_col"].apply(_extract_item_short_from_col)
    long["reponse_num"] = long["reponse"].apply(_recode_questionnaire_response)
    long = long[long["dimension"].isin(QUESTIONNAIRE_RELIABILITY_DIMENSIONS)].copy()
    if long.empty:
        return pd.DataFrame()

    invert_mask = long["item_code"].isin(QUESTIONNAIRE_ICC_INVERT_SHORT) & long["reponse_num"].notna()
    long.loc[invert_mask, "reponse_num"] = 10 - long.loc[invert_mask, "reponse_num"]

    group_col = meta.get("Groupe")
    session_col = meta.get("Session")
    modalite_col = meta.get("Modalite")
    scenario_col = meta.get("Scenario")

    long["group_id"] = long[group_col].apply(lambda x: _normalize_questionnaire_meta_local(x, "group")) if group_col else ""
    long["timepoint"] = long[session_col].apply(lambda x: _normalize_questionnaire_meta_local(x, "timepoint")) if session_col else ""
    long["modalite"] = long[modalite_col].apply(lambda x: _normalize_questionnaire_meta_local(x, "modalite")) if modalite_col else ""
    long["scenario"] = long[scenario_col].apply(lambda x: _normalize_questionnaire_meta_local(x, "scenario")) if scenario_col else ""
    long["rater"] = long[role_col].astype(str).str.strip() if role_col else long["Participant"].astype(str).str.strip()
    long.loc[long["rater"].isin(["", "nan", "<NA>", "None"]), "rater"] = long["Participant"].astype(str).str.strip()
    long["_group_key"] = long["group_id"] + "|" + long["timepoint"] + "|" + long["modalite"] + "|" + long["scenario"]
    long = long[long["_group_key"].isin(reference_keys)].copy()
    long = long[long["reponse_num"].notna()].copy()
    return long.reset_index(drop=True)


def compute_questionnaire_dimension_icc_matrix_from_survey(
    q_scores_reference: pd.DataFrame,
    survey_path: Path | None,
    removed_item_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Compatibilité : renvoie désormais la synthèse ICC correcte par dimension.

    Ancienne logique supprimée : les items ne sont plus utilisés comme cibles
    ICC et aucun ICC par groupe n'est calculé.
    """
    return compute_questionnaire_icc_by_dimension_from_survey(
        q_scores_reference=q_scores_reference,
        survey_path=survey_path,
        removed_item_cols=removed_item_cols,
    )


def compute_questionnaire_icc_by_dimension_from_survey(
    q_scores_reference: pd.DataFrame,
    survey_path: Path | None,
    removed_item_cols: set[str] | None = None,
    missing: str = "drop",
    include_optional_icc: bool = True,
) -> pd.DataFrame:
    """
    Wrapper rapport pour `compute_icc_by_dimension`.

    Il part du questionnaire brut item-level, applique le même recodage et le
    même pruning exploratoire que la section ICC existante, puis produit un
    résumé par dimension. L'unité d'analyse reste la triade/condition, et les
    raters sont les participants du groupe.
    """
    long = _prepare_questionnaire_item_long_for_agreement(
        q_scores_reference=q_scores_reference,
        survey_path=survey_path,
        removed_item_cols=removed_item_cols,
    )
    if long.empty:
        return pd.DataFrame()

    group_keys = ["group_id", "timepoint", "modalite", "scenario"]
    work = long.copy()
    work["_icc_unit_id"] = work[group_keys].astype(str).agg("|".join, axis=1)
    item_level = work.rename(
        columns={
            "rater": "participant_id",
            "item_code": "item_id",
            "reponse_num": "score",
        }
    )

    out = compute_icc_by_dimension(
        item_level,
        group_col="_icc_unit_id",
        participant_col="participant_id",
        item_col="item_id",
        score_col="score",
        dimension_col="dimension",
        missing=missing,
        include_optional_icc=include_optional_icc,
    )
    if out.empty:
        return out

    out["construct"] = out["dimension"].map(
        {**{d: "TMS" for d in TMS_DIMENSIONS}, **{d: "Cohesion" for d in COHESION_COMPONENTS}}
    ).fillna("")
    ordered = TMS_DIMENSIONS + COHESION_COMPONENTS
    rank = {dim: i for i, dim in enumerate(ordered)}
    out = out.sort_values(
        by="dimension",
        key=lambda s: s.map(rank).fillna(999),
    ).reset_index(drop=True)
    preferred_cols = [
        "dimension", "construct", "n_groups_total", "n_groups_used",
        "n_groups_excluded_lt2_participants", "n_groups_dropped_missing",
        "n_raters", "ICC2k", "ICC21", "ICC1", "interpretation",
    ]
    return out[[c for c in preferred_cols if c in out.columns]].copy()


def compute_questionnaire_icc_summary(q_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Conservé pour compatibilité, mais l'ancien ICC par groupe est désactivé.
    return pd.DataFrame(), pd.DataFrame()


def analyze_questionnaire_icc_criteria(
    icc_matrix_df: pd.DataFrame,
    poor_threshold: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Conservé pour compatibilité, mais l'ancien filtrage par ICC de groupe est supprimé.
    return pd.DataFrame(), pd.DataFrame()


def _uniform_null_variance(n_categories: int) -> float:
    if n_categories < 2:
        return np.nan
    return float((n_categories**2 - 1) / 12.0)


def _compute_rwg_j_from_item_block(block: pd.DataFrame) -> tuple[float, float, float, int, int]:
    """
    Calcule rwg(j) de James et al. (1984) sur un bloc d'items d'un construit.

    Retourne :
    - rwg(j)
    - étendue max-min des scores moyens par rater
    - écart-type des scores moyens par rater
    - nombre de raters
    - nombre d'items
    """
    if block is None or block.empty:
        return np.nan, np.nan, np.nan, 0, 0

    pivot = (
        block.groupby(["item_code", "rater"], dropna=False)["reponse_num"]
        .mean()
        .reset_index()
        .pivot(index="item_code", columns="rater", values="reponse_num")
        .dropna(axis=0, how="any")
    )
    if pivot.shape[0] < 1 or pivot.shape[1] < 2:
        return np.nan, np.nan, np.nan, int(pivot.shape[1]), int(pivot.shape[0])

    obs_var_mean = float(pivot.var(axis=1, ddof=1).mean())
    sigma_eu = _uniform_null_variance(QUESTIONNAIRE_AGREEMENT_N_CATEGORIES)
    if not np.isfinite(obs_var_mean) or not np.isfinite(sigma_eu) or sigma_eu <= 0:
        rwg_j = np.nan
    else:
        j = int(pivot.shape[0])
        ratio = obs_var_mean / sigma_eu
        denom = (j * (1.0 - ratio)) + ratio
        rwg_j = np.nan if abs(float(denom)) < 1e-12 else float((j * (1.0 - ratio)) / denom)
        rwg_j = _stabilize_icc_value(rwg_j)
        if np.isfinite(rwg_j):
            # Le rwg(j) est interprété sur [0, 1] ; les sorties brutes hors bornes
            # sont tronquées pour éviter des décisions illisibles.
            rwg_j = float(np.clip(rwg_j, 0.0, 1.0))

    rater_scores = pivot.mean(axis=0)
    score_range = float(rater_scores.max() - rater_scores.min()) if len(rater_scores) >= 2 else np.nan
    score_sd = float(rater_scores.std(ddof=1)) if len(rater_scores) >= 2 else np.nan
    return rwg_j, score_range, score_sd, int(pivot.shape[1]), int(pivot.shape[0])


def compute_questionnaire_rwg_construct_matrix_from_survey(
    q_scores_reference: pd.DataFrame,
    survey_path: Path | None,
    removed_item_cols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Calcule un diagnostic rwg(j) par groupe pour chaque dimension du
    questionnaire, ainsi qu'en complément pour les construits TMS et Cohesion.

    Le rwg(j) est calculé sur les items retenus du bloc considéré ; l'étendue
    max-min et l'écart-type sont calculés sur les scores moyens par rater.
    """
    long = _prepare_questionnaire_item_long_for_agreement(
        q_scores_reference=q_scores_reference,
        survey_path=survey_path,
        removed_item_cols=removed_item_cols,
    )
    if long.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_keys = ["group_id", "timepoint", "modalite", "scenario"]
    dimension_order = [d for d in TMS_DIMENSIONS + COHESION_COMPONENTS if d in QUESTIONNAIRE_RELIABILITY_DIMENSIONS]
    construct_map = {
        "TMS": TMS_DIMENSIONS,
        "Cohesion": COHESION_COMPONENTS,
    }

    for keys, sub in long.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_keys, keys))

        for dim in dimension_order:
            block = sub[sub["dimension"] == dim].copy()
            rwg_j, score_range, score_sd, n_raters, n_items = _compute_rwg_j_from_item_block(block)
            if pd.isna(rwg_j):
                decision = "non estimable"
            elif rwg_j < QUESTIONNAIRE_RWG_THRESHOLD:
                decision = "eviter l'agregation"
            elif np.isfinite(score_range) and score_range <= QUESTIONNAIRE_RWG_MAX_RANGE_OK:
                decision = "moyenne OK"
            else:
                decision = "prudence"

            row[f"rwg_{dim}"] = rwg_j
            row[f"range_{dim}"] = score_range
            row[f"sd_{dim}"] = score_sd
            row[f"n_raters_{dim}"] = n_raters
            row[f"n_items_{dim}"] = n_items
            row[f"decision_{dim}"] = decision

        for construct_name, dims in construct_map.items():
            block = sub[sub["dimension"].isin(dims)].copy()
            rwg_j, score_range, score_sd, n_raters, n_items = _compute_rwg_j_from_item_block(block)
            if pd.isna(rwg_j):
                decision = "non estimable"
            elif rwg_j < QUESTIONNAIRE_RWG_THRESHOLD:
                decision = "eviter l'agregation"
            elif np.isfinite(score_range) and score_range <= QUESTIONNAIRE_RWG_MAX_RANGE_OK:
                decision = "moyenne OK"
            else:
                decision = "prudence"

            row[f"rwg_{construct_name}"] = rwg_j
            row[f"range_{construct_name}"] = score_range
            row[f"sd_{construct_name}"] = score_sd
            row[f"n_raters_{construct_name}"] = n_raters
            row[f"n_items_{construct_name}"] = n_items
            row[f"decision_{construct_name}"] = decision

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["group_id", "timepoint", "modalite", "scenario"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def summarize_questionnaire_rwg_construct_matrix(rwg_matrix_df: pd.DataFrame) -> pd.DataFrame:
    if rwg_matrix_df is None or rwg_matrix_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    summary_specs = (
        [("dimension", dim) for dim in (TMS_DIMENSIONS + COHESION_COMPONENTS) if f"rwg_{dim}" in rwg_matrix_df.columns]
        + [("construct", name) for name in ["TMS", "Cohesion"] if f"rwg_{name}" in rwg_matrix_df.columns]
    )
    for level, target in summary_specs:
        decision_col = f"decision_{target}"
        rwg_col = f"rwg_{target}"
        if decision_col not in rwg_matrix_df.columns or rwg_col not in rwg_matrix_df.columns:
            continue

        decision_counts = rwg_matrix_df[decision_col].astype(str).value_counts()
        rows.append({
            "level": level,
            "target": target,
            "n_groups": int(len(rwg_matrix_df)),
            "rwg_mean": float(pd.to_numeric(rwg_matrix_df[rwg_col], errors="coerce").mean()),
            "n_moyenne_ok": int(decision_counts.get("moyenne OK", 0)),
            "n_prudence": int(decision_counts.get("prudence", 0)),
            "n_eviter_agregation": int(decision_counts.get("eviter l'agregation", 0)),
            "n_non_estimable": int(decision_counts.get("non estimable", 0)),
        })

    return pd.DataFrame(rows)


def _questionnaire_icc_key_series(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    if df is None or df.empty or not key_cols:
        return pd.Series(dtype=str)
    work = df.copy()
    parts: list[pd.Series] = []
    for col in key_cols:
        if col in work.columns:
            vals = work[col].where(pd.notna(work[col]), "")
            parts.append(vals.astype(str).str.strip().str.lower())
        else:
            parts.append(pd.Series([""] * len(work), index=work.index, dtype=str))
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def filter_questionnaire_scores_by_icc(
    q_scores: pd.DataFrame,
    poor_threshold: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Exclut des analyses questionnaire aval les groupes dont l'ICC2k est "poor".

    Retourne :
    - q_scores filtré
    - détail des groupes exclus
    - détail ICC complet (avant exclusion)
    - résumé ICC complet (avant exclusion)
    """
    if q_scores is None or q_scores.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    icc_detail_df, icc_summary_df = compute_questionnaire_icc_summary(q_scores)
    if icc_detail_df.empty or "icc2k" not in icc_detail_df.columns:
        return q_scores.copy(), pd.DataFrame(), icc_detail_df, icc_summary_df

    excluded_groups_df = icc_detail_df.loc[
        pd.to_numeric(icc_detail_df["icc2k"], errors="coerce") < poor_threshold
    ].copy()
    if excluded_groups_df.empty:
        return q_scores.copy(), excluded_groups_df, icc_detail_df, icc_summary_df

    key_cols = [c for c in ["group_id", "timepoint", "modalite", "scenario"] if c in q_scores.columns and c in excluded_groups_df.columns]
    if not key_cols:
        return q_scores.copy(), excluded_groups_df, icc_detail_df, icc_summary_df

    work = q_scores.copy()
    work["_icc_group_key"] = _questionnaire_icc_key_series(work, key_cols)
    excluded_keys = set(_questionnaire_icc_key_series(excluded_groups_df, key_cols).tolist())
    filtered = work.loc[~work["_icc_group_key"].isin(excluded_keys)].drop(columns=["_icc_group_key"])
    return filtered.reset_index(drop=True), excluded_groups_df.reset_index(drop=True), icc_detail_df, icc_summary_df


def render_questionnaire_icc_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    q_scores: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    section_num: str = "1.4.5",
    icc_by_dimension_df: pd.DataFrame | None = None,
    icc_detail_df: pd.DataFrame | None = None,
    icc_summary_df: pd.DataFrame | None = None,
    excluded_groups_df: pd.DataFrame | None = None,
    n_groups_retained: int | None = None,
    icc_matrix_df: pd.DataFrame | None = None,
    icc_after_pruning: bool = False,
    rwg_matrix_df: pd.DataFrame | None = None,
    rwg_summary_df: pd.DataFrame | None = None,
):
    title = f"{section_num} Accord inter-membres avant agrégation (ICC)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    icc_show = (
        icc_by_dimension_df.copy()
        if icc_by_dimension_df is not None and not icc_by_dimension_df.empty
        else pd.DataFrame()
    )
    if icc_show.empty and icc_detail_df is not None and not icc_detail_df.empty:
        icc_show = icc_detail_df.copy()

    note = (
        "L'ICC est désormais calculé au niveau inter-membres, après agrégation "
        "des items en score moyen par participant et par dimension. Pour chaque "
        "dimension, la matrice ICC est `groupes x participants` : les groupes "
        "sont les cibles, les participants sont les raters. Les items ne sont "
        "donc pas utilisés comme répétitions ICC ; leur cohérence relève de "
        "l'alpha de Cronbach."
    )
    lines.append(note + "\n\n")
    pdf_elems.append(Paragraph(note.replace("`", ""), styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    if icc_show.empty:
        msg = "ICC inter-membres non calculable avec les données questionnaire disponibles."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
    else:
        for col in icc_show.columns:
            if icc_show[col].dtype.kind in {"f", "i"}:
                icc_show[col] = pd.to_numeric(icc_show[col], errors="coerce").round(3)
        show_cols = [
            c for c in [
                "dimension", "construct", "n_groups_total", "n_groups_used",
                "n_groups_excluded_lt2_participants", "n_groups_dropped_missing",
                "n_raters", "ICC2k", "ICC21", "ICC1", "interpretation",
            ] if c in icc_show.columns
        ]
        icc_show = icc_show[show_cols].copy()
        lines.append(md_table_fn(icc_show))
        pdf_elems.append(pdf_table_fn(icc_show, max_rows=12))
        pdf_elems.append(Spacer(1, 0.12 * inch))

    if rwg_matrix_df is not None and not rwg_matrix_df.empty:
        rwg_title = "Complément rwg(j) par dimension"
        rwg_note = (
            f"Le rwg(j) reste reporté comme diagnostic descriptif complémentaire "
            f"de consensus intra-groupe, selon James et al. (1984). Règle de lecture : "
            f"`rwg >= {QUESTIONNAIRE_RWG_THRESHOLD:.2f}` et `max-min <= {QUESTIONNAIRE_RWG_MAX_RANGE_OK:.0f}` "
            f"=> `moyenne OK` ; `rwg >= {QUESTIONNAIRE_RWG_THRESHOLD:.2f}` mais dispersion forte "
            f"=> `prudence` ; `rwg < {QUESTIONNAIRE_RWG_THRESHOLD:.2f}` => `eviter l'agregation`. "
            f"Les valeurs brutes hors intervalle sont tronquées à `[0, 1]`."
        )
        lines.append(f"**{rwg_title}**\n\n")
        lines.append(rwg_note + "\n\n")
        pdf_elems.append(Paragraph(rwg_title, styles["Normal"]))
        pdf_elems.append(Paragraph(rwg_note.replace("`", ""), styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

        if rwg_summary_df is None or rwg_summary_df.empty:
            rwg_summary_df = summarize_questionnaire_rwg_construct_matrix(rwg_matrix_df)
        rwg_show = rwg_summary_df.copy() if rwg_summary_df is not None else pd.DataFrame()
        if not rwg_show.empty and "level" in rwg_show.columns:
            rwg_show = rwg_show[rwg_show["level"].astype(str).eq("dimension")].copy()
        if not rwg_show.empty:
            for col in ["rwg_mean"]:
                if col in rwg_show.columns:
                    rwg_show[col] = pd.to_numeric(rwg_show[col], errors="coerce").round(3)
            rwg_show = rwg_show[
                [c for c in [
                    "level", "target", "n_groups", "rwg_mean",
                    "n_moyenne_ok", "n_prudence",
                    "n_eviter_agregation", "n_non_estimable",
                ] if c in rwg_show.columns]
            ].copy()
            lines.append(md_table_fn(rwg_show))
            pdf_elems.append(pdf_table_fn(rwg_show, max_rows=20))
            pdf_elems.append(Spacer(1, 0.12 * inch))

        # Tableau par groupe et par dimension : conserve le diagnostic rwg,
        # sans réintroduire l'ancien ICC par groupe.
        dim_cols = [f"rwg_{dim}" for dim in (TMS_DIMENSIONS + COHESION_COMPONENTS)]
        rwg_group_cols = [
            "group_id", "timepoint", "modalite", "scenario",
            *dim_cols,
        ]
        rwg_group_show = rwg_matrix_df[[c for c in rwg_group_cols if c in rwg_matrix_df.columns]].copy()
        if not rwg_group_show.empty:
            group_title = "rwg(j) par groupe et par dimension"
            lines.append(f"**{group_title}**\n\n")
            pdf_elems.append(Paragraph(group_title, styles["Normal"]))
            for col in dim_cols:
                if col in rwg_group_show.columns:
                    rwg_group_show[col] = pd.to_numeric(rwg_group_show[col], errors="coerce").round(3)
            lines.append(md_table_fn(rwg_group_show))
            pdf_elems.append(pdf_table_fn(rwg_group_show, max_rows=30))
            pdf_elems.append(Spacer(1, 0.12 * inch))

    return


def render_questionnaire_modality_effect_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    q_scores: pd.DataFrame,
    fig_dir: Path,
    add_two_plots_row_fn: Callable[[list, list[Path], Any], None],
    safe_filename_fn: Callable[[str], str],
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    section_num: str = "1.4.8",
):
    title = f"{section_num} Effet de la modalité sur les questionnaires"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    effects = compute_questionnaire_modality_effects(q_scores)
    if effects.empty:
        msg = "Comparaisons questionnaire PC vs VR indisponibles."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return

    show = effects.copy()
    rank = {dim: i for i, dim in enumerate(_questionnaire_modality_effect_dimensions())}
    if "dimension" in show.columns:
        show = show.sort_values(by="dimension", key=lambda s: s.map(rank).fillna(999)).reset_index(drop=True)
    for col in ["mean_pc", "sd_pc", "mean_vr", "sd_vr", "stat", "p_value", "cohens_d"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce").round(3)

    lines.append(md_table_fn(show))
    pdf_elems.append(pdf_table_fn(show, max_rows=12))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    q_long = build_questionnaire_group_modality_long(q_scores)
    combined_outpng = fig_dir / "questionnaire_modalite_dimensions_boxplot.png"
    if plot_questionnaire_dimensions_by_modalite(q_long, combined_outpng):
        lines.append("**Boxplot synthétique des dimensions questionnaire par modalité**\n\n")
        lines.append(f"![]({combined_outpng.name})\n\n")
        pdf_elems.append(Paragraph("Boxplot synthétique des dimensions questionnaire par modalité", styles["Heading4"]))
        pdf_elems.append(Image(str(combined_outpng), width=6.7 * inch, height=3.3 * inch))
        pdf_elems.append(Spacer(1, 0.12 * inch))

    made = []
    for dim in _questionnaire_modality_effect_dimensions():
        outpng = fig_dir / f"questionnaire_modalite_{safe_filename_fn(dim)}.png"
        ok = plot_questionnaire_by_modalite(
            q_long,
            dim=dim,
            outpng=outpng,
            title=f"{dim} — comparaison PC vs VR",
        )
        if ok:
            made.append((dim, outpng))

    for i in range(0, len(made), 2):
        pair = made[i:i + 2]
        add_two_plots_row_fn(pdf_elems, [p for _, p in pair], styles)
        pdf_elems.append(Spacer(1, 0.12 * inch))


def questionnaire_group_condition_scores(
    q_scores: pd.DataFrame,
    normalize_group_fn: Callable[[pd.DataFrame | None], pd.DataFrame | None],
    harmonize_timepoint_fn: Callable[..., pd.DataFrame | None],
    normalize_timepoint_fn: Callable[[pd.DataFrame | None], pd.DataFrame | None],
) -> pd.DataFrame:
    """Construit une table questionnaire agrégée au niveau groupe × condition × dimension."""
    if q_scores is None or q_scores.empty:
        return pd.DataFrame()

    df = q_scores.copy()
    ren = {}
    if "Modalite" in df.columns and "modalite" not in df.columns:
        ren["Modalite"] = "modalite"
    if "Scenario" in df.columns and "scenario" not in df.columns:
        ren["Scenario"] = "scenario"
    if ren:
        df = df.rename(columns=ren)

    required = {"group_id", "dimension", "score", "modalite", "scenario"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = normalize_group_fn(df)
    df = harmonize_timepoint_fn(df, session_col="session", raw_group_candidates=["group_id"])
    df = normalize_timepoint_fn(df)

    df["dimension"] = df["dimension"].astype(str).str.strip()
    df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["group_id", "dimension", "score", "modalite", "scenario"])
    if df.empty:
        return pd.DataFrame()

    group_cols = ["group_id", "modalite", "scenario", "dimension"]
    if "timepoint" in df.columns:
        group_cols = ["group_id", "timepoint", "modalite", "scenario", "dimension"]

    return df.groupby(group_cols, dropna=False)["score"].mean().reset_index()


def render_questionnaire_group_condition_plots(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    q_scores: pd.DataFrame,
    fig_dir: Path,
    normalize_group_fn: Callable[[pd.DataFrame | None], pd.DataFrame | None],
    harmonize_timepoint_fn: Callable[..., pd.DataFrame | None],
    normalize_timepoint_fn: Callable[[pd.DataFrame | None], pd.DataFrame | None],
    add_two_plots_row_fn: Callable[[list, list[Path], Any], None],
    safe_filename_fn: Callable[[str], str],
    dims_order: list[str] | None = None,
    section_num: str = "1.4.5",
):
    title = f"{section_num} Questionnaire par condition (moyenne par groupe)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    q_group_cond = questionnaire_group_condition_scores(
        q_scores,
        normalize_group_fn=normalize_group_fn,
        harmonize_timepoint_fn=harmonize_timepoint_fn,
        normalize_timepoint_fn=normalize_timepoint_fn,
    )
    if q_group_cond is None or q_group_cond.empty:
        lines.append("_(données insuffisantes pour un plot agrégé par groupe)_\n\n")
        pdf_elems.append(Paragraph("(données insuffisantes pour un plot agrégé par groupe)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    dims = sorted(q_group_cond["dimension"].dropna().astype(str).unique())
    if dims_order:
        dims = [d for d in dims_order if d in dims] + [d for d in dims if d not in dims_order]

    made = []
    for dim in dims:
        outpng = fig_dir / f"questionnaire_condition_group_mean_{safe_filename_fn(dim)}.png"
        ok = plot_questionnaire_by_condition(
            q_group_cond,
            dim=dim,
            outpng=outpng,
            title=f"Questionnaire – {dim} par condition (moyenne groupe)",
        )
        if ok:
            made.append((dim, outpng))

    if not made:
        lines.append("_(aucun plot group-level généré)_\n\n")
        pdf_elems.append(Paragraph("(aucun plot group-level généré)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    for i in range(0, len(made), 2):
        pair = made[i:i + 2]
        paths = [p for _, p in pair]
        add_two_plots_row_fn(pdf_elems, paths, styles)
        pdf_elems.append(Spacer(1, 0.12 * inch))

    lines.append(f"{len(made)} plot(s) group-level ajoutés au PDF.\n\n")


def render_questionnaire_condition_plots(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    q_scores: pd.DataFrame,
    fig_dir: Path,
    add_two_plots_row_fn: Callable[[list, list[Path], Any], None],
    safe_filename_fn: Callable[[str], str],
    dims_order: list[str] | None = None,
    section_num: str = "1.4.4",
):
    title = f"{section_num} Questionnaire par condition (modalité × scénario)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    if q_scores is None or q_scores.empty:
        lines.append("_(q_scores vide)_\n\n")
        pdf_elems.append(Paragraph("(q_scores vide)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    df = q_scores.copy()
    ren = {}
    if "Modalite" in df.columns and "modalite" not in df.columns:
        ren["Modalite"] = "modalite"
    if "Scenario" in df.columns and "scenario" not in df.columns:
        ren["Scenario"] = "scenario"
    if ren:
        df = df.rename(columns=ren)

    needed = {"dimension", "score", "modalite", "scenario"}
    if not needed.issubset(df.columns):
        msg = f"Colonnes manquantes pour les plots questionnaire: {sorted(needed - set(df.columns))}"
        lines.append(msg + "\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    df["dimension"] = df["dimension"].astype(str).str.strip()
    dims = sorted(df["dimension"].dropna().unique())
    if dims_order:
        dims = [d for d in dims_order if d in dims] + [d for d in dims if d not in dims_order]

    made = []
    for dim in dims:
        outpng = fig_dir / f"questionnaire_condition_{safe_filename_fn(dim)}.png"
        ok = plot_questionnaire_by_condition(df, dim=dim, outpng=outpng, title=f"Questionnaire – {dim} par condition")
        if ok:
            made.append((dim, outpng))

    if not made:
        lines.append("_(aucun plot questionnaire généré)_\n\n")
        pdf_elems.append(Paragraph("(aucun plot questionnaire généré)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    for i in range(0, len(made), 2):
        pair = made[i:i + 2]
        paths = [p for _, p in pair]
        add_two_plots_row_fn(pdf_elems, paths, styles)
        pdf_elems.append(Spacer(1, 0.12 * inch))

    lines.append(f"{len(made)} plot(s) questionnaire ajoutés au PDF.\n\n")


def render_questionnaire_profile_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    profile_df: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    section_num: str = "1.2",
):
    title = f"{section_num} Profil des participants questionnaire"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))

    if profile_df is None or profile_df.empty:
        msg = "Profil participant indisponible dans les sorties questionnaire."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return

    work = profile_df.copy()
    n_participants = len(work)
    n_groups = work["group_id"].replace("", np.nan).dropna().nunique() if "group_id" in work.columns else np.nan
    age_vals = pd.to_numeric(work["age"], errors="coerce").dropna() if "age" in work.columns else pd.Series(dtype=float)
    team_vals = pd.to_numeric(work["team_familiarity_mean_score"], errors="coerce").dropna() if "team_familiarity_mean_score" in work.columns else pd.Series(dtype=float)

    intro = (
        f"Le profil participant ci-dessous est calculé sur {n_participants} répondant(s)"
        f"{f' appartenant à {int(n_groups)} groupe(s)' if pd.notna(n_groups) else ''} "
        "retenus dans le périmètre courant du rapport."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    overview = pd.DataFrame([{
        "n_participants": int(n_participants),
        "n_groups": int(n_groups) if pd.notna(n_groups) else np.nan,
        "age_mean": float(age_vals.mean()) if not age_vals.empty else np.nan,
        "age_sd": float(age_vals.std(ddof=1)) if len(age_vals) >= 2 else np.nan,
        "age_min": float(age_vals.min()) if not age_vals.empty else np.nan,
        "age_max": float(age_vals.max()) if not age_vals.empty else np.nan,
        "team_familiarity_mean": float(team_vals.mean()) if not team_vals.empty else np.nan,
    }])
    for col in ["age_mean", "age_sd", "age_min", "age_max", "team_familiarity_mean"]:
        if col in overview.columns:
            overview[col] = pd.to_numeric(overview[col], errors="coerce").round(2)

    lines.append(md_table_fn(overview))
    pdf_elems.append(pdf_table_fn(overview, max_rows=8))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    count_specs = [
        ("gender", None, "Répartition déclarée femme/homme/non-binaire"),
        ("vr_familiarity", PROFILE_FAMILIARITY_ORDER, "Familiarité déclarée avec la VR"),
        ("team_familiarity_mean_label", PROFILE_FAMILIARITY_ORDER, "Familiarité préalable avec les coéquipiers (moyenne des deux items)"),
    ]

    for col, order, subtitle in count_specs:
        if col not in work.columns:
            continue
        vals = work[col].replace("", np.nan).dropna()
        if vals.empty:
            continue
        counts = vals.value_counts(dropna=False)
        levels = order if order is not None else list(counts.index)
        rows = []
        for level in levels:
            n = int(counts.get(level, 0))
            if n == 0:
                continue
            rows.append({"level": level, "n": n, "pct": round(100 * n / len(vals), 1)})
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        lines.append(f"**{subtitle}**\n\n")
        lines.append(md_table_fn(tab))
        pdf_elems.append(Paragraph(subtitle, styles["Normal"]))
        pdf_elems.append(pdf_table_fn(tab, max_rows=12))
        pdf_elems.append(Spacer(1, 0.08 * inch))


def render_questionnaire_comments_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    comments_df: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    section_num: str = "1.2b",
):
    title = f"{section_num} Commentaires libres questionnaire"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))

    if comments_df is None or comments_df.empty:
        msg = "Aucun commentaire libre exploitable dans le périmètre courant."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return

    work = comments_df.copy()
    if "comment_type" not in work.columns:
        msg = "Structure des commentaires libres invalide."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return

    summary_rows = []
    for comment_type, label in [("task", "Tâche"), ("group", "Groupe")]:
        sub = work[work["comment_type"] == comment_type].copy()
        if sub.empty:
            continue
        summary_rows.append({
            "bloc": label,
            "n_commentaires": int(len(sub)),
            "n_commentaires_substantiels": int((~sub.get("is_trivial", False).astype(bool)).sum()) if "is_trivial" in sub.columns else int(len(sub)),
        })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        lines.append(md_table_fn(summary_df))
        pdf_elems.append(pdf_table_fn(summary_df, max_rows=8))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    theme_rows = []
    for comment_type, label in [("task", "Tâche"), ("group", "Groupe")]:
        sub = work[work["comment_type"] == comment_type].copy()
        if sub.empty or "themes" not in sub.columns:
            continue
        theme_counts: dict[str, int] = {}
        for themes in sub["themes"].astype(str):
            for theme in [t.strip() for t in themes.split(";") if t.strip()]:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        if not theme_counts:
            continue
        for theme, count in sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            theme_rows.append({
                "bloc": label,
                "theme": theme,
                "n": int(count),
                "pct": round(100 * count / len(sub), 1),
            })

    if theme_rows:
        theme_df = pd.DataFrame(theme_rows)
        lines.append("**Themes les plus frequents dans les commentaires libres**\n\n")
        lines.append(md_table_fn(theme_df, max_rows=16))
        pdf_elems.append(Paragraph("Themes les plus frequents dans les commentaires libres", styles["Normal"]))
        pdf_elems.append(pdf_table_fn(theme_df, max_rows=16))
        pdf_elems.append(Spacer(1, 0.08 * inch))

    exemplar_rows = []
    for comment_type, label in [("task", "Tâche"), ("group", "Groupe")]:
        sub = work[work["comment_type"] == comment_type].copy()
        if sub.empty:
            continue
        if "is_trivial" in sub.columns:
            sub = sub[~sub["is_trivial"].astype(bool)].copy()
        if sub.empty:
            continue
        sub = sub.drop_duplicates(subset=["text"] if "text" in sub.columns else None)
        for _, row in sub.head(3).iterrows():
            exemplar_rows.append({
                "bloc": label,
                "extrait": row.get("excerpt", row.get("text", "")),
            })

    if exemplar_rows:
        intro = (
            "Lecture qualitative breve : les commentaires portent surtout sur la clarte "
            "ou la difficulte de la tache, ainsi que sur la dynamique de collaboration "
            "et l'ecoute dans le groupe."
        )
        lines.append(intro + "\n\n")
        pdf_elems.append(Paragraph(intro, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

        ex_df = pd.DataFrame(exemplar_rows)
        lines.append("**Extraits illustratifs (anonymises)**\n\n")
        lines.append(md_table_fn(ex_df, max_rows=6))
        pdf_elems.append(Paragraph("Extraits illustratifs (anonymises)", styles["Normal"]))
        pdf_elems.append(pdf_table_fn(ex_df, max_rows=6))
        pdf_elems.append(Spacer(1, 0.12 * inch))


def render_questionnaire_pruning_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    explo_summary: pd.DataFrame,
    alpha_comp: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    max_rows_md: int,
    max_rows_pdf: int,
    section_num: str = "1.4.2",
):
    """Rend la sous-section pruning exploratoire."""
    title = f"{section_num} Optimisation exploratoire avant exclusion ICC (item pruning)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    if alpha_comp is not None and not alpha_comp.empty:
        lines.append("**Comparaison alpha avant / après pruning :**\n\n")
        pdf_elems.append(Paragraph("Comparaison alpha avant / après pruning", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch))

        show_cols = [c for c in ["dimension", "label", "n_items", "alpha_avant", "alpha_apres",
                                  "gain_alpha", "n_items_retires"] if c in alpha_comp.columns]
        tab = alpha_comp[show_cols].copy()
        for c in ["alpha_avant", "alpha_apres", "gain_alpha"]:
            if c in tab.columns:
                tab[c] = pd.to_numeric(tab[c], errors="coerce").round(3)
        if "n_items_retires" in tab.columns:
            tab["n_items_retires"] = pd.to_numeric(tab["n_items_retires"], errors="coerce").fillna(0).astype(int)

        if "dimension" in tab.columns:
            order = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]
            rank = {k: i for i, k in enumerate(order)}
            tab = tab.sort_values(by=["dimension"], key=lambda s: s.map(rank).fillna(999)).reset_index(drop=True)

        lines.append(md_table_fn(tab, max_rows=max_rows_md))
        pdf_elems.append(pdf_table_fn(tab, max_rows=max_rows_pdf))
        pdf_elems.append(Spacer(1, 0.15 * inch))
    else:
        lines.append("_(comparaison alpha non disponible)_\n\n")
        pdf_elems.append(Paragraph("(comparaison alpha non disponible)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    if alpha_comp is not None and not alpha_comp.empty and "items_retires" in alpha_comp.columns:
        dropped = alpha_comp[alpha_comp["items_retires"].fillna("").astype(str).str.strip().ne("")]
        if not dropped.empty:
            lines.append("**Items retirés pour améliorer l'alpha :**\n\n")
            pdf_elems.append(Paragraph("Items retirés pour améliorer l'alpha :", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.08 * inch))

            for _, row in dropped.iterrows():
                dim = row.get("dimension", "?")
                items_str = str(row.get("items_retires", ""))
                items_list = [it.strip() for it in items_str.split(";") if it.strip()]
                items_short = []
                for it in items_list:
                    m = re.search(r"\[([A-Z]{3}\d{2})\]", it)
                    items_short.append(m.group(1) if m else it[:50])

                txt = f"• {dim} : {', '.join(items_short)}"
                lines.append(txt + "\n")
                pdf_elems.append(Paragraph(txt, styles["Normal"]))

            pdf_elems.append(Spacer(1, 0.15 * inch))
            lines.append("\n")

def render_questionnaire_descriptif_table(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    desc_df: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    fmt2_fn: Callable[[Any], str],
    max_rows_md: int,
    max_rows_pdf: int,
    title_suffix: str = "",
    section_num: str = "1.4.3",
):
    """Rend un tableau descriptif des dimensions."""
    title = f"{section_num} Descriptifs par dimension{title_suffix}"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    if desc_df is None or desc_df.empty:
        lines.append("_(descriptifs non disponibles)_\n\n")
        pdf_elems.append(Paragraph("(descriptifs non disponibles)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    show_cols = [c for c in ["dimension", "label", "n_responses", "n_participants", "n_items", "mean", "sd", "median", "min", "max"] if c in desc_df.columns]
    if not show_cols:
        lines.append("_(colonnes descriptives introuvables)_\n\n")
        pdf_elems.append(Paragraph("(colonnes descriptives introuvables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    tab = desc_df[show_cols].copy()
    for c in ["mean", "sd", "median", "min", "max"]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce").apply(fmt2_fn)

    if "dimension" in tab.columns:
        order = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]
        rank = {k: i for i, k in enumerate(order)}
        tab = tab.sort_values(by=["dimension"], key=lambda s: s.map(rank).fillna(999)).reset_index(drop=True)

    lines.append(md_table_fn(tab, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_fn(tab, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.15 * inch))


def render_questionnaire_cronbach_table(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    cronbach_df: pd.DataFrame,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    max_rows_md: int,
    max_rows_pdf: int,
    section_num: str = "1.4.1",
):
    title = f"{section_num} Fiabilité interne initiale (avant pruning, avant exclusion ICC)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))

    if cronbach_df is None or cronbach_df.empty:
        lines.append("_(table alpha de Cronbach non disponible)_\n\n")
        pdf_elems.append(Paragraph("(table alpha de Cronbach non disponible)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    cols = [c for c in ["dimension", "label", "n_items", "alpha"] if c in cronbach_df.columns]
    if not cols:
        lines.append("_(colonnes alpha de Cronbach introuvables)_\n\n")
        pdf_elems.append(Paragraph("(colonnes alpha de Cronbach introuvables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    tab = cronbach_df[cols].copy()
    if "dimension" in tab.columns:
        tab["dimension"] = tab["dimension"].astype(str).str.upper().str.strip()
    if "alpha" in tab.columns:
        tab["alpha"] = pd.to_numeric(tab["alpha"], errors="coerce").round(3)
    if "n_items" in tab.columns:
        tab["n_items"] = pd.to_numeric(tab["n_items"], errors="coerce")

    if "dimension" in tab.columns:
        order = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]
        rank = {k: i for i, k in enumerate(order)}
        tab = tab.sort_values(by=["dimension"], key=lambda s: s.map(rank).fillna(999)).reset_index(drop=True)

    lines.append(md_table_fn(tab, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_fn(tab, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.15 * inch))
