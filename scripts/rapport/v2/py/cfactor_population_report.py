# -*- coding: utf-8 -*-
"""
Rapport autonome pour le c-factor projete sur la population parente Riedl 2021.

Ce module produit un PDF dedie ainsi que les tables / figures associees.
Il reutilise les memes briques de rendu que le pipeline v2 (ReportLab +
matplotlib) afin de conserver un format homogene avec les rapports existants.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.stats import bootstrap, linregress, spearmanr

_scripts_dir = _Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))

from py.corr import bh_fdr, pdf_table_from_df
from py.data import normalize_group, read_csv_auto


REPORT_NAME = "test_cfactor_global_population"
SEED = 42
TMS_DIM_ORDER = ["tms_coordination", "tms_credibilite", "tms_specialisation"]
COHESION_DIM_ORDER = ["cohesion_sociale", "cohesion_tache", "cohesion_communication"]
QUESTIONNAIRE_OUTCOMES = TMS_DIM_ORDER + COHESION_DIM_ORDER
TCI_ID_COLS = {
    "group_id",
    "c_score",
    "rme_mean",
    "rme_max",
    "rme_min",
    "c_factor_pop",
    "c_factor_sample",
    "rank_pop",
    "rank_sample",
    "rank_delta",
}


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "Title": styles["Title"],
        "Heading1": styles["Heading1"],
        "Heading2": styles["Heading2"],
        "Body": styles["BodyText"],
        "Small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10.5,
            spaceAfter=4,
        ),
        "Box": ParagraphStyle(
            "Box",
            parent=styles["BodyText"],
            fontSize=9,
            leading=11,
            backColor=colors.whitesmoke,
            borderWidth=0.6,
            borderColor=colors.lightgrey,
            borderPadding=6,
            spaceAfter=6,
        ),
    }


def _paragraph(text: str, styles: dict[str, ParagraphStyle], style: str = "Body") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style])


def _sample_box(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles["Box"])


def _save_pdf(path: Path, elems: list[Any]) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    doc.build(elems)


def _read_and_normalize(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_group(df)


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _find_existing(results_dir: Path, rel_candidates: list[str]) -> Path | None:
    for rel in rel_candidates:
        p = results_dir / rel
        if p.exists():
            return p
    return None


def _load_tci_population(results_dir: Path, scope: str) -> tuple[pd.DataFrame, Path]:
    scope_norm = str(scope).strip().lower()
    suffix = "allowed" if scope_norm == "allowed" else "all"
    path = results_dir / "TCI" / f"c_scores_{suffix}_pop.csv"
    if not path.exists():
        raise FileNotFoundError(f"Fichier TCI population introuvable: {path}")
    df = _read_and_normalize(path)
    numeric_cols = [
        "c_score",
        "rme_mean",
        "rme_max",
        "rme_min",
        "c_factor_pop",
        "c_factor_sample",
        "rank_pop",
        "rank_sample",
        "rank_delta",
    ]
    task_cols = [c for c in df.columns if c not in {"group_id"}]
    return _coerce_numeric(df, list(set(numeric_cols + task_cols))), path


def _load_tci_population_with_tasks(results_dir: Path, scope: str) -> tuple[pd.DataFrame, Path]:
    scope_norm = str(scope).strip().lower()
    suffix = "allowed" if scope_norm == "allowed" else "all"
    path = results_dir / "TCI" / f"c_scores_{suffix}_pop_with_tasks.csv"
    if not path.exists():
        # Fallback sur le fichier principal, qui contient deja les taches dans
        # la passe actuelle.
        return _load_tci_population(results_dir, scope)
    df = _read_and_normalize(path)
    numeric_cols = [c for c in df.columns if c != "group_id"]
    return _coerce_numeric(df, numeric_cols), path


def _load_task_reference_stats(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "TCI" / "task_reference_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    return _coerce_numeric(_read_and_normalize(path), [c for c in _read_and_normalize(path).columns if c not in {"scope", "task_sample", "task_full_csv", "reference_pop", "flag", "parent_pca_full_task", "parent_pca_orientation_source"}])


def _load_parent_loadings(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "TCI" / "c_factor_parent_loadings.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_and_normalize(path)
    num_cols = [c for c in df.columns if c not in {"task_sample", "task_full_csv", "orientation_source"}]
    return _coerce_numeric(df, num_cols)


def _load_population_log(results_dir: Path) -> list[str]:
    path = results_dir / "TCI" / "cfactor_population_log.txt"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _load_riedl_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "indices_collab" / "riedl_group_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_and_normalize(path)
    if "GroupID" in df.columns and "group_id" not in df.columns:
        df = df.rename(columns={"GroupID": "group_id"})
    num_cols = [c for c in df.columns if c not in {"group_id", "analysis_effort_mode", "analysis_congruence_mode", "analysis_congruence_effort_scope", "analysis_sudoku_denominator", "analysis_reference_full_csv", "effort_norm_pop_source", "strategy_norm_pop_source"}]
    return _coerce_numeric(df, num_cols)


def _load_profile(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "questionnaire" / "global" / "participant_profile_responses.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_and_normalize(path)
    for col in ["vr_familiarity_score", "team_familiarity_mean_score", "age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()
    return df


def _load_performance(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "performance_task" / "recap_scores_all.csv"
    if not path.exists():
        return pd.DataFrame()
    df = _read_and_normalize(path)
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()
    if "Score_perf_tsk" in df.columns:
        df["Score_perf_tsk"] = pd.to_numeric(df["Score_perf_tsk"], errors="coerce")
    return df


def _load_questionnaire_group(results_dir: Path) -> pd.DataFrame:
    path = _find_existing(
        results_dir,
        [
            "questionnaire/global/pruned/scores_dimension_par_participant_pruned.csv",
            "questionnaire/analyse/scores_dimension_par_participant.csv",
        ],
    )
    if path is None:
        return pd.DataFrame()
    df = _read_and_normalize(path)
    rename_map = {
        "Groupe": "group_id",
        "Modalite": "modalite",
        "Scenario": "scenario",
        "Session": "session",
        "Participant": "participant",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "score" not in df.columns or "dimension" not in df.columns:
        return pd.DataFrame()

    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()

    dim_map = {
        "COR": "tms_coordination",
        "CRE": "tms_credibilite",
        "SPE": "tms_specialisation",
        "SOC": "cohesion_sociale",
        "TSK": "cohesion_tache",
        "COM": "cohesion_communication",
    }
    df["dimension_std"] = df["dimension"].astype(str).str.upper().str.strip().map(dim_map)
    df = df.dropna(subset=["group_id", "dimension_std", "score"])
    group_cols = ["group_id", "dimension_std"]
    if "modalite" in df.columns:
        meta = (
            df.groupby("group_id", dropna=False)
            .agg(
                modalite=("modalite", "first"),
                scenario=("scenario", "first") if "scenario" in df.columns else ("group_id", "first"),
            )
            .reset_index()
        )
    else:
        meta = pd.DataFrame({"group_id": df["group_id"].drop_duplicates()})

    wide = (
        df.groupby(group_cols, dropna=False)["score"]
        .mean()
        .unstack("dimension_std")
        .reset_index()
    )
    for col in COHESION_DIM_ORDER:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["cohesion_globale"] = wide[COHESION_DIM_ORDER].mean(axis=1, skipna=True)
    return wide.merge(meta, on="group_id", how="left")


def _spearman_pair(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    if x not in df.columns or y not in df.columns:
        return {"x": x, "y": y, "n": 0, "rho": np.nan, "p": np.nan}
    sub = df[[x, y]].dropna()
    n = int(len(sub))
    if n < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        return {"x": x, "y": y, "n": n, "rho": np.nan, "p": np.nan}
    rho, p = spearmanr(sub[x], sub[y])
    return {"x": x, "y": y, "n": n, "rho": float(rho), "p": float(p)}


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 5000,
) -> tuple[float, float]:
    if len(x) < 4:
        return np.nan, np.nan

    def _stat(a: np.ndarray, b: np.ndarray, axis: int = -1) -> float:
        return stat_fn(np.asarray(a), np.asarray(b))

    try:
        res = bootstrap(
            (x, y),
            _stat,
            paired=True,
            vectorized=False,
            method="BCa",
            n_resamples=n_resamples,
            random_state=SEED,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return np.nan, np.nan


def _stat_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    if len(np.unique(x[~np.isnan(x)])) < 2 or len(np.unique(y[~np.isnan(y)])) < 2:
        return np.nan
    rho, _ = spearmanr(x, y)
    return float(rho)


def _standardized_linear_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 3:
        return {"beta": np.nan, "r2": np.nan, "p": np.nan}
    x_std = np.nanstd(x, ddof=0)
    y_std = np.nanstd(y, ddof=0)
    if not np.isfinite(x_std) or not np.isfinite(y_std) or x_std == 0 or y_std == 0:
        return {"beta": np.nan, "r2": np.nan, "p": np.nan}
    zx = (x - np.nanmean(x)) / x_std
    zy = (y - np.nanmean(y)) / y_std
    reg = linregress(zx, zy)
    return {"beta": float(reg.slope), "r2": float(reg.rvalue ** 2), "p": float(reg.pvalue)}


def _stat_beta(x: np.ndarray, y: np.ndarray) -> float:
    return _standardized_linear_metrics(x, y)["beta"]


def _fdr_by_outcome(df: pd.DataFrame, p_col: str = "p_spearman", group_col: str = "outcome") -> pd.DataFrame:
    if df.empty or p_col not in df.columns or group_col not in df.columns:
        return df
    out = df.copy()
    out["p_fdr"] = np.nan
    for outcome, idx in out.groupby(group_col).groups.items():
        vals = pd.to_numeric(out.loc[idx, p_col], errors="coerce").values
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        corrected = bh_fdr(vals[mask])
        out.loc[np.array(list(idx))[mask], "p_fdr"] = corrected
    return out


def _add_sig(df: pd.DataFrame, p_col: str = "p_fdr") -> pd.DataFrame:
    out = df.copy()
    out["sig"] = ""
    if p_col not in out.columns:
        return out
    mask_01 = pd.to_numeric(out[p_col], errors="coerce") < 0.01
    mask_05 = (pd.to_numeric(out[p_col], errors="coerce") < 0.05) & ~mask_01
    out.loc[mask_01, "sig"] = "**"
    out.loc[mask_05, "sig"] = "*"
    return out


def _task_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TCI_ID_COLS]


def _comparison_summary(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = _spearman_pair(df, "c_factor_pop", "c_factor_sample")
    sub = df[["c_factor_pop", "c_factor_sample"]].dropna()
    pearson = sub["c_factor_pop"].corr(sub["c_factor_sample"], method="pearson") if not sub.empty else np.nan
    rows.append(
        {
            "scope": scope,
            "n_groups": int(df["group_id"].nunique()) if "group_id" in df.columns else len(df),
            "n_complete_pairs": int(len(sub)),
            "rho_spearman": metrics["rho"],
            "p_spearman": metrics["p"],
            "r_pearson": pearson,
            "mean_abs_rank_delta": pd.to_numeric(df.get("rank_delta"), errors="coerce").abs().mean(),
            "max_abs_rank_delta": pd.to_numeric(df.get("rank_delta"), errors="coerce").abs().max(),
        }
    )
    out = pd.DataFrame(rows)
    for col in ["rho_spearman", "p_spearman", "r_pearson", "mean_abs_rank_delta", "max_abs_rank_delta"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def _task_associations(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_cols = _task_columns(df)
    targets = ["c_factor_pop", "c_factor_sample", "rme_mean"]
    for task in task_cols:
        for target in targets:
            rec = _spearman_pair(df, task, target)
            rec["scope"] = scope
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.rename(columns={"x": "task", "y": "target"})
    out = out.sort_values(["target", "p", "rho"], ascending=[True, True, False]).reset_index(drop=True)
    for col in ["rho", "p"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def _rme_correlations(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = [
        dict(_spearman_pair(df, "c_factor_pop", y), scope=scope, predictor="c_factor_pop")
        for y in ["rme_mean", "rme_max", "rme_min"]
    ]
    rows += [
        dict(_spearman_pair(df, "c_factor_sample", y), scope=scope, predictor="c_factor_sample")
        for y in ["rme_mean", "rme_max", "rme_min"]
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.rename(columns={"y": "outcome"})
    out = out[["scope", "predictor", "outcome", "n", "rho", "p"]]
    for col in ["rho", "p"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def _riedl_descriptives(riedl: pd.DataFrame) -> pd.DataFrame:
    if riedl.empty:
        return pd.DataFrame()
    wanted = [
        "skill_mean",
        "skill_max",
        "skill_congruence_mean",
        "skill_congruence_mean_core",
        "strategy_ratio_mean",
        "strategy_norm",
        "strategy_norm_pop",
        "effort_task_sum",
        "effort_task_norm",
        "effort_norm_pop",
    ]
    rows: list[dict[str, Any]] = []
    for col in wanted:
        if col not in riedl.columns:
            continue
        vals = pd.to_numeric(riedl[col], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append(
            {
                "variable": col,
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "sd": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["mean", "sd", "min", "max"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    return out


def _riedl_cfactor_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    predictors = [
        "skill_mean",
        "skill_max",
        "skill_congruence_mean",
        "skill_congruence_mean_core",
        "strategy_ratio_mean",
        "strategy_norm",
        "strategy_norm_pop",
        "effort_task_sum",
        "effort_task_norm",
        "effort_norm_pop",
    ]
    for predictor in predictors:
        if predictor not in merged.columns:
            continue
        for outcome in ["c_factor_pop", "c_factor_sample"]:
            rec = _spearman_pair(merged, predictor, outcome)
            rec["predictor"] = predictor
            rec["outcome"] = outcome
            rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[["predictor", "outcome", "n", "rho", "p"]]
    for col in ["rho", "p"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out


def _build_group_familiarity(profile: pd.DataFrame) -> pd.DataFrame:
    if profile.empty or "group_id" not in profile.columns:
        return pd.DataFrame()
    numeric_candidates = [c for c in ["vr_familiarity_score", "team_familiarity_mean_score", "age"] if c in profile.columns]
    meta_cols = [c for c in ["group_id", "modalite", "scenario"] if c in profile.columns]
    agg: dict[str, tuple[str, str]] = {}
    if "vr_familiarity_score" in numeric_candidates:
        agg.update(
            {
                "vr_familiarity_mean": ("vr_familiarity_score", "mean"),
                "vr_familiarity_sd": ("vr_familiarity_score", "std"),
                "vr_familiarity_min": ("vr_familiarity_score", "min"),
                "vr_familiarity_max": ("vr_familiarity_score", "max"),
            }
        )
    if "team_familiarity_mean_score" in numeric_candidates:
        agg.update(
            {
                "team_familiarity_mean": ("team_familiarity_mean_score", "mean"),
                "team_familiarity_sd": ("team_familiarity_mean_score", "std"),
                "team_familiarity_min": ("team_familiarity_mean_score", "min"),
                "team_familiarity_max": ("team_familiarity_mean_score", "max"),
            }
        )
    if "age" in numeric_candidates:
        agg.update({"age_mean": ("age", "mean")})

    group_df = profile.groupby("group_id", dropna=False).agg(**agg).reset_index() if agg else pd.DataFrame({"group_id": profile["group_id"].drop_duplicates()})
    for col in ["modalite", "scenario"]:
        if col in profile.columns:
            first_meta = (
                profile.groupby("group_id", dropna=False)[col]
                .agg(lambda s: next((str(v).strip().upper() for v in s if pd.notna(v) and str(v).strip()), np.nan))
                .reset_index(name=col)
            )
            group_df = group_df.merge(first_meta, on="group_id", how="left")

    for col in [c for c in group_df.columns if c != "group_id" and c not in {"modalite", "scenario"}]:
        group_df[col] = pd.to_numeric(group_df[col], errors="coerce")
    return group_df


def _familiarity_analysis(
    fam_group: pd.DataFrame,
    tci_allowed: pd.DataFrame,
    perf: pd.DataFrame,
    q_group: pd.DataFrame,
) -> pd.DataFrame:
    if fam_group.empty:
        return pd.DataFrame()
    predictors = [c for c in fam_group.columns if "familiarity" in c and c not in {"modalite", "scenario"}]
    if not predictors:
        return pd.DataFrame()

    merge_frames = [fam_group.copy()]
    keep_tci = [c for c in ["group_id", "c_factor_pop", "c_factor_sample"] if c in tci_allowed.columns]
    if keep_tci:
        merge_frames.append(tci_allowed[keep_tci].drop_duplicates(subset=["group_id"]))
    keep_perf = [c for c in ["group_id", "Score_perf_tsk", "modalite", "scenario"] if c in perf.columns]
    if keep_perf:
        merge_frames.append(perf[keep_perf].drop_duplicates(subset=["group_id"]))
    keep_q = ["group_id"] + [c for c in QUESTIONNAIRE_OUTCOMES if c in q_group.columns]
    if len(keep_q) > 1:
        merge_frames.append(q_group[keep_q].drop_duplicates(subset=["group_id"]))

    merged = merge_frames[0]
    for other in merge_frames[1:]:
        merged = merged.merge(other, on="group_id", how="left", suffixes=("", "_dup"))
    if "modalite" in merged.columns:
        merged = merged.loc[merged["modalite"].astype(str).str.upper().eq("VR")].copy()

    outcomes = [c for c in ["c_factor_pop", "c_factor_sample", "Score_perf_tsk"] + QUESTIONNAIRE_OUTCOMES if c in merged.columns]
    rows: list[dict[str, Any]] = []

    for predictor in predictors:
        for outcome in outcomes:
            sub = merged[[predictor, outcome]].dropna()
            n = int(len(sub))
            rec: dict[str, Any] = {
                "predictor": predictor,
                "outcome": outcome,
                "n": n,
                "rho_spearman": np.nan,
                "p_spearman": np.nan,
                "rho_ci_low": np.nan,
                "rho_ci_high": np.nan,
                "beta_std": np.nan,
                "beta_p": np.nan,
                "beta_ci_low": np.nan,
                "beta_ci_high": np.nan,
                "r2": np.nan,
            }
            if n < 3 or sub[predictor].nunique(dropna=True) < 2 or sub[outcome].nunique(dropna=True) < 2:
                rows.append(rec)
                continue

            x = sub[predictor].to_numpy(dtype=float)
            y = sub[outcome].to_numpy(dtype=float)
            rho, p = spearmanr(x, y)
            rec["rho_spearman"] = float(rho)
            rec["p_spearman"] = float(p)
            rec["rho_ci_low"], rec["rho_ci_high"] = _bootstrap_ci(x, y, _stat_spearman)

            lm = _standardized_linear_metrics(x, y)
            rec["beta_std"] = lm["beta"]
            rec["beta_p"] = lm["p"]
            rec["r2"] = lm["r2"]
            rec["beta_ci_low"], rec["beta_ci_high"] = _bootstrap_ci(x, y, _stat_beta)
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _fdr_by_outcome(out, p_col="p_spearman", group_col="outcome")
    out = _add_sig(out, p_col="p_fdr")
    round_cols = ["rho_spearman", "p_spearman", "rho_ci_low", "rho_ci_high", "beta_std", "beta_p", "beta_ci_low", "beta_ci_high", "r2", "p_fdr"]
    for col in round_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)
    return out.sort_values(["outcome", "p_fdr", "p_spearman"], ascending=[True, True, True]).reset_index(drop=True)


def _plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> bool:
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return False
    plt.figure(figsize=(5.2, 4.1))
    plt.scatter(sub[x], sub[y], color="#2c7fb8", alpha=0.8)
    if sub[x].nunique(dropna=True) >= 2:
        coef = np.polyfit(sub[x], sub[y], 1)
        xx = np.linspace(sub[x].min(), sub[x].max(), 100)
        plt.plot(xx, coef[0] * xx + coef[1], color="black", linewidth=1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _plot_rank_delta_distribution(df: pd.DataFrame, path: Path) -> bool:
    vals = pd.to_numeric(df.get("rank_delta"), errors="coerce").dropna().abs().sort_values(ascending=False).to_numpy()
    if vals.size == 0:
        return False
    plt.figure(figsize=(5.2, 3.6))
    plt.bar(np.arange(1, vals.size + 1), vals, color="#dd8452")
    plt.xlabel("Groupes tries par |delta rang|")
    plt.ylabel("|delta rang|")
    plt.title("Amplitude des changements de rang")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _plot_parent_loadings(loadings: pd.DataFrame, path: Path) -> bool:
    if loadings.empty or "parent_pc1_loading" not in loadings.columns:
        return False
    df = loadings.copy().sort_values("parent_pc1_loading", ascending=True)
    labels = df["task_sample"].fillna("").astype(str)
    labels = np.where(labels == "", df["task_full_csv"].astype(str), labels + " -> " + df["task_full_csv"].astype(str))
    plt.figure(figsize=(6.8, 4.8))
    plt.barh(labels, df["parent_pc1_loading"], color="#4c72b0")
    plt.xlabel("Loading PC1 parent")
    plt.title("Loadings de la PC1 parentale (Riedl 2021)")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _plot_task_heatmap(task_assoc: pd.DataFrame, path: Path) -> bool:
    if task_assoc.empty:
        return False
    pivot = task_assoc.pivot_table(index="task", columns="target", values="rho", aggfunc="mean")
    wanted_cols = [c for c in ["c_factor_pop", "c_factor_sample", "rme_mean"] if c in pivot.columns]
    if pivot.empty or not wanted_cols:
        return False
    pivot = pivot[wanted_cols].sort_index()
    plt.figure(figsize=(6.4, max(3.6, 0.42 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=15, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Associations Spearman taches / c-factor / RME")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _plot_familiarity_forest(df: pd.DataFrame, path: Path) -> bool:
    if df.empty:
        return False
    keep = df.loc[df["outcome"].isin(["c_factor_pop", "c_factor_sample", "Score_perf_tsk"])].copy()
    keep = keep.dropna(subset=["beta_std"])
    if keep.empty:
        return False
    keep["label"] = keep["predictor"] + " -> " + keep["outcome"]
    keep = keep.sort_values("beta_std")
    plt.figure(figsize=(7.0, max(3.8, 0.42 * len(keep))))
    y = np.arange(len(keep))
    beta = keep["beta_std"].to_numpy(dtype=float)
    low = beta - keep["beta_ci_low"].to_numpy(dtype=float)
    high = keep["beta_ci_high"].to_numpy(dtype=float) - beta
    plt.errorbar(beta, y, xerr=[low, high], fmt="o", color="#2c7fb8", ecolor="#7f7f7f", capsize=3)
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.yticks(y, keep["label"])
    plt.xlabel("Beta standardise (IC bootstrap BCa 95 %)")
    plt.title("Familiarite VR -> issues VR")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return True


def _sem_environment_check() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pkg in ["semopy", "pyplspm", "plspm", "rpy2"]:
        try:
            __import__(pkg)
            status = "OK"
        except Exception as exc:
            status = f"FAIL: {type(exc).__name__}"
        rows.append({"environment": "python", "package": pkg, "status": status})

    rscript = Path(r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe")
    if rscript.exists():
        try:
            cmd = [
                str(rscript),
                "-e",
                "pkgs <- c('lavaan','seminr','plspm'); for (p in pkgs) cat(p, ifelse(requireNamespace(p, quietly=TRUE), ':OK', ':FAIL'), '\\n')",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            for line in lines:
                pkg, status = [part.strip() for part in line.split(":", 1)]
                rows.append({"environment": "R", "package": pkg, "status": status})
        except Exception as exc:
            rows.append({"environment": "R", "package": "Rscript", "status": f"FAIL: {type(exc).__name__}"})
    else:
        rows.append({"environment": "R", "package": "Rscript", "status": "FAIL: introuvable"})

    return pd.DataFrame(rows)


def _add_section_title(elems: list[Any], text: str, styles: dict[str, ParagraphStyle], level: int = 2) -> None:
    key = "Heading1" if level == 1 else "Heading2"
    elems.append(Paragraph(text, styles[key]))
    elems.append(Spacer(1, 0.1 * inch))


def _add_table(elems: list[Any], title: str, df: pd.DataFrame, styles: dict[str, ParagraphStyle], max_rows: int = 40) -> None:
    elems.append(_paragraph(title, styles, "Body"))
    elems.append(Spacer(1, 0.04 * inch))
    elems.append(pdf_table_from_df(df, max_rows=max_rows))
    elems.append(Spacer(1, 0.1 * inch))


def generate_cfactor_population_report(results_dir: Path, output_dir: Path | None = None) -> dict[str, Path]:
    output_dir = output_dir or (results_dir / "rapport_v2" / "cfactor_population")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    log_lines: list[str] = [f"results_dir={results_dir}", f"output_dir={output_dir}", f"seed={SEED}"]

    tci_allowed, tci_allowed_path = _load_tci_population(results_dir, "allowed")
    tci_all, tci_all_path = _load_tci_population(results_dir, "all")
    tci_allowed_tasks, _ = _load_tci_population_with_tasks(results_dir, "allowed")
    tci_all_tasks, _ = _load_tci_population_with_tasks(results_dir, "all")
    task_ref = _load_task_reference_stats(results_dir)
    parent_loadings = _load_parent_loadings(results_dir)
    pop_log_lines = _load_population_log(results_dir)
    riedl = _load_riedl_summary(results_dir)
    profile = _load_profile(results_dir)
    perf = _load_performance(results_dir)
    q_group = _load_questionnaire_group(results_dir)
    sem_env = _sem_environment_check()

    log_lines.append(f"tci_allowed_path={tci_allowed_path}")
    log_lines.append(f"tci_all_path={tci_all_path}")
    log_lines.append(f"n_allowed={len(tci_allowed)}")
    log_lines.append(f"n_all={len(tci_all)}")

    comp_allowed = _comparison_summary(tci_allowed, "allowed")
    comp_all = _comparison_summary(tci_all, "all")
    comp_summary = pd.concat([comp_allowed, comp_all], ignore_index=True)
    rank_changes_all = tci_all[["group_id", "rank_pop", "rank_sample", "rank_delta"]].copy() if {"group_id", "rank_pop", "rank_sample", "rank_delta"}.issubset(tci_all.columns) else pd.DataFrame()
    if not rank_changes_all.empty:
        rank_changes_all["abs_rank_delta"] = pd.to_numeric(rank_changes_all["rank_delta"], errors="coerce").abs()
        rank_changes_all = rank_changes_all.sort_values("abs_rank_delta", ascending=False).reset_index(drop=True)

    rme_corr_allowed = _rme_correlations(tci_allowed, "allowed")
    rme_corr_all = _rme_correlations(tci_all, "all")
    rme_corr = pd.concat([rme_corr_allowed, rme_corr_all], ignore_index=True)

    task_assoc_allowed = _task_associations(tci_allowed_tasks, "allowed")
    task_assoc_all = _task_associations(tci_all_tasks, "all")
    task_assoc = pd.concat([task_assoc_allowed, task_assoc_all], ignore_index=True)

    riedl_desc = _riedl_descriptives(riedl)
    riedl_merged = tci_all[["group_id", "c_factor_pop", "c_factor_sample"]].merge(
        riedl, on="group_id", how="inner"
    ) if not riedl.empty else pd.DataFrame()
    riedl_corr = _riedl_cfactor_correlations(riedl_merged)

    fam_group = _build_group_familiarity(profile)
    fam_results = _familiarity_analysis(fam_group, tci_allowed, perf, q_group)

    comp_summary.to_csv(tables_dir / "cfactor_population_comparison_summary.csv", index=False, encoding="utf-8")
    rank_changes_all.to_csv(tables_dir / "cfactor_population_rank_changes_all.csv", index=False, encoding="utf-8")
    rme_corr.to_csv(tables_dir / "cfactor_population_rme_correlations.csv", index=False, encoding="utf-8")
    task_assoc.to_csv(tables_dir / "cfactor_population_task_associations.csv", index=False, encoding="utf-8")
    riedl_desc.to_csv(tables_dir / "riedl_population_descriptives.csv", index=False, encoding="utf-8")
    riedl_corr.to_csv(tables_dir / "riedl_population_cfactor_correlations.csv", index=False, encoding="utf-8")
    fam_results.to_csv(tables_dir / "familiarity_vr_results.csv", index=False, encoding="utf-8")
    sem_env.to_csv(tables_dir / "sem_environment_check.csv", index=False, encoding="utf-8")

    scatter_pop_sample = figures_dir / "cfactor_pop_vs_sample_all.png"
    rank_delta_fig = figures_dir / "cfactor_rank_delta_distribution.png"
    loadings_fig = figures_dir / "cfactor_parent_loadings.png"
    rme_fig = figures_dir / "cfactor_pop_vs_rme_mean_all.png"
    heatmap_fig = figures_dir / "cfactor_task_heatmap_all.png"
    fam_fig = figures_dir / "familiarity_vr_forest.png"

    _plot_scatter(tci_all, "c_factor_pop", "c_factor_sample", "c-factor population parente vs echantillon", scatter_pop_sample)
    _plot_rank_delta_distribution(tci_all, rank_delta_fig)
    _plot_parent_loadings(parent_loadings, loadings_fig)
    _plot_scatter(tci_all, "c_factor_pop", "rme_mean", "c-factor population parente vs RME moyen", rme_fig)
    _plot_task_heatmap(task_assoc_all, heatmap_fig)
    _plot_familiarity_forest(fam_results, fam_fig)

    log_lines.extend(pop_log_lines)
    log_lines.append(f"sem_environment_rows={len(sem_env)}")
    log_path = output_dir / "cfactor_population_report_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    pdf_path = output_dir / f"{REPORT_NAME}.pdf"
    elems: list[Any] = []
    elems.append(Paragraph("Rapport autonome — C-factor population parente", styles["Title"]))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(
        _sample_box(
            (
                f"Sources TCI :<br/>allowed = {tci_allowed_path}<br/>all = {tci_all_path}<br/>"
                f"Riedl reference = {results_dir / 'TCI' / 'task_reference_stats.csv'}<br/>"
                f"Effectifs observes : allowed = {len(tci_allowed)} groupes ; all = {len(tci_all)} groupes."
            ),
            styles,
        )
    )
    elems.append(
        _paragraph(
            "Ce rapport documente la variante de c-factor projete sur la PC1 apprise dans la population parente de Riedl (2021). "
            "Le score local n'est donc plus obtenu en refittant une PCA sur l'echantillon courant : les groupes sont projetes sur des "
            "loadings parentaux, avec imputation au niveau moyen parent pour les taches sans observation locale equivalente.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.08 * inch))

    _add_section_title(elems, "1. Methode et mapping des taches", styles)
    elems.append(
        _paragraph(
            "Le recalcul s'appuie sur `full.csv` de Riedl 2021. Les taches locales ont ete mappees explicitement vers les taches parentes quand un equivalent strict etait defendable ; "
            "les taches sans equivalent ont ete conservees comme sorties descriptives mais n'entrent pas dans la projection parentale. "
            "L'orientation de signe de la PC1 parentale a ete fixee une seule fois sur le jeu `all` pour garantir la coherence entre scores, loadings exportes et comparaisons inter-jeux.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.06 * inch))
    if not task_ref.empty:
        wanted_ref = [
            c
            for c in [
                "scope",
                "task_sample",
                "task_full_csv",
                "reference_pop",
                "flag",
                "used_for_parent_pca",
                "parent_pc1_loading",
            ]
            if c in task_ref.columns
        ]
        _add_table(elems, "Tableau de mapping et de reference par tache", task_ref[wanted_ref], styles, max_rows=20)
    if not parent_loadings.empty:
        wanted_load = [
            c
            for c in [
                "task_sample",
                "task_full_csv",
                "parent_pc1_weight",
                "parent_pc1_loading",
                "mean_pop",
                "sd_pop",
                "n_pop",
            ]
            if c in parent_loadings.columns
        ]
        _add_table(elems, "Loadings parentaux exportes", parent_loadings[wanted_load], styles, max_rows=12)
        if loadings_fig.exists():
            elems.append(Image(str(loadings_fig), width=6.2 * inch, height=4.3 * inch))
            elems.append(Spacer(1, 0.1 * inch))

    _add_section_title(elems, "2. Comparaison c-factor population parente vs echantillon", styles)
    _add_table(elems, "Resume des correlations et changements de rang", comp_summary, styles, max_rows=10)
    elems.append(
        _paragraph(
            "Cette section compare directement `c_factor_pop` et `c_factor_sample`. Le point critique a surveiller est la stabilite du rang des groupes, "
            "car une faible correlation entre les deux scores indique que la normalisation parentale et l'utilisation de loadings externes modifient substantiellement la structure du construit local.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.06 * inch))
    if scatter_pop_sample.exists():
        elems.append(Image(str(scatter_pop_sample), width=5.8 * inch, height=4.4 * inch))
        elems.append(Spacer(1, 0.08 * inch))
    if rank_delta_fig.exists():
        elems.append(Image(str(rank_delta_fig), width=5.8 * inch, height=3.9 * inch))
        elems.append(Spacer(1, 0.08 * inch))

    _add_section_title(elems, "3. Relations avec le RME et les taches TCI", styles)
    if not rme_corr.empty:
        _add_table(elems, "Correlations Spearman c-factor vs RME", rme_corr, styles, max_rows=12)
    if rme_fig.exists():
        elems.append(Image(str(rme_fig), width=5.8 * inch, height=4.3 * inch))
        elems.append(Spacer(1, 0.08 * inch))
    if not task_assoc_all.empty:
        task_assoc_pdf = task_assoc_all.loc[task_assoc_all["target"].isin(["c_factor_pop", "c_factor_sample", "rme_mean"])].copy()
        task_assoc_pdf = task_assoc_pdf.sort_values(["target", "p", "rho"], ascending=[True, True, False])
        _add_table(elems, "Associations taches / c-factor / RME (jeu all)", task_assoc_pdf, styles, max_rows=24)
    if heatmap_fig.exists():
        elems.append(Image(str(heatmap_fig), width=6.0 * inch, height=4.6 * inch))
        elems.append(Spacer(1, 0.08 * inch))

    _add_section_title(elems, "4. Indicateurs Riedl recalcules", styles)
    elems.append(
        _paragraph(
            "Les indicateurs Riedl ont ete recalcules avec le traitement explicite des taches `typing_oriented` : "
            "`skill_i` reste non definissable sur ces taches, `skill_congruence_task` y est force a 1, et `skill_congruence_mean_core` exclut ces taches de l'agregation inter-taches. "
            "Les variantes `effort_norm_pop` et `strategy_norm_pop` sont exportees lorsque les colonnes de reference parentale sont exploitables ; sinon la source de normalisation reste l'echantillon.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.06 * inch))
    if not riedl_desc.empty:
        _add_table(elems, "Descriptifs Riedl", riedl_desc, styles, max_rows=20)
    if not riedl_corr.empty:
        _add_table(elems, "Correlations indicateurs Riedl vs c-factor", riedl_corr, styles, max_rows=20)

    elems.append(PageBreak())
    _add_section_title(elems, "5. Familiarite VR", styles)
    vr_n = int(fam_group.loc[fam_group["modalite"].astype(str).str.upper().eq("VR"), "group_id"].nunique()) if not fam_group.empty and "modalite" in fam_group.columns else 0
    elems.append(
        _sample_box(
            f"Echantillon VR disponible pour la familiarite : N = {vr_n} groupes. "
            "Les tests restent exploratoires et doivent etre interpretes avec prudence lorsque n < 8.",
            styles,
        )
    )
    elems.append(
        _paragraph(
            "Toutes les variables de familiarite disponibles dans le questionnaire profil ont ete agregees au niveau groupe, puis testees contre `c_factor_pop`, `c_factor_sample`, `Score_perf_tsk` et les dimensions questionnaire disponibles en VR. "
            "Les correlations sont de type Spearman avec IC bootstrap BCa 95 %, et les p-values sont corrigees par FDR au sein de chaque variable dependante.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.06 * inch))
    if not fam_results.empty:
        fam_pdf = fam_results[
            [
                c
                for c in [
                    "predictor",
                    "outcome",
                    "n",
                    "rho_spearman",
                    "p_spearman",
                    "p_fdr",
                    "beta_std",
                    "r2",
                    "sig",
                ]
                if c in fam_results.columns
            ]
        ]
        _add_table(elems, "Resultats synthese familiarite VR", fam_pdf, styles, max_rows=40)
        if fam_fig.exists():
            elems.append(Image(str(fam_fig), width=6.3 * inch, height=4.6 * inch))
            elems.append(Spacer(1, 0.08 * inch))
    else:
        elems.append(_paragraph("Aucune analyse de familiarite VR n'a pu etre calculee a partir des fichiers disponibles.", styles))
        elems.append(Spacer(1, 0.08 * inch))

    _add_section_title(elems, "6. Audit d'environnement SEM", styles)
    elems.append(
        _paragraph(
            "La presente passe se concentre sur le recalcul du c-factor population parente et sur ses analyses descriptives / correlatives. "
            "L'audit ci-dessous documente simplement l'etat de l'environnement pour les futurs blocs PLS-SEM et multilevel SEM ; aucune sortie SEM n'est interpretee ici.",
            styles,
        )
    )
    elems.append(Spacer(1, 0.06 * inch))
    if not sem_env.empty:
        _add_table(elems, "Etat des dependances SEM", sem_env, styles, max_rows=20)

    _save_pdf(pdf_path, elems)

    return {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "log_path": log_path,
    }
