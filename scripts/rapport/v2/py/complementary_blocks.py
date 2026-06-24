# -*- coding: utf-8 -*-
"""
Analyses complémentaires dédiées aux blocs 1-4 demandés en plus du rapport v2.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from scipy.stats import mannwhitneyu, shapiro, spearmanr

from pathlib import Path as _Path

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))
_scripts_dir = _Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from py.corr import bh_fdr, pdf_table_from_df  # noqa: E402
from py.data import harmonize_timepoint, normalize_group, normalize_timepoint, read_csv_auto  # noqa: E402
from py.regression import compute_cronbach_alpha  # noqa: E402

try:
    import statsmodels.api as sm  # noqa: E402
    from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: E402
    from statsmodels.tools.sm_exceptions import ConvergenceWarning  # noqa: E402

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    sm = None
    variance_inflation_factor = None
    ConvergenceWarning = Warning


RIEDL_SOURCE_MAP: dict[str, str] = {
    "skill_mean": "skill_mean",
    "effort_mean": "effort_task_mean",
    "strategy_ratio_mean": "strategy_ratio_mean",
    "skill_congruence_mean": "skill_congruence_mean",
    "strategy_norm": "strategy_norm",
}

QUESTIONNAIRE_DIM_MAP: dict[str, str] = {
    "COR": "tms_coordination",
    "CRE": "tms_credibilite",
    "SPE": "tms_specialisation",
    "SOC": "cohesion_sociale",
    "TSK": "cohesion_tache",
    "COM": "cohesion_communication",
}

QUESTIONNAIRE_DIM_ORDER = [
    "tms_credibilite",
    "tms_specialisation",
    "tms_coordination",
    "cohesion_sociale",
    "cohesion_tache",
    "cohesion_communication",
    "cohesion_globale",
]

TMS_DIMENSIONS = ["tms_credibilite", "tms_specialisation", "tms_coordination"]
COHESION_DIMENSIONS = ["cohesion_sociale", "cohesion_tache", "cohesion_communication"]


def _first_nonempty(values: pd.Series) -> str | float | None:
    for val in values:
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if sval and sval.lower() not in {"nan", "none"}:
            return val
    return np.nan


def _infer_timepoint_from_group_id(group_id: Any) -> str | None:
    if pd.isna(group_id):
        return None
    s = str(group_id).strip().lower()
    if not s:
        return None
    return "T2" if s.endswith("_2") else "T1"


def _normalize_modality(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if "VR" in s:
        return "VR"
    if "PC" in s:
        return "PC"
    return s


def _normalize_scenario(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if "S1" in s:
        return "S1"
    if "S2" in s:
        return "S2"
    return s


def _read_and_normalize(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    if df is None or df.empty:
        return pd.DataFrame()
    return normalize_group(df)


def _resolve_tci_file(results_dir: Path, scope: str, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    scope = str(scope).strip().lower()
    candidates = (
        [results_dir / "TCI" / "c_scores_all.csv", results_dir / "TCI" / "c_scores.csv"]
        if scope == "all"
        else [results_dir / "TCI" / "c_scores_allowed.csv", results_dir / "TCI" / "c_scores.csv"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Fichier TCI introuvable pour scope='{scope}' dans {results_dir / 'TCI'}")


def _load_tci_scores(results_dir: Path, scope: str, explicit_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    path = _resolve_tci_file(results_dir, scope, explicit_path)
    df = _read_and_normalize(path)
    if df.empty or "c_score" not in df.columns:
        return pd.DataFrame(), path

    keep_cols = [c for c in ["group_id", "c_score", "rme_mean", "rme_max", "rme_min"] if c in df.columns]
    out = df[keep_cols].copy()
    out["c_factor"] = pd.to_numeric(out["c_score"], errors="coerce")
    for col in ["rme_mean", "rme_max", "rme_min"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=["group_id"]).reset_index(drop=True), path


def _load_riedl_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "indices_collab" / "riedl_group_summary.csv"
    df = _read_and_normalize(path)
    if df.empty:
        return pd.DataFrame()

    for source_name, source_col in RIEDL_SOURCE_MAP.items():
        if source_col in df.columns:
            df[source_name] = pd.to_numeric(df[source_col], errors="coerce")

    keep_cols = ["group_id"] + [c for c in RIEDL_SOURCE_MAP.keys() if c in df.columns]
    return df[keep_cols].drop_duplicates(subset=["group_id"]).reset_index(drop=True)


def _load_performance(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "performance_task" / "recap_scores_all.csv"
    df = _read_and_normalize(path)
    if df.empty:
        return pd.DataFrame()

    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].apply(_normalize_modality)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].apply(_normalize_scenario)
    df["timepoint"] = df["group_id"].apply(_infer_timepoint_from_group_id)
    if "Score_perf_tsk" in df.columns:
        df["Score_perf_tsk"] = pd.to_numeric(df["Score_perf_tsk"], errors="coerce")
    keep_cols = [c for c in ["group_id", "timepoint", "modalite", "scenario", "Score_perf_tsk"] if c in df.columns]
    return df[keep_cols].drop_duplicates(subset=["group_id"]).reset_index(drop=True)


def _load_profile(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "questionnaire" / "global" / "participant_profile_responses.csv"
    df = _read_and_normalize(path)
    if df.empty:
        return pd.DataFrame()

    for col in ["timepoint", "modalite", "scenario"]:
        if col in df.columns:
            if col == "modalite":
                df[col] = df[col].apply(_normalize_modality)
            elif col == "scenario":
                df[col] = df[col].apply(_normalize_scenario)
            else:
                df[col] = df[col].astype(str).str.upper().str.strip()
    if "vr_familiarity_score" in df.columns:
        df["vr_familiarity_score"] = pd.to_numeric(df["vr_familiarity_score"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "Participant" in df.columns and "participant" not in df.columns:
        df["participant"] = df["Participant"]
    return df.reset_index(drop=True)


def _load_questionnaire_scores_long(results_dir: Path, use_pruned: bool = True) -> pd.DataFrame:
    p_pruned = results_dir / "questionnaire" / "global" / "pruned" / "scores_dimension_par_participant_pruned.csv"
    p_orig = results_dir / "questionnaire" / "analyse" / "scores_dimension_par_participant.csv"
    path = p_pruned if use_pruned and p_pruned.exists() else p_orig
    df = _read_and_normalize(path)
    if df.empty:
        return pd.DataFrame()

    if "Participant" in df.columns and "participant" not in df.columns:
        df["participant"] = df["Participant"]
    if "Session" in df.columns and "session" not in df.columns:
        df["session"] = df["Session"]
    if "Scenario" in df.columns and "scenario" not in df.columns:
        df["scenario"] = df["Scenario"]
    if "Modalite" in df.columns and "modalite" not in df.columns:
        df["modalite"] = df["Modalite"]
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "n_items" in df.columns:
        df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce")
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].apply(_normalize_modality)
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].apply(_normalize_scenario)

    df = harmonize_timepoint(df, session_col="session", raw_group_candidates=["group_id", "Groupe", "groupe"])
    df = normalize_timepoint(df)
    keep_cols = [c for c in ["group_id", "timepoint", "modalite", "scenario", "participant", "dimension", "score", "n_items"] if c in df.columns]
    return df[keep_cols].dropna(subset=["group_id", "dimension"]).reset_index(drop=True)


def _build_group_metadata(*dfs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep = ["group_id", "timepoint", "modalite", "scenario"]
    for df in dfs:
        if df is None or df.empty or "group_id" not in df.columns:
            continue
        cols = [c for c in keep if c in df.columns]
        if not cols or "group_id" not in cols:
            continue
        frames.append(df[cols].copy())
    if not frames:
        return pd.DataFrame(columns=keep)

    raw = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = raw.groupby("group_id", dropna=False, as_index=False).agg(
        timepoint=("timepoint", _first_nonempty),
        modalite=("modalite", _first_nonempty),
        scenario=("scenario", _first_nonempty),
    )
    out["timepoint"] = out["timepoint"].where(out["timepoint"].notna(), out["group_id"].apply(_infer_timepoint_from_group_id))
    return out.reset_index(drop=True)


def _build_questionnaire_participant_wide(q_long: pd.DataFrame) -> pd.DataFrame:
    if q_long is None or q_long.empty:
        return pd.DataFrame()

    idx_cols = [c for c in ["group_id", "participant", "timepoint", "modalite", "scenario"] if c in q_long.columns]
    wide = q_long.pivot_table(index=idx_cols, columns="dimension", values="score", aggfunc="mean").reset_index()
    wide.columns.name = None
    rename_dims = {src: dst for src, dst in QUESTIONNAIRE_DIM_MAP.items() if src in wide.columns}
    wide = wide.rename(columns=rename_dims)
    return wide.reset_index(drop=True)


def _build_questionnaire_group_wide(q_participant_wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if q_participant_wide is None or q_participant_wide.empty:
        return pd.DataFrame()

    dim_cols = [c for c in QUESTIONNAIRE_DIM_MAP.values() if c in q_participant_wide.columns]
    if not dim_cols:
        return pd.DataFrame()

    out = q_participant_wide.groupby("group_id", dropna=False)[dim_cols].mean().reset_index()
    if set(COHESION_DIMENSIONS).issubset(out.columns):
        out["cohesion_globale"] = out[COHESION_DIMENSIONS].mean(axis=1)
    out = out.merge(meta, on="group_id", how="left")
    return out.reset_index(drop=True)


def _safe_shapiro(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if len(vals) < 3 or vals.nunique(dropna=True) < 3:
        return np.nan
    try:
        return float(shapiro(vals).pvalue)
    except Exception:
        return np.nan


def _safe_spearman(df: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    if x not in df.columns or y not in df.columns:
        return {"rho": np.nan, "p": np.nan, "n": 0}
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        return {"rho": np.nan, "p": np.nan, "n": int(len(sub))}
    rho, pval = spearmanr(sub[x], sub[y])
    return {"rho": float(rho), "p": float(pval), "n": int(len(sub))}


def _extract_named_value(values: Any, exog_names: list[str], name: str) -> float:
    if isinstance(values, pd.Series):
        return float(values.get(name, np.nan))
    if isinstance(values, np.ndarray):
        try:
            idx = exog_names.index(name)
            return float(values[idx])
        except Exception:
            return np.nan
    try:
        return float(values[name])
    except Exception:
        return np.nan


def _fit_ols(df: pd.DataFrame, x: str, y: str, cluster_col: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "beta": np.nan,
        "se": np.nan,
        "p": np.nan,
        "r2": np.nan,
        "r2_adj": np.nan,
        "n": 0,
        "warning": "",
        "shapiro_x_p": np.nan,
        "shapiro_y_p": np.nan,
        "clustered": False,
    }
    if not HAS_STATSMODELS or x not in df.columns or y not in df.columns:
        out["warning"] = "statsmodels indisponible ou variables manquantes"
        return out

    keep_cols = [x, y] + ([cluster_col] if cluster_col and cluster_col in df.columns else [])
    sub = df[keep_cols].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])
    out["n"] = int(len(sub))
    if len(sub) < 3 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        out["warning"] = "données insuffisantes pour OLS"
        return out

    out["shapiro_x_p"] = _safe_shapiro(sub[x])
    out["shapiro_y_p"] = _safe_shapiro(sub[y])

    X = sm.add_constant(sub[[x]], has_constant="add")
    yv = sub[y]
    try:
        base_fit = sm.OLS(yv, X).fit()
    except Exception as exc:
        out["warning"] = f"échec OLS: {exc}"
        return out

    fit = base_fit
    if cluster_col and cluster_col in sub.columns and sub[cluster_col].nunique(dropna=True) >= 2:
        try:
            fit = base_fit.get_robustcov_results(cov_type="cluster", groups=sub[cluster_col])
            out["clustered"] = True
        except Exception:
            fit = base_fit

    exog_names = list(getattr(fit.model, "exog_names", []))
    out["beta"] = _extract_named_value(fit.params, exog_names, x)
    out["se"] = _extract_named_value(fit.bse, exog_names, x)
    out["p"] = _extract_named_value(fit.pvalues, exog_names, x)
    out["r2"] = float(getattr(base_fit, "rsquared", np.nan))
    out["r2_adj"] = float(getattr(base_fit, "rsquared_adj", np.nan))
    if len(sub) < 8:
        out["warning"] = "n faible — résultats exploratoires"
    return out


def _fit_mixed_lmm(df: pd.DataFrame, x: str, y: str, group_col: str) -> dict[str, Any]:
    out = {"beta": np.nan, "se": np.nan, "p": np.nan, "n": 0, "n_groups": 0, "warning": "", "converged": False}
    if not HAS_STATSMODELS:
        out["warning"] = "statsmodels indisponible"
        return out
    if any(col not in df.columns for col in [x, y, group_col]):
        out["warning"] = "variables manquantes pour MixedLM"
        return out

    sub = df[[x, y, group_col]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub[group_col] = sub[group_col].astype(str)
    sub = sub.dropna(subset=[x, y, group_col])
    out["n"] = int(len(sub))
    out["n_groups"] = int(sub[group_col].nunique(dropna=True))
    if out["n_groups"] < 6:
        out["warning"] = "moins de 6 groupes — LMM non estimé"
        return out
    if len(sub) < 6 or sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
        out["warning"] = "données insuffisantes pour MixedLM"
        return out

    exog = sm.add_constant(sub[[x]], has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            fit = sm.MixedLM(sub[y], exog, groups=sub[group_col]).fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        exog_names = list(getattr(fit.model, "exog_names", []))
        out["beta"] = _extract_named_value(fit.params, exog_names, x)
        out["se"] = _extract_named_value(fit.bse, exog_names, x)
        out["p"] = _extract_named_value(fit.pvalues, exog_names, x)
        out["converged"] = bool(getattr(fit, "converged", False))
        if (
            np.isfinite(out["beta"])
            and np.isfinite(out["se"])
            and abs(out["beta"]) < 1e-6
            and abs(out["se"]) < 1e-3
        ):
            out["warning"] = "MixedLM dégénéré / solution singulière"
    except Exception as exc:
        out["warning"] = f"MixedLM non convergent: {exc}"
    return out


def _mannwhitney_effect(df: pd.DataFrame, value_col: str, group_col: str = "modalite", group_a: str = "VR", group_b: str = "PC") -> dict[str, Any]:
    out = {"U": np.nan, "p": np.nan, "r": np.nan, "n_a": 0, "n_b": 0, "direction": "", "warning": "", "median_a": np.nan, "median_b": np.nan}
    if value_col not in df.columns or group_col not in df.columns:
        out["warning"] = "variables manquantes pour Mann-Whitney"
        return out

    sub = df[[value_col, group_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub[group_col] = sub[group_col].astype(str)
    a = sub.loc[sub[group_col] == group_a, value_col].dropna()
    b = sub.loc[sub[group_col] == group_b, value_col].dropna()
    out["n_a"] = int(len(a))
    out["n_b"] = int(len(b))
    out["median_a"] = float(a.median()) if len(a) else np.nan
    out["median_b"] = float(b.median()) if len(b) else np.nan
    if len(a) < 2 or len(b) < 2:
        out["warning"] = "effectifs insuffisants pour Mann-Whitney"
        return out

    try:
        u_stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        mean_u = len(a) * len(b) / 2.0
        sd_u = math.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12.0)
        z_val = 0.0 if sd_u == 0 else (u_stat - mean_u) / sd_u
        r_val = z_val / math.sqrt(len(a) + len(b))
        out["U"] = float(u_stat)
        out["p"] = float(p_val)
        out["r"] = float(r_val)
        out["direction"] = f"{group_a} > {group_b}" if a.median() > b.median() else f"{group_b} > {group_a}"
        if min(len(a), len(b)) < 5:
            out["warning"] = "Interprétation limitée"
        return out
    except Exception as exc:
        out["warning"] = f"échec Mann-Whitney: {exc}"
        return out


def _pairwise_spearman_table(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, x in enumerate(variables):
        for y in variables[i + 1:]:
            stat = _safe_spearman(df, x, y)
            rows.append({"var_x": x, "var_y": y, "rho": stat["rho"], "p": stat["p"], "n": stat["n"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = bh_fdr(out["p"].values)
    out["strength"] = np.where(out["rho"].abs() >= 0.5, "fort", np.where(out["rho"].abs() >= 0.3, "modéré", "faible"))
    out["sig"] = np.where(out["p_fdr"] < 0.01, "**", np.where(out["p_fdr"] < 0.05, "*", ""))
    return out.sort_values(["p_fdr", "p", "rho"], ascending=[True, True, False]).reset_index(drop=True)


def _fit_forward_stepwise_aic(df: pd.DataFrame, y: str, predictors: list[str]) -> dict[str, Any]:
    out = {"selected_predictors": [], "history": pd.DataFrame(), "model_table": pd.DataFrame(), "metrics": {}, "warning": ""}
    if not HAS_STATSMODELS:
        out["warning"] = "statsmodels indisponible"
        return out

    cols = [y] + predictors
    sub = df[cols].copy()
    for col in cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if len(sub) < max(6, len(predictors) + 2):
        out["warning"] = "n trop faible pour une régression multiple stable"
        return out

    remaining = list(predictors)
    selected: list[str] = []
    history: list[dict[str, Any]] = []
    current_fit = sm.OLS(sub[y], sm.add_constant(pd.DataFrame(index=sub.index), has_constant="add")).fit()
    current_aic = float(current_fit.aic)

    while remaining:
        candidates: list[tuple[float, str, Any]] = []
        for pred in remaining:
            cols_try = selected + [pred]
            X_try = sm.add_constant(sub[cols_try], has_constant="add")
            try:
                fit_try = sm.OLS(sub[y], X_try).fit()
            except Exception:
                continue
            candidates.append((float(fit_try.aic), pred, fit_try))
        if not candidates:
            break

        best_aic, best_pred, best_fit = sorted(candidates, key=lambda item: item[0])[0]
        history.append({"step": len(selected) + 1, "candidate": best_pred, "aic": best_aic})
        if best_aic + 1e-9 < current_aic:
            selected.append(best_pred)
            remaining.remove(best_pred)
            current_aic = best_aic
            current_fit = best_fit
        else:
            break

    out["selected_predictors"] = selected
    out["history"] = pd.DataFrame(history)
    if not selected:
        out["warning"] = "Aucun prédicteur n'améliore l'AIC par rapport au modèle nul"
        out["metrics"] = {"n": int(len(sub)), "r2": 0.0, "r2_adj": 0.0, "aic": float(current_fit.aic)}
        return out

    X_final = sm.add_constant(sub[selected], has_constant="add")
    fit = sm.OLS(sub[y], X_final).fit()
    vif_rows: list[dict[str, Any]] = []
    if len(selected) == 1:
        vif_rows = [{"predictor": selected[0], "vif": 1.0}]
    elif variance_inflation_factor is not None:
        for idx, pred in enumerate(selected, start=1):
            try:
                vif_val = float(variance_inflation_factor(X_final.values, idx))
            except Exception:
                vif_val = np.nan
            vif_rows.append({"predictor": pred, "vif": vif_val})
    vif_df = pd.DataFrame(vif_rows)

    model_rows: list[dict[str, Any]] = []
    for pred in selected:
        vif_val = np.nan
        if not vif_df.empty and pred in set(vif_df["predictor"]):
            vif_val = float(vif_df.loc[vif_df["predictor"] == pred, "vif"].iloc[0])
        model_rows.append({"predictor": pred, "beta": float(fit.params.get(pred, np.nan)), "se": float(fit.bse.get(pred, np.nan)), "p": float(fit.pvalues.get(pred, np.nan)), "vif": vif_val})

    out["model_table"] = pd.DataFrame(model_rows)
    out["metrics"] = {"n": int(len(sub)), "r2": float(fit.rsquared), "r2_adj": float(fit.rsquared_adj), "aic": float(fit.aic)}
    if not vif_df.empty and vif_df["vif"].gt(5).any():
        out["warning"] = "multicolinéarité détectée (VIF > 5)"
    elif len(sub) < 8:
        out["warning"] = "n faible — résultats exploratoires"
    return out


def _save_scatter_with_fit(df: pd.DataFrame, x: str, y: str, path: Path, title: str, annotate_col: str = "group_id") -> None:
    sub = df[[c for c in [x, y, annotate_col] if c in df.columns]].copy()
    sub[x] = pd.to_numeric(sub[x], errors="coerce")
    sub[y] = pd.to_numeric(sub[y], errors="coerce")
    sub = sub.dropna(subset=[x, y])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(sub[x], sub[y], color="#1f77b4", s=50, alpha=0.9)
    if len(sub) >= 2 and sub[x].nunique() >= 2:
        coef = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(float(sub[x].min()), float(sub[x].max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#d62728", linewidth=2)
    if annotate_col in sub.columns:
        for _, row in sub.iterrows():
            ax.annotate(str(row[annotate_col]), (row[x], row[y]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    data = matrix.copy()
    fig_w = max(5.4, 0.95 * max(1, data.shape[1]) + 2.0)
    fig_h = max(4.2, 0.55 * max(1, data.shape[0]) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    arr = data.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(list(data.columns), rotation=35, ha="right")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(list(data.index))
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _with_rounded_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) and not pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].round(digits)
    return out


def _sample_box(styles, text: str) -> Table:
    tbl = Table([[Paragraph(text, styles["Normal"])]], colWidths=[17.5 * cm])
    tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke), ("BOX", (0, 0), (-1, -1), 0.6, colors.grey), ("INNERPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return tbl


def _paragraph(text: str, styles, style_name: str = "Normal") -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def _build_styles():
    styles = getSampleStyleSheet()
    if "Small" not in styles:
        styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=9, leading=11))
    return styles


def _add_table_section(pdf_elems: list, styles, title: str, df: pd.DataFrame) -> None:
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.06 * inch))
    pdf_elems.append(pdf_table_from_df(_with_rounded_numeric(df), max_rows=max(len(df), 1)))
    pdf_elems.append(Spacer(1, 0.14 * inch))


def _save_pdf(path: Path, elements: list) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    doc.build(elements)


def _build_block1(profile: pd.DataFrame, perf: pd.DataFrame, analysis_groups: set[str], figures_dir: Path, tables_dir: Path) -> dict[str, Any]:
    merged = profile.merge(perf[[c for c in ["group_id", "Score_perf_tsk", "modalite", "scenario", "timepoint"] if c in perf.columns]], on="group_id", how="inner", suffixes=("", "_perf"))
    merged = merged[merged["group_id"].isin(analysis_groups)].copy()
    merged["modalite"] = merged["modalite"].apply(_normalize_modality)
    merged["vr_familiarity_score"] = pd.to_numeric(merged["vr_familiarity_score"], errors="coerce")
    merged["Score_perf_tsk"] = pd.to_numeric(merged["Score_perf_tsk"], errors="coerce")
    merged = merged.dropna(subset=["vr_familiarity_score", "Score_perf_tsk", "group_id"]).reset_index(drop=True)

    group_df = merged.groupby("group_id", dropna=False).agg(vr_familiarity_mean=("vr_familiarity_score", "mean"), Score_perf_tsk=("Score_perf_tsk", "first"), modalite=("modalite", "first"), scenario=("scenario", "first"), timepoint=("timepoint", "first")).reset_index()

    indiv_spear = _safe_spearman(merged, "vr_familiarity_score", "Score_perf_tsk")
    indiv_ols = _fit_ols(merged, "vr_familiarity_score", "Score_perf_tsk", cluster_col="group_id")
    group_spear = _safe_spearman(group_df, "vr_familiarity_mean", "Score_perf_tsk")
    group_ols = _fit_ols(group_df, "vr_familiarity_mean", "Score_perf_tsk")
    mixed = _fit_mixed_lmm(merged, "vr_familiarity_score", "Score_perf_tsk", "group_id")

    summary = pd.DataFrame([
        {"niveau": "individu (OLS cluster-robust)", "rho_spearman": indiv_spear["rho"], "beta": indiv_ols["beta"], "se": indiv_ols["se"], "p_value": indiv_ols["p"], "r2": indiv_ols["r2"], "n": indiv_ols["n"], "warning": indiv_ols["warning"]},
        {"niveau": "groupe (Spearman + OLS)", "rho_spearman": group_spear["rho"], "beta": group_ols["beta"], "se": group_ols["se"], "p_value": group_ols["p"], "r2": group_ols["r2"], "n": group_ols["n"], "warning": group_ols["warning"]},
        {"niveau": "multiniveau (MixedLM)", "rho_spearman": np.nan, "beta": mixed["beta"], "se": mixed["se"], "p_value": mixed["p"], "r2": np.nan, "n": mixed["n"], "warning": mixed["warning"]},
    ])

    summary_path = tables_dir / "bloc1_vr_familiarity_regression.csv"
    _with_rounded_numeric(summary).to_csv(summary_path, index=False)
    scatter_path = figures_dir / "bloc1_scatter_vrfam_perf.png"
    _save_scatter_with_fit(group_df, "vr_familiarity_mean", "Score_perf_tsk", scatter_path, "VR familiarity_mean vs Score_perf_tsk (toutes modalités)")

    return {
        "individual_df": merged,
        "group_df": group_df,
        "summary": summary,
        "table_path": summary_path,
        "scatter_path": scatter_path,
        "n_groups": int(group_df["group_id"].nunique()),
        "n_individuals": int(len(merged)),
        "n_vr": int(group_df["modalite"].eq("VR").sum()),
        "n_pc": int(group_df["modalite"].eq("PC").sum()),
    }


def _build_block2(tci_all: pd.DataFrame, riedl: pd.DataFrame, meta: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> dict[str, Any]:
    full = tci_all.merge(riedl, on="group_id", how="inner").merge(meta, on="group_id", how="left")
    analysis_vars = ["c_factor"] + [k for k in RIEDL_SOURCE_MAP.keys() if k in full.columns]

    desc_rows: list[dict[str, Any]] = []
    for var in analysis_vars:
        vals = pd.to_numeric(full[var], errors="coerce").dropna()
        desc_rows.append({"variable": var, "mean": vals.mean(), "sd": vals.std(ddof=1), "min": vals.min(), "max": vals.max(), "n": len(vals)})
    descriptives = pd.DataFrame(desc_rows)
    descriptives.to_csv(tables_dir / "bloc2_tci_riedl_descriptives.csv", index=False)

    corr_table = _pairwise_spearman_table(full, analysis_vars)
    corr_table.to_csv(tables_dir / "bloc2_tci_riedl_correlations.csv", index=False)

    heatmap_matrix = full[analysis_vars].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    heatmap_path = figures_dir / "bloc2_heatmap_tci_riedl.png"
    _save_heatmap(heatmap_matrix, heatmap_path, f"Bloc 2 - Spearman TCI/Riedl (N={len(full)})")

    stepwise = _fit_forward_stepwise_aic(full, "c_factor", ["skill_mean", "effort_mean", "strategy_norm", "skill_congruence_mean"])
    if not stepwise["model_table"].empty:
        stepwise["model_table"].to_csv(tables_dir / "bloc2_tci_stepwise_final.csv", index=False)
    if not stepwise["history"].empty:
        stepwise["history"].to_csv(tables_dir / "bloc2_tci_stepwise_history.csv", index=False)

    modality_known = full.dropna(subset=["modalite"]).copy()
    pc_vr_rows: list[dict[str, Any]] = []
    for var in [k for k in RIEDL_SOURCE_MAP.keys() if k in modality_known.columns]:
        test = _mannwhitney_effect(modality_known, var, group_col="modalite", group_a="VR", group_b="PC")
        pc_vr_rows.append({"variable": var, "n_vr": test["n_a"], "n_pc": test["n_b"], "U": test["U"], "p": test["p"], "r_rank": test["r"], "direction": test["direction"], "warning": test["warning"]})
    pc_vr = pd.DataFrame(pc_vr_rows)
    if not pc_vr.empty:
        pc_vr.to_csv(tables_dir / "bloc2_riedl_pc_vs_vr.csv", index=False)

    return {"full_df": full, "descriptives": descriptives, "corr_table": corr_table, "heatmap_path": heatmap_path, "stepwise": stepwise, "pc_vr": pc_vr, "n_groups": int(full["group_id"].nunique()), "n_modalite_known": int(modality_known["group_id"].nunique())}


def _build_block3(perf_pool: pd.DataFrame, figures_dir: Path, tables_dir: Path, block2: dict[str, Any]) -> dict[str, Any]:
    data = perf_pool.copy()
    predictors = [k for k in RIEDL_SOURCE_MAP.keys() if k in data.columns]

    corr_rows: list[dict[str, Any]] = []
    for pred in predictors:
        stat = _safe_spearman(data, pred, "Score_perf_tsk")
        corr_rows.append({"predictor": pred, "target": "Score_perf_tsk", "rho": stat["rho"], "p": stat["p"], "n": stat["n"]})
    if "c_factor" in data.columns:
        cf_stat = _safe_spearman(data, "c_factor", "Score_perf_tsk")
        corr_rows.append({"predictor": "c_factor", "target": "Score_perf_tsk", "rho": cf_stat["rho"], "p": cf_stat["p"], "n": cf_stat["n"]})

    corr_table = pd.DataFrame(corr_rows)
    if not corr_table.empty:
        corr_table["p_fdr"] = bh_fdr(corr_table["p"].values)
        corr_table["sig"] = np.where(corr_table["p_fdr"] < 0.01, "**", np.where(corr_table["p_fdr"] < 0.05, "*", ""))
    corr_table.to_csv(tables_dir / "bloc3_perf_riedl_correlations.csv", index=False)

    ols_rows: list[dict[str, Any]] = []
    for pred in predictors:
        stat = _safe_spearman(data, pred, "Score_perf_tsk")
        ols = _fit_ols(data, pred, "Score_perf_tsk")
        ols_rows.append({"predictor": pred, "rho_spearman": stat["rho"], "p_spearman": stat["p"], "beta": ols["beta"], "se": ols["se"], "p_ols": ols["p"], "r2": ols["r2"], "n": ols["n"], "warning": ols["warning"]})
    ols_table = pd.DataFrame(ols_rows)
    ols_table.to_csv(tables_dir / "bloc3_perf_riedl_ols.csv", index=False)

    heatmap_index = [row["predictor"] for row in corr_rows]
    heatmap_vals = [row["rho"] for row in corr_rows]
    heatmap_matrix = pd.DataFrame({"Score_perf_tsk": heatmap_vals}, index=heatmap_index)
    heatmap_path = figures_dir / "bloc3_heatmap_perf_riedl.png"
    _save_heatmap(heatmap_matrix, heatmap_path, f"Bloc 3 - Spearman vers Score_perf_tsk (N={len(data)})")

    mw_perf = _mannwhitney_effect(data, "Score_perf_tsk", group_col="modalite", group_a="VR", group_b="PC")
    mw_df = pd.DataFrame([{"variable": "Score_perf_tsk", "n_vr": mw_perf["n_a"], "n_pc": mw_perf["n_b"], "U": mw_perf["U"], "p": mw_perf["p"], "r_rank": mw_perf["r"], "direction": mw_perf["direction"], "warning": mw_perf["warning"]}])
    mw_df.to_csv(tables_dir / "bloc3_perf_pc_vs_vr.csv", index=False)

    block2_cfactor = block2["corr_table"].copy()
    block2_cfactor = block2_cfactor.loc[((block2_cfactor["var_x"] == "c_factor") & (block2_cfactor["var_y"].isin(predictors))) | ((block2_cfactor["var_y"] == "c_factor") & (block2_cfactor["var_x"].isin(predictors)))].copy()
    if not block2_cfactor.empty:
        block2_cfactor["predictor"] = np.where(block2_cfactor["var_x"] == "c_factor", block2_cfactor["var_y"], block2_cfactor["var_x"])
        cmp = block2_cfactor[["predictor", "rho"]].rename(columns={"rho": "rho_cfactor"}).merge(corr_table.loc[corr_table["predictor"].isin(predictors), ["predictor", "rho"]].rename(columns={"rho": "rho_perf"}), on="predictor", how="inner")
        cmp["same_direction"] = np.sign(cmp["rho_cfactor"]) == np.sign(cmp["rho_perf"])
        same = int(cmp["same_direction"].sum())
        total = int(len(cmp))
        comparison_note = f"{same}/{total} prédicteur(s) Riedl montrent une direction cohérente entre c_factor et Score_perf_tsk." if total else "Comparaison bloc 2 / bloc 3 indisponible."
    else:
        comparison_note = "Comparaison bloc 2 / bloc 3 indisponible."

    return {"data": data, "corr_table": corr_table, "ols_table": ols_table, "heatmap_path": heatmap_path, "mw_df": mw_df, "comparison_note": comparison_note, "n_groups": int(data["group_id"].nunique()), "n_vr": int((data["modalite"] == "VR").sum()), "n_pc": int((data["modalite"] == "PC").sum())}


def _build_block4(q_group_all: pd.DataFrame, perf_pool: pd.DataFrame, tci_all: pd.DataFrame, figures_dir: Path, tables_dir: Path, q_participant: pd.DataFrame) -> dict[str, Any]:
    q_group = q_group_all.copy()
    q_perf = q_group.loc[q_group["group_id"].isin(set(perf_pool["group_id"]))].copy()
    q_perf = q_perf.merge(perf_pool[["group_id", "Score_perf_tsk"]], on="group_id", how="left")
    q_tci = q_group.loc[q_group["group_id"].isin(set(tci_all["group_id"]))].copy()
    q_tci = q_tci.merge(tci_all[["group_id", "c_factor"]], on="group_id", how="left")
    q_pcvr = q_group.dropna(subset=["modalite"]).copy()

    dims_perf = [d for d in QUESTIONNAIRE_DIM_ORDER if d in q_perf.columns]
    dims_tci = [d for d in QUESTIONNAIRE_DIM_ORDER if d in q_tci.columns]
    dims_all = [d for d in QUESTIONNAIRE_DIM_ORDER if d in q_group.columns]

    corr_rows: list[dict[str, Any]] = []
    for dim in dims_perf:
        stat = _safe_spearman(q_perf, dim, "Score_perf_tsk")
        corr_rows.append({"dimension": dim, "target": "Score_perf_tsk", "rho": stat["rho"], "p": stat["p"], "n": stat["n"]})
    for dim in dims_tci:
        stat = _safe_spearman(q_tci, dim, "c_factor")
        corr_rows.append({"dimension": dim, "target": "c_factor", "rho": stat["rho"], "p": stat["p"], "n": stat["n"]})
    corr_table = pd.DataFrame(corr_rows)
    if not corr_table.empty:
        corr_table["p_fdr"] = bh_fdr(corr_table["p"].values)
        corr_table["sig"] = np.where(corr_table["p_fdr"] < 0.01, "**", np.where(corr_table["p_fdr"] < 0.05, "*", ""))

    interdim = _pairwise_spearman_table(q_group, dims_all)
    redundant_pairs = interdim.loc[interdim["rho"].abs() > 0.80, ["var_x", "var_y", "rho", "n"]].reset_index(drop=True)

    pc_vr_rows: list[dict[str, Any]] = []
    for dim in dims_all:
        test = _mannwhitney_effect(q_pcvr, dim, group_col="modalite", group_a="VR", group_b="PC")
        pc_vr_rows.append({"dimension": dim, "n_vr": test["n_a"], "n_pc": test["n_b"], "U": test["U"], "p": test["p"], "r_rank": test["r"], "direction": test["direction"], "warning": test["warning"]})
    pc_vr = pd.DataFrame(pc_vr_rows)

    perf_reg_rows: list[dict[str, Any]] = []
    for dim in dims_perf:
        stat = _safe_spearman(q_perf, dim, "Score_perf_tsk")
        ols = _fit_ols(q_perf, dim, "Score_perf_tsk")
        perf_reg_rows.append({"dimension": dim, "rho_spearman": stat["rho"], "p_spearman": stat["p"], "beta": ols["beta"], "se": ols["se"], "p_ols": ols["p"], "r2": ols["r2"], "n": ols["n"], "warning": ols["warning"]})
    perf_reg = pd.DataFrame(perf_reg_rows)

    heatmap_matrix = pd.DataFrame(index=QUESTIONNAIRE_DIM_ORDER, columns=["Score_perf_tsk", "c_factor"], dtype=float)
    for _, row in corr_table.iterrows():
        if row["dimension"] in heatmap_matrix.index and row["target"] in heatmap_matrix.columns:
            heatmap_matrix.loc[row["dimension"], row["target"]] = row["rho"]
    heatmap_path = figures_dir / "bloc4_heatmap_dimensions.png"
    _save_heatmap(heatmap_matrix, heatmap_path, "Bloc 4 - Questionnaire vs Score_perf_tsk / c_factor")

    tms_alpha = compute_cronbach_alpha(q_participant[TMS_DIMENSIONS].dropna().to_numpy(dtype=float)) if set(TMS_DIMENSIONS).issubset(q_participant.columns) else np.nan
    cohesion_alpha = compute_cronbach_alpha(q_participant[COHESION_DIMENSIONS].dropna().to_numpy(dtype=float)) if set(COHESION_DIMENSIONS).issubset(q_participant.columns) else np.nan
    tms_alpha_vr = compute_cronbach_alpha(q_participant.loc[q_participant["modalite"] == "VR", TMS_DIMENSIONS].dropna().to_numpy(dtype=float)) if set(TMS_DIMENSIONS).issubset(q_participant.columns) else np.nan
    cohesion_alpha_vr = compute_cronbach_alpha(q_participant.loc[q_participant["modalite"] == "VR", COHESION_DIMENSIONS].dropna().to_numpy(dtype=float)) if set(COHESION_DIMENSIONS).issubset(q_participant.columns) else np.nan

    questionnaire_export = corr_table.copy()
    if not perf_reg.empty:
        perf_reg_export = perf_reg.copy()
        perf_reg_export["target"] = "Score_perf_tsk"
        perf_reg_export["analysis"] = "ols_perf"
        questionnaire_export["analysis"] = "spearman"
        questionnaire_export = pd.concat([questionnaire_export, perf_reg_export], ignore_index=True, sort=False)
    questionnaire_export.to_csv(tables_dir / "bloc4_questionnaire_dimensions.csv", index=False)
    interdim.to_csv(tables_dir / "bloc4_questionnaire_interdimensions.csv", index=False)
    redundant_pairs.to_csv(tables_dir / "bloc4_questionnaire_redondances.csv", index=False)
    pc_vr.to_csv(tables_dir / "bloc4_questionnaire_pc_vs_vr.csv", index=False)
    perf_reg.to_csv(tables_dir / "bloc4_questionnaire_perf_ols.csv", index=False)

    note = (
        f"cohesion_globale = agrégat validé (alpha={cohesion_alpha:.3f}, VR={cohesion_alpha_vr:.3f}) ; "
        f"tms_global non calculé (alpha insuffisant : global={tms_alpha:.3f}, VR={tms_alpha_vr:.3f})."
    )
    if np.isfinite(tms_alpha) and tms_alpha < 0.70:
        note += " Warning: fiabilité insuffisante pour agrégation TMS."
    if np.isfinite(cohesion_alpha) and cohesion_alpha < 0.70:
        note += " Warning: fiabilité insuffisante pour l'agrégat de cohésion."

    return {
        "q_group": q_group,
        "q_perf": q_perf,
        "q_tci": q_tci,
        "corr_table": corr_table,
        "interdim": interdim,
        "redundant_pairs": redundant_pairs,
        "pc_vr": pc_vr,
        "perf_reg": perf_reg,
        "heatmap_path": heatmap_path,
        "note": note,
        "alpha_summary": pd.DataFrame([{"construct": "TMS", "alpha_global": tms_alpha, "alpha_vr": tms_alpha_vr, "aggregable": False}, {"construct": "Cohesion", "alpha_global": cohesion_alpha, "alpha_vr": cohesion_alpha_vr, "aggregable": bool(np.isfinite(cohesion_alpha) and cohesion_alpha >= 0.75)}]),
    }


def _render_report_blocks_1_3(out_path: Path, block1: dict[str, Any], block3: dict[str, Any], tci_allowed_path: Path, tci_all_path: Path) -> None:
    styles = _build_styles()
    elems: list = []

    elems.append(Paragraph("Rapport complémentaire — Blocs 1 et 3", styles["Heading1"]))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(_paragraph(f"Source TCI allowed : {tci_allowed_path}<br/>Source TCI all : {tci_all_path}<br/>Les effectifs sont détectés automatiquement à partir des intersections réellement observées.", styles, "Small"))
    elems.append(Spacer(1, 0.12 * inch))

    elems.append(Paragraph("Bloc 1 — Régression : familiarité VR → performance", styles["Heading2"]))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_sample_box(styles, f"Échantillon : N groupes = {block1['n_groups']} | Niveau individu = {block1['n_individuals']} réponses | Modalité(s) : PC + VR | PC = {block1['n_pc']} | VR = {block1['n_vr']} | Variables VD : Score_perf_tsk"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_paragraph("La familiarité VR est ici analysée indépendamment de la modalité expérimentale : tous les groupes disposant à la fois du questionnaire profil et de la performance sont inclus. Le niveau individuel répète le score de performance du groupe pour chacun des membres et estime un modèle OLS avec erreurs robustes clusterisées par groupe lorsque disponible. Le niveau groupe utilise la moyenne intra-groupe de familiarité VR, avec Spearman et OLS simple ; un MixedLM est tenté si le nombre de groupes est suffisant.", styles))
    elems.append(Spacer(1, 0.08 * inch))
    _add_table_section(elems, styles, "Tableau synthétique bloc 1", block1["summary"])
    elems.append(Image(str(block1["scatter_path"]), width=6.6 * inch, height=4.5 * inch))
    elems.append(Spacer(1, 0.14 * inch))

    elems.append(PageBreak())
    elems.append(Paragraph("Bloc 3 — Analyse performance Score_perf_tsk", styles["Heading2"]))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_sample_box(styles, f"Échantillon : N = {block3['n_groups']} | Modalité(s) : PC + VR | Variables VD : Score_perf_tsk | PC = {block3['n_pc']} | VR = {block3['n_vr']}"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_paragraph("Les corrélations Riedl et c_factor → Score_perf_tsk sont calculées sans imputation. Les régressions sont univariées afin de limiter l'instabilité liée au faible effectif. La comparaison PC vs VR sur Score_perf_tsk utilise Mann-Whitney avec taille d'effet r.", styles))
    elems.append(Spacer(1, 0.08 * inch))
    _add_table_section(elems, styles, "Corrélations bloc 3", block3["corr_table"])
    _add_table_section(elems, styles, "Régressions univariées bloc 3", block3["ols_table"])
    _add_table_section(elems, styles, "Comparaison PC vs VR sur Score_perf_tsk", block3["mw_df"])
    elems.append(_paragraph(block3["comparison_note"], styles))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(Image(str(block3["heatmap_path"]), width=5.9 * inch, height=4.6 * inch))

    _save_pdf(out_path, elems)


def _render_report_blocks_2_4(out_path: Path, block2: dict[str, Any], block4: dict[str, Any], tci_allowed_path: Path, tci_all_path: Path) -> None:
    styles = _build_styles()
    elems: list = []

    elems.append(Paragraph("Rapport complémentaire — Blocs 2 et 4", styles["Heading1"]))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(_paragraph(f"Source TCI allowed : {tci_allowed_path}<br/>Source TCI all : {tci_all_path}<br/>Ce rapport consomme explicitement les fichiers TCI fournis afin d'éviter toute ambiguïté de pipeline entre variantes sample et population-parente.", styles, "Small"))
    elems.append(Spacer(1, 0.12 * inch))

    elems.append(Paragraph("Bloc 2 — Analyse TCI-Riedl sur le dataset complet", styles["Heading2"]))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_sample_box(styles, f"Échantillon : N = {block2['n_groups']} | Modalité(s) : toutes modalités confondues | Variables VD : c_factor"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_paragraph("Cette section utilise tous les groupes pour lesquels les sources TCI et Riedl sont présentes dans les fichiers fournis. Les comparaisons PC/VR sur les indicateurs Riedl sont restreintes au sous-échantillon où la modalité est réellement documentée.", styles))
    elems.append(Spacer(1, 0.08 * inch))
    _add_table_section(elems, styles, "Descriptifs bloc 2", block2["descriptives"])
    _add_table_section(elems, styles, "Corrélations Spearman bloc 2", block2["corr_table"])
    if not block2["stepwise"]["model_table"].empty:
        _add_table_section(elems, styles, "Régression stepwise finale bloc 2", block2["stepwise"]["model_table"])
        metrics_df = pd.DataFrame([block2["stepwise"]["metrics"]])
        _add_table_section(elems, styles, "Métriques du modèle stepwise", metrics_df)
    else:
        elems.append(_paragraph(f"Régression stepwise : {block2['stepwise']['warning']}", styles))
        elems.append(Spacer(1, 0.08 * inch))
    if not block2["pc_vr"].empty:
        _add_table_section(elems, styles, f"Comparaison PC vs VR des indicateurs Riedl (N modalité connue = {block2['n_modalite_known']})", block2["pc_vr"])
    elems.append(Image(str(block2["heatmap_path"]), width=6.1 * inch, height=4.9 * inch))

    elems.append(PageBreak())
    elems.append(Paragraph("Bloc 4 — Dimensions de questionnaire", styles["Heading2"]))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_sample_box(styles, f"Échantillon performance : N = {len(block4['q_perf'])} | Échantillon c_factor : N = {len(block4['q_tci'])} | Échantillon questionnaire inter-dimensions : N = {len(block4['q_group'])}"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_paragraph(block4["note"], styles))
    elems.append(Spacer(1, 0.08 * inch))
    _add_table_section(elems, styles, "Alpha de construit", block4["alpha_summary"])
    _add_table_section(elems, styles, "Corrélations dimensions → Score_perf_tsk / c_factor", block4["corr_table"])
    _add_table_section(elems, styles, "Corrélations inter-dimensions", block4["interdim"])
    if not block4["redundant_pairs"].empty:
        _add_table_section(elems, styles, "Paires de dimensions redondantes (|ρ| > 0.80)", block4["redundant_pairs"])
    _add_table_section(elems, styles, "Comparaison PC vs VR par dimension", block4["pc_vr"])
    _add_table_section(elems, styles, "Régressions univariées dimension → Score_perf_tsk", block4["perf_reg"])
    elems.append(Image(str(block4["heatmap_path"]), width=6.2 * inch, height=5.2 * inch))

    _save_pdf(out_path, elems)


def generate_complementary_reports(results_dir: Path, out_dir: Path, *, verbose: bool = False, tci_allowed_file: Path | None = None, tci_all_file: Path | None = None) -> dict[str, Path]:
    """
    Génère les deux rapports complémentaires et leurs sorties associées.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tci_allowed, tci_allowed_path = _load_tci_scores(results_dir, "allowed", explicit_path=tci_allowed_file)
    tci_all, tci_all_path = _load_tci_scores(results_dir, "all", explicit_path=tci_all_file)
    riedl = _load_riedl_summary(results_dir)
    perf = _load_performance(results_dir)
    profile = _load_profile(results_dir)
    q_long = _load_questionnaire_scores_long(results_dir, use_pruned=True)

    meta = _build_group_metadata(perf, profile, q_long)
    tci_allowed = tci_allowed.merge(meta, on="group_id", how="left")
    tci_all = tci_all.merge(meta, on="group_id", how="left")
    riedl = riedl.merge(meta, on="group_id", how="left")

    q_participant = _build_questionnaire_participant_wide(q_long)
    q_group = _build_questionnaire_group_wide(q_participant, meta)

    allowed_groups = set(tci_allowed["group_id"].dropna().astype(str))
    riedl_groups = set(riedl["group_id"].dropna().astype(str))
    perf_groups = set(perf["group_id"].dropna().astype(str))
    q_groups = set(q_group["group_id"].dropna().astype(str))
    profile_groups = set(profile["group_id"].dropna().astype(str))

    perf_pool_groups = allowed_groups & riedl_groups & perf_groups & q_groups
    block1_groups = perf_groups & profile_groups
    perf_pool = perf.merge(riedl, on="group_id", how="inner").merge(tci_allowed[["group_id", "c_factor"]], on="group_id", how="inner").merge(q_group, on="group_id", how="inner", suffixes=("", "_q"))
    perf_pool = perf_pool.loc[perf_pool["group_id"].isin(perf_pool_groups)].drop_duplicates(subset=["group_id"]).reset_index(drop=True)

    if verbose:
        print(f"[Complementary] TCI allowed path : {tci_allowed_path}")
        print(f"[Complementary] TCI all path     : {tci_all_path}")
        print(f"[Complementary] Block 2 N        : {len(set(tci_all['group_id']) & set(riedl['group_id']))}")
        print(f"[Complementary] Block 1 N        : {len(block1_groups)}")
        print(f"[Complementary] Block 3 N        : {len(perf_pool_groups)}")
        print(f"[Complementary] Block 4 TCI N    : {len(set(q_group['group_id']) & set(tci_all['group_id']))}")
        print(f"[Complementary] Block 4 Q-only N : {len(set(q_group['group_id']))}")

    block1 = _build_block1(profile, perf, block1_groups, figures_dir, tables_dir)
    block2 = _build_block2(tci_all, riedl, meta, figures_dir, tables_dir)
    block3 = _build_block3(perf_pool, figures_dir, tables_dir, block2)
    block4 = _build_block4(q_group, perf_pool, tci_all, figures_dir, tables_dir, q_participant)

    pdf_vr = out_dir / "rapport_VR_blocs1_3.pdf"
    pdf_global = out_dir / "rapport_global_blocs2_4.pdf"
    _render_report_blocks_1_3(pdf_vr, block1, block3, tci_allowed_path, tci_all_path)
    _render_report_blocks_2_4(pdf_global, block2, block4, tci_allowed_path, tci_all_path)

    return {"rapport_vr_pdf": pdf_vr, "rapport_global_pdf": pdf_global, "tables_dir": tables_dir, "figures_dir": figures_dir}
