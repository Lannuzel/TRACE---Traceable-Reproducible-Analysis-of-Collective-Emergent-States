# -*- coding: utf-8 -*-
"""
Bloc H2b : moderation de la relation performance ~ predicteurs par modalite.

Ce module ajoute des analyses exploratoires sur le sous-echantillon disposant
simultanement d'un score de performance et du c-score (N attendu = 12 ; PC=4,
VR=8 dans l'etat actuel du pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer, Table
from scipy.stats import spearmanr, f as f_dist

try:
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


H2B_PREDICTOR_ORDER = [
    "c_score",
    "c_factor_pop",
    "effort_task_sum",
    "skill_mean",
    "strategy_ratio_mean",
    "contribution_mean",
    "skill_congruence_mean",
]

H2B_FIGURE_ONLY_PREDICTORS: list[str] = []

H2B_LABELS = {
    "c_score": "C-factor (c_score)",
    "c_factor_pop": "C-factor population parente",
    "effort_task_sum": "Effort collectif (effort_task_sum)",
    "skill_mean": "Skill moyen",
    "strategy_ratio_mean": "Strategie moyenne",
    "contribution_mean": "Contribution moyenne",
    "skill_congruence_mean": "Congruence skill-effort",
    "Score_perf_tsk": "Performance finale",
    "C(modalite)": "Modalite",
    "modalite": "Modalite",
}

H2B_COLORS = {
    "PC": "#d95f02",
    "VR": "#1b9e77",
}


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_modality(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.upper()
    values = values.replace({"RV": "VR"})
    return values


def prepare_h2b_dataframe(merged_master: pd.DataFrame | None) -> pd.DataFrame:
    """Prepare le dataset group-level utilise par les analyses H2b."""
    if merged_master is None or merged_master.empty:
        return pd.DataFrame()

    keep_cols = [
        c
        for c in ["group_id", "timepoint", "modalite", "scenario", "Score_perf_tsk", *H2B_PREDICTOR_ORDER, *H2B_FIGURE_ONLY_PREDICTORS]
        if c in merged_master.columns
    ]
    if not {"group_id", "Score_perf_tsk"}.issubset(set(keep_cols)):
        return pd.DataFrame()

    df = merged_master[keep_cols].copy()
    if "modalite" not in df.columns:
        return pd.DataFrame()

    df["group_id"] = df["group_id"].astype(str).str.strip().str.lower()
    df["modalite"] = _normalize_modality(df["modalite"])
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()

    numeric_cols = [c for c in ["Score_perf_tsk", *H2B_PREDICTOR_ORDER, *H2B_FIGURE_ONLY_PREDICTORS] if c in df.columns]
    df = _safe_numeric(df, numeric_cols)

    dedup_keys = [c for c in ["group_id", "timepoint"] if c in df.columns]
    if not dedup_keys:
        dedup_keys = ["group_id"]
    df = df.drop_duplicates(subset=dedup_keys, keep="first").reset_index(drop=True)
    return df


def _add_parent_population_cfactor(df: pd.DataFrame, parent_tci_path: Path | None) -> pd.DataFrame:
    """Ajoute `c_factor_pop` depuis le fichier TCI population parente si disponible."""
    if df is None or df.empty or "c_factor_pop" in df.columns:
        return df
    if parent_tci_path is None or not parent_tci_path.exists():
        return df

    try:
        pop = pd.read_csv(parent_tci_path)
    except Exception:
        return df
    if not {"group_id", "c_factor_pop"}.issubset(pop.columns):
        return df

    pop = pop[["group_id", "c_factor_pop"]].copy()
    pop["group_id"] = pop["group_id"].astype(str).str.strip().str.lower()
    pop["c_factor_pop"] = pd.to_numeric(pop["c_factor_pop"], errors="coerce")
    pop = pop.dropna(subset=["group_id"]).drop_duplicates(subset=["group_id"], keep="first")

    out = df.merge(pop, on="group_id", how="left")
    return out


def _partial_eta_squared(anova_table: pd.DataFrame, term: str) -> float:
    if term not in anova_table.index or "Residual" not in anova_table.index:
        return np.nan
    ss_effect = pd.to_numeric(anova_table.loc[term, "sum_sq"], errors="coerce")
    ss_resid = pd.to_numeric(anova_table.loc["Residual", "sum_sq"], errors="coerce")
    if not np.isfinite(ss_effect) or not np.isfinite(ss_resid) or (ss_effect + ss_resid) <= 0:
        return np.nan
    return float(ss_effect / (ss_effect + ss_resid))


def _resolve_interaction_term(index_values: list[str], predictor: str) -> str | None:
    exact = f"{predictor}:C(modalite)"
    if exact in index_values:
        return exact
    reversed_exact = f"C(modalite):{predictor}"
    if reversed_exact in index_values:
        return reversed_exact
    for idx in index_values:
        if ":" not in idx:
            continue
        if predictor in idx and "C(modalite)" in idx:
            return idx
    return None


def _effect_stats(anova_table: pd.DataFrame, term: str | None) -> tuple[float, float, float]:
    if term is None or term not in anova_table.index:
        return np.nan, np.nan, np.nan
    f_val = pd.to_numeric(anova_table.loc[term, "F"], errors="coerce")
    p_val = pd.to_numeric(anova_table.loc[term, "PR(>F)"], errors="coerce")
    eta = _partial_eta_squared(anova_table, term)
    return float(f_val) if np.isfinite(f_val) else np.nan, float(p_val) if np.isfinite(p_val) else np.nan, eta


def _ols_fit_stats(y: np.ndarray, x_matrix: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x_matrix, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or y.ndim != 1 or len(y) != x.shape[0]:
        raise ValueError("Dimensions OLS invalides.")

    rank = int(np.linalg.matrix_rank(x))
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    df_resid = int(len(y) - rank)
    return {
        "beta": beta,
        "fitted": fitted,
        "resid": resid,
        "rss": rss,
        "tss": tss,
        "rank": rank,
        "df_resid": df_resid,
    }


def _safe_f_test(ss_effect: float, df_effect: int, rss_full: float, df_resid_full: int) -> tuple[float, float]:
    if (
        not np.isfinite(ss_effect)
        or df_effect <= 0
        or not np.isfinite(rss_full)
        or df_resid_full <= 0
    ):
        return np.nan, np.nan
    ms_effect = ss_effect / df_effect
    ms_error = rss_full / df_resid_full
    if ms_error <= 0:
        return np.nan, np.nan
    f_val = ms_effect / ms_error
    p_val = float(f_dist.sf(f_val, df_effect, df_resid_full))
    return float(f_val), p_val


def _manual_interaction_model(df: pd.DataFrame, predictor: str, formula: str, model_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    sub = df[[predictor, "modalite", "Score_perf_tsk"]].dropna().copy()
    sub["mod_vr"] = (sub["modalite"].astype(str).str.upper() == "VR").astype(float)
    x = pd.to_numeric(sub[predictor], errors="coerce").to_numpy(dtype=float)
    mod = sub["mod_vr"].to_numpy(dtype=float)
    y = pd.to_numeric(sub["Score_perf_tsk"], errors="coerce").to_numpy(dtype=float)

    x_full = np.column_stack([np.ones(len(sub)), x, mod, x * mod])
    x_add = np.column_stack([np.ones(len(sub)), x, mod])
    x_mod_only = np.column_stack([np.ones(len(sub)), mod])
    x_pred_only = np.column_stack([np.ones(len(sub)), x])

    fit_full = _ols_fit_stats(y, x_full)
    fit_add = _ols_fit_stats(y, x_add)
    fit_mod_only = _ols_fit_stats(y, x_mod_only)
    fit_pred_only = _ols_fit_stats(y, x_pred_only)

    ss_pred = max(0.0, fit_mod_only["rss"] - fit_add["rss"])
    ss_mod = max(0.0, fit_pred_only["rss"] - fit_add["rss"])
    ss_inter = max(0.0, fit_add["rss"] - fit_full["rss"])

    f_pred, p_pred = _safe_f_test(ss_pred, 1, fit_full["rss"], fit_full["df_resid"])
    f_mod, p_mod = _safe_f_test(ss_mod, 1, fit_full["rss"], fit_full["df_resid"])
    f_inter, p_inter = _safe_f_test(ss_inter, 1, fit_full["rss"], fit_full["df_resid"])

    def _eta(ss_effect: float) -> float:
        denom = ss_effect + fit_full["rss"]
        if denom <= 0:
            return np.nan
        return float(ss_effect / denom)

    r2 = np.nan
    r2_adj = np.nan
    if fit_full["tss"] > 0:
        r2 = 1.0 - (fit_full["rss"] / fit_full["tss"])
        n_obs = len(sub)
        p_params = fit_full["rank"] - 1
        if n_obs - p_params - 1 > 0:
            r2_adj = 1.0 - (1.0 - r2) * ((n_obs - 1) / (n_obs - p_params - 1))

    row = {
        "model_name": model_name,
        "predictor": predictor,
        "formula": formula,
        "n": int(len(sub)),
        "r2": r2,
        "r2_adj": r2_adj,
        "F_pred": f_pred,
        "p_pred": p_pred,
        "eta2p_pred": _eta(ss_pred),
        "F_modalite": f_mod,
        "p_modalite": p_mod,
        "eta2p_modalite": _eta(ss_mod),
        "F_interaction": f_inter,
        "p_interaction": p_inter,
        "eta2p_interaction": _eta(ss_inter),
        "F_c_score": np.nan,
        "p_c_score": np.nan,
        "eta2p_c_score": np.nan,
    }

    long_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": predictor,
                "sum_sq": ss_pred,
                "df": 1,
                "F": f_pred,
                "p": p_pred,
                "eta2p": _eta(ss_pred),
            },
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": "C(modalite)",
                "sum_sq": ss_mod,
                "df": 1,
                "F": f_mod,
                "p": p_mod,
                "eta2p": _eta(ss_mod),
            },
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": f"{predictor}:C(modalite)",
                "sum_sq": ss_inter,
                "df": 1,
                "F": f_inter,
                "p": p_inter,
                "eta2p": _eta(ss_inter),
            },
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": "Residual",
                "sum_sq": fit_full["rss"],
                "df": fit_full["df_resid"],
                "F": np.nan,
                "p": np.nan,
                "eta2p": np.nan,
            },
        ]
    )
    return row, long_df


def _manual_additive_predictor_model(
    df: pd.DataFrame,
    predictor: str,
    formula: str,
    model_name: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    sub = df[[predictor, "modalite", "Score_perf_tsk"]].dropna().copy()
    sub["mod_vr"] = (sub["modalite"].astype(str).str.upper() == "VR").astype(float)
    predictor_values = pd.to_numeric(sub[predictor], errors="coerce").to_numpy(dtype=float)
    mod = sub["mod_vr"].to_numpy(dtype=float)
    y = pd.to_numeric(sub["Score_perf_tsk"], errors="coerce").to_numpy(dtype=float)

    x_full = np.column_stack([np.ones(len(sub)), mod, predictor_values])
    x_predictor_only = np.column_stack([np.ones(len(sub)), predictor_values])
    x_mod_only = np.column_stack([np.ones(len(sub)), mod])

    fit_full = _ols_fit_stats(y, x_full)
    fit_predictor_only = _ols_fit_stats(y, x_predictor_only)
    fit_mod_only = _ols_fit_stats(y, x_mod_only)

    ss_mod = max(0.0, fit_predictor_only["rss"] - fit_full["rss"])
    ss_predictor = max(0.0, fit_mod_only["rss"] - fit_full["rss"])

    f_mod, p_mod = _safe_f_test(ss_mod, 1, fit_full["rss"], fit_full["df_resid"])
    f_predictor, p_predictor = _safe_f_test(ss_predictor, 1, fit_full["rss"], fit_full["df_resid"])

    def _eta(ss_effect: float) -> float:
        denom = ss_effect + fit_full["rss"]
        if denom <= 0:
            return np.nan
        return float(ss_effect / denom)

    r2 = np.nan
    r2_adj = np.nan
    if fit_full["tss"] > 0:
        r2 = 1.0 - (fit_full["rss"] / fit_full["tss"])
        n_obs = len(sub)
        p_params = fit_full["rank"] - 1
        if n_obs - p_params - 1 > 0:
            r2_adj = 1.0 - (1.0 - r2) * ((n_obs - 1) / (n_obs - p_params - 1))

    row = {
        "model_name": model_name,
        "predictor": predictor,
        "formula": formula,
        "n": int(len(sub)),
        "r2": r2,
        "r2_adj": r2_adj,
        "F_pred": np.nan,
        "p_pred": np.nan,
        "eta2p_pred": np.nan,
        "F_modalite": f_mod,
        "p_modalite": p_mod,
        "eta2p_modalite": _eta(ss_mod),
        "F_interaction": np.nan,
        "p_interaction": np.nan,
        "eta2p_interaction": np.nan,
        "F_c_score": f_predictor,
        "p_c_score": p_predictor,
        "eta2p_c_score": _eta(ss_predictor),
    }

    long_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": "C(modalite)",
                "sum_sq": ss_mod,
                "df": 1,
                "F": f_mod,
                "p": p_mod,
                "eta2p": _eta(ss_mod),
            },
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": predictor,
                "sum_sq": ss_predictor,
                "df": 1,
                "F": f_predictor,
                "p": p_predictor,
                "eta2p": _eta(ss_predictor),
            },
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(sub)),
                "term": "Residual",
                "sum_sq": fit_full["rss"],
                "df": fit_full["df_resid"],
                "F": np.nan,
                "p": np.nan,
                "eta2p": np.nan,
            },
        ]
    )
    return row, long_df


def _fit_type2_ancova(df: pd.DataFrame, formula: str, predictor: str, model_name: str) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if df is None or df.empty:
        return None, pd.DataFrame()

    if not HAS_STATSMODELS:
        if "* C(modalite)" in formula:
            return _manual_interaction_model(df, predictor, formula, model_name)
        if formula.strip().startswith("Score_perf_tsk ~ C(modalite) + "):
            return _manual_additive_predictor_model(df, predictor, formula, model_name)
        return None, pd.DataFrame()

    try:
        model = ols(formula, data=df).fit()
        anova = anova_lm(model, typ=2)
    except Exception:
        return None, pd.DataFrame()

    interaction_term = _resolve_interaction_term(list(anova.index), predictor)
    f_pred, p_pred, eta_pred = _effect_stats(anova, predictor)
    f_mod, p_mod, eta_mod = _effect_stats(anova, "C(modalite)")
    f_int, p_int, eta_int = _effect_stats(anova, interaction_term)
    f_cscore, p_cscore, eta_cscore = _effect_stats(anova, predictor)

    row = {
        "model_name": model_name,
        "predictor": predictor,
        "formula": formula,
        "n": int(len(df)),
        "r2": float(model.rsquared) if np.isfinite(model.rsquared) else np.nan,
        "r2_adj": float(model.rsquared_adj) if np.isfinite(model.rsquared_adj) else np.nan,
        "F_pred": f_pred,
        "p_pred": p_pred,
        "eta2p_pred": eta_pred,
        "F_modalite": f_mod,
        "p_modalite": p_mod,
        "eta2p_modalite": eta_mod,
        "F_interaction": f_int,
        "p_interaction": p_int,
        "eta2p_interaction": eta_int,
        "F_c_score": f_cscore,
        "p_c_score": p_cscore,
        "eta2p_c_score": eta_cscore,
    }

    long_rows = []
    for term in anova.index:
        long_rows.append(
            {
                "model_name": model_name,
                "predictor": predictor,
                "formula": formula,
                "n": int(len(df)),
                "term": term,
                "sum_sq": pd.to_numeric(anova.loc[term, "sum_sq"], errors="coerce"),
                "df": pd.to_numeric(anova.loc[term, "df"], errors="coerce"),
                "F": pd.to_numeric(anova.loc[term, "F"], errors="coerce"),
                "p": pd.to_numeric(anova.loc[term, "PR(>F)"], errors="coerce"),
                "eta2p": _partial_eta_squared(anova, term) if term != "Residual" else np.nan,
            }
        )

    return row, pd.DataFrame(long_rows)


def _direction_from_rho(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    if rho > 0:
        return "positive"
    if rho < 0:
        return "negative"
    return "nulle"


def _stratified_spearman(df: pd.DataFrame, predictor: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "predictor": predictor,
        "label": H2B_LABELS.get(predictor, predictor),
    }
    for modality in ["PC", "VR"]:
        sub = df.loc[df["modalite"] == modality, [predictor, "Score_perf_tsk"]].dropna()
        n_obs = int(len(sub))
        rho = np.nan
        p_val = np.nan
        if n_obs >= 3 and sub[predictor].nunique(dropna=True) >= 2 and sub["Score_perf_tsk"].nunique(dropna=True) >= 2:
            rho, p_val = spearmanr(sub[predictor], sub["Score_perf_tsk"])
        row[f"n_{modality.lower()}"] = n_obs
        row[f"rho_{modality.lower()}"] = float(rho) if np.isfinite(rho) else np.nan
        row[f"p_{modality.lower()}"] = float(p_val) if np.isfinite(p_val) else np.nan
        row[f"direction_{modality.lower()}"] = _direction_from_rho(rho)

    rho_pc = row.get("rho_pc", np.nan)
    rho_vr = row.get("rho_vr", np.nan)
    row["delta_rho_vr_minus_pc"] = (
        float(rho_vr - rho_pc) if np.isfinite(rho_vr) and np.isfinite(rho_pc) else np.nan
    )
    return row


def _plot_by_modality(df: pd.DataFrame, predictor: str, out_path: Path) -> bool:
    sub = df[[c for c in ["group_id", "modalite", predictor, "Score_perf_tsk"] if c in df.columns]].dropna().copy()
    if sub.empty or len(sub) < 3:
        return False

    fig, ax = plt.subplots(figsize=(6.6, 4.4))

    for modality in ["PC", "VR"]:
        part = sub[sub["modalite"] == modality].copy()
        if part.empty:
            continue
        color = H2B_COLORS.get(modality, "#4c4c4c")
        ax.scatter(
            part[predictor],
            part["Score_perf_tsk"],
            s=52,
            alpha=0.9,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=f"{modality} (n={len(part)})",
        )
        for _, row in part.iterrows():
            if "group_id" in row and pd.notna(row["group_id"]):
                ax.annotate(
                    str(row["group_id"]),
                    (row[predictor], row["Score_perf_tsk"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                    color=color,
                    alpha=0.9,
                )
        if len(part) >= 2 and part[predictor].nunique(dropna=True) >= 2:
            coef = np.polyfit(part[predictor], part["Score_perf_tsk"], 1)
            xx = np.linspace(part[predictor].min(), part[predictor].max(), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, color=color, linewidth=1.8, alpha=0.9)

    if len(sub) >= 2 and sub[predictor].nunique(dropna=True) >= 2:
        coef = np.polyfit(sub[predictor], sub["Score_perf_tsk"], 1)
        xx = np.linspace(sub[predictor].min(), sub[predictor].max(), 120)
        yy = coef[0] * xx + coef[1]
        ax.plot(
            xx,
            yy,
            color="#222222",
            linewidth=2.1,
            linestyle="--",
            alpha=0.85,
            label=f"Tous (n={len(sub)})",
        )

    ax.set_xlabel(H2B_LABELS.get(predictor, predictor))
    ax.set_ylabel(H2B_LABELS["Score_perf_tsk"])
    ax.set_title(f"{H2B_LABELS.get(predictor, predictor)} × performance par modalite")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def run_h2b_analyses(
    merged_master: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    parent_tci_path: Path | None = None,
) -> dict[str, Any]:
    df = prepare_h2b_dataframe(merged_master)
    df = _add_parent_population_cfactor(df, parent_tci_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    primary_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    long_rows: list[pd.DataFrame] = []

    # Sous-echantillon central H2b : performance + c_score + modalite.
    h2b_core = df[[c for c in ["group_id", "modalite", "scenario", "Score_perf_tsk", "c_score", "c_factor_pop"] if c in df.columns]].dropna(subset=["Score_perf_tsk", "c_score"]).copy()
    sample_n = int(len(h2b_core))
    n_pc = int((h2b_core["modalite"] == "PC").sum()) if not h2b_core.empty else 0
    n_vr = int((h2b_core["modalite"] == "VR").sum()) if not h2b_core.empty else 0

    for predictor in H2B_PREDICTOR_ORDER:
        needed = [c for c in ["group_id", "modalite", "Score_perf_tsk", predictor] if c in df.columns]
        sub = df[needed].dropna().copy() if needed else pd.DataFrame()
        if sub.empty or predictor not in sub.columns or len(sub) < 6 or sub["modalite"].nunique(dropna=True) < 2:
            continue
        formula = f"Score_perf_tsk ~ {predictor} * C(modalite)"
        row, long_df = _fit_type2_ancova(
            sub,
            formula=formula,
            predictor=predictor,
            model_name=f"interaction_{predictor}",
        )
        if row is not None:
            primary_rows.append(row)
        if not long_df.empty:
            long_rows.append(long_df)

    for predictor in ["c_score", "c_factor_pop"]:
        if {"Score_perf_tsk", "modalite", predictor}.issubset(df.columns):
            robust_df = df[["group_id", "modalite", "Score_perf_tsk", predictor]].dropna().copy()
            if len(robust_df) >= 6 and robust_df["modalite"].nunique(dropna=True) == 2:
                robust_row, robust_long = _fit_type2_ancova(
                    robust_df,
                    formula=f"Score_perf_tsk ~ C(modalite) + {predictor}",
                    predictor=predictor,
                    model_name=f"robustesse_modalite_plus_{predictor}",
                )
                if robust_row is not None:
                    robustness_rows.append(robust_row)
                if not robust_long.empty:
                    long_rows.append(robust_long)

    strat_rows = []
    for predictor in H2B_PREDICTOR_ORDER:
        needed = [c for c in ["modalite", predictor, "Score_perf_tsk"] if c in df.columns]
        sub = df[needed].dropna().copy() if needed else pd.DataFrame()
        if sub.empty or predictor not in sub.columns:
            continue
        strat_rows.append(_stratified_spearman(sub, predictor))

    primary_df = pd.DataFrame(primary_rows)
    robustness_df = pd.DataFrame(robustness_rows)
    stratified_df = pd.DataFrame(strat_rows)
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    if not primary_df.empty:
        primary_df = primary_df.sort_values(
            by="predictor",
            key=lambda s: s.map({k: i for i, k in enumerate(H2B_PREDICTOR_ORDER)}).fillna(999),
        ).reset_index(drop=True)
    if not stratified_df.empty:
        stratified_df = stratified_df.sort_values(
            by="predictor",
            key=lambda s: s.map({k: i for i, k in enumerate(H2B_PREDICTOR_ORDER)}).fillna(999),
        ).reset_index(drop=True)
    if not robustness_df.empty:
        robustness_df = robustness_df.sort_values(
            by="predictor",
            key=lambda s: s.map({k: i for i, k in enumerate(H2B_PREDICTOR_ORDER)}).fillna(999),
        ).reset_index(drop=True)

    _h2bdir = out_dir / "data_h2b"
    _h2bdir.mkdir(parents=True, exist_ok=True)
    interactions_csv = _h2bdir / "h2b_ancova_interactions.csv"
    robustness_csv = _h2bdir / "h2b_ancova_robustesse.csv"
    stratified_csv = _h2bdir / "h2b_stratified_correlations.csv"
    long_csv = _h2bdir / "h2b_ancova_terms_long.csv"
    sample_csv = _h2bdir / "h2b_analysis_sample.csv"

    primary_df.to_csv(interactions_csv, index=False, encoding="utf-8-sig")
    robustness_df.to_csv(robustness_csv, index=False, encoding="utf-8-sig")
    stratified_df.to_csv(stratified_csv, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    h2b_core.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    figure_paths: dict[str, Path] = {}
    for predictor in ["c_score", "c_factor_pop", "effort_task_sum", "skill_mean"]:
        if predictor not in df.columns:
            continue
        fig_path = fig_dir / f"h2b_{predictor}_vs_performance_by_modalite.png"
        if _plot_by_modality(df, predictor, fig_path):
            figure_paths[predictor] = fig_path

    return {
        "data": df,
        "core_sample": h2b_core,
        "sample_n": sample_n,
        "n_pc": n_pc,
        "n_vr": n_vr,
        "interactions_df": primary_df,
        "robustness_df": robustness_df,
        "stratified_df": stratified_df,
        "long_df": long_df,
        "figure_paths": figure_paths,
        "paths": {
            "interactions": interactions_csv,
            "robustness": robustness_csv,
            "stratified": stratified_csv,
            "long": long_csv,
            "sample": sample_csv,
        },
    }


def _display_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["modele"] = out["predictor"].map(H2B_LABELS).fillna(out["predictor"])
    cols = [
        "modele",
        "n",
        "F_pred",
        "p_pred",
        "eta2p_pred",
        "F_modalite",
        "p_modalite",
        "eta2p_modalite",
        "F_interaction",
        "p_interaction",
        "eta2p_interaction",
    ]
    out = out[cols].rename(
        columns={
            "F_pred": "F_pred",
            "p_pred": "p_pred",
            "eta2p_pred": "η²p_pred",
            "F_modalite": "F_mod",
            "p_modalite": "p_mod",
            "eta2p_modalite": "η²p_mod",
            "F_interaction": "F_inter",
            "p_interaction": "p_inter",
            "eta2p_interaction": "η²p_inter",
        }
    )
    for col in [c for c in out.columns if c != "modele" and c != "n"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def _display_robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["modele"] = out["predictor"].map(
        lambda x: f"Robustesse: modalite + {H2B_LABELS.get(x, x)}"
    )
    cols = [
        "modele",
        "n",
        "F_modalite",
        "p_modalite",
        "eta2p_modalite",
        "F_c_score",
        "p_c_score",
        "eta2p_c_score",
    ]
    out = out[cols].rename(
        columns={
            "F_modalite": "F_mod",
            "p_modalite": "p_mod",
            "eta2p_modalite": "η²p_mod",
            "F_c_score": "F_covariable",
            "p_c_score": "p_covariable",
            "eta2p_c_score": "η²p_covariable",
        }
    )
    for col in [c for c in out.columns if c != "modele" and c != "n"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def _display_stratified_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["predicteur"] = out["predictor"].map(H2B_LABELS).fillna(out["predictor"])
    cols = [
        "predicteur",
        "n_pc",
        "rho_pc",
        "n_vr",
        "rho_vr",
        "delta_rho_vr_minus_pc",
        "direction_pc",
        "direction_vr",
    ]
    out = out[cols].rename(
        columns={
            "n_pc": "n_PC",
            "rho_pc": "ρ_PC",
            "n_vr": "n_VR",
            "rho_vr": "ρ_VR",
            "delta_rho_vr_minus_pc": "Δρ(VR-PC)",
            "direction_pc": "sens_PC",
            "direction_vr": "sens_VR",
        }
    )
    for col in ["ρ_PC", "ρ_VR", "Δρ(VR-PC)"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    return out


def _build_h2b_comment(bundle: dict[str, Any]) -> str:
    n_total = int(bundle.get("sample_n", 0))
    n_pc = int(bundle.get("n_pc", 0))
    n_vr = int(bundle.get("n_vr", 0))
    parts = [
        f"Le bloc H2b repose sur le sous-echantillon disposant simultanement d'un `c_score` et d'un `Score_perf_tsk` (n = {n_total} groupes ; PC = {n_pc}, VR = {n_vr}).",
        "Compte tenu de la taille d'echantillon, ces tests doivent etre lus comme exploratoires et indicatifs plutot que confirmatoires.",
    ]

    interactions = bundle.get("interactions_df", pd.DataFrame())
    if not interactions.empty:
        c_row = interactions.loc[interactions["predictor"] == "c_score"]
        if not c_row.empty:
            c_row = c_row.iloc[0]
            p_inter = pd.to_numeric(c_row.get("p_interaction"), errors="coerce")
            if np.isfinite(p_inter):
                if p_inter < 0.05:
                    parts.append(
                        "Le modele `Score_perf_tsk ~ c_score * modalite` suggere une interaction statistiquement detectable entre pente du c-score et modalite."
                    )
                else:
                    parts.append(
                        "Le modele `Score_perf_tsk ~ c_score * modalite` ne soutient pas une interaction nette c-score × modalite ; l'interpretation repose donc surtout sur les tailles d'effet et les corrélations stratifiées."
                    )

    stratified = bundle.get("stratified_df", pd.DataFrame())
    if not stratified.empty:
        comparable = stratified.dropna(subset=["rho_pc", "rho_vr"]).copy()
        if not comparable.empty:
            comparable["abs_delta"] = comparable["delta_rho_vr_minus_pc"].abs()
            strongest = comparable.sort_values("abs_delta", ascending=False).iloc[0]
            pred_label = H2B_LABELS.get(strongest["predictor"], strongest["predictor"])
            parts.append(
                f"Descriptivement, le contraste PC/VR le plus marque concerne `{pred_label}` (ρ_PC = {strongest['rho_pc']:.2f}, ρ_VR = {strongest['rho_vr']:.2f})."
            )

    robustness = bundle.get("robustness_df", pd.DataFrame())
    if not robustness.empty:
        row = robustness.iloc[0]
        p_mod = pd.to_numeric(row.get("p_modalite"), errors="coerce")
        if np.isfinite(p_mod):
            if p_mod < 0.05:
                parts.append(
                    "Dans le modele de robustesse `Score_perf_tsk ~ modalite + c_score`, l'effet de modalite persiste apres ajustement sur le c-score."
                )
            else:
                parts.append(
                    "Dans le modele de robustesse `Score_perf_tsk ~ modalite + c_score`, l'ajustement sur le c-score attenue l'effet de modalite au-dessous du seuil conventionnel."
                )

    return " ".join(parts)


def render_h2b_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    merged_master: pd.DataFrame | None,
    out_dir: Path,
    fig_dir: Path,
    *,
    md_table_fn,
    pdf_table_fn,
    section_num: str = "2.5",
    parent_tci_path: Path | None = None,
) -> dict[str, Any]:
    """Genere et insere la section H2b dans le rapport principal."""
    title = f"{section_num} Tests complementaires H2b (performance, modalite et potentiel collectif)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    bundle = run_h2b_analyses(
        merged_master,
        out_dir=out_dir,
        fig_dir=fig_dir,
        parent_tci_path=parent_tci_path,
    )
    if bundle["sample_n"] == 0:
        msg = "Sous-echantillon H2b indisponible : aucune intersection exploitable entre performance, c_score et modalite."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return bundle

    intro = _build_h2b_comment(bundle)
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro.replace("`", ""), styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    interaction_display = _display_interaction_table(bundle["interactions_df"])
    if not interaction_display.empty:
        lines.append("**ANCOVA type II : performance ~ predicteur * modalite**\n\n")
        lines.append(md_table_fn(interaction_display, max_rows=20))
        pdf_elems.append(Paragraph("ANCOVA type II : performance ~ predicteur * modalite", styles["Heading4"]))
        pdf_elems.append(pdf_table_fn(interaction_display, max_rows=20))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    robustness_display = _display_robustness_table(bundle["robustness_df"])
    if not robustness_display.empty:
        lines.append("**Controle de robustesse : modalite + C-factor**\n\n")
        lines.append(md_table_fn(robustness_display, max_rows=10))
        pdf_elems.append(Paragraph("Controle de robustesse : modalite + C-factor", styles["Heading4"]))
        pdf_elems.append(pdf_table_fn(robustness_display, max_rows=10))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    stratified_display = _display_stratified_table(bundle["stratified_df"])
    if not stratified_display.empty:
        note = (
            "Les corrélations suivantes sont rapportées descriptivement par modalite. "
            "Aucune comparaison formelle des coefficients (type Fisher z) n'est tentee compte tenu des effectifs PC = 4 et VR = 8."
        )
        lines.append(note + "\n\n")
        lines.append("**Corrélations stratifiées PC vs VR**\n\n")
        lines.append(md_table_fn(stratified_display, max_rows=20))
        pdf_elems.append(Paragraph(note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.06 * inch))
        pdf_elems.append(Paragraph("Corrélations stratifiées PC vs VR", styles["Heading4"]))
        pdf_elems.append(pdf_table_fn(stratified_display, max_rows=20))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    figure_paths = [bundle["figure_paths"].get(k) for k in ["c_score", "c_factor_pop", "effort_task_sum", "skill_mean"]]
    figure_paths = [p for p in figure_paths if p is not None and p.exists()]
    if figure_paths:
        lines.append("**Figures H2b**\n\n")
        for fig_path in figure_paths:
            lines.append(f"![]({fig_path.name})\n\n")

        pdf_elems.append(Paragraph("Figures H2b", styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))

        img_objs = [Image(str(path), width=3.15 * inch, height=2.35 * inch) for path in figure_paths]
        for i in range(0, len(img_objs), 2):
            pair = img_objs[i:i + 2]
            if len(pair) == 2:
                pdf_elems.append(Table([pair], colWidths=[3.2 * inch, 3.2 * inch]))
            else:
                pdf_elems.append(pair[0])
            pdf_elems.append(Spacer(1, 0.08 * inch))
        pdf_elems.append(Spacer(1, 0.12 * inch))

    export_note = (
        "Exports H2b generes : "
        f"`{bundle['paths']['interactions'].name}`, "
        f"`{bundle['paths']['robustness'].name}`, "
        f"`{bundle['paths']['stratified'].name}`."
    )
    lines.append(export_note + "\n\n")
    pdf_elems.append(Paragraph(export_note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.12 * inch))
    return bundle
