from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from py.behavioral_indices_analyze import (
    _compute_vif,
    _fit_full_ols,
    _plot_heatmap,
    bh_fdr,
    bootstrap_spearman_ci,
    cronbach_alpha,
    icc2k,
    omega_total,
    permutation_spearman_p,
    safe_spearman,
)
from py.behavioral_indices_build import RunLogger, SEED, build_behavioral_indices_pipeline


OUTPUT_DIR = PROJECT_ROOT / "results" / "rapport_v2" / "VR_only" / "behavioral_indices_v2"
FIGURES_DIRNAME = "figures"
INDEX_ORDER = ["I_TMS_b", "I_TAS_b", "I_TRS_b"]
PRIMARY_TARGETS = {
    "I_TMS_b": "CRE",
    "I_TAS_b": "TSK",
    "I_TRS_b": "COM",
}


def load_base_bundle() -> dict[str, Any]:
    """
    Recharge le bundle v1 sans ecrire de nouvelles sorties.
    """
    return build_behavioral_indices_pipeline(write_outputs=False)


def zscore_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (numeric - mean) / std


def build_signed_matrix(
    df: pd.DataFrame,
    feature_signs: list[tuple[str, int]],
    *,
    label: str | None = None,
) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for feature_name, sign in feature_signs:
        col_name = f"{label}::{feature_name}" if label else feature_name
        if feature_name not in df.columns:
            cols[col_name] = pd.Series(np.nan, index=df.index, dtype=float)
        else:
            cols[col_name] = zscore_series(df[feature_name]) * int(sign)
    return pd.DataFrame(cols, index=df.index)


def build_composite_score(
    df: pd.DataFrame,
    feature_signs: list[tuple[str, int]],
    *,
    min_features: int | None = None,
    label: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    matrix = build_signed_matrix(df, feature_signs, label=label)
    available = matrix.notna().sum(axis=1)
    required = int(min_features) if min_features is not None else len(feature_signs)
    score = matrix.mean(axis=1, skipna=True)
    score = pd.Series(np.where(available >= required, score, np.nan), index=df.index, dtype=float)
    return score, matrix, available


def perf_sample(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["c_factor"] = pd.to_numeric(out["c_score_allowed"], errors="coerce")
    return out[out["c_factor"].notna() & out["Score_perf_tsk"].notna()].copy()


def ratio_m1_m2(df: pd.DataFrame) -> pd.Series:
    m1 = pd.to_numeric(df["M1"], errors="coerce")
    m2 = pd.to_numeric(df["M2"], errors="coerce")
    denom = m2.where(m2 > 0, np.nan)
    return m1 / denom


def loo_range(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    sub = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()
    if len(sub) < 5:
        return (np.nan, np.nan)
    values = sub.to_numpy(dtype=float)
    estimates: list[float] = []
    for i in range(len(values)):
        leave = np.delete(values, i, axis=0)
        if np.unique(leave[:, 0]).size < 2 or np.unique(leave[:, 1]).size < 2:
            continue
        rho = safe_spearman(pd.Series(leave[:, 0]), pd.Series(leave[:, 1]))["rho"]
        if math.isfinite(rho):
            estimates.append(float(rho))
    if not estimates:
        return (np.nan, np.nan)
    return (float(np.min(estimates)), float(np.max(estimates)))


def correlation_metrics(
    x: pd.Series,
    y: pd.Series,
    *,
    use_bootstrap: bool = True,
    use_permutation: bool = False,
    use_loo: bool = False,
) -> dict[str, Any]:
    stats = safe_spearman(x, y)
    ci_low, ci_high = (np.nan, np.nan)
    if use_bootstrap:
        ci_low, ci_high = bootstrap_spearman_ci(x, y, seed=SEED)
    perm_p = permutation_spearman_p(x, y, seed=SEED) if use_permutation else np.nan
    loo_low, loo_high = loo_range(x, y) if use_loo else (np.nan, np.nan)
    return {
        "n": stats["n"],
        "rho": stats["rho"],
        "p_value": stats["p_value"],
        "ci95_low_boot": ci_low,
        "ci95_high_boot": ci_high,
        "perm_p_value": perm_p,
        "loo_min_rho": loo_low,
        "loo_max_rho": loo_high,
    }


def sign_spec_to_text(feature_signs: list[tuple[str, int]]) -> str:
    return "; ".join(f"{name}:{'+' if int(sign) > 0 else '-'}1" for name, sign in feature_signs)


def sign_combo_id(feature_signs: list[tuple[str, int]]) -> str:
    return "".join("+" if int(sign) > 0 else "-" for _, sign in feature_signs)


def metric_snapshot(
    index_name: str,
    score: pd.Series,
    item_matrix: pd.DataFrame,
    questionnaire_series: pd.Series,
    perf_df: pd.DataFrame,
) -> dict[str, Any]:
    alpha = cronbach_alpha(item_matrix)
    omega = omega_total(item_matrix)
    icc_val, icc_n = icc2k(item_matrix)
    q_metrics = correlation_metrics(score, questionnaire_series, use_bootstrap=True)
    p_metrics = correlation_metrics(
        score.loc[perf_df.index],
        perf_df["Score_perf_tsk"],
        use_bootstrap=True,
        use_permutation=True,
        use_loo=True,
    )
    return {
        "index_name": index_name,
        "alpha_cronbach": alpha,
        "omega_total": omega,
        "icc2k": icc_val,
        "icc2k_n_complete": icc_n,
        "questionnaire_target": questionnaire_series.name,
        "rho_questionnaire": q_metrics["rho"],
        "p_questionnaire": q_metrics["p_value"],
        "ci95_low_questionnaire": q_metrics["ci95_low_boot"],
        "ci95_high_questionnaire": q_metrics["ci95_high_boot"],
        "n_questionnaire": q_metrics["n"],
        "rho_score_perf": p_metrics["rho"],
        "p_score_perf": p_metrics["p_value"],
        "ci95_low_score_perf": p_metrics["ci95_low_boot"],
        "ci95_high_score_perf": p_metrics["ci95_high_boot"],
        "perm_p_score_perf": p_metrics["perm_p_value"],
        "loo_min_score_perf": p_metrics["loo_min_rho"],
        "loo_max_score_perf": p_metrics["loo_max_rho"],
        "n_score_perf": p_metrics["n"],
    }


def mannwhitney_rank_biserial(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    a_num = pd.to_numeric(a, errors="coerce").dropna()
    b_num = pd.to_numeric(b, errors="coerce").dropna()
    if len(a_num) == 0 or len(b_num) == 0:
        return {
            "n_a": int(len(a_num)),
            "n_b": int(len(b_num)),
            "u_stat": np.nan,
            "p_value": np.nan,
            "rank_biserial": np.nan,
            "median_a": np.nan,
            "median_b": np.nan,
        }
    u_stat, p_value = mannwhitneyu(a_num, b_num, alternative="two-sided")
    rank_biserial = (2.0 * float(u_stat) / (len(a_num) * len(b_num))) - 1.0
    return {
        "n_a": int(len(a_num)),
        "n_b": int(len(b_num)),
        "u_stat": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial": float(rank_biserial),
        "median_a": float(a_num.median()),
        "median_b": float(b_num.median()),
    }


def save_labeled_scatter(
    path: Path,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    sub = df[[x_col, y_col, "group_id"]].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return path
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(sub[x_col], sub[y_col], s=48, alpha=0.9)
    if len(sub) >= 2 and sub[x_col].nunique() >= 2:
        coeffs = np.polyfit(sub[x_col], sub[y_col], deg=1)
        xx = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, color="black", linewidth=1.2)
    for _, row in sub.iterrows():
        ax.annotate(row["group_id"], (row[x_col], row[y_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_boxplot(
    path: Path,
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    title: str,
    ylabel: str,
) -> Path:
    sub = df[[group_col, value_col]].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    groups = [g for g in sub[group_col].dropna().astype(str).unique().tolist()]
    if not groups:
        return path
    data = [sub.loc[sub[group_col] == grp, value_col].dropna().to_numpy(dtype=float) for grp in groups]
    if not any(len(arr) for arr in data):
        return path
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.boxplot(data, labels=groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_grid_histograms(path: Path, grid_df: pd.DataFrame) -> Path:
    if grid_df.empty:
        return path
    fig, axes = plt.subplots(len(INDEX_ORDER), 1, figsize=(6.8, 3.0 * len(INDEX_ORDER)))
    if len(INDEX_ORDER) == 1:
        axes = [axes]
    for ax, index_name in zip(axes, INDEX_ORDER):
        sub = grid_df.loc[grid_df["index_name"] == index_name, "rho_score_perf"].dropna().astype(float)
        ax.hist(sub, bins=min(10, max(4, len(sub))), color="#6B8EAD", edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(index_name)
        ax.set_xlabel("rho avec Score_perf_tsk")
        ax.set_ylabel("n combinaisons")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_v2_questionnaire_heatmap(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        return path
    df = pd.DataFrame(rows)
    heatmap = df.pivot(index="row_label", columns="questionnaire_target", values="rho_questionnaire")
    annot = heatmap.copy().astype(object)
    for idx in heatmap.index:
        for col in heatmap.columns:
            val = heatmap.loc[idx, col]
            annot.loc[idx, col] = "" if pd.isna(val) else f"{val:.2f}"
    _plot_heatmap(
        heatmap,
        annot,
        path,
        title="Comparaison v1 / v1-inv / v2 sur les correlations questionnaires principales",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    return path


def ensure_output_dirs(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, figures_dir


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def fit_ols_zscored(df: pd.DataFrame, dv: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    work = df.copy()
    z_predictors: list[str] = []
    for predictor in predictors:
        z_name = f"{predictor}_z"
        work[z_name] = zscore_series(work[predictor])
        z_predictors.append(z_name)
    coef_table, summary = _fit_full_ols(work, dv, z_predictors)
    vif_df = _compute_vif(work, z_predictors) if not coef_table.empty else pd.DataFrame()
    return coef_table, summary, vif_df
