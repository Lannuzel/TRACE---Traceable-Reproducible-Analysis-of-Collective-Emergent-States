from __future__ import annotations

import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

_V2_DIR = Path(__file__).resolve().parents[1]
if str(_V2_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_DIR))

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

from py.behavioral_indices_build import (
    INDEX_SPEC,
    OUTPUT_DIR,
    SEED,
    RunLogger,
    build_behavioral_indices_pipeline,
)

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
    sm = None
    variance_inflation_factor = None

try:
    from sklearn.decomposition import FactorAnalysis

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
    FactorAnalysis = None


QUESTIONNAIRE_TARGETS = [
    ("I_TMS_b", "COR", "Convergence positive attendue avec TMS-COR."),
    ("I_TMS_b", "CRE", "Convergence positive attendue avec TMS-CRE."),
    ("I_TMS_b", "SPE", "Convergence positive attendue avec TMS-SPE."),
    ("I_TAS_b", "TSK", "Convergence positive attendue avec la cohesion tache."),
    ("I_TAS_b", "COM", "Convergence positive attendue avec la communication."),
    ("I_TRS_b", "SOC", "Convergence positive attendue avec la cohesion sociale."),
    ("I_TRS_b", "COM", "Convergence positive attendue avec la communication."),
    ("I_TRS_b", "strategy_norm", "Convergence positive attendue avec la coordination strategique."),
]

DISCRIMINANT_TARGETS = [
    ("I_TMS_b", "effort_task_norm", "Discriminant attendu modere avec l'effort."),
    ("I_TMS_b", "skill_congruence_mean", "Discriminant attendu modere avec la congruence skill-effort."),
    ("I_TAS_b", "effort_task_norm", "Discriminant attendu modere avec l'effort."),
    ("I_TAS_b", "skill_congruence_mean", "Discriminant attendu modere avec la congruence skill-effort."),
    ("I_TRS_b", "effort_task_norm", "Discriminant attendu modere avec l'effort."),
    ("I_TRS_b", "skill_congruence_mean", "Discriminant attendu modere avec la congruence skill-effort."),
]

PERFORMANCE_TARGETS = ["Score_perf_tsk", "M1", "M2", "c_factor"]
INDEX_ORDER = ["I_TMS_b", "I_TAS_b", "I_TRS_b"]
QUESTIONNAIRE_ORDER = ["COR", "CRE", "SPE", "SOC", "TSK", "COM", "strategy_norm", "effort_task_norm", "skill_congruence_mean"]


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
        if math.isfinite(val):
            return val
    except Exception:
        return None
    return None


def _p_to_stars(p_value: float | None) -> str:
    if p_value is None or not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _fmt_log_float(value: Any) -> str:
    val = _safe_float(value)
    return f"{val:.3f}" if val is not None else "NA"


def bh_fdr(p_values: list[float | None]) -> list[float | None]:
    arr = np.array([np.nan if p is None else float(p) for p in p_values], dtype=float)
    valid = np.where(np.isfinite(arr))[0]
    out = np.array([np.nan] * len(arr), dtype=float)
    if valid.size == 0:
        return out.tolist()

    ranked = valid[np.argsort(arr[valid])]
    m = float(len(ranked))
    prev = 1.0
    for reverse_rank, idx in enumerate(ranked[::-1], start=1):
        rank = m - reverse_rank + 1.0
        adjusted = min(prev, arr[idx] * m / rank)
        out[idx] = adjusted
        prev = adjusted
    return out.tolist()


def _pairwise_complete(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()


def safe_spearman(x: pd.Series, y: pd.Series) -> dict[str, float]:
    sub = _pairwise_complete(x, y)
    n = int(len(sub))
    if n < 3:
        return {"n": n, "rho": np.nan, "p_value": np.nan}
    if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return {"n": n, "rho": np.nan, "p_value": np.nan}
    rho, p_value = spearmanr(sub["x"], sub["y"])
    return {"n": n, "rho": float(rho), "p_value": float(p_value)}


def bootstrap_spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 1000, seed: int = SEED) -> tuple[float, float]:
    sub = _pairwise_complete(x, y)
    n = len(sub)
    if n < 4:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    values = sub.to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = values[idx]
        if np.unique(boot[:, 0]).size < 2 or np.unique(boot[:, 1]).size < 2:
            continue
        rho, _ = spearmanr(boot[:, 0], boot[:, 1])
        if math.isfinite(rho):
            estimates.append(float(rho))
    if len(estimates) < max(100, n_boot // 10):
        return (np.nan, np.nan)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return (float(low), float(high))


def permutation_spearman_p(x: pd.Series, y: pd.Series, n_perm: int = 1000, seed: int = SEED) -> float:
    sub = _pairwise_complete(x, y)
    n = len(sub)
    if n < 4:
        return np.nan
    observed = safe_spearman(sub["x"], sub["y"])["rho"]
    if not math.isfinite(observed):
        return np.nan

    rng = np.random.default_rng(seed)
    x_vals = sub["x"].to_numpy()
    y_vals = sub["y"].to_numpy()
    hits = 0
    for _ in range(n_perm):
        permuted = rng.permutation(y_vals)
        rho, _ = spearmanr(x_vals, permuted)
        if math.isfinite(rho) and abs(rho) >= abs(observed):
            hits += 1
    return float((hits + 1) / (n_perm + 1))


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    mat = df_items.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    k = mat.shape[1]
    if mat.shape[0] < 2 or k < 2:
        return np.nan
    item_vars = mat.var(axis=0, ddof=1)
    total = mat.sum(axis=1)
    total_var = total.var(ddof=1)
    if not math.isfinite(total_var) or total_var == 0:
        return np.nan
    alpha = (k / (k - 1.0)) * (1.0 - (item_vars.sum() / total_var))
    return float(alpha)


def omega_total(df_items: pd.DataFrame) -> float:
    if not HAS_SKLEARN:
        return np.nan
    mat = df_items.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if mat.shape[0] < 3 or mat.shape[1] < 2:
        return np.nan
    try:
        fa = FactorAnalysis(n_components=1, random_state=SEED)
        fa.fit(mat.to_numpy())
        loadings = np.asarray(fa.components_[0], dtype=float)
        uniqueness = np.asarray(fa.noise_variance_, dtype=float)
        numerator = float(np.square(loadings.sum()))
        denominator = numerator + float(uniqueness.sum())
        return float(numerator / denominator) if denominator > 0 else np.nan
    except Exception:
        return np.nan


def mean_interitem_spearman(df_items: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    mat = df_items.apply(pd.to_numeric, errors="coerce")
    corr = mat.corr(method="spearman")
    if corr.shape[0] < 2:
        return np.nan, corr
    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    values = tri.stack().dropna().to_numpy()
    if len(values) == 0:
        return np.nan, corr
    return float(np.mean(values)), corr


def icc2k(df_items: pd.DataFrame) -> tuple[float, int]:
    mat = df_items.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    n, k = mat.shape
    if n < 2 or k < 2:
        return (np.nan, n)
    x = mat.to_numpy(dtype=float)
    grand_mean = x.mean()
    mean_targets = x.mean(axis=1, keepdims=True)
    mean_raters = x.mean(axis=0, keepdims=True)
    ss_target = k * float(((mean_targets - grand_mean) ** 2).sum())
    ss_rater = n * float(((mean_raters - grand_mean) ** 2).sum())
    ss_total = float(((x - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_target - ss_rater
    df_target = n - 1
    df_rater = k - 1
    df_error = (n - 1) * (k - 1)
    if df_target <= 0 or df_rater <= 0 or df_error <= 0:
        return (np.nan, n)
    msb = ss_target / df_target
    msj = ss_rater / df_rater
    mse = ss_error / df_error
    denom = msb + (msj - mse) / n
    if not math.isfinite(denom) or denom == 0:
        return (np.nan, n)
    return (float((msb - mse) / denom), n)


def corrected_item_total(df_items: pd.DataFrame, index_scores: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = df_items.apply(pd.to_numeric, errors="coerce")
    for col in numeric.columns:
        others = numeric.drop(columns=[col])
        other_score = others.mean(axis=1, skipna=True) if others.shape[1] else pd.Series(np.nan, index=numeric.index)
        stats = safe_spearman(numeric[col], other_score)
        rows.append(
            {
                "feature": col,
                "n": int(pd.to_numeric(numeric[col], errors="coerce").notna().sum()),
                "mean": float(pd.to_numeric(numeric[col], errors="coerce").mean()),
                "sd": float(pd.to_numeric(numeric[col], errors="coerce").std(ddof=1)),
                "item_total_rho": stats["rho"],
                "item_total_p_value": stats["p_value"],
                "index_rho_raw": safe_spearman(numeric[col], index_scores)["rho"],
            }
        )
    return pd.DataFrame(rows)


def _prepare_signed_item_matrix(bundle: dict[str, Any], index_name: str) -> pd.DataFrame:
    feature_z = bundle["feature_z_df"].copy()
    retained = bundle["resolved_specs"][index_name]["retained_features"]
    cols = [f"{feat['feature_name']}__signed_z" for feat in retained]
    return feature_z[["group_id"] + cols].set_index("group_id")


def analyze_internal_consistency(bundle: dict[str, Any], output_dir: Path, log: RunLogger) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    item_tables: dict[str, pd.DataFrame] = {}
    interitem_paths: dict[str, Path] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for index_name in INDEX_ORDER:
        item_matrix = _prepare_signed_item_matrix(bundle, index_name)
        item_cols = list(item_matrix.columns)
        index_scores = bundle["indices_df"].set_index("group_id")[index_name]
        alpha = cronbach_alpha(item_matrix)
        omega = omega_total(item_matrix)
        mean_r, corr_matrix = mean_interitem_spearman(item_matrix)
        icc_val, icc_n = icc2k(item_matrix)
        item_table = corrected_item_total(item_matrix, index_scores)
        item_table.insert(0, "index_name", index_name)
        item_tables[index_name] = item_table

        corr_path = output_dir / f"interitem_spearman_{index_name}.csv"
        corr_matrix.to_csv(corr_path, encoding="utf-8")
        interitem_paths[index_name] = corr_path

        rows.append(
            {
                "index_name": index_name,
                "n_groups_complete": int(item_matrix.dropna(axis=0, how="any").shape[0]),
                "n_features": len(item_cols),
                "alpha_cronbach": alpha,
                "omega_total": omega,
                "mean_interitem_spearman": mean_r,
                "icc2k": icc_val,
                "icc2k_n_complete": icc_n,
            }
        )
        log.info(
            f"Cohérence {index_name}: n_complete={rows[-1]['n_groups_complete']}, "
            f"alpha={_fmt_log_float(alpha)}, icc2k={_fmt_log_float(icc_val)}."
        )

    summary = pd.DataFrame(rows)
    item_total = pd.concat(item_tables.values(), ignore_index=True) if item_tables else pd.DataFrame()
    summary.to_csv(output_dir / "internal_consistency_indices.csv", index=False, encoding="utf-8")
    item_total.to_csv(output_dir / "internal_consistency_item_total.csv", index=False, encoding="utf-8")
    return {
        "summary": summary,
        "item_total": item_total,
        "interitem_paths": interitem_paths,
    }


def _correlation_rows(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str]],
    family: str,
    n_boot: int = 1000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index_name, target, hypothesis in pairs:
        stats = safe_spearman(df[index_name], df[target])
        ci_low, ci_high = bootstrap_spearman_ci(df[index_name], df[target], n_boot=n_boot, seed=SEED)
        rows.append(
            {
                "family": family,
                "index_name": index_name,
                "target": target,
                "hypothesis": hypothesis,
                "n": stats["n"],
                "rho_spearman": stats["rho"],
                "p_value": stats["p_value"],
                "ci95_low_boot": ci_low,
                "ci95_high_boot": ci_high,
            }
        )
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_value"].tolist())
    out["signif"] = out["p_fdr"].apply(_p_to_stars)
    return out


def analyze_questionnaire_convergence(bundle: dict[str, Any], output_dir: Path, figures_dir: Path, log: RunLogger) -> dict[str, Any]:
    df = bundle["analysis_df"].copy()
    corr_df = _correlation_rows(df, QUESTIONNAIRE_TARGETS + DISCRIMINANT_TARGETS, family="questionnaires")
    corr_path = output_dir / "correlations_indices_questionnaires.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8")
    log.info(f"Validite convergente questionnaire: {len(corr_df)} correlations exportees.")

    heatmap_targets = [t for _, t, _ in QUESTIONNAIRE_TARGETS + DISCRIMINANT_TARGETS]
    unique_targets = [t for t in QUESTIONNAIRE_ORDER if t in heatmap_targets]
    heatmap = pd.DataFrame(index=INDEX_ORDER, columns=unique_targets, dtype=float)
    annot = pd.DataFrame(index=INDEX_ORDER, columns=unique_targets, dtype=object)
    for _, row in corr_df.iterrows():
        if row["target"] in heatmap.columns:
            heatmap.loc[row["index_name"], row["target"]] = row["rho_spearman"]
            val = row["rho_spearman"]
            annot.loc[row["index_name"], row["target"]] = (
                f"{val:.2f}{row['signif']}" if pd.notna(val) else ""
            )

    fig_path = figures_dir / "heatmap_indices_questionnaires.png"
    _plot_heatmap(
        heatmap,
        annot,
        fig_path,
        title="Indices comportementaux vs questionnaires / Riedl",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    return {"corr_df": corr_df, "heatmap_path": fig_path}


def _compute_vif(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    if len(predictors) < 2:
        return pd.DataFrame(columns=["predictor", "vif"])
    sub = df[predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.shape[0] < 3:
        return pd.DataFrame(columns=["predictor", "vif"])

    if HAS_STATSMODELS:
        x = sm.add_constant(sub)
        rows = []
        for i, predictor in enumerate(["const"] + predictors):
            if predictor == "const":
                continue
            rows.append({"predictor": predictor, "vif": float(variance_inflation_factor(x.values, i))})
        return pd.DataFrame(rows)

    rows = []
    for predictor in predictors:
        others = [p for p in predictors if p != predictor]
        if not others:
            continue
        y = sub[predictor].to_numpy(dtype=float)
        x = sub[others].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            vif = np.nan
        else:
            r2 = 1.0 - (ss_res / ss_tot)
            vif = np.nan if r2 >= 1 else float(1.0 / max(1e-12, 1.0 - r2))
        rows.append({"predictor": predictor, "vif": vif})
    return pd.DataFrame(rows)


def _fit_full_ols(df: pd.DataFrame, dv: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    sub = df[[dv] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(predictors) + 2:
        return pd.DataFrame(), {"n": len(sub), "warning": "n insuffisant pour OLS"}
    if HAS_STATSMODELS:
        x = sm.add_constant(sub[predictors], has_constant="add")
        model = sm.OLS(sub[dv], x).fit()
        coef_table = pd.DataFrame(
            {
                "term": model.params.index,
                "beta": model.params.values,
                "se": model.bse.values,
                "t": model.tvalues.values,
                "p_value": model.pvalues.values,
                "ci_low": model.conf_int()[0].values,
                "ci_high": model.conf_int()[1].values,
            }
        )
        summary = {
            "n": int(len(sub)),
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "aic": float(model.aic),
            "bic": float(model.bic),
        }
        return coef_table, summary

    y = sub[dv].to_numpy(dtype=float)
    x_pred = sub[predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x_pred)), x_pred])
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ x.T @ y
    y_hat = x @ beta
    resid = y - y_hat
    n = len(y)
    p = x.shape[1]
    df_resid = n - p
    if df_resid <= 0:
        return pd.DataFrame(), {"n": n, "warning": "ddl residuels insuffisants pour OLS"}

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    mse = rss / df_resid if df_resid > 0 else np.nan
    cov_beta = xtx_inv * mse
    se = np.sqrt(np.clip(np.diag(cov_beta), a_min=0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = beta / se
    p_values = 2.0 * (1.0 - student_t.cdf(np.abs(t_values), df=df_resid))
    t_crit = student_t.ppf(0.975, df=df_resid)
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se
    r2 = np.nan if tss == 0 else float(1.0 - rss / tss)
    r2_adj = np.nan if n - p == 0 else float(1.0 - (1.0 - r2) * ((n - 1) / (n - p)))
    sigma2 = rss / n if n > 0 else np.nan
    if sigma2 and sigma2 > 0:
        aic = float(n * np.log(sigma2) + 2 * p)
        bic = float(n * np.log(sigma2) + p * np.log(n))
    else:
        aic = np.nan
        bic = np.nan

    terms = ["const"] + predictors
    coef_table = pd.DataFrame(
        {
            "term": terms,
            "beta": beta,
            "se": se,
            "t": t_values,
            "p_value": p_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    summary = {"n": int(n), "r2": r2, "r2_adj": r2_adj, "aic": aic, "bic": bic}
    return coef_table, summary


def analyze_performance(bundle: dict[str, Any], output_dir: Path, figures_dir: Path, log: RunLogger) -> dict[str, Any]:
    df = bundle["analysis_df"].copy()
    df["c_factor"] = pd.to_numeric(df["c_score_allowed"], errors="coerce")
    perf_df = df[df["c_factor"].notna() & df["Score_perf_tsk"].notna()].copy()
    log.info(
        f"Validite predictive: sous-echantillon VR perf+TCI = n={len(perf_df)} "
        "(les groupes VR avec performance mais sans c_score sont exclus ici pour conserver un echantillon constant)."
    )

    rows: list[dict[str, Any]] = []
    for index_name in INDEX_ORDER:
        for target in PERFORMANCE_TARGETS:
            stats = safe_spearman(perf_df[index_name], perf_df[target])
            ci_low, ci_high = bootstrap_spearman_ci(perf_df[index_name], perf_df[target], seed=SEED)
            perm_p = permutation_spearman_p(perf_df[index_name], perf_df[target], seed=SEED) if target in {"Score_perf_tsk", "M1", "M2"} else np.nan
            rows.append(
                {
                    "index_name": index_name,
                    "target": target,
                    "n": stats["n"],
                    "rho_spearman": stats["rho"],
                    "p_value": stats["p_value"],
                    "ci95_low_boot": ci_low,
                    "ci95_high_boot": ci_high,
                    "perm_p_value": perm_p,
                    "warning_n_lt_8": stats["n"] < 8,
                }
            )
    corr_df = pd.DataFrame(rows)
    corr_df["p_fdr"] = bh_fdr(corr_df["p_value"].tolist())
    corr_df["signif"] = corr_df["p_fdr"].apply(_p_to_stars)

    loo_rows: list[dict[str, Any]] = []
    for index_name in INDEX_ORDER:
        for target in ["Score_perf_tsk", "c_factor"]:
            sub = _pairwise_complete(perf_df[index_name], perf_df[target])
            if len(sub) < 5:
                loo_rows.append(
                    {
                        "index_name": index_name,
                        "target": target,
                        "loo_min_rho": np.nan,
                        "loo_max_rho": np.nan,
                    }
                )
                continue
            estimates: list[float] = []
            values = sub.to_numpy()
            for i in range(len(values)):
                leave = np.delete(values, i, axis=0)
                if np.unique(leave[:, 0]).size < 2 or np.unique(leave[:, 1]).size < 2:
                    continue
                rho, _ = spearmanr(leave[:, 0], leave[:, 1])
                if math.isfinite(rho):
                    estimates.append(float(rho))
            loo_rows.append(
                {
                    "index_name": index_name,
                    "target": target,
                    "loo_min_rho": float(np.min(estimates)) if estimates else np.nan,
                    "loo_max_rho": float(np.max(estimates)) if estimates else np.nan,
                }
            )
    loo_df = pd.DataFrame(loo_rows)
    corr_df = corr_df.merge(loo_df, on=["index_name", "target"], how="left")

    corr_path = output_dir / "correlations_indices_performance.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8")

    coef_table, model_summary = _fit_full_ols(perf_df, "Score_perf_tsk", INDEX_ORDER)
    coef_path = output_dir / "ols_indices_performance.csv"
    if not coef_table.empty:
        vif_df = _compute_vif(perf_df, INDEX_ORDER)
        coef_table.to_csv(coef_path, index=False, encoding="utf-8")
        vif_df.to_csv(output_dir / "ols_indices_performance_vif.csv", index=False, encoding="utf-8")
    else:
        pd.DataFrame().to_csv(coef_path, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(output_dir / "ols_indices_performance_vif.csv", index=False, encoding="utf-8")
        vif_df = pd.DataFrame()

    heatmap = pd.DataFrame(index=INDEX_ORDER, columns=PERFORMANCE_TARGETS, dtype=float)
    annot = pd.DataFrame(index=INDEX_ORDER, columns=PERFORMANCE_TARGETS, dtype=object)
    for _, row in corr_df.iterrows():
        heatmap.loc[row["index_name"], row["target"]] = row["rho_spearman"]
        val = row["rho_spearman"]
        annot.loc[row["index_name"], row["target"]] = f"{val:.2f}{row['signif']}" if pd.notna(val) else ""
    heatmap_path = figures_dir / "heatmap_indices_performance.png"
    _plot_heatmap(heatmap, annot, heatmap_path, title="Indices comportementaux vs performance / c-factor", cmap="coolwarm", vmin=-1, vmax=1)

    scatter_path = figures_dir / "scatter_indices_score_perf.png"
    _plot_index_scatter(perf_df, scatter_path)

    return {
        "perf_df": perf_df,
        "corr_df": corr_df,
        "coef_table": coef_table,
        "model_summary": model_summary,
        "vif_df": vif_df,
        "heatmap_path": heatmap_path,
        "scatter_path": scatter_path,
    }


def analyze_configuration(bundle: dict[str, Any], performance_bundle: dict[str, Any], output_dir: Path, log: RunLogger) -> pd.DataFrame:
    df = performance_bundle["perf_df"].copy()
    if df.empty:
        out = pd.DataFrame()
        out.to_csv(output_dir / "configuration_profiles.csv", index=False, encoding="utf-8")
        return out

    for index_name in INDEX_ORDER:
        median_val = df[index_name].median(skipna=True)
        df[f"{index_name}_level"] = np.where(df[index_name] >= median_val, "haut", "bas")
    df["configuration_profile"] = df[[f"{name}_level" for name in INDEX_ORDER]].agg("/".join, axis=1)

    group_cols = ["configuration_profile"]
    summary = (
        df.groupby(group_cols)
        .agg(
            n_groupes=("group_id", "count"),
            groups=("group_id", lambda s: "; ".join(sorted(s.astype(str)))),
            mean_score_perf=("Score_perf_tsk", "mean"),
            mean_c_factor=("c_factor", "mean"),
        )
        .reset_index()
        .sort_values(["n_groupes", "configuration_profile"], ascending=[False, True])
    )
    summary.to_csv(output_dir / "configuration_profiles.csv", index=False, encoding="utf-8")
    log.info(f"Analyse configurationnelle exportee ({len(summary)} profils observes).")
    return summary


def analyze_sensitivity(bundle: dict[str, Any], performance_bundle: dict[str, Any], output_dir: Path, figures_dir: Path) -> dict[str, pd.DataFrame]:
    indices = bundle["indices_df"].copy()
    rows = []
    for index_name in INDEX_ORDER:
        median_col = f"{index_name}_median"
        stats = safe_spearman(indices[index_name], indices[median_col])
        rows.append(
            {
                "index_name": index_name,
                "n": stats["n"],
                "rho_mean_vs_median": stats["rho"],
                "p_value": stats["p_value"],
            }
        )
    mean_median_df = pd.DataFrame(rows)
    mean_median_df.to_csv(output_dir / "sensitivity_mean_vs_median.csv", index=False, encoding="utf-8")

    heatmap = pd.DataFrame(index=INDEX_ORDER, columns=["mean_vs_median"], dtype=float)
    annot = pd.DataFrame(index=INDEX_ORDER, columns=["mean_vs_median"], dtype=object)
    for _, row in mean_median_df.iterrows():
        heatmap.loc[row["index_name"], "mean_vs_median"] = row["rho_mean_vs_median"]
        annot.loc[row["index_name"], "mean_vs_median"] = f"{row['rho_mean_vs_median']:.2f}" if pd.notna(row["rho_mean_vs_median"]) else ""
    heatmap_path = figures_dir / "heatmap_mean_vs_median.png"
    _plot_heatmap(heatmap, annot, heatmap_path, title="Sensibilite moyenne vs mediane", cmap="viridis", vmin=0, vmax=1)

    return {"mean_median_df": mean_median_df, "heatmap_path": heatmap_path}


def _plot_heatmap(
    df: pd.DataFrame,
    annot: pd.DataFrame,
    path: Path,
    title: str,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    plot_annot = annot.copy()
    if len(plot_df.columns) > max(4, int(len(plot_df.index) * 1.5)):
        plot_df = plot_df.T
        plot_annot = plot_annot.T

    def _label(value: Any, width: int = 14) -> str:
        txt = str(value).replace("_", " ")
        return textwrap.fill(txt, width=width)

    n_rows, n_cols = plot_df.shape
    fig_w = min(8.0, max(5.2, 0.9 * max(2, n_cols) + 1.4))
    fig_h = min(10.5, max(4.2, 0.65 * max(2, n_rows) + 1.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_df.to_numpy(dtype=float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels([_label(c) for c in plot_df.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels([_label(i) for i in plot_df.index])
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            label = plot_annot.iloc[i, j] if i < plot_annot.shape[0] and j < plot_annot.shape[1] else ""
            if isinstance(label, str) and label:
                ax.text(j, i, label, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_index_scatter(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(len(INDEX_ORDER), 1, figsize=(7.0, 3.6 * len(INDEX_ORDER)))
    if len(INDEX_ORDER) == 1:
        axes = [axes]
    for ax, index_name in zip(axes, INDEX_ORDER):
        x = pd.to_numeric(df[index_name], errors="coerce")
        y = pd.to_numeric(df["Score_perf_tsk"], errors="coerce")
        sub = pd.DataFrame({"x": x, "y": y, "group_id": df["group_id"]}).dropna()
        ax.scatter(sub["x"], sub["y"], s=45, alpha=0.9)
        if len(sub) >= 2:
            coeffs = np.polyfit(sub["x"], sub["y"], deg=1)
            xx = np.linspace(sub["x"].min(), sub["x"].max(), 100)
            yy = coeffs[0] * xx + coeffs[1]
            ax.plot(xx, yy, color="black", linewidth=1.2)
        for _, row in sub.iterrows():
            ax.annotate(row["group_id"], (row["x"], row["y"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(index_name)
        ax.set_ylabel("Score_perf_tsk")
        ax.set_title(index_name)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_all_analyses(
    bundle: dict[str, Any] | None = None,
    output_dir: Path = OUTPUT_DIR,
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    bundle = bundle or build_behavioral_indices_pipeline(output_dir=output_dir, write_outputs=True)
    log = logger or bundle["log"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Le rapport ne distingue plus les scenarios S1/S2 : on retire toute
    # visualisation specifique scenario pour ne pas sur-interpreter une
    # difference non significative.
    legacy_scenario_figure = figures_dir / "indices_by_scenario.png"
    if legacy_scenario_figure.exists():
        legacy_scenario_figure.unlink()

    internal = analyze_internal_consistency(bundle, output_dir, log)
    questionnaire = analyze_questionnaire_convergence(bundle, output_dir, figures_dir, log)
    performance = analyze_performance(bundle, output_dir, figures_dir, log)
    configuration = analyze_configuration(bundle, performance, output_dir, log)
    sensitivity = analyze_sensitivity(bundle, performance, output_dir, figures_dir)

    log.write(output_dir / "log_indices.txt")
    return {
        "bundle": bundle,
        "internal": internal,
        "questionnaire": questionnaire,
        "performance": performance,
        "configuration": configuration,
        "sensitivity": sensitivity,
        "figures_dir": figures_dir,
    }


def main() -> None:
    results = run_all_analyses()
    print(f"[OK] Corr questionnaires : {OUTPUT_DIR / 'correlations_indices_questionnaires.csv'}")
    print(f"[OK] Corr performance   : {OUTPUT_DIR / 'correlations_indices_performance.csv'}")
    print(f"[OK] Cohérence interne : {OUTPUT_DIR / 'internal_consistency_indices.csv'}")
    print(f"[OK] Figures           : {results['figures_dir']}")


if __name__ == "__main__":
    main()
