# -*- coding: utf-8 -*-
"""
Helpers pour les analyses de fiabilité TMS / cohésion et les sections de
régression du rapport.
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import norm, shapiro, spearmanr, t as student_t
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from pathlib import Path as _Path

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))
_scripts_dir = _Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    COHESION_SCORE_COL,
    INV_FEATURES,
    PERFORMANCE_ANALYSIS_COLS,
    QUESTIONNAIRE_ANALYSIS_COLS,
    TMS_DIMENSIONS,
    COHESION_COMPONENTS,
    infer_family_from_name,
)

try:
    import statsmodels.api as sm  # noqa: F401
    from statsmodels.formula.api import ols
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import RepeatedKFold
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


QUESTIONNAIRE_ALPHA_THRESHOLD = 0.70


def compute_cronbach_alpha(items: np.ndarray) -> float:
    """Calcule l'alpha de Cronbach pour une matrice d'items."""
    items = np.asarray(items)
    if items.ndim != 2:
        return np.nan

    n_items = items.shape[1]
    if n_items < 2:
        return np.nan

    valid_rows = ~np.any(np.isnan(items), axis=1)
    items = items[valid_rows]
    if len(items) < 3:
        return np.nan

    item_variances = np.var(items, axis=0, ddof=1)
    total_scores = np.sum(items, axis=1)
    total_variance = np.var(total_scores, ddof=1)
    if total_variance == 0:
        return np.nan

    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return alpha


def compute_score_m1m3_and_alpha(perf: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Calcule le score Performance_M1M3 et son alpha."""
    m1_col = "M1_consignes_%"
    m3_col = "M3_precision_%"

    if m1_col not in perf.columns or m3_col not in perf.columns:
        return perf, np.nan

    m1 = pd.to_numeric(perf[m1_col], errors="coerce")
    m3 = pd.to_numeric(perf[m3_col], errors="coerce")

    perf = perf.copy()
    perf["Score_perf_M1M3"] = (m1 + m3) / 2

    items = np.column_stack([m1.values, m3.values])
    alpha = compute_cronbach_alpha(items)
    return perf, alpha


def compute_aggregated_scores(
    q_group: pd.DataFrame,
    *,
    alpha_threshold: float | None = None,
    include_tms_score: bool = True,
    include_cohesion_score: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Calcule les alphas TMS / cohésion et, optionnellement, leurs scores agrégés.

    Les scores latents ne sont créés que si le construit est suffisamment
    cohérent lorsque `alpha_threshold` est renseigné. Cette contrainte évite de
    propager en aval un score moyen pour un construit psychométriquement fragile.
    """
    if q_group is None or q_group.empty:
        return q_group, {"TMS": np.nan, "COHESION": np.nan}

    q_group = q_group.copy()
    alphas = {}

    tms_cols = [c for c in TMS_DIMENSIONS if c in q_group.columns]
    if len(tms_cols) >= 2:
        tms_values = q_group[tms_cols].apply(pd.to_numeric, errors="coerce")
        alphas["TMS"] = compute_cronbach_alpha(tms_values.values)
        if include_tms_score and (
            alpha_threshold is None or
            (not np.isnan(alphas["TMS"]) and alphas["TMS"] >= alpha_threshold)
        ):
            q_group["TMS_score"] = tms_values.mean(axis=1)
    else:
        alphas["TMS"] = np.nan

    coh_cols = [c for c in COHESION_COMPONENTS if c in q_group.columns]
    if len(coh_cols) >= 2:
        coh_values = q_group[coh_cols].apply(pd.to_numeric, errors="coerce")
        alphas["COHESION"] = compute_cronbach_alpha(coh_values.values)
        if include_cohesion_score and (
            alpha_threshold is None or
            (not np.isnan(alphas["COHESION"]) and alphas["COHESION"] >= alpha_threshold)
        ):
            q_group[COHESION_SCORE_COL] = coh_values.mean(axis=1)
    else:
        alphas["COHESION"] = np.nan

    return q_group, alphas


def analyze_tms_reliability(q_group: pd.DataFrame) -> dict:
    """Analyse approfondie de la fiabilité du TMS."""
    results = {
        "alpha_global": np.nan,
        "alpha_by_dimension": {},
        "correlations": {},
        "alpha_if_deleted": {},
        "recommendations": [],
        "n_valid": 0,
    }

    if q_group is None or q_group.empty:
        return results

    tms_cols = [c for c in TMS_DIMENSIONS if c in q_group.columns]
    if len(tms_cols) < 2:
        results["recommendations"].append("Données insuffisantes : moins de 2 dimensions TMS disponibles.")
        return results

    tms_values = q_group[tms_cols].apply(pd.to_numeric, errors="coerce")
    valid_rows = ~tms_values.isna().any(axis=1)
    tms_clean = tms_values[valid_rows]
    results["n_valid"] = len(tms_clean)

    if len(tms_clean) < 5:
        results["recommendations"].append(f"Échantillon trop petit (n={len(tms_clean)}) pour une analyse fiable.")
        return results

    items = tms_clean.values
    results["alpha_global"] = compute_cronbach_alpha(items)

    corr_matrix = tms_clean.corr(method="spearman")
    for i, col1 in enumerate(tms_cols):
        for col2 in tms_cols[i + 1:]:
            results["correlations"][f"{col1}-{col2}"] = corr_matrix.loc[col1, col2]

    for col in tms_cols:
        remaining_cols = [c for c in tms_cols if c != col]
        if len(remaining_cols) >= 2:
            sub_items = tms_clean[remaining_cols].values
            results["alpha_if_deleted"][col] = compute_cronbach_alpha(sub_items)

    alpha_global = results["alpha_global"]
    if np.isnan(alpha_global):
        results["recommendations"].append("Alpha global non calculable.")
        return results

    if alpha_global >= 0.70:
        results["recommendations"].append(f"Alpha global ({alpha_global:.3f}) acceptable (≥0.70).")
    elif alpha_global >= 0.60:
        results["recommendations"].append(f"Alpha global ({alpha_global:.3f}) limite (0.60-0.70). Interprétation prudente recommandée.")
    else:
        results["recommendations"].append(f"Alpha global ({alpha_global:.3f}) faible (<0.60). Fiabilité insuffisante.")

    problematic_dims = []
    for dim, alpha_without in results["alpha_if_deleted"].items():
        if alpha_without > alpha_global + 0.02:
            problematic_dims.append((dim, alpha_without))

    if problematic_dims:
        for dim, alpha_without in problematic_dims:
            results["recommendations"].append(
                f"Dimension problématique : {dim}. α si retirée = {alpha_without:.3f} (amélioration +{alpha_without - alpha_global:.3f})"
            )

    weak_correlations = []
    for pair, rho in results["correlations"].items():
        if abs(rho) < 0.30:
            weak_correlations.append((pair, rho))

    if weak_correlations:
        for pair, rho in weak_correlations:
            results["recommendations"].append(f"Corrélation faible : {pair} (ρ = {rho:.2f})")

    results["improvement_paths"] = []
    if alpha_global < 0.70:
        results["improvement_paths"].append(
            "1. Agréger au niveau de l'équipe (moyenne des 3 membres) pour réduire l'erreur de mesure."
        )
        results["improvement_paths"].append(
            "2. Utiliser la médiane plutôt que la moyenne pour robustifier face aux outliers."
        )
        if problematic_dims:
            results["improvement_paths"].append(
                f"3. Considérer retirer la dimension {problematic_dims[0][0]} de l'indice composite."
            )
        results["improvement_paths"].append(
            "4. Évaluer si les dimensions mesurent le même construit latent (analyse factorielle)."
        )
        results["improvement_paths"].append(
            "5. Augmenter la taille de l'échantillon pour stabiliser les estimations."
        )

    return results


def render_tms_reliability_analysis(
    lines: list[str],
    pdf_elems: list,
    styles,
    q_group: pd.DataFrame,
    section_num: str = "1.4.x",
):
    """Génère la section d'analyse de la fiabilité TMS dans le rapport."""
    title = f"{section_num} Analyse de fiabilité TMS"
    lines.append(f"\n### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    analysis = analyze_tms_reliability(q_group)

    alpha_text = f"**Alpha de Cronbach global** : {analysis['alpha_global']:.3f}" if not np.isnan(analysis["alpha_global"]) else "**Alpha de Cronbach** : non calculable"
    lines.append(f"{alpha_text} (n = {analysis['n_valid']})\n\n")
    alpha_html = alpha_text.replace("**", "<b>", 1).replace("**", "</b>", 1)
    pdf_elems.append(Paragraph(alpha_html + f" (n = {analysis['n_valid']})", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.05 * inch))

    if analysis["correlations"]:
        lines.append("**Corrélations inter-dimensions (Spearman) :**\n")
        pdf_elems.append(Paragraph("<b>Corrélations inter-dimensions :</b>", styles["Normal"]))
        corr_text = ", ".join([f"{pair}: ρ={rho:.2f}" for pair, rho in analysis["correlations"].items()])
        lines.append(f"{corr_text}\n\n")
        pdf_elems.append(Paragraph(corr_text, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))

    if analysis["alpha_if_deleted"]:
        lines.append("**Alpha si dimension retirée :**\n")
        pdf_elems.append(Paragraph("<b>Alpha si dimension retirée :</b>", styles["Normal"]))
        alpha_del_rows = [["Dimension retirée", "Alpha résultant", "Δα"]]
        for dim, alpha_without in analysis["alpha_if_deleted"].items():
            delta = alpha_without - analysis["alpha_global"] if not np.isnan(analysis["alpha_global"]) else np.nan
            alpha_del_rows.append([dim, f"{alpha_without:.3f}", f"{delta:+.3f}" if not np.isnan(delta) else "NA"])
            lines.append(f"- {dim} : α = {alpha_without:.3f} (Δ = {delta:+.3f})\n")

        alpha_del_table = Table(alpha_del_rows, colWidths=[1.8 * inch, 1.2 * inch, 0.8 * inch])
        alpha_del_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        pdf_elems.append(alpha_del_table)
        pdf_elems.append(Spacer(1, 0.1 * inch))

    # if analysis["recommendations"]:
    #     lines.append("\n**Diagnostic :**\n")
    #     pdf_elems.append(Paragraph("<b>Diagnostic :</b>", styles["Normal"]))
    #     for rec in analysis["recommendations"]:
    #         lines.append(f"- {rec}\n")
    #         pdf_elems.append(Paragraph(f"• {rec}", styles["Normal"]))
    #     pdf_elems.append(Spacer(1, 0.08 * inch))

    if analysis.get("improvement_paths"):
        lines.append("\n**Pistes d'amélioration :**\n")
        pdf_elems.append(Paragraph("<b>Pistes d'amélioration :</b>", styles["Normal"]))
        for path in analysis["improvement_paths"]:
            lines.append(f"- {path}\n")
            pdf_elems.append(Paragraph(f"• {path}", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    lines.append("\n")


def run_regression_analysis(
    df: pd.DataFrame,
    y_col: str,
    predictors: list[str],
    title: str = "Régression",
) -> dict | None:
    """Exécute une régression OLS et retourne les résultats."""
    if not HAS_STATSMODELS:
        print("  [SKIP] statsmodels non disponible pour la régression")
        return None

    if y_col not in df.columns:
        return None

    valid_preds = [p for p in predictors if p in df.columns]
    if not valid_preds:
        return None

    sub = df[[y_col] + valid_preds].dropna()
    if len(sub) < 5:
        return None

    formula_parts = []
    for p in valid_preds:
        if sub[p].dtype == "object" or sub[p].nunique() <= 5:
            formula_parts.append(f"C({p})")
        else:
            formula_parts.append(p)

    formula = f"{y_col} ~ " + " + ".join(formula_parts)

    try:
        model = ols(formula, data=sub).fit()
        results = {
            "title": title,
            "formula": formula,
            "n": len(sub),
            "r2": model.rsquared,
            "r2_adj": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "coefficients": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "conf_int": model.conf_int().to_dict(),
        }

        try:
            from statsmodels.stats.anova import anova_lm
            anova_table = anova_lm(model, typ=2)
            results["anova"] = anova_table.to_dict()

            ss_resid = anova_table["sum_sq"].get("Residual", 0)
            partial_r2 = {}
            for idx in anova_table.index:
                if idx != "Residual":
                    ss_effect = anova_table.loc[idx, "sum_sq"]
                    if ss_effect + ss_resid > 0:
                        partial_r2[idx] = ss_effect / (ss_effect + ss_resid)
            results["partial_r2"] = partial_r2
        except Exception:
            pass

        return results
    except Exception as e:
        print(f"  [WARN] Régression {title} : {e}")
        return None


def render_regression_section(
    lines: list[str],
    pdf_elems: list,
    styles,
    df: pd.DataFrame,
    section_num: str = "1.3c",
):
    """Génère la section des analyses de régression sur la performance."""
    title = f"{section_num} Analyses de régression — Performance"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    if not HAS_STATSMODELS:
        msg = "(Analyses de régression indisponibles — statsmodels non installé)"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    if df is None or df.empty:
        msg = "(Données de performance indisponibles)"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    df = df.copy()
    for cand in ["condition", "modalite", "modality"]:
        if cand in df.columns:
            df["modalite"] = df[cand].astype(str).str.upper().str.strip()
            break

    for cand in ["scenario", "Scenario"]:
        if cand in df.columns:
            df["scenario"] = df[cand].astype(str).str.upper().str.strip()
            break

    perf_cols = [cand for cand in ["Score_perf_tsk", "Score_perf_tsk_mean", "Score_perf_M1M3"] if cand in df.columns]
    predictors = ["modalite", "scenario"]

    all_results = []
    for y_col in perf_cols[:1]:
        res = run_regression_analysis(df, y_col, predictors, f"Performance ({y_col})")
        if res:
            all_results.append(res)

    if not all_results:
        msg = "(Aucune régression calculée — données insuffisantes)"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    for res in all_results:
        lines.append(f"**{res['title']}**\n\n")
        pdf_elems.append(Paragraph(f"<b>{res['title']}</b>", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))

        stats_text = (
            f"n = {res['n']}, "
            f"R² = {res['r2']:.3f}, "
            f"R² ajusté = {res['r2_adj']:.3f}, "
            f"F = {res['f_stat']:.2f}, "
            f"p = {res['f_pvalue']:.4f}"
        )
        lines.append(f"{stats_text}\n\n")
        pdf_elems.append(Paragraph(stats_text, styles["Heading4"]))

        coef_rows = [["Prédicteur", "Coefficient", "p-value"]]
        for pred, coef in res["coefficients"].items():
            pval = res["pvalues"].get(pred, np.nan)
            coef_rows.append([
                pred.replace("C(", "").replace(")", "").replace("[T.", " vs "),
                f"{coef:.3f}",
                f"{pval:.4f}" if not np.isnan(pval) else "NA",
            ])

        lines.append("| Prédicteur | Coefficient | p-value |\n|---|---|---|\n")
        for row in coef_rows[1:]:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        lines.append("\n")

        pdf_table = Table(coef_rows, colWidths=[2.5 * inch, 1.2 * inch, 1.2 * inch])
        pdf_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        pdf_elems.append(pdf_table)
        pdf_elems.append(Spacer(1, 0.1 * inch))

        if "partial_r2" in res and res["partial_r2"]:
            lines.append("**Variance absorbée (R² partiel) :**\n\n")
            pdf_elems.append(Paragraph("<b>Variance absorbée (R² partiel) :</b>", styles["Heading4"]))

            pr2_rows = [["Effet", "R² partiel", "% variance"]]
            for effect, pr2 in res["partial_r2"].items():
                pr2_rows.append([
                    effect.replace("C(", "").replace(")", ""),
                    f"{pr2:.4f}",
                    f"{pr2*100:.1f}%",
                ])

            lines.append("| Effet | R² partiel | % variance |\n|---|---|---|\n")
            for row in pr2_rows[1:]:
                lines.append(f"| {row[0]} | {row[1]} | {row[2]} |\n")
            lines.append("\n")

            pdf_table2 = Table(pr2_rows, colWidths=[2.0 * inch, 1.2 * inch, 1.2 * inch])
            pdf_table2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            pdf_elems.append(pdf_table2)
            pdf_elems.append(Spacer(1, 0.15 * inch))

    lines.append("\n")


def plot_regression_tms_cohesion(
    df: pd.DataFrame,
    fig_path: Path,
    y_col: str,
    y_label: str,
    tms_col: str = "TMS_mean",
    cohesion_col: str = "Cohesion_mean",
) -> bool:
    """Génère un graphique de régression bivariée TMS / Cohésion → outcome."""
    if df is None or df.empty or y_col not in df.columns:
        return False

    has_tms = tms_col in df.columns
    has_cohesion = cohesion_col in df.columns
    if not has_tms and not has_cohesion:
        return False

    df_work = df.copy()
    cols_needed = [y_col]
    if has_tms:
        cols_needed.append(tms_col)
    if has_cohesion:
        cols_needed.append(cohesion_col)

    df_valid = df_work[cols_needed].dropna()
    if len(df_valid) < 5:
        return False

    n_plots = int(has_tms) + int(has_cohesion)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0
    if has_tms:
        ax = axes[ax_idx]
        x = df_valid[tms_col].values
        y = df_valid[y_col].values
        ax.scatter(x, y, alpha=0.7, s=50, c="#3498db", edgecolors="white", linewidth=0.5)
        if len(x) >= 3:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.8)
                rho, pval = spearmanr(x, y)
                r2 = rho ** 2
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                ax.text(
                    0.05, 0.95, f"ρ = {rho:.2f}{sig}\nR² = {r2:.2f}\nn = {len(x)}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            except Exception:
                pass
        ax.set_xlabel("TMS (moyenne)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f"TMS → {y_label}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    if has_cohesion:
        ax = axes[ax_idx]
        x = df_valid[cohesion_col].values
        y = df_valid[y_col].values
        ax.scatter(x, y, alpha=0.7, s=50, c="#2ecc71", edgecolors="white", linewidth=0.5)
        if len(x) >= 3:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.8)
                rho, pval = spearmanr(x, y)
                r2 = rho ** 2
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                ax.text(
                    0.05, 0.95, f"ρ = {rho:.2f}{sig}\nR² = {r2:.2f}\nn = {len(x)}",
                    transform=ax.transAxes, fontsize=9, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
            except Exception:
                pass
        ax.set_xlabel("Cohésion (moyenne)", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(f"Cohésion → {y_label}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    return True


def render_regression_graphs_section(
    lines: list[str],
    pdf_elems: list,
    styles,
    df_merged: pd.DataFrame,
    fig_dir: Path,
    section_num: str = "2.5",
):
    """Génère la section des graphiques TMS/Cohésion → Performance et C-factor."""
    if df_merged is None or df_merged.empty:
        title = f"{section_num} Régressions questionnaire → Performance"
        lines.append(f"\n### {title}\n")
        pdf_elems.append(Paragraph(title, styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))
        msg = "(Données fusionnées indisponibles pour les régressions)"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    df = df_merged.copy()
    if "TMS_mean" not in df.columns and "TMS_score" in df.columns:
        df["TMS_mean"] = pd.to_numeric(df["TMS_score"], errors="coerce")

    if "Cohesion_mean" not in df.columns:
        if COHESION_SCORE_COL in df.columns:
            df["Cohesion_mean"] = pd.to_numeric(df[COHESION_SCORE_COL], errors="coerce")
        elif "COHESION_score" in df.columns:
            df["Cohesion_mean"] = pd.to_numeric(df["COHESION_score"], errors="coerce")

    has_tms = "TMS_mean" in df.columns and df["TMS_mean"].notna().any()
    has_cohesion = "Cohesion_mean" in df.columns and df["Cohesion_mean"].notna().any()
    if has_tms and has_cohesion:
        title = f"{section_num} Régressions TMS/Cohésion → Performance"
    elif has_cohesion:
        title = f"{section_num} Régressions Cohésion → Performance"
    elif has_tms:
        title = f"{section_num} Régressions TMS → Performance"
    else:
        title = f"{section_num} Régressions questionnaire → Performance"

    lines.append(f"\n### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading4"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    perf_col = next((cand for cand in ["Score_perf_tsk", "Score_perf_tsk_mean", "Score_perf_M1M3", "performance"] if cand in df.columns), None)
    cfactor_col = next((cand for cand in ["C_score", "c_score", "C_factor", "c_factor"] if cand in df.columns), None)

    if has_tms and has_cohesion:
        note = (
            "Ces graphiques montrent les relations bivariées entre les scores "
            "questionnaire agrégés disponibles (TMS, cohésion) et les outcomes "
            "(Performance de la tâche, C-factor de l'intelligence collective)."
        )
    elif has_cohesion:
        note = (
            "Ces graphiques montrent les relations bivariées entre le score "
            "agrégé de cohésion questionnaire et les outcomes "
            "(Performance de la tâche, C-factor de l'intelligence collective)."
        )
    elif has_tms:
        note = (
            "Ces graphiques montrent les relations bivariées entre le score "
            "agrégé TMS disponible et les outcomes "
            "(Performance de la tâche, C-factor de l'intelligence collective)."
        )
    else:
        note = (
            "Aucun score questionnaire agrégé exploitable n'est disponible pour "
            "les graphiques de régression."
        )
    lines.append(f"{note}\n\n")
    pdf_elems.append(Paragraph(note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    graphs_generated = []
    if perf_col is not None:
        fig_perf = fig_dir / "regression_tms_cohesion_performance.png"
        ok_perf = plot_regression_tms_cohesion(df, fig_perf, y_col=perf_col, y_label="Performance")
        if ok_perf:
            lines.append("**TMS / Cohésion → Performance**\n\n")
            lines.append(f"![]({fig_perf.name})\n\n")
            pdf_elems.append(Paragraph("<b>TMS / Cohésion → Performance</b>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(Image(str(fig_perf), width=5.8 * inch, height=3.2 * inch))
            pdf_elems.append(Spacer(1, 0.15 * inch))
            graphs_generated.append("Performance")

    if cfactor_col is not None:
        fig_cfactor = fig_dir / "regression_tms_cohesion_cfactor.png"
        ok_cfactor = plot_regression_tms_cohesion(df, fig_cfactor, y_col=cfactor_col, y_label="C-factor")
        if ok_cfactor:
            lines.append("**TMS / Cohésion → C-factor**\n\n")
            lines.append(f"![]({fig_cfactor.name})\n\n")
            pdf_elems.append(Paragraph("<b>TMS / Cohésion → C-factor</b>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(Image(str(fig_cfactor), width=5.8 * inch, height=3.2 * inch))
            pdf_elems.append(Spacer(1, 0.15 * inch))
            graphs_generated.append("C-factor")

    if not graphs_generated:
        msg = "(Graphiques de régression non générés — données insuffisantes)"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
    else:
        print(f"  [OK] Graphiques générés pour : {', '.join(graphs_generated)}")


def _qq_correlation(residuals: np.ndarray) -> float:
    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if len(resid) < 3:
        return np.nan
    ordered = np.sort(resid)
    theoretical = norm.ppf((np.arange(1, len(ordered) + 1) - 0.5) / len(ordered))
    if np.std(ordered) == 0 or np.std(theoretical) == 0:
        return np.nan
    return float(np.corrcoef(ordered, theoretical)[0, 1])


def compute_model_diagnostics(model) -> dict[str, float | bool]:
    resid = np.asarray(model.resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    shapiro_p = np.nan
    bp_p = np.nan
    qq_corr = _qq_correlation(resid)

    if len(resid) >= 3:
        try:
            shapiro_p = float(shapiro(resid).pvalue)
        except Exception:
            shapiro_p = np.nan

    try:
        bp_p = float(het_breuschpagan(model.resid, model.model.exog)[3])
    except Exception:
        bp_p = np.nan

    bp_ok = (not np.isfinite(bp_p)) or bp_p > 0.05
    assumptions_ok = (
        np.isfinite(shapiro_p) and shapiro_p > 0.05
        and bp_ok
        and np.isfinite(qq_corr) and qq_corr >= 0.90
    )
    return {
        "shapiro_p": shapiro_p,
        "breusch_pagan_p": bp_p,
        "qq_corr": qq_corr,
        "assumptions_ok": assumptions_ok,
    }


def _fit_ols_subset(df: pd.DataFrame, y_col: str, feature_cols: list[str]):
    sub = df[[y_col] + feature_cols].dropna().copy()
    if len(sub) < max(8, len(feature_cols) + 3):
        return None, pd.DataFrame()
    if HAS_STATSMODELS:
        X = sm.add_constant(sub[feature_cols], has_constant="add")
        model = sm.OLS(sub[y_col], X).fit()
        return model, sub

    # Fallback autonome quand statsmodels n'est pas disponible dans l'environnement.
    # Il reproduit l'OLS standard nécessaire à la sélection stepwise.
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    X_raw = sub[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(X_raw).all():
        return None, pd.DataFrame()
    X = np.column_stack([np.ones(len(sub)), X_raw])
    n_obs, n_params = X.shape
    df_resid = n_obs - n_params
    if df_resid <= 0:
        return None, pd.DataFrame()

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    sse = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    if tss <= 0:
        return None, pd.DataFrame()

    mse = sse / df_resid
    cov_beta = mse * np.linalg.pinv(X.T @ X)
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    t_vals = np.divide(beta, se_beta, out=np.full_like(beta, np.nan), where=se_beta > 0)
    p_vals = 2.0 * (1.0 - student_t.cdf(np.abs(t_vals), df_resid))
    r2 = 1.0 - (sse / tss)
    r2_adj = 1.0 - ((1.0 - r2) * (n_obs - 1) / df_resid)

    k_predictors = n_params - 1
    if k_predictors > 0 and (1.0 - r2) > 0:
        f_value = (r2 / k_predictors) / ((1.0 - r2) / df_resid)
        f_pvalue = 1.0 - f_dist.cdf(f_value, k_predictors, df_resid)
    else:
        f_pvalue = np.nan

    index = ["const"] + list(feature_cols)
    model = SimpleNamespace(
        resid=resid,
        model=SimpleNamespace(exog=X),
        params=pd.Series(beta, index=index),
        pvalues=pd.Series(p_vals, index=index),
        rsquared=float(r2),
        rsquared_adj=float(r2_adj),
        f_pvalue=float(f_pvalue) if np.isfinite(f_pvalue) else np.nan,
    )
    return model, sub


def _cross_validated_metrics(df: pd.DataFrame, y_col: str, feature_cols: list[str]) -> tuple[float, float, float]:
    if not HAS_SKLEARN:
        return np.nan, np.nan, np.nan
    sub = df[[y_col] + feature_cols].dropna().copy()
    if len(sub) < 8:
        return np.nan, np.nan, np.nan

    X = sub[feature_cols].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    rkf = RepeatedKFold(n_splits=min(5, len(sub)), n_repeats=10, random_state=42)
    scores: list[float] = []
    rmses: list[float] = []
    for train_idx, test_idx in rkf.split(X):
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) < 2:
            continue
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        scores.append(float(model.score(X[test_idx], y[test_idx])))
        rmses.append(float(np.sqrt(np.mean((y[test_idx] - y_pred) ** 2))))
    if not scores:
        return np.nan, np.nan, np.nan
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(rmses)) if rmses else np.nan


def forward_stepwise_inv_models(
    df: pd.DataFrame,
    y_col: str,
    candidate_cols: list[str],
    max_features: int = 4,
    p_enter: float = 0.05,
) -> list[dict[str, Any]]:
    """Sélection forward stepwise OLS.

    p_enter : seuil p pour qu'un prédicteur reste dans le modèle (chaque prédicteur
    doit vérifier p < p_enter). Défaut 0.05 (M9, revue : harmonisé avec la
    régression sur scores PCA, qui utilisait déjà 0.05).
    """
    selected: list[str] = []
    remaining = [c for c in candidate_cols if c in df.columns]
    results: list[dict[str, Any]] = []

    while remaining and len(selected) < max_features:
        best_entry: dict[str, Any] | None = None
        best_feature: str | None = None

        for feature in remaining:
            feats = selected + [feature]
            model, sub = _fit_ols_subset(df, y_col, feats)
            if model is None or sub.empty:
                continue
            pvals = model.pvalues.drop(labels=["const"], errors="ignore")
            if pvals.empty or (pvals >= p_enter).any():
                continue
            params = model.params.drop(labels=["const"], errors="ignore")
            y_sd = float(pd.to_numeric(sub[y_col], errors="coerce").std(ddof=1))
            standardized_coefficients: dict[str, float] = {}
            for pred, coef in params.to_dict().items():
                x_sd = float(pd.to_numeric(sub[pred], errors="coerce").std(ddof=1))
                standardized_coefficients[pred] = (
                    float(coef) * x_sd / y_sd
                    if np.isfinite(coef) and np.isfinite(x_sd) and np.isfinite(y_sd) and y_sd > 0
                    else np.nan
                )
            diag = compute_model_diagnostics(model)
            cv_mean, cv_sd, cv_rmse = _cross_validated_metrics(df, y_col, feats)
            entry = {
                "n_features": len(feats),
                "features": feats,
                "r2": float(model.rsquared),
                "r2_adj": float(model.rsquared_adj),
                "model_p": float(model.f_pvalue) if np.isfinite(model.f_pvalue) else np.nan,
                "predictor_pvalues": {k: float(v) for k, v in pvals.to_dict().items()},
                "predictor_coefficients": {k: float(v) for k, v in params.to_dict().items()},
                "predictor_betas_std": standardized_coefficients,
                "n_obs": int(len(sub)),
                "cv_r2_mean": cv_mean,
                "cv_r2_sd": cv_sd,
                "cv_rmse_mean": cv_rmse,
                **diag,
            }
            if best_entry is None or entry["r2"] > best_entry["r2"]:
                best_entry = entry
                best_feature = feature

        if best_entry is None or best_feature is None:
            break

        selected.append(best_feature)
        remaining = [c for c in remaining if c != best_feature]
        results.append(best_entry)

    return results


def _plot_inv_model_scatter(
    df: pd.DataFrame,
    dv: str,
    features: list[str],
    model_num: int,
    fig_path: Path,
) -> bool:
    """Scatter plots feature → DV pour un modèle stepwise (une colonne par feature)."""
    n_feats = len(features)
    if n_feats == 0:
        return False
    valid_features = [f for f in features if f in df.columns and dv in df.columns]
    if not valid_features:
        return False

    fig, axes = plt.subplots(1, n_feats, figsize=(4.2 * n_feats, 4.0), squeeze=False)
    axes = axes[0]

    colors_seq = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    for i, feat in enumerate(features):
        ax = axes[i]
        if feat not in df.columns:
            ax.set_visible(False)
            continue
        sub = df[[feat, dv]].dropna()
        if len(sub) < 4:
            ax.set_visible(False)
            continue
        x = sub[feat].values.astype(float)
        y = sub[dv].values.astype(float)
        c = colors_seq[i % len(colors_seq)]
        ax.scatter(x, y, alpha=0.75, s=55, c=c, edgecolors="white", linewidth=0.6, zorder=3)
        try:
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color="#e74c3c", linewidth=1.8, alpha=0.85, zorder=2)
            rho, pval = spearmanr(x, y)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            ax.text(
                0.05, 0.95,
                f"ρ = {rho:.2f}{sig}\nR² = {rho**2:.2f}\nn = {len(x)}",
                transform=ax.transAxes, fontsize=8.5, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            )
        except Exception:
            pass
        short_feat = feat.replace("audio_", "").replace("face_", "").replace("gaze_", "")
        ax.set_xlabel(short_feat, fontsize=9, labelpad=4)
        ax.set_ylabel(dv, fontsize=9, labelpad=4)
        ax.set_title(f"M{model_num} : {short_feat} → {dv}", fontsize=9, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.8)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def render_inv_stepwise_regression_section(
    lines: list[str],
    pdf_elems: list,
    styles,
    df: pd.DataFrame,
    predictor_groups: dict[str, list[str]],
    section_num: str = "3.5",
    figs_dir: "Path | None" = None,
) -> set[str]:
    title = f"{section_num} Régressions multiples stepwise sur les INV pruned"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    if df is None or df.empty:
        msg = "Données insuffisantes pour les régressions INV."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    # Les dimensions de cohésion sont traitées comme VD séparées, au même titre
    # que les dimensions TMS ; le score global de cohésion reste conservé.
    dv_candidates = (
        ["Score_perf_tsk", COHESION_SCORE_COL]
        + TMS_DIMENSIONS
        + [c for c in COHESION_COMPONENTS if c not in TMS_DIMENSIONS]
    )
    dv_cols = [c for c in dv_candidates if c in df.columns]
    if not dv_cols:
        msg = "Aucune variable dépendante exploitable pour les régressions INV."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    intro = (
        "Pour chaque variable dépendante, les modèles sont estimés par sélection "
        "pas-à-pas avant (1 à 4 prédicteurs) sur les features INV pruned, séparés "
        "par modalité ou combinaison de modalités. Parmi les modèles calculés, seuls "
        "sont retenus ceux pour lesquels chaque prédicteur vérifie p < 0.05 (seuil "
        "harmonisé avec la régression sur scores PCA) ; pour "
        "chaque nombre de prédicteurs, le modèle affiché est celui qui présente le "
        "R² le plus élevé. Les diagnostics de résidus et la validation croisée "
        "répétée 10-run 5-fold sont rapportés pour ces modèles retenus."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    def _modality_abbrev(label: str) -> str:
        mapping = {
            "Audio/Speech": "A",
            "Face": "F",
            "Gaze": "G",
            "Audio/Speech + Face": "A+F",
            "Audio/Speech + Gaze": "A+G",
            "Face + Gaze": "F+G",
            "Audio/Speech + Face + Gaze": "A+F+G",
        }
        return mapping.get(label, label)

    def _modalities_from_features(features: list[str]) -> str:
        fam_to_abbr = {"audio": "A", "face": "F", "gaze": "G"}
        ordered = ["A", "F", "G"]
        seen: set[str] = set()
        for feat in features:
            fam = infer_family_from_name(feat)
            abbr = fam_to_abbr.get(str(fam).lower()) if fam is not None else None
            if abbr:
                seen.add(abbr)
        actual = [abbr for abbr in ordered if abbr in seen]
        return "+".join(actual) if actual else "?"

    def _sig_stars(p: float) -> str:
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.10:
            return "†"  # tendance
        return ""

    def _format_pvalue(p: float) -> str:
        if not np.isfinite(p):
            return "NA"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _format_feature_list(features: list[str], pvals: dict[str, float]) -> str:
        parts = []
        for feat in features:
            p = pvals.get(feat, np.nan)
            parts.append(f"{feat}{_sig_stars(p)}")
        return "; ".join(parts)

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

    def _format_sign_magnitude(features: list[str], betas: dict[str, float]) -> str:
        parts: list[str] = []
        for feat in features:
            beta = betas.get(feat, np.nan)
            if not np.isfinite(beta):
                parts.append(f"{feat}: NA")
                continue
            sign = "+" if beta >= 0 else "-"
            parts.append(f"{feat}: {sign} {_effect_magnitude(beta)} (βstd={beta:.2f})")
        return "; ".join(parts)

    def _feature_nom_and_method(feature: str) -> tuple[str, str, str]:
        meta = INV_FEATURES.get(str(feature), {})
        description = str(meta.get("description", "")).strip()
        calc_method = str(meta.get("calc_method", "")).strip()
        reference = str(meta.get("reference", "")).strip()
        if description:
            name = f"{description} ({feature})"
        else:
            name = str(feature)
        if not calc_method:
            calc_method = "Méthode de calcul non documentée dans la config."
        return name, calc_method, reference

    def _feature_stats_rows(features: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feat in features:
            if feat not in df.columns:
                continue
            vals = pd.to_numeric(df[feat], errors="coerce").dropna()
            if vals.empty:
                continue
            name, calc_method, reference = _feature_nom_and_method(feat)
            rows.append({
                "Nom": name,
                "Référence": reference,
                "Méthode de calcul": calc_method,
                "Mean": round(float(vals.mean()), 3),
                "SD": round(float(vals.std(ddof=1)), 3) if len(vals) >= 2 else np.nan,
                "Min": round(float(vals.min()), 3),
                "Max": round(float(vals.max()), 3),
            })
        return rows

    base_feature_rows: list[dict[str, Any]] = []
    base_feature_specs: list[tuple[str, list[str]]] = []
    for base_name in ["Audio/Speech", "Face", "Gaze"]:
        features = [f for f in predictor_groups.get(base_name, []) if f in df.columns]
        if features:
            base_feature_specs.append((base_name, features))
            base_feature_rows.append({
                "Modalité INV": _modality_abbrev(base_name),
                "Bloc": base_name,
                "n": len(features),
                "Features analysées": ", ".join(features),
            })

    if base_feature_specs:
        desc_cell_style = styles["BodyText"].clone("InvDescCell")
        desc_cell_style.fontName = "Helvetica"
        desc_cell_style.fontSize = 6.5
        desc_cell_style.leading = 7.5
        desc_cell_style.wordWrap = "CJK"

        num_cell_style = styles["BodyText"].clone("InvDescNumCell")
        num_cell_style.fontName = "Helvetica"
        num_cell_style.fontSize = 6.5
        num_cell_style.leading = 7.5

        lines.append(f"#### {section_num}.1 Variables utilisées dans la régression INV\n\n")
        lines.append(
            "Le tableau ci-dessous résume les variables effectivement utilisées dans la régression stepwise, "
            "avec leur méthode de calcul documentée dans la configuration centrale des INV et leurs statistiques descriptives "
            "sur le sous-ensemble VR analysé.\n\n"
        )
        pdf_elems.append(Paragraph(f"{section_num}.1 Variables utilisées dans la régression INV", styles["Heading4"]))
        pdf_elems.append(
            Paragraph(
                "Le tableau ci-dessous résume les variables effectivement utilisées dans la régression stepwise, "
                "avec leur méthode de calcul documentée dans la configuration centrale des INV et leurs statistiques descriptives "
                "sur le sous-ensemble VR analysé.",
                styles["Normal"],
            )
        )
        pdf_elems.append(Spacer(1, 0.08 * inch))

        for base_name, features in base_feature_specs:
            feature_stats = _feature_stats_rows(features)
            if not feature_stats:
                continue
            stats_df = pd.DataFrame(feature_stats)
            lines.append(f"##### {base_name}\n\n")
            lines.append(stats_df.to_markdown(index=False) + "\n\n")
            pdf_elems.append(Paragraph(base_name, styles["Heading4"]))
            table_rows = [list(stats_df.columns)]
            for _, row in stats_df.iterrows():
                table_rows.append([
                    Paragraph(str(row["Nom"]), desc_cell_style),
                    Paragraph(str(row["Référence"]), desc_cell_style),
                    Paragraph(str(row["Méthode de calcul"]), desc_cell_style),
                    Paragraph("" if pd.isna(row["Mean"]) else f"{row['Mean']}", num_cell_style),
                    Paragraph("" if pd.isna(row["SD"]) else f"{row['SD']}", num_cell_style),
                    Paragraph("" if pd.isna(row["Min"]) else f"{row['Min']}", num_cell_style),
                    Paragraph("" if pd.isna(row["Max"]) else f"{row['Max']}", num_cell_style),
                ])
            stats_tbl = Table(
                table_rows,
                repeatRows=1,
                colWidths=[1.6 * inch, 1.4 * inch, 2.6 * inch, 0.45 * inch, 0.45 * inch, 0.45 * inch, 0.45 * inch],
            )
            stats_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            pdf_elems.append(stats_tbl)
            pdf_elems.append(Spacer(1, 0.08 * inch))

    if base_feature_rows:
        n_base_features = int(sum(int(row["n"]) for row in base_feature_rows))
        cell_style = styles["BodyText"].clone("InvFeatureCell")
        cell_style.fontName = "Helvetica"
        cell_style.fontSize = 7
        cell_style.leading = 8
        cell_style.wordWrap = "CJK"

        lines.append(f"#### {section_num}.2 INV analysés avant la régression\n\n")
        lines.append(
            "Les modèles combinés (`A+F`, `A+G`, `F+G`, `A+F+G`) sont construits comme unions de ces blocs de variables.\n\n"
        )
        lines.append(
            f"_Important : cette régression v2 repose ici sur **{n_base_features} features prunées** "
            "effectivement disponibles dans les tables `Audio/Speech`, `Face` et `Gaze`, après application "
            "des exclusions analytiques v2. Ce jeu peut différer de l'espace PCA, qui reste plus large "
            "et plus descriptif._\n\n"
        )
        feature_df = pd.DataFrame(base_feature_rows)
        lines.append(feature_df.to_markdown(index=False) + "\n\n")
        pdf_elems.append(Paragraph(f"{section_num}.2 INV analysés avant la régression", styles["Heading4"]))
        pdf_elems.append(
            Paragraph(
                "Les modèles combinés (A+F, A+G, F+G, A+F+G) sont construits comme unions de ces blocs de variables.",
                styles["Normal"],
            )
        )
        pdf_elems.append(
            Paragraph(
                f"<i>Important : cette régression v2 repose ici sur <b>{n_base_features} features prunées</b> "
                "effectivement disponibles dans les tables Audio/Speech, Face et Gaze, après application "
                "des exclusions analytiques v2. Ce jeu peut différer de l'espace PCA, qui reste plus large "
                "et plus descriptif.</i>",
                styles["Normal"],
            )
        )
        pdf_elems.append(Spacer(1, 0.08 * inch))
        feature_rows = [list(feature_df.columns)]
        for _, row in feature_df.iterrows():
            feature_rows.append([
                Paragraph(str(row["Modalité INV"]), cell_style),
                Paragraph(str(row["Bloc"]), cell_style),
                Paragraph(str(row["n"]), cell_style),
                Paragraph(str(row["Features analysées"]).replace(", ", ",<br/>"), cell_style),
            ])
        feature_tbl = Table(
            feature_rows,
            repeatRows=1,
            colWidths=[0.7 * inch, 1.2 * inch, 0.38 * inch, 4.72 * inch],
        )
        feature_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (2, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        pdf_elems.append(feature_tbl)
        pdf_elems.append(Spacer(1, 0.1 * inch))

    model_counter = 1
    table_lines: list[str] = []
    pdf_tables: list[Any] = []
    retained_features: set[str] = set()

    for dv in dv_cols:
        model_cell_style = styles["BodyText"].clone(f"InvRegCell_{dv}")
        model_cell_style.fontName = "Helvetica"
        model_cell_style.fontSize = 7
        model_cell_style.leading = 8
        model_cell_style.wordWrap = "CJK"

        rows: list[dict[str, Any]] = []
        for group_name, features in predictor_groups.items():
            features = [f for f in features if f in df.columns]
            if not features:
                continue
            for model_info in forward_stepwise_inv_models(df, dv, features, max_features=4):
                rows.append({
                    "Modalities": _modalities_from_features(model_info["features"]),
                    "Search_pool": _modality_abbrev(group_name),
                    "# Features": model_info["n_features"],
                    "Significant features": _format_feature_list(model_info["features"], model_info["predictor_pvalues"]),
                    "Signe / magnitude": _format_sign_magnitude(
                        model_info["features"],
                        model_info.get("predictor_betas_std", {}),
                    ),
                    "R2": round(model_info["r2"], 3),
                    "p-value": _format_pvalue(model_info["model_p"]),
                    "RMSE": round(model_info["cv_rmse_mean"], 3) if np.isfinite(model_info["cv_rmse_mean"]) else np.nan,
                    "R2_CV_mean": round(model_info["cv_r2_mean"], 3) if np.isfinite(model_info["cv_r2_mean"]) else np.nan,
                    "R2_CV_sd": round(model_info["cv_r2_sd"], 3) if np.isfinite(model_info["cv_r2_sd"]) else np.nan,
                    "Shapiro_p": round(model_info["shapiro_p"], 4) if np.isfinite(model_info["shapiro_p"]) else np.nan,
                    "BP_p": round(model_info["breusch_pagan_p"], 4) if np.isfinite(model_info["breusch_pagan_p"]) else np.nan,
                    "QQ_r": round(model_info["qq_corr"], 3) if np.isfinite(model_info["qq_corr"]) else np.nan,
                    "OLS_ok": "yes" if model_info["assumptions_ok"] else "no",
                    "n": model_info["n_obs"],
                    "feature_signature": tuple(model_info["features"]),
                })

        if not rows:
            table_lines.append(f"**{dv}**\n\n_Aucun modèle stepwise significatif._\n\n")
            pdf_tables.append(Paragraph(dv, styles["Heading4"]))
            pdf_tables.append(Paragraph("Aucun modèle stepwise significatif.", styles["Normal"]))
            pdf_tables.append(Spacer(1, 0.08 * inch))
            continue

        res_df = pd.DataFrame(rows)
        res_df = res_df.sort_values(["# Features", "R2", "Modalities", "Search_pool"], ascending=[True, False, True, True]).reset_index(drop=True)
        res_df = res_df.drop_duplicates(subset=["# Features", "feature_signature"], keep="first").reset_index(drop=True)
        best_by_size = res_df.groupby("# Features", dropna=False).head(1).reset_index(drop=True)
        best_by_size.insert(0, "Model", range(model_counter, model_counter + len(best_by_size)))
        model_counter += len(best_by_size)

        # Accumuler les features de tous les meilleurs modèles retenus
        for feat_sig in best_by_size["feature_signature"]:
            if feat_sig:
                retained_features.update(feat_sig)

        display_df = best_by_size[
            ["Model", "Modalities", "# Features", "Significant features", "Signe / magnitude", "R2", "p-value", "RMSE"]
        ].copy()
        display_df = display_df.rename(columns={"Modalities": "Modalities"})
        table_lines.append(f"**{dv}**\n\n")
        table_lines.append(display_df.to_markdown(index=False) + "\n\n")
        table_lines.append(
            "_Diagnostics_: "
            + "; ".join(
                [
                    f"M{int(row['Model'])}: n={int(row_n['n'])}, SW p={row_n['Shapiro_p']}, BP p={row_n['BP_p']}, QQ r={row_n['QQ_r']}, OLS={row_n['OLS_ok']}, R² CV={row_n['R2_CV_mean']}±{row_n['R2_CV_sd']}"
                    for (_, row), (_, row_n) in zip(display_df.iterrows(), best_by_size.iterrows())
                ]
            )
            + "\n\n"
        )

        pdf_tables.append(Paragraph(dv, styles["Heading4"]))
        table_rows = [list(display_df.columns)]
        for _, row in display_df.iterrows():
            table_rows.append([
                Paragraph(str(row["Model"]), model_cell_style),
                Paragraph(str(row["Modalities"]), model_cell_style),
                Paragraph(str(row["# Features"]), model_cell_style),
                Paragraph(str(row["Significant features"]).replace("; ", ";<br/>"), model_cell_style),
                Paragraph(str(row["Signe / magnitude"]).replace("; ", ";<br/>"), model_cell_style),
                Paragraph(str(row["R2"]), model_cell_style),
                Paragraph(str(row["p-value"]), model_cell_style),
                Paragraph(str(row["RMSE"]), model_cell_style),
            ])
        tbl = Table(
            table_rows,
            repeatRows=1,
            colWidths=[0.45 * inch, 0.7 * inch, 0.6 * inch, 2.05 * inch, 2.2 * inch, 0.45 * inch, 0.65 * inch, 0.5 * inch],
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (2, -1), "CENTER"),
            ("ALIGN", (5, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        pdf_tables.append(tbl)
        diag_lines = [
            f"M{int(row['Model'])}: n={int(row_n['n'])}, SW p={row_n['Shapiro_p']}, BP p={row_n['BP_p']}, QQ r={row_n['QQ_r']}, OLS={row_n['OLS_ok']}, R² CV={row_n['R2_CV_mean']}±{row_n['R2_CV_sd']}"
            for (_, row), (_, row_n) in zip(display_df.iterrows(), best_by_size.iterrows())
        ]
        diag_note = "Diagnostics:<br/>" + "<br/>".join(diag_lines)
        pdf_tables.append(Spacer(1, 0.05 * inch))
        pdf_tables.append(Paragraph(diag_note, styles["Normal"]))
        pdf_tables.append(Spacer(1, 0.08 * inch))

        # --- Scatter plots feature → DV pour chaque modèle retenu ---
        if figs_dir is not None:
            dv_slug = re.sub(r"[^\w]", "_", dv)
            for _, model_row in best_by_size.iterrows():
                m_num = int(model_row["Model"])
                feat_sig = model_row["feature_signature"]
                feats = list(feat_sig) if feat_sig else []
                if not feats:
                    continue
                fig_fname = figs_dir / f"scatter_M{m_num}_{dv_slug}.png"
                ok = _plot_inv_model_scatter(df, dv, feats, m_num, fig_fname)
                if ok and fig_fname.exists():
                    n_feats = len(feats)
                    img_w = min(7.0, 4.2 * n_feats) * inch
                    img_h = 4.0 * inch
                    rel_path = f"{figs_dir.name}/{fig_fname.name}"
                    table_lines.append(f"![M{m_num} scatter]({rel_path})\n\n")
                    pdf_tables.append(Image(str(fig_fname), width=img_w, height=img_h))
                    pdf_tables.append(Spacer(1, 0.08 * inch))

    if table_lines:
        lines.append(f"#### {section_num}.3 Meilleurs modèles stepwise significatifs par variable dépendante\n\n")
        lines.extend(table_lines)
        lines.append(
            "_Seuls les modèles pour lesquels chaque prédicteur vérifie p < 0.05 sont retenus "
            "(seuil harmonisé avec la régression sur scores PCA). "
            "Pour chaque nombre de prédicteurs, le modèle affiché est celui qui présente le R² le plus élevé. "
            "Le dernier indicateur (RMSE) correspond au RMSE moyen de la validation croisée répétée 10-run 5-fold. "
            "Les marqueurs indiquent le statut des prédicteurs individuels (* significatif p < 0.05 ; ** p < 0.01 ; *** p < 0.001). "
            "La colonne `Signe / magnitude` rapporte le sens de l'effet et sa taille selon |βstd| "
            "(négligeable < 0.10 ; faible < 0.30 ; modérée < 0.50 ; forte ≥ 0.50). "
            "La colonne `Modalities` correspond aux familles réellement représentées par les prédicteurs retenus._"
            "\n\n"
        )
        pdf_elems.append(Paragraph(f"{section_num}.3 Meilleurs modèles stepwise significatifs par variable dépendante", styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))
        pdf_elems.extend(pdf_tables)
        pdf_elems.append(
            Paragraph(
                "Seuls les modèles pour lesquels chaque prédicteur vérifie p < 0.05 sont retenus "
                "(seuil harmonisé avec la régression sur scores PCA). "
                "Pour chaque nombre de prédicteurs, le modèle affiché est celui qui présente le R² le plus élevé. "
                "Le dernier indicateur (RMSE) correspond au RMSE moyen de la validation croisée répétée 10-run 5-fold. "
                "Les marqueurs indiquent le statut des prédicteurs individuels (* = significatif p < 0.05 ; ** p < 0.01 ; *** p < 0.001). "
                "La colonne Signe / magnitude rapporte le sens de l'effet et sa taille selon |βstd| "
                "(négligeable < 0.10 ; faible < 0.30 ; modérée < 0.50 ; forte ≥ 0.50). "
                "La colonne Modalities correspond aux familles reellement representees par les predicteurs retenus.",
                styles["Normal"],
            )
        )
        pdf_elems.append(Spacer(1, 0.1 * inch))

    return retained_features
