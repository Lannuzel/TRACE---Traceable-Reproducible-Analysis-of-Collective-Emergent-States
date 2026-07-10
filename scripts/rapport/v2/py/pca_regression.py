# -*- coding: utf-8 -*-
"""
pca_regression.py
=================
Régressions sur scores PCA (composantes issues de analyze_inv_structure.py).

Méthode : identique à la régression INV (regression.py) —
    - Sélection forward stepwise (1 à 4 prédicteurs, p < 0.05 pour chaque prédicteur)
    - OLS via statsmodels (fallback numpy lstsq)
    - Coefficients bêta standardisés (βstd)
    - Diagnostics : Shapiro-Wilk (SW), Breusch-Pagan (BP), QQ r
    - Validation croisée répétée 10-run 5-fold (R² CV ± SD)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

_v2_dir = Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))
_scripts_dir = Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    COHESION_SCORE_COL,
    TMS_DIMENSIONS,
    COHESION_COMPONENTS,
)
from config.inv_features_config import infer_family_from_name
from py.regression import (
    forward_stepwise_inv_models,
    compute_model_diagnostics,
    _fit_ols_subset,
    _cross_validated_metrics,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# PC_LABELS: dict[str, str] = {
#     "PC1": "Intensité et équilibre de la participation verbale (31.5%)",
#     "PC2": "Dispersion vs focalisation du regard (20.1%)",
#     "PC3": "Résonance affective positive (16.2%)",
#     "PC4": "Régulation structurée des échanges (10.4%)",
#     "PC5": "Joie active individuelle (7.4%)",
#     "PC6": "Affect négatif vs attention aux objets partagés (5.5%)",
# }

# PC_FAMILY: dict[str, list[str]] = {
#     "A": ["PC1", "PC4"],
#     "F": ["PC3", "PC5", "PC6"],
#     "G": ["PC2"],
# }

# # Modèles bivariés niveau 2  (A×G et F×G — G=PC2 Gaze)
# BIVARIATE_MODELS: list[tuple[str, list[str]]] = [
#     ("PC1+PC2", ["PC1", "PC2"]),
#     ("PC3+PC2", ["PC3", "PC2"]),
#     ("PC4+PC2", ["PC4", "PC2"]),
# ]

# # Modèles multi-familles niveau 3
# MULTIFAMILY_MODELS: list[tuple[str, list[str]]] = [
#     ("A",     ["PC1", "PC4"]),
#     ("F",     ["PC3", "PC5", "PC6"]),
#     ("G",     ["PC2"]),
#     ("A+G",   ["PC1", "PC4", "PC2"]),
#     ("F+G",   ["PC3", "PC5", "PC6", "PC2"]),
#     ("A+F+G", ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]),
# ]

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

PC_LABELS: dict[str, str] = {
    "PC1": "",
    "PC2": "",
    "PC3": "",
    "PC4": "",
    "PC5": "",
    "PC6": "",
}

PC_FAMILY: dict[str, list[str]] = {
    "A": [""],
    "F": [""],
    "G": [""],
}

# ---------------------------------------------------------------------------
# Modèles bivariés niveau 2
# ---------------------------------------------------------------------------

BIVARIATE_MODELS: list[tuple[str, list[str]]] = [

    # --- A × G ---
    ("PC1+PC2", ["PC1", "PC2"]),
    ("PC4+PC2", ["PC4", "PC2"]),

    # --- F × G ---
    ("PC3+PC2", ["PC3", "PC2"]),
    ("PC5+PC2", ["PC5", "PC2"]),
    ("PC6+PC2", ["PC6", "PC2"]),

    # --- A × F ---
    ("PC1+PC3", ["PC1", "PC3"]),
    ("PC1+PC5", ["PC1", "PC5"]),
    ("PC1+PC6", ["PC1", "PC6"]),

    ("PC4+PC3", ["PC4", "PC3"]),
    ("PC4+PC5", ["PC4", "PC5"]),
    ("PC4+PC6", ["PC4", "PC6"]),

    # --- intra-famille A ---
    ("PC1+PC4", ["PC1", "PC4"]),

    # --- intra-famille F ---
    ("PC3+PC5", ["PC3", "PC5"]),
    ("PC3+PC6", ["PC3", "PC6"]),
    ("PC5+PC6", ["PC5", "PC6"]),
]

# ---------------------------------------------------------------------------
# Modèles multi-familles niveau 3
# ---------------------------------------------------------------------------

MULTIFAMILY_MODELS: list[tuple[str, list[str]]] = [

    # Familles seules
    ("A", ["PC1", "PC4"]),
    ("F", ["PC3", "PC5", "PC6"]),
    ("G", ["PC2"]),

    # Familles combinées
    ("A+G", ["PC1", "PC4", "PC2"]),
    ("F+G", ["PC3", "PC5", "PC6", "PC2"]),
    ("A+F", ["PC1", "PC4", "PC3", "PC5", "PC6"]),

    # Triplets ciblés
    ("PC1+PC3+PC2", ["PC1", "PC3", "PC2"]),
    ("PC1+PC5+PC2", ["PC1", "PC5", "PC2"]),
    ("PC1+PC6+PC2", ["PC1", "PC6", "PC2"]),

    ("PC4+PC3+PC2", ["PC4", "PC3", "PC2"]),
    ("PC4+PC5+PC2", ["PC4", "PC5", "PC2"]),
    ("PC4+PC6+PC2", ["PC4", "PC6", "PC2"]),

    # Modèle global
    ("A+F+G", ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]),
]

# VD utilisées dans le rapport
VD_REGRESSION = [
    "Score_perf_tsk",
    "Cohesion_questionnaire_score",
    "COR", "CRE", "SPE",
    "SOC", "TSK", "COM",
]
VD_LABELS: dict[str, str] = {
    "Cohesion_questionnaire_score": "Cohésion (score global)",
    "Score_perf_tsk": "Performance tâche",
    "COR": "TMS-Coordination",
    "CRE": "TMS-Crédibilité",
    "SPE": "TMS-Spécialisation",
    "SOC": "Cohésion sociale",
    "TSK": "Cohésion tâche",
    "COM": "Cohésion communication",
}

_FAMILY_LABEL_FR = {
    "audio": "conversationnel/audio",
    "face": "expressivité faciale",
    "affect": "affect",
    "gaze": "attention visuelle",
}


def _build_pc_interpretations(ld_csv: Path, pc_cols: list[str], top_k: int = 3) -> dict[str, str]:
    """Dérive un libellé d'interprétation court pour chaque PC à partir des top loadings.

    Point 5 : remplit la colonne « Interprétation » des tables 5.5.1. Le libellé
    combine la famille dominante des features à fort loading et la polarité (+/−)
    des 2 features les plus chargées, ex. « Face+ : synchronie & sourire ».
    """
    out: dict[str, str] = {}
    try:
        ld = pd.read_csv(ld_csv)
    except Exception:
        return {pc: "" for pc in pc_cols}
    if ld.empty:
        return {pc: "" for pc in pc_cols}
    feat_col = ld.columns[0]
    for pc in pc_cols:
        if pc not in ld.columns:
            out[pc] = ""
            continue
        sub = ld[[feat_col, pc]].copy()
        sub["abs"] = sub[pc].abs()
        sub = sub.nlargest(top_k, "abs")
        if sub.empty:
            out[pc] = ""
            continue
        # C8 : familles parmi les top loadings — détecter les axes MIXTES (aucune
        # famille nettement dominante) au lieu de forcer une étiquette monolithique.
        _short_fam = {"audio": "audio", "face": "face", "affect": "face", "gaze": "gaze"}
        _fam_of = lambda f: _short_fam.get(str(infer_family_from_name(str(f))), str(infer_family_from_name(str(f))))
        # Les deux loadings les plus forts (sub est déjà trié par |loading| décroissant).
        _f1_name = sub.iloc[0][feat_col]
        _f1_fam = _fam_of(_f1_name)
        _f1_abs = abs(float(sub.iloc[0][pc]))
        _f2_fam = None
        _f2_abs = 0.0
        if len(sub) >= 2:
            _f2_fam = _fam_of(sub.iloc[1][feat_col])
            _f2_abs = abs(float(sub.iloc[1][pc]))
        # Axe mixte : les 2 loadings dominants sont de familles différentes et de
        # magnitude comparable (le 2e vaut ≥ 85 % du 1er) → aucune famille ne domine.
        if _f2_fam and _f2_fam != _f1_fam and _f1_abs > 0 and (_f2_abs / _f1_abs) >= 0.85:
            fam_label = f"mixte {_f1_fam}-{_f2_fam}"
        else:
            fam_label = _FAMILY_LABEL_FR.get(_f1_fam, _f1_fam)
        # polarité de la feature la plus chargée
        top1 = sub.iloc[0]
        sign = "+" if float(top1[pc]) >= 0 else "−"
        feats_short = ", ".join(str(f).replace("audio_", "").replace("face_", "").replace("gaze_", "")
                                for f in sub[feat_col].head(2))
        out[pc] = f"{fam_label} {sign} ({feats_short})"
    return out


# ---------------------------------------------------------------------------
# Chargement données
# ---------------------------------------------------------------------------

def _load_pca_scores(inv_dir: Path) -> pd.DataFrame | None:
    """Charge inv_dimensions.csv — contient group_id, scenario, timepoint, PC1..PCn."""
    p = inv_dir / "inv_dimensions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    if not pc_cols:
        return None
    return df


def _load_vd(results_dir: Path) -> pd.DataFrame | None:
    """
    Charge les variables dépendantes depuis les fichiers résultats.
    Priorité : sem/path_analysis_vr > behavioral_indices_v2 > merged_dataset.
    """
    vd_cols_needed = (
        VD_REGRESSION
        + TMS_DIMENSIONS
        + COHESION_COMPONENTS
        + ["group_id", "scenario", "timepoint"]
    )

    candidates = [
        results_dir / "sem" / "path_analysis_vr" / "path_analysis_vr_dataset.csv",
        results_dir / "rapport_v2" / "VR_only" / "behavioral_indices_v2" / "silent_division_profiles.csv",
        results_dir / "merged_dataset" / "with_pruning" / "merged_dataset_complete_vr.csv",
        results_dir / "merged_dataset" / "without_pruning" / "merged_dataset_complete_vr.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            keep = [c for c in vd_cols_needed if c in df.columns]
            if "group_id" not in keep:
                continue
            if len([c for c in VD_REGRESSION if c in keep]) == 0:
                continue
            return df[list(dict.fromkeys(keep))].copy()
        except Exception:
            continue
    return None


def _merge_pc_vd(pc_df: pd.DataFrame, vd_df: pd.DataFrame) -> pd.DataFrame | None:
    """Fusionne scores PC et VD sur group_id + scenario + timepoint."""
    join_keys = [k for k in ["group_id", "scenario", "timepoint"]
                 if k in pc_df.columns and k in vd_df.columns]
    if not join_keys:
        return None
    merged = pc_df.merge(vd_df, on=join_keys, how="inner")
    return merged if not merged.empty else None


# ---------------------------------------------------------------------------
# Helpers formatage — identiques à regression.py
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


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


def _format_feature_list(features: list[str], pvals: dict[str, float]) -> str:
    parts = []
    for feat in features:
        p = pvals.get(feat, np.nan)
        parts.append(f"{feat}{_sig_stars(p)}")
    return "; ".join(parts)


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


def _modalities_from_pcs(pcs: list[str]) -> str:
    families: list[str] = []
    for fam, fam_pcs in PC_FAMILY.items():
        if any(pc in fam_pcs for pc in pcs):
            families.append(fam)
    return "+".join(families) if families else "–"


def _df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Aucun résultat disponible._\n"
    lines = []
    cols = list(df.columns)
    lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines.append("|" + "|".join([":--"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) if pd.notna(row[c]) else "" for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _df_to_pdf_table(df: pd.DataFrame, styles_map: Any, col_widths=None) -> Table | None:
    if df.empty:
        return None
    data_rows = [list(df.columns)] + [list(row) for _, row in df.iterrows()]
    col_w = col_widths or ([1.0 * inch] * len(df.columns))

    def _para(text, bold=False):
        style = styles_map["BodyText"].clone(f"PcaRegCell_{id(df)}")
        style.fontName = "Helvetica"
        style.fontSize = 7
        style.leading = 8
        style.wordWrap = "CJK"
        t = str(text) if pd.notna(text) else ""
        if bold or t.startswith("**"):
            t = t.replace("**", "")
            return Paragraph(f"<b>{t}</b>", style)
        return Paragraph(t, style)

    table_data = []
    for i, row in enumerate(data_rows):
        is_header = i == 0
        table_data.append([_para(cell, bold=is_header) for cell in row])

    tbl = Table(table_data, colWidths=col_w, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Construction tableaux résultats (méthode stepwise identique à INV)
# ---------------------------------------------------------------------------

def _render_stepwise_results(
    data: pd.DataFrame,
    candidate_pcs: list[str],
    vd: str,
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    section_label: str,
    section_label_pdf: str,
    model_counter: int,
) -> tuple[bool, int]:
    """
    Lance forward_stepwise_inv_models sur les scores PC et affiche les résultats
    dans le même format que la régression INV (tableau + diagnostics).
    Retourne True si au moins un modèle significatif trouvé.
    """
    avail = [pc for pc in candidate_pcs if pc in data.columns]
    if not avail:
        return False, model_counter

    # M9 : côté PCA le seuil d'entrée stepwise est p<0.05 (cohérent avec le préambule).
    models = forward_stepwise_inv_models(data, vd, avail, max_features=min(4, len(avail)), p_enter=0.05)
    if not models:
        return False, model_counter

    if section_label:
        lines.append(f"**{section_label}**\n\n")
    if section_label_pdf:
        pdf_elems.append(Paragraph(section_label_pdf, styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.04 * inch))

    rows: list[dict] = []
    for m in models:
        feats = m["features"]
        rows.append({
            "Modalities": _modalities_from_pcs(feats),
            "# Features": m.get("n_features", len(feats)),
            "Significant features": _format_feature_list(feats, m.get("predictor_pvalues", {})),
            "Signe / magnitude": _format_sign_magnitude(feats, m.get("predictor_betas_std", {})),
            "R2": round(float(m.get("r2", np.nan)), 3) if np.isfinite(m.get("r2", np.nan)) else np.nan,
            "p-value": _format_pvalue(m.get("model_p", np.nan)),
            "RMSE": round(float(m.get("cv_rmse_mean", np.nan)), 3) if np.isfinite(m.get("cv_rmse_mean", np.nan)) else np.nan,
            "R2_CV_mean": round(float(m.get("cv_r2_mean", np.nan)), 3) if np.isfinite(m.get("cv_r2_mean", np.nan)) else np.nan,
            "R2_CV_sd": round(float(m.get("cv_r2_sd", np.nan)), 3) if np.isfinite(m.get("cv_r2_sd", np.nan)) else np.nan,
            "Shapiro_p": round(float(m.get("shapiro_p", np.nan)), 4) if np.isfinite(m.get("shapiro_p", np.nan)) else np.nan,
            "BP_p": round(float(m.get("breusch_pagan_p", np.nan)), 4) if np.isfinite(m.get("breusch_pagan_p", np.nan)) else np.nan,
            "QQ_r": round(float(m.get("qq_corr", np.nan)), 3) if np.isfinite(m.get("qq_corr", np.nan)) else np.nan,
            "OLS_ok": "yes" if m.get("assumptions_ok", False) else "no",
            "n": m.get("n_obs", "?"),
            "feature_signature": tuple(feats),
        })

    if not rows:
        return False, model_counter

    res_df = pd.DataFrame(rows)
    res_df = res_df.sort_values(["# Features", "R2", "Modalities"], ascending=[True, False, True]).reset_index(drop=True)
    res_df = res_df.drop_duplicates(subset=["# Features", "feature_signature"], keep="first").reset_index(drop=True)
    best_by_size = res_df.groupby("# Features", dropna=False).head(1).reset_index(drop=True)
    best_by_size.insert(0, "Model", range(model_counter, model_counter + len(best_by_size)))
    model_counter += len(best_by_size)

    display_df = best_by_size[
        ["Model", "Modalities", "# Features", "Significant features", "Signe / magnitude", "R2", "p-value", "RMSE"]
    ].copy()

    lines.append(_df_to_md(display_df))
    lines.append("\n")

    model_cell_style = styles["BodyText"].clone(f"PcaRegStepCell_{vd}")
    model_cell_style.fontName = "Helvetica"
    model_cell_style.fontSize = 7
    model_cell_style.leading = 8
    model_cell_style.wordWrap = "CJK"

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
    _style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (2, -1), "CENTER"),
        ("ALIGN", (5, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    # Point 8 : griser les modèles dont les hypothèses OLS échouent (Shapiro,
    # Breusch-Pagan < 0.05, ou QQ < 0.90) — même filtre visuel que le rapport INV.
    _grey_diag = colors.HexColor("#e0e0e0")
    _n_grey = 0
    for _i, (_, _brow) in enumerate(best_by_size.iterrows(), start=1):
        if str(_brow.get("OLS_ok", "no")) != "yes":
            _style_cmds.append(("BACKGROUND", (0, _i), (-1, _i), _grey_diag))
            _style_cmds.append(("TEXTCOLOR", (0, _i), (-1, _i), colors.HexColor("#666666")))
            _n_grey += 1
    tbl.setStyle(TableStyle(_style_cmds))
    pdf_elems.append(tbl)
    if _n_grey:
        _grey_note = (
            f"_{_n_grey} modèle(s) grisé(s) : hypothèses OLS non satisfaites "
            "(Shapiro-Wilk p<0.05, Breusch-Pagan p<0.05 ou QQ r<0.90) — coefficients non interprétables._"
        )
        lines.append(_grey_note + "\n\n")
        pdf_elems.append(Paragraph(_grey_note, styles["Normal"]))

    diag_parts = [
        f"M{int(row['Model'])}: n={int(row_n['n'])}, SW p={row_n['Shapiro_p']}, "
        f"BP p={row_n['BP_p']}, QQ r={row_n['QQ_r']}, OLS={row_n['OLS_ok']}, "
        f"R² CV={row_n['R2_CV_mean']}±{row_n['R2_CV_sd']}"
        for (_, row), (_, row_n) in zip(display_df.iterrows(), best_by_size.iterrows())
    ]
    if diag_parts:
        diag_str = "_Diagnostics_: " + "; ".join(diag_parts)
        lines.append(f"{diag_str}\n\n")
        pdf_elems.append(Paragraph(diag_str, styles["Normal"]))

    pdf_elems.append(Spacer(1, 0.1 * inch))
    return True, model_counter


def _render_fixed_model(
    data: pd.DataFrame,
    name: str,
    preds: list[str],
    vd: str,
    lines: list[str],
    pdf_elems: list,
    styles: Any,
) -> bool:
    """
    Ajuste un modèle OLS fixe (prédicteurs imposés) et affiche diagnostics — même
    format que INV (coefficients, diagnostics, R²CV).
    """
    avail = [p for p in preds if p in data.columns and vd in data.columns]
    if not avail:
        return False

    model, sub = _fit_ols_subset(data, vd, avail)
    if model is None or sub.empty:
        return False

    n_obs = len(sub)
    diag = compute_model_diagnostics(model)
    cv_mean, cv_sd, _ = _cross_validated_metrics(data, vd, avail)

    try:
        pvals = model.pvalues.drop(labels=["const"], errors="ignore")
        params = model.params.drop(labels=["const"], errors="ignore")
        r2_adj = float(model.rsquared_adj)
        f_pval = float(model.f_pvalue) if hasattr(model, "f_pvalue") else np.nan
    except Exception:
        return False

    y_sd = float(pd.to_numeric(sub[vd], errors="coerce").std(ddof=1))
    betas_std: dict[str, float] = {}
    for pred, coef in params.to_dict().items():
        x_sd = float(pd.to_numeric(sub[pred], errors="coerce").std(ddof=1))
        betas_std[pred] = float(coef) * x_sd / y_sd if np.isfinite(coef) and y_sd > 0 and x_sd > 0 else np.nan

    feat_parts = []
    for feat in avail:
        p = pvals.get(feat, np.nan)
        bstd = betas_std.get(feat, np.nan)
        bstd_str = f"βstd={bstd:.2f}" if np.isfinite(bstd) else ""
        feat_parts.append(f"{feat}{_sig_stars(p)} ({bstd_str})")

    row = {
        "Modèle": name,
        "Prédicteurs (βstd)": "; ".join(feat_parts),
        "n": n_obs,
        "R²_adj": f"{r2_adj:.3f}" if np.isfinite(r2_adj) else "–",
        "p_glob": _format_pvalue(f_pval),
        "R²CV": f"{cv_mean:.3f}±{cv_sd:.3f}" if np.isfinite(cv_mean) else "–",
    }
    ols_ok = diag.get("assumptions_ok", False)
    df_res = pd.DataFrame([row])
    # C7 : marquer visuellement le modèle si les hypothèses OLS échouent (BP<0.05…).
    if not ols_ok:
        df_res.insert(0, "⚠", "OLS✗")
    lines.append(_df_to_md(df_res))
    lines.append("\n")
    _fixed_widths = ([0.5 * inch] if not ols_ok else []) + [
        0.7 * inch, 3.3 * inch, 0.4 * inch, 0.7 * inch, 0.6 * inch, 1.2 * inch
    ]
    pdf_tbl = _df_to_pdf_table(df_res, styles, col_widths=_fixed_widths)
    if pdf_tbl:
        # Griser toute la ligne de données quand OLS non satisfait.
        if not ols_ok:
            try:
                pdf_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e0e0e0")),
                    ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#666666")),
                ]))
            except Exception:
                pass
        pdf_elems.append(pdf_tbl)

    sw_p = diag.get("shapiro_p", np.nan)
    bp_p = diag.get("breusch_pagan_p", np.nan)
    qq_r = diag.get("qq_corr", np.nan)
    diag_str = (
        f"_Diagnostics {name}_: n={n_obs}, "
        f"SW p={sw_p:.4f}, BP p={bp_p:.4f}, QQ r={qq_r:.3f}, "
        f"OLS={'yes' if ols_ok else 'no'}, "
        f"R² CV={cv_mean:.3f}±{cv_sd:.3f}"
        if np.isfinite(cv_mean) else
        f"_Diagnostics {name}_: n={n_obs}, "
        f"SW p={sw_p:.4f}, BP p={bp_p:.4f}, QQ r={qq_r:.3f}, "
        f"OLS={'yes' if ols_ok else 'no'}"
    )
    lines.append(f"{diag_str}\n\n")
    pdf_elems.append(Paragraph(diag_str, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))
    return True


# ---------------------------------------------------------------------------
# Rendu rapport
# ---------------------------------------------------------------------------

def render_pca_regression_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    results_dir: Path,
    inv_subdir: str = "results_inv_structure_vr_only/with_pruning",
    section_num: str = "5.5",
    pca_rotation: str = "none",
):
    """
    Génère la section 'Régressions PCA' dans le rapport INV.

    Méthode identique à la régression INV (regression.py) :
    - Niveau 1 : forward stepwise sur toutes les PC (p<0.05 pour chaque prédicteur)
    - Niveau 2 : modèles bivariés fixés (A×G, F×G, A×F) — OLS avec diagnostics
    - Niveau 3 : modèles multi-familles fixés — OLS avec diagnostics

    Diagnostics systématiques : SW, BP, QQ r, R²CV (10-run 5-fold).
    """
    title = f"{section_num} Régressions sur scores PCA"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading2"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # — Chargement données —
    inv_dir = results_dir / inv_subdir
    _main_pca_candidates = [
        results_dir / inv_subdir.replace("_regression", ""),
        inv_dir,
    ]
    pc_df = None
    for _candidate in _main_pca_candidates:
        pc_df = _load_pca_scores(_candidate)
        if pc_df is not None and len([c for c in pc_df.columns if c.startswith("PC")]) >= 5:
            inv_dir = _candidate
            break
    if pc_df is None:
        msg = f"inv_dimensions.csv absent dans {inv_subdir}. Relancer analyze_inv_structure.py."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    vd_df = _load_vd(results_dir)
    if vd_df is None:
        msg = "Fichiers VD introuvables (questionnaire / performance). Section ignorée."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    data = _merge_pc_vd(pc_df, vd_df)
    if data is None or data.empty:
        msg = "Aucune correspondance entre scores PC et VD après fusion. Section ignorée."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    pc_cols = sorted([c for c in data.columns if c.startswith("PC") and c[2:].isdigit()],
                     key=lambda x: int(x[2:]))
    vd_available = [vd for vd in VD_REGRESSION if vd in data.columns]
    n_obs = len(data)

    # --- Interprétation des composantes dérivée des top loadings (point 5) ---
    # Calculée UNE fois pour alimenter les colonnes « Interprétation » des tables.
    _rotation_pre = str(pca_rotation).lower().strip()
    _ld_csv_pre = inv_dir / ("pca_loadings_varimax.csv" if _rotation_pre == "varimax" else "pca_loadings.csv")
    if not _ld_csv_pre.exists():
        _ld_csv_pre = inv_dir / "pca_loadings_raw.csv"
        if not _ld_csv_pre.exists():
            _ld_csv_pre = inv_dir / "pca_loadings.csv"
    pc_interpretations = _build_pc_interpretations(_ld_csv_pre, pc_cols)

    # — Préambule —
    intro = (
        f"Les scores PC sont issus de la PCA sur {n_obs} groupes VR. "
        f"Composantes disponibles : {', '.join(pc_cols)}. "
        "Méthode : sélection forward stepwise (1 à 4 prédicteurs, p < 0.05 pour chaque prédicteur), "
        "OLS (statsmodels). Diagnostics : Shapiro-Wilk (résidus), Breusch-Pagan (hétéroscédasticité), "
        "corrélation QQ (normalité graphique). Validation croisée répétée 10-run 5-fold (R² CV ± SD). "
        "Étoiles : * p<.05 ** p<.01 *** p<.001. "
        "Note d'imputation (audit) : un groupe VR sans features audio complètes a été imputé "
        "(médiane) pour atteindre n=12 dans la PCA et ces régressions ; les composantes à dominante "
        "audio et les modèles qui les mobilisent reposent donc sur n=11 observations audio réelles + 1 imputée."
    )
    lines.append(f"{intro}\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    # — Tableau descriptif des scores PC (équivalent 3.1.1 INV) —
    pc_desc_rows = []
    for pc in pc_cols:
        vals = pd.to_numeric(data[pc], errors="coerce").dropna()
        if vals.empty:
            continue
        pc_desc_rows.append({
            "Composante": pc,
            "Interprétation": pc_interpretations.get(pc, ""),
            "Famille": next((fam for fam, pcs in PC_FAMILY.items() if pc in pcs), "–"),
            "N": int(len(vals)),
            "Moy": round(float(vals.mean()), 3),
            "ET": round(float(vals.std(ddof=1)), 3),
            "Min": round(float(vals.min()), 3),
            "Max": round(float(vals.max()), 3),
        })
    if pc_desc_rows:
        pc_desc_df = pd.DataFrame(pc_desc_rows)
        desc_title_md = f"#### {section_num}.1 Scores PCA utilisés dans la régression"
        desc_title_pdf = f"{section_num}.1 Scores PCA utilisés dans la régression"
        desc_note = (
            "Le tableau ci-dessous présente les statistiques descriptives des scores PCA "
            "sur le sous-ensemble VR analysé (n=" + str(n_obs) + " groupes). "
            "Les scores sont centrés (μ≈0) mais NON standardisés : leur variance égale "
            "l'eigenvalue de la composante (sortie PCA sklearn sans whitening), de sorte "
            "que l'ET décroît de PC1 à PC_k. Les coefficients OLS ci-dessous étant estimés "
            "sur variables standardisées en interne, cette échelle n'affecte pas les β_std."
        )
        lines.append(f"{desc_title_md}\n\n")
        lines.append(f"{desc_note}\n\n")
        lines.append(_df_to_md(pc_desc_df))
        lines.append("\n")
        pdf_elems.append(Paragraph(desc_title_pdf, styles["Heading4"]))
        pdf_elems.append(Paragraph(desc_note, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))

        # Tableau PDF avec colonnes larges pour Interprétation
        desc_tbl = _df_to_pdf_table(pc_desc_df, styles, col_widths=[
            0.5*inch, 2.5*inch, 0.5*inch, 0.4*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch
        ])
        if desc_tbl:
            pdf_elems.append(desc_tbl)
        pdf_elems.append(Spacer(1, 0.15 * inch))

    # — Tableau interprétation des composantes avec top loadings —
    top_loadings: dict[str, str] = {}
    rotation = str(pca_rotation).lower().strip()
    if rotation == "varimax":
        ld_csv = inv_dir / "pca_loadings_varimax.csv"
        if not ld_csv.exists():
            ld_csv = inv_dir / "pca_loadings.csv"
    else:
        ld_csv = inv_dir / "pca_loadings.csv"
        if not ld_csv.exists():
            ld_csv = inv_dir / "pca_loadings_raw.csv"

    loadings_label = None
    if ld_csv.exists():
        try:
            loadings_label = "varimax" if rotation == "varimax" and ld_csv.name.endswith("varimax.csv") else "raw"
            ld = pd.read_csv(ld_csv)
            feat_col = ld.columns[0]
            for pc in pc_cols:
                if pc not in ld.columns:
                    continue
                top = ld[[feat_col, pc]].copy()
                top["abs"] = top[pc].abs()
                top = top.nlargest(4, "abs")
                parts = []
                for _, row in top.iterrows():
                    sign = "+" if row[pc] > 0 else "-"
                    parts.append(f"{sign}{abs(row[pc]):.2f} {row[feat_col]}")
                top_loadings[pc] = " | ".join(parts)
        except Exception:
            pass

    top_label = "Top loadings (|λ| décroissant)"
    if loadings_label:
        top_label = f"{top_label} — {loadings_label}"
    label_rows = [
        {
            "PC": pc,
            "Interprétation": pc_interpretations.get(pc, ""),
            "Fam.": next((fam for fam, pcs in PC_FAMILY.items() if pc in pcs), "–"),
            top_label: top_loadings.get(pc, "–"),
        }
        for pc in pc_cols
    ]
    if label_rows:
        ldf = pd.DataFrame(label_rows)
        lines.append("**Interprétation des composantes :**\n\n")
        lines.append(_df_to_md(ldf))
        pdf_elems.append(Paragraph("Interprétation des composantes :", styles["Heading4"]))
        tbl = _df_to_pdf_table(ldf, styles, col_widths=[0.5*inch, 2.2*inch, 0.5*inch, 3.6*inch])
        if tbl:
            pdf_elems.append(tbl)
        pdf_elems.append(Spacer(1, 0.1 * inch))

    # — Boucle sur VD —
    model_counter = 1
    for vd in vd_available:
        vd_label = VD_LABELS.get(vd, vd)

        lines.append(f"\n---\n\n#### {vd_label}\n\n")
        pdf_elems.append(Paragraph(vd_label, styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))

        has_l1, model_counter = _render_stepwise_results(
            data, pc_cols, vd, lines, pdf_elems, styles,
            section_label="",
            section_label_pdf="",
            model_counter=model_counter,
        )
        if not has_l1:
            lines.append("_Aucun modèle stepwise significatif._\n\n")
            pdf_elems.append(Paragraph("Aucun modèle stepwise significatif.", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.06 * inch))

        pdf_elems.append(Spacer(1, 0.1 * inch))

    # — Note méthodologique —
    method_note = (
        "_Seuls les modèles pour lesquels chaque prédicteur vérifie p < 0.05 sont retenus. "
        "Pour chaque nombre de prédicteurs, le modèle affiché est celui qui présente le R² le plus élevé. "
        "Le dernier indicateur (RMSE) correspond au RMSE moyen de la validation croisée répétée 10-run 5-fold. "
        "Les étoiles indiquent la significativité des prédicteurs individuels (* p < 0.05; ** p < 0.01; *** p < 0.001). "
        "La colonne `Signe / magnitude` rapporte le sens de l'effet et sa taille selon |βstd| "
        "(négligeable < 0.10 ; faible < 0.30 ; modérée < 0.50 ; forte ≥ 0.50). "
        "La colonne `Modalities` correspond aux familles réellement représentées par les prédicteurs retenus._"
    )
    lines.append(f"{method_note}\n\n")
    pdf_elems.append(Paragraph(
        "Seuls les modèles pour lesquels chaque prédicteur vérifie p < 0.05 sont retenus. "
        "Pour chaque nombre de prédicteurs, le modèle affiché est celui qui présente le R² le plus élevé. "
        "Le dernier indicateur (RMSE) correspond au RMSE moyen de la validation croisée répétée 10-run 5-fold. "
        "Les étoiles indiquent la significativité des prédicteurs individuels (* p < 0.05; ** p < 0.01; *** p < 0.001). "
        "La colonne Signe / magnitude rapporte le sens de l'effet et sa taille selon |βstd| "
        "(négligeable < 0.10 ; faible < 0.30 ; modérée < 0.50 ; forte ≥ 0.50). "
        "La colonne Modalities correspond aux familles réellement représentées par les prédicteurs retenus.",
        styles["Normal"],
    ))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    # — Export CSV des résultats —
    _export_regression_csv(data, pc_cols, vd_available, inv_dir)


def _export_regression_csv(
    data: pd.DataFrame,
    pc_cols: list[str],
    vd_list: list[str],
    out_dir: Path,
) -> None:
    """Exporte un CSV récapitulatif de tous les modèles."""
    all_rows: list[dict] = []
    model_specs = (
        [("L1_stepwise", "stepwise", pc_cols)]
        + [("L2", name, preds) for name, preds in BIVARIATE_MODELS]
        + [("L3", name, preds) for name, preds in MULTIFAMILY_MODELS]
    )
    for vd in vd_list:
        if vd not in data.columns:
            continue
        for level, name, preds in model_specs:
            avail = [p for p in preds if p in data.columns]
            if not avail:
                continue
            if level == "L1_stepwise":
                models = forward_stepwise_inv_models(data, vd, avail, max_features=4)
                for m in models:
                    for feat in m["features"]:
                        all_rows.append({
                            "level": level,
                            "model": f"stepwise_{len(m['features'])}pred",
                            "vd": vd,
                            "predictor": feat,
                            "n_eff": m.get("n_obs"),
                            "r2_adj": round(m.get("r2_adj", np.nan), 4),
                            "beta_std": round(m["predictor_betas_std"].get(feat, np.nan), 4),
                            "p_pred": round(m["predictor_pvalues"].get(feat, np.nan), 4),
                            "cv_r2_mean": round(m.get("cv_r2_mean", np.nan), 4),
                            "shapiro_p": round(m.get("shapiro_p", np.nan), 4),
                        })
            else:
                model, sub = _fit_ols_subset(data, vd, avail)
                if model is None or sub.empty:
                    continue
                try:
                    pvals = model.pvalues.drop(labels=["const"], errors="ignore")
                    params = model.params.drop(labels=["const"], errors="ignore")
                    r2_adj = float(model.rsquared_adj)
                except Exception:
                    continue
                y_sd = float(pd.to_numeric(sub[vd], errors="coerce").std(ddof=1))
                for feat in avail:
                    coef = float(params.get(feat, np.nan))
                    x_sd = float(pd.to_numeric(sub[feat], errors="coerce").std(ddof=1))
                    bstd = coef * x_sd / y_sd if np.isfinite(coef) and y_sd > 0 and x_sd > 0 else np.nan
                    all_rows.append({
                        "level": level,
                        "model": name,
                        "vd": vd,
                        "predictor": feat,
                        "n_eff": len(sub),
                        "r2_adj": round(r2_adj, 4),
                        "beta_std": round(bstd, 4),
                        "p_pred": round(float(pvals.get(feat, np.nan)), 4),
                        "cv_r2_mean": np.nan,
                        "shapiro_p": np.nan,
                    })
    if all_rows:
        try:
            pd.DataFrame(all_rows).to_csv(out_dir / "pca_regression_results.csv", index=False)
        except Exception:
            pass
