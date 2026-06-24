#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
familiarity_vr_analysis.py

Livrable 2 — Mise à jour du bloc 1 (familiarité) dans rapport_VR_blocs1_3.pdf.

Population : groupes VR uniquement (modalite == "VR"), après exclusions documentées.
Analyses : Spearman + bootstrap BCa (5000 réplicats, seed=42) + régression linéaire
simple + correction FDR Benjamini-Hochberg par VD.
VD prioritaires : c_factor_pop, c_factor_sample, score_perf_tsk, COR, CRE, SPE, SOC, TSK, COM.

Sorties :
- familiarity_vr_results.csv (tables/)
- familiarity_vr_forest_plot.png (figures/)
- rapport_VR_blocs1_3.pdf mis à jour (même chemin)
"""

from __future__ import annotations

import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*[Cc]onstant.*")
warnings.filterwarnings("ignore", message=".*BCa confidence interval cannot.*")
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, linregress, spearmanr

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_V2_DIR = _HERE.parents[2]
_SCRIPTS_DIR = _HERE.parents[4]
_PROJECT_ROOT = _HERE.parents[5]

for _p in [str(_V2_DIR), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from py.corr import bh_fdr  # noqa: E402

RESULTS_DIR = _PROJECT_ROOT / "results"
REPORT_DIR = RESULTS_DIR / "rapport_v2" / "complementary_blocks"
TABLES_DIR = REPORT_DIR / "tables"
FIGURES_DIR = REPORT_DIR / "figures"
PDF_PATH = REPORT_DIR / "rapport_VR_blocs1_3.pdf"

EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}
SEED = 42
N_BOOTSTRAP = 5000
P_ALPHA = 0.05

# Variables dépendantes à analyser (dans cet ordre de priorité)
VD_ORDER = [
    "c_factor_pop",
    "c_factor_sample",
    "score_perf_tsk",
    "COR",
    "CRE",
    "SPE",
    "SOC",
    "TSK",
    "COM",
]

# Labels lisibles pour les VD
VD_LABELS: dict[str, str] = {
    "c_factor_pop": "C-factor (pop. parente)",
    "c_factor_sample": "C-factor (sample local)",
    "score_perf_tsk": "Performance (Score_perf_tsk)",
    "COR": "TMS Coordination",
    "CRE": "TMS Crédibilité",
    "SPE": "TMS Spécialisation",
    "SOC": "Cohésion sociale",
    "TSK": "Cohésion tâche",
    "COM": "Cohésion communication",
}


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Retourne un DataFrame group-level VR avec toutes les VD et prédicteurs."""

    # -- Profil questionnaire (source familiarité)
    profile_path = RESULTS_DIR / "questionnaire" / "global" / "participant_profile_responses.csv"
    profile = pd.read_csv(profile_path, low_memory=False)
    profile["modalite"] = profile["modalite"].astype(str).str.upper().str.strip()

    # Filtrer VR + exclusions
    vr_profile = profile[
        (profile["modalite"] == "VR") & (~profile["group_id"].isin(EXCLUDED_GROUPS))
    ].copy()

    # Agréger au niveau groupe
    group_fam = vr_profile.groupby("group_id", dropna=False).agg(
        vr_familiarity_mean=("vr_familiarity_score", "mean"),
        vr_familiarity_sd=("vr_familiarity_score", "std"),
        vr_familiarity_min=("vr_familiarity_score", "min"),
        vr_familiarity_max=("vr_familiarity_score", "max"),
        team_familiarity_mean=("team_familiarity_mean_score", "mean"),
        n_members=("vr_familiarity_score", "count"),
    ).reset_index()

    # -- Performance (la colonne groupe_id peut s'appeler "groupe" ou "group_id")
    perf_path = RESULTS_DIR / "performance_task" / "recap_scores_all.csv"
    perf = pd.read_csv(perf_path, low_memory=False)
    perf["modalite"] = perf["modalite"].astype(str).str.upper().str.strip()
    if "group_id" not in perf.columns and "groupe" in perf.columns:
        perf = perf.rename(columns={"groupe": "group_id"})
    vr_perf = perf[
        (perf["modalite"] == "VR") & (~perf["group_id"].isin(EXCLUDED_GROUPS))
    ][["group_id", "Score_perf_tsk"]].copy()
    vr_perf["score_perf_tsk"] = pd.to_numeric(vr_perf["Score_perf_tsk"], errors="coerce")

    # -- c_factor_pop et c_factor_sample
    pop_path = RESULTS_DIR / "TCI" / "c_scores_allowed_pop.csv"
    pop = pd.read_csv(pop_path, low_memory=False)
    pop = pop[~pop["group_id"].isin(EXCLUDED_GROUPS)][
        ["group_id", "c_factor_pop", "c_factor_sample"]
    ].copy()

    # -- Dimensions questionnaire (COR, CRE, SPE, SOC, TSK, COM) au niveau groupe
    merged_path = RESULTS_DIR / "merged_dataset" / "with_pruning" / "merged_dataset_complete_all.csv"
    merged = pd.read_csv(merged_path, low_memory=False)
    merged["modalite"] = merged["modalite"].astype(str).str.upper().str.strip()
    vr_merged = merged[
        (merged["modalite"] == "VR") & (~merged["group_id"].isin(EXCLUDED_GROUPS))
    ].copy()
    quest_cols = [c for c in ["group_id", "COR", "CRE", "SPE", "SOC", "TSK", "COM"] if c in vr_merged.columns]
    vr_quest = vr_merged[quest_cols].drop_duplicates(subset=["group_id"])

    # -- Jointure
    df = group_fam.merge(vr_perf[["group_id", "score_perf_tsk"]], on="group_id", how="left")
    df = df.merge(pop, on="group_id", how="left")
    df = df.merge(vr_quest, on="group_id", how="left")

    return df


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(df: pd.DataFrame) -> list[str]:
    logs = []
    logs.append(f"N groupes VR retenus (après exclusions) : {df['group_id'].nunique()}")
    logs.append(f"  (exclusions appliquées : {sorted(EXCLUDED_GROUPS)})")

    fam_pred = [c for c in df.columns if "familiarity" in c and c != "n_members"]
    logs.append(f"Variables de familiarité disponibles : {fam_pred}")
    for pred in fam_pred:
        n_valid = df[pred].notna().sum()
        n_na = df[pred].isna().sum()
        logs.append(f"  {pred} : {n_valid} valides, {n_na} NaN")

    for vd in VD_ORDER:
        if vd in df.columns:
            n_valid = df[vd].notna().sum()
            n_na = df[vd].isna().sum()
            logs.append(f"  VD {vd} : {n_valid} valides, {n_na} NaN")
        else:
            logs.append(f"  VD {vd} : ABSENTE")

    # Vérification critique
    if "c_factor_pop" in df.columns:
        n_valid_pop = df["c_factor_pop"].notna().sum()
        if n_valid_pop == 0:
            logs.append("[BLOCAGE] c_factor_pop : aucune valeur valide pour les groupes VR.")
        else:
            logs.append(f"  c_factor_pop disponible pour {n_valid_pop}/{len(df)} groupes VR.")

    return logs


# ---------------------------------------------------------------------------
# Analyse statistique
# ---------------------------------------------------------------------------

def _bootstrap_spearman_bca(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """IC 95% BCa sur ρ de Spearman par bootstrap."""
    def spearman_stat(x, y):
        rho, _ = spearmanr(x, y)
        return rho

    try:
        res = bootstrap(
            (x, y),
            spearman_stat,
            n_resamples=n_boot,
            paired=True,
            confidence_level=0.95,
            method="BCa",
            random_state=SEED,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return float("nan"), float("nan")


def _bootstrap_beta_bca(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP) -> tuple[float, float]:
    """IC 95% BCa sur β standardisé par bootstrap."""
    x_std = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    y_std = (y - y.mean()) / (y.std(ddof=1) + 1e-12)

    def beta_stat(x_s, y_s):
        if len(x_s) < 3:
            return float("nan")
        slope, _, _, _, _ = linregress(x_s, y_s)
        return slope

    try:
        res = bootstrap(
            (x_std, y_std),
            beta_stat,
            n_resamples=n_boot,
            paired=True,
            confidence_level=0.95,
            method="BCa",
            random_state=SEED,
        )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        return float("nan"), float("nan")


def analyze_pair(df: pd.DataFrame, predictor: str, vd: str) -> dict | None:
    """Analyse une paire (predictor, VD) : Spearman + bootstrap + régression linéaire simple."""
    sub = df[[predictor, vd]].dropna()
    n = len(sub)
    if n < 5:
        return None

    x = sub[predictor].values.astype(float)
    y = sub[vd].values.astype(float)

    # Ignorer les variables constantes
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None

    # Spearman
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p_raw = spearmanr(x, y)
    if not np.isfinite(rho):
        rho, p_raw = float("nan"), float("nan")

    # IC bootstrap BCa sur ρ
    ic_rho_low, ic_rho_high = _bootstrap_spearman_bca(x, y)

    # Régression standardisée
    x_std = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    y_std = (y - y.mean()) / (y.std(ddof=1) + 1e-12)

    if np.std(x_std) < 1e-12 or np.std(y_std) < 1e-12:
        slope_std, r_value = float("nan"), float("nan")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                slope_std, _, r_value, _, _ = linregress(x_std, y_std)
        except ValueError:
            slope_std, r_value = float("nan"), float("nan")

    r2 = r_value ** 2 if np.isfinite(r_value) else float("nan")
    ic_beta_low, ic_beta_high = _bootstrap_beta_bca(x, y)

    return {
        "predictor": predictor,
        "vd": vd,
        "n": n,
        "rho_spearman": round(float(rho), 4),
        "p_raw": round(float(p_raw), 4),
        "ic_rho_low": round(ic_rho_low, 4) if np.isfinite(ic_rho_low) else float("nan"),
        "ic_rho_high": round(ic_rho_high, 4) if np.isfinite(ic_rho_high) else float("nan"),
        "beta_std": round(float(slope_std), 4),
        "r2": round(float(r2), 4),
        "ic_beta_low": round(ic_beta_low, 4) if np.isfinite(ic_beta_low) else float("nan"),
        "ic_beta_high": round(ic_beta_high, 4) if np.isfinite(ic_beta_high) else float("nan"),
    }


def run_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lance toutes les paires (predictor × VD) et applique FDR par VD.
    Retourne le DataFrame complet des résultats.
    """
    # Identifier prédicteurs disponibles
    fam_predictors = [c for c in df.columns if "familiarity" in c and c != "n_members"]
    available_vds = [vd for vd in VD_ORDER if vd in df.columns and df[vd].notna().any()]

    print(f"  Prédicteurs : {fam_predictors}")
    print(f"  VD disponibles : {available_vds}")

    rows: list[dict] = []
    for vd in available_vds:
        vd_rows: list[dict] = []
        for pred in fam_predictors:
            result = analyze_pair(df, pred, vd)
            if result:
                vd_rows.append(result)

        # FDR par VD
        if vd_rows:
            p_raws = np.array([r["p_raw"] for r in vd_rows])
            p_fdrs = bh_fdr(p_raws)
            for r, p_fdr in zip(vd_rows, p_fdrs):
                r["p_fdr"] = round(float(p_fdr), 4) if np.isfinite(p_fdr) else float("nan")
            rows.extend(vd_rows)

    results_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return results_df


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def plot_forest(results_df: pd.DataFrame, out_path: Path) -> bool:
    """Forest plot des β standardisés avec IC bootstrap, groupé par VD."""
    if results_df.empty:
        print("  [SKIP] forest plot : résultats vides")
        return False

    available_vds = [vd for vd in VD_ORDER if vd in results_df["vd"].values]
    if not available_vds:
        return False

    n_vd = len(available_vds)
    fig, axes = plt.subplots(1, n_vd, figsize=(3.5 * n_vd, 6), sharey=False)
    if n_vd == 1:
        axes = [axes]

    for ax, vd in zip(axes, available_vds):
        sub = results_df[results_df["vd"] == vd].copy().reset_index(drop=True)
        if sub.empty:
            ax.set_visible(False)
            continue

        y_pos = np.arange(len(sub))
        colors = []
        for _, row in sub.iterrows():
            if pd.notna(row.get("p_fdr")) and row["p_fdr"] <= P_ALPHA:
                colors.append("#e15759")
            elif pd.notna(row.get("p_raw")) and row["p_raw"] <= P_ALPHA:
                colors.append("#f28e2b")
            else:
                colors.append("#9c9c9c")

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        for i, (_, row) in enumerate(sub.iterrows()):
            beta = row.get("beta_std", float("nan"))
            lo = row.get("ic_beta_low", float("nan"))
            hi = row.get("ic_beta_high", float("nan"))
            if np.isfinite(beta):
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.plot([lo, hi], [y_pos[i], y_pos[i]], color=colors[i], linewidth=2, alpha=0.7)
                ax.scatter([beta], [y_pos[i]], color=colors[i], s=50, zorder=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["predictor"].replace("_", " ") for _, r in sub.iterrows()], fontsize=8)
        ax.set_xlabel("β standardisé", fontsize=8)
        ax.set_title(VD_LABELS.get(vd, vd), fontsize=9, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)

        # Annoter p_fdr
        for i, (_, row) in enumerate(sub.iterrows()):
            if np.isfinite(row.get("p_fdr", float("nan"))) and row["p_fdr"] <= P_ALPHA:
                beta = row.get("beta_std", float("nan"))
                if np.isfinite(beta):
                    ax.text(beta, y_pos[i] + 0.15, "†", ha="center", va="bottom",
                            fontsize=9, color="#e15759")

    # Légende
    legend_handles = [
        mpatches.Patch(color="#e15759", label=f"p_fdr ≤ {P_ALPHA}"),
        mpatches.Patch(color="#f28e2b", label=f"p_raw ≤ {P_ALPHA}"),
        mpatches.Patch(color="#9c9c9c", label="non significatif"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"Familiarité VR → VD (β std. + IC bootstrap 95% BCa, n_boot={N_BOOTSTRAP})\n"
        f"N groupes VR ≈ {results_df['n'].max()} | FDR par VD (Benjamini-Hochberg) | † = p_fdr ≤ {P_ALPHA}",
        fontsize=10, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")
    return True


# ---------------------------------------------------------------------------
# Génération du PDF
# ---------------------------------------------------------------------------

def _build_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    styles = getSampleStyleSheet()
    custom = {
        "Heading1": styles["Heading1"],
        "Heading2": styles["Heading2"],
        "Heading3": styles["Heading3"],
        "Body": styles["BodyText"],
        "Small": ParagraphStyle(
            "Small", parent=styles["BodyText"],
            fontSize=8, leading=10, spaceAfter=3,
        ),
        "Box": ParagraphStyle(
            "Box", parent=styles["BodyText"],
            fontSize=9, leading=11,
            backColor=colors.lightblue,
            borderPad=4, spaceAfter=6,
        ),
        "Warning": ParagraphStyle(
            "Warning", parent=styles["BodyText"],
            fontSize=9, leading=11,
            backColor=colors.lightyellow,
            borderPad=4, spaceAfter=6,
        ),
    }
    return custom


def _para(text: str, styles, style_name: str = "Body"):
    from reportlab.platypus import Paragraph
    return Paragraph(text, styles[style_name])


def _df_to_table(df: pd.DataFrame, col_widths=None):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch

    if df.empty:
        return None
    data = [list(df.columns)] + [[str(v)[:60] if pd.notna(v) else "" for v in row] for row in df.values]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4e79a7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def rebuild_pdf_with_new_block1(
    results_df: pd.DataFrame,
    forest_path: Path,
    sanity_logs: list[str],
    n_vr_groups: int,
    df_group: pd.DataFrame,
) -> None:
    """
    Recrée rapport_VR_blocs1_3.pdf avec le bloc 1 mis à jour.
    Conserve les blocs 2 et 3 en appelant le pipeline existant.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table

    # Backup avant écrasement
    if PDF_PATH.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = PDF_PATH.parent / f"rapport_VR_blocs1_3_backup_{ts}.pdf"
        shutil.copy2(PDF_PATH, backup)
        print(f"  [OK] Backup créé : {backup.name}")

    styles = _build_styles()

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    elems: list = []

    # Titre
    elems.append(_para("Rapport complémentaire — Blocs 1 et 3 (v2 mise à jour)", styles, "Heading1"))
    elems.append(Spacer(1, 0.15 * inch))
    elems.append(_para(
        f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')} — Bloc 1 mis à jour : "
        "filtrage VR uniquement, c_factor_pop prioritaire, bootstrap BCa (5000 réplicats, seed=42), "
        "FDR Benjamini-Hochberg par VD.",
        styles, "Small"
    ))
    elems.append(Spacer(1, 0.1 * inch))

    # Sanity checks
    elems.append(_para("Sanity checks — chargement des données", styles, "Heading3"))
    for line in sanity_logs:
        elems.append(_para(f"• {line}", styles, "Small"))
    elems.append(Spacer(1, 0.1 * inch))

    # -------------------------------------------------------------------------
    # BLOC 1 — Familiarité VR (mis à jour)
    # -------------------------------------------------------------------------
    elems.append(_para("Bloc 1 — Familiarité VR → VD (groupes VR uniquement)", styles, "Heading2"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_para(
        f"Échantillon : N groupes VR = {n_vr_groups} | Population : groupes VR uniquement (modalite == 'VR') | "
        "Groupes exclus : bim002, bim032, bim065_2, bim075.",
        styles, "Box"
    ))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_para(
        "Analyses statistiques : (1) Spearman avec IC bootstrap 95% BCa (5000 réplicats, seed=42) ; "
        "(2) régression linéaire simple VD ~ familiarity_X avec β standardisé et IC bootstrap 95% BCa ; "
        "(3) correction FDR Benjamini-Hochberg appliquée par VD (famille = toutes les variables de "
        "familiarité pour une VD donnée). Les VD traitées sont : c_factor_pop (prioritaire), "
        "c_factor_sample, score_perf_tsk, TMS (COR, CRE, SPE), cohésion (SOC, TSK, COM). "
        "TMS global exclu (α = -.022 sur VR).",
        styles
    ))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_para(
        "⚠ Limites statistiques : N ≈ 8–12 groupes. La puissance statistique est faible ; "
        "les intervalles de confiance bootstrap sont instables. L'absence de significativité ne "
        "doit pas être interprétée comme une absence d'effet. Aucun modèle multivarié n'est produit.",
        styles, "Warning"
    ))
    elems.append(Spacer(1, 0.1 * inch))

    # Tableau synthétique
    elems.append(_para("Tableau synthétique — Spearman + régression + FDR par VD", styles, "Heading3"))
    if not results_df.empty:
        display_cols = [
            "predictor", "vd", "n", "rho_spearman", "p_raw", "p_fdr",
            "ic_rho_low", "ic_rho_high", "beta_std", "r2", "ic_beta_low", "ic_beta_high"
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]
        display_df = results_df[display_cols].copy()
        for col in ["rho_spearman", "p_raw", "p_fdr", "ic_rho_low", "ic_rho_high",
                    "beta_std", "r2", "ic_beta_low", "ic_beta_high"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        t = _df_to_table(display_df)
        if t:
            elems.append(t)
    else:
        elems.append(_para("(Aucun résultat — données insuffisantes)", styles))
    elems.append(Spacer(1, 0.12 * inch))

    # Forest plot
    if forest_path.exists():
        elems.append(_para("Forest plot — β standardisés avec IC bootstrap 95% BCa", styles, "Heading3"))
        elems.append(Spacer(1, 0.06 * inch))
        try:
            img_w = 6.5 * inch
            from PIL import Image as PILImage
            with PILImage.open(str(forest_path)) as img_obj:
                w, h = img_obj.size
            aspect = h / max(w, 1)
            img_h = img_w * aspect
            elems.append(Image(str(forest_path), width=img_w, height=min(img_h, 7 * inch)))
        except Exception:
            elems.append(Image(str(forest_path), width=6.5 * inch, height=5 * inch))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_para(
            "† = p_fdr ≤ 0.05. Barres : IC bootstrap 95% BCa (5000 réplicats, seed=42). "
            "Points rouges = p_fdr ≤ 0.05 ; orange = p_raw ≤ 0.05 seulement ; gris = non significatif.",
            styles, "Small"
        ))
    else:
        elems.append(_para("(Forest plot non disponible)", styles))

    # -------------------------------------------------------------------------
    # BLOC 3 — Régénéré depuis le pipeline existant
    # -------------------------------------------------------------------------
    elems.append(PageBreak())
    elems.append(_para("Bloc 3 — Analyse performance Score_perf_tsk", styles, "Heading2"))
    elems.append(Spacer(1, 0.08 * inch))
    elems.append(_para(
        "Le contenu de ce bloc est inchangé par rapport à la version précédente. "
        "Les données proviennent du pipeline complementary_blocks.py (blocs 2 et 3). "
        "Se référer au fichier rapport_global_blocs2_4.pdf pour les blocs 2 et 4.",
        styles, "Box"
    ))
    elems.append(Spacer(1, 0.08 * inch))

    # Essayer de charger les tables existantes du bloc 3
    _append_bloc3_from_existing(elems, styles)

    doc.build(elems)
    print(f"  [OK] PDF généré : {PDF_PATH.name}")


def _append_bloc3_from_existing(elems: list, styles: dict) -> None:
    """Ajoute le contenu bloc 3 depuis les CSV existants."""
    from reportlab.platypus import Spacer, Image
    from reportlab.lib.units import inch

    bloc3_corr = TABLES_DIR / "bloc3_perf_riedl_correlations.csv"
    bloc3_ols = TABLES_DIR / "bloc3_perf_riedl_ols.csv"
    bloc3_pc_vr = TABLES_DIR / "bloc3_perf_pc_vs_vr.csv"
    bloc3_heatmap = FIGURES_DIR / "bloc3_heatmap_perf_riedl.png"

    if bloc3_corr.exists():
        elems.append(_para("Corrélations bloc 3 (Riedl/TCI → Score_perf_tsk)", styles, "Heading3"))
        df3 = pd.read_csv(bloc3_corr)
        t = _df_to_table(df3)
        if t:
            elems.append(t)
        elems.append(Spacer(1, 0.08 * inch))

    if bloc3_ols.exists():
        elems.append(_para("Régressions univariées bloc 3", styles, "Heading3"))
        df3 = pd.read_csv(bloc3_ols)
        t = _df_to_table(df3)
        if t:
            elems.append(t)
        elems.append(Spacer(1, 0.08 * inch))

    if bloc3_pc_vr.exists():
        elems.append(_para("Comparaison PC vs VR sur Score_perf_tsk", styles, "Heading3"))
        df3 = pd.read_csv(bloc3_pc_vr)
        t = _df_to_table(df3)
        if t:
            elems.append(t)
        elems.append(Spacer(1, 0.08 * inch))

    if bloc3_heatmap.exists():
        elems.append(_para("Heatmap Riedl/TCI vs Performance", styles, "Heading3"))
        try:
            elems.append(Image(str(bloc3_heatmap), width=5.9 * inch, height=4.6 * inch))
        except Exception:
            pass
        elems.append(Spacer(1, 0.08 * inch))

    if not any(p.exists() for p in [bloc3_corr, bloc3_ols, bloc3_pc_vr]):
        elems.append(_para(
            "Tables du bloc 3 non trouvées dans les répertoires attendus. "
            "Relancer le pipeline complementary_blocks pour régénérer ces tables.",
            styles, "Warning"
        ))


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Chargement des données VR ===")
    df = load_data()

    print("\n=== Sanity checks ===")
    sanity_logs = sanity_check(df)
    for line in sanity_logs:
        print(f"  {line}")

    # Blocage si c_factor_pop absent
    if "c_factor_pop" not in df.columns or df["c_factor_pop"].notna().sum() == 0:
        print("\n[BLOCAGE] c_factor_pop absent ou entièrement NaN pour les groupes VR. Arrêt.")
        sys.exit(1)

    n_vr_groups = df["group_id"].nunique()

    # Vérifier cohérence Spearman sample vs pop sur VR
    if all(c in df.columns for c in ["c_factor_sample", "c_factor_pop"]):
        sub = df[["c_factor_sample", "c_factor_pop"]].dropna()
        if len(sub) >= 5:
            rho_check, p_check = spearmanr(sub["c_factor_sample"], sub["c_factor_pop"])
            print(f"\n  Spearman c_factor_sample × c_factor_pop sur VR : rho={rho_check:.3f}, p={p_check:.4f}, n={len(sub)}")
            if abs(rho_check) < 0.1 and len(sub) >= 8:
                print("  [AVERTISSEMENT] Corrélation VR très faible — c_factor_pop instable sur cet échantillon.")

    print(f"\n=== Analyse familiarité VR (N={n_vr_groups} groupes) ===")
    results_df = run_analysis(df)

    if not results_df.empty:
        csv_out = TABLES_DIR / "familiarity_vr_results.csv"
        results_df.to_csv(csv_out, index=False, encoding="utf-8")
        print(f"  [OK] {csv_out.name} ({len(results_df)} paires analysées)")
        print(f"  Bootstrap seed={SEED}, n_boot={N_BOOTSTRAP} — reproductible")

        # Vérifier FDR par VD
        for vd in results_df["vd"].unique():
            sub_vd = results_df[results_df["vd"] == vd]
            print(f"  FDR confirmé par VD '{vd}' : {len(sub_vd)} tests corrigés")
    else:
        print("  [WARN] Aucun résultat — vérifier les données VR")

    print("\n=== Forest plot ===")
    forest_path = FIGURES_DIR / "familiarity_vr_forest_plot.png"
    plot_forest(results_df, forest_path)

    print("\n=== Génération PDF ===")
    rebuild_pdf_with_new_block1(results_df, forest_path, sanity_logs, n_vr_groups, df)

    print("\n=== Sorties ===")
    for p in [TABLES_DIR / "familiarity_vr_results.csv", forest_path, PDF_PATH]:
        status = "[OK]" if p.exists() else "[MANQUANT]"
        print(f"  {status} {p}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
