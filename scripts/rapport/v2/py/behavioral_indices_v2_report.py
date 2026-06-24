from __future__ import annotations

import hashlib
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    _orig_md5 = hashlib.md5

    def _compat_md5(*args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return _orig_md5(*args, **kwargs)

    hashlib.md5 = _compat_md5

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

from py.behavioral_indices_v2_build import build_behavioral_indices_v2
from py.behavioral_indices_v2_common import OUTPUT_DIR


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    if val == 0 or abs(val) < 0.5 * (10 ** (-digits)):
        return f"{0:.{digits}f}"
    if abs(val) >= 1000 or abs(val) < (10 ** (-digits)) / 10:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def _format_display_df(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in out.columns:
        col_low = str(col).lower()
        if pd.api.types.is_numeric_dtype(out[col]):
            series = pd.to_numeric(out[col], errors="coerce")
            should_int = col_low == "n" or col_low.startswith("n_") or col_low.endswith("_n")
            if should_int:
                out[col] = series.apply(lambda x: "NA" if pd.isna(x) else str(int(round(float(x)))))
            else:
                out[col] = series.apply(lambda x: _fmt(x, digits=digits))
    return out.fillna("NA")


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_Aucune donnée disponible._\n"
    out = _format_display_df(df.head(max_rows) if max_rows is not None else df.copy())
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        vals = [str(v).replace("\n", " ").replace("|", "/") for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _paragraph(text: str, styles, style_name: str = "Normal"):
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def _wrap_text_for_display(value: Any, width: int = 36) -> str:
    text = str(value) if value is not None else "NA"
    if text == "nan":
        text = "NA"
    return "<br/>".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def _table_from_df(df: pd.DataFrame, max_rows: int | None = None):
    if df is None or df.empty:
        df = pd.DataFrame({"Info": ["Aucune donnée disponible"]})
    out = _format_display_df(df.head(max_rows) if max_rows is not None else df.copy())
    header_style = ParagraphStyle(name="CellHeaderV2", fontName="Helvetica-Bold", fontSize=7, leading=8, alignment=1)
    body_style = ParagraphStyle(name="CellBodyV2", fontName="Helvetica", fontSize=7, leading=8)
    cols = [str(c) for c in out.columns]
    data = [[Paragraph(_wrap_text_for_display(c, width=24), header_style) for c in cols]]
    for _, row in out.iterrows():
        data.append([Paragraph(_wrap_text_for_display(v), body_style) for v in row.tolist()])
    weights = []
    for col in cols:
        max_len = max([len(str(col))] + [len(str(v)) for v in out[col].tolist()])
        weights.append(min(max(max_len, 8), 40))
    total_width = 18.0 * cm
    weight_sum = sum(weights) if weights else 1
    col_widths = [total_width * (w / weight_sum) for w in weights]
    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _pdf_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=16, fontweight="bold", va="top")
    y = 0.92
    for raw_line in lines:
        wrapped = textwrap.wrap(raw_line, width=112) if len(raw_line) > 112 else [raw_line]
        for line in wrapped:
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                fig.text(0.06, 0.96, f"{title} (suite)", fontsize=16, fontweight="bold", va="top")
                y = 0.92
            fig.text(0.06, y, line, fontsize=10, va="top")
            y -= 0.022
        if not raw_line.strip():
            y -= 0.01
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_table_pages(pdf: PdfPages, title: str, df: pd.DataFrame, rows_per_page: int = 18) -> None:
    if df is None or df.empty:
        _pdf_text_page(pdf, title, ["Aucune donnée disponible."])
        return
    start = 0
    page_idx = 1
    while start < len(df):
        chunk = _format_display_df(df.iloc[start : start + rows_per_page].copy())
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax.axis("off")
        heading = title if page_idx == 1 else f"{title} (suite {page_idx})"
        fig.text(0.06, 0.96, heading, fontsize=15, fontweight="bold", va="top")
        table = ax.table(
            cellText=chunk.astype(str).values,
            colLabels=[str(c) for c in chunk.columns],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.2)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            if row == 0:
                cell.set_facecolor("#D9E2F3")
                cell.set_text_props(weight="bold")
        plt.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.04)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        start += rows_per_page
        page_idx += 1


def _pdf_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=15, fontweight="bold", va="top")
    img = plt.imread(str(image_path))
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.78])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def _md_image(image_path: Path, alt: str) -> str:
    return f"![{alt}](figures/{image_path.name})\n\n_{alt}_\n"


def _executive_lines(results: dict[str, Any]) -> list[str]:
    hypotheses_df = results["hypotheses_df"]
    decision_df = results["decision_df"]
    support = {row["hypothesis"]: row["support_level"] for _, row in hypotheses_df.iterrows()}
    silent_corr = results["silent"]["correlations_df"].set_index("target")
    rho_perf = float(silent_corr.loc["Score_perf_tsk", "rho"]) if "Score_perf_tsk" in silent_corr.index else np.nan
    rho_cf = float(silent_corr.loc["c_factor", "rho"]) if "c_factor" in silent_corr.index else np.nan
    tas_decision = decision_df.loc[decision_df["construct"] == "TAS", "decision"].iloc[0]
    trs_decision = decision_df.loc[decision_df["construct"] == "TRS", "decision"].iloc[0]
    return [
        "1. H2 est l'hypothese la mieux soutenue empiriquement : la coordination explicite totale (I_TAS_b + I_TRS_b) est associee negativement a la performance et au c-factor.",
        f"2. H1 n'est soutenue que partiellement : les inversions ciblees aident surtout TMS, sans corriger le pattern inverse global (H1={support.get('H1', 'NA')}).",
        f"3. H3 est egalement fortement soutenue : TMS v1 ne tient pas comme indice latent et TAS v1 est structurellement confondu par la redondance interne (H3={support.get('H3', 'NA')}).",
        f"4. Spec v2 recommandee : TMS en features separees ; TAS={tas_decision} ; TRS={trs_decision}.",
        f"5. Effet cle : rho(I_TAS_b + I_TRS_b, Score_perf_tsk)={_fmt(rho_perf)} et rho(I_TAS_b + I_TRS_b, c_factor)={_fmt(rho_cf)} sur n=8 ; ce resultat est compatible avec une logique de division silencieuse du travail.",
    ]


def _grid_extremes(grid_df: pd.DataFrame, index_name: str, ascending: bool) -> pd.DataFrame:
    sub = grid_df[grid_df["index_name"] == index_name].copy()
    return sub.sort_values(["rho_score_perf", "alpha_cronbach"], ascending=[ascending, ascending]).head(5)


def write_markdown_report(results: dict[str, Any], output_dir: Path) -> Path:
    signs_targeted = results["signs"]["targeted_df"].copy()
    signs_grid = results["signs"]["grid_df"].copy()
    tms_df = results["tms"]["decomposition_df"].copy()
    tms_ols = results["tms"]["ols_df"].copy()
    tms_ols_summary = results["tms"]["ols_summary_df"].copy()
    tms_vif = results["tms"]["vif_df"].copy()
    tas_summary = results["tas"]["summary_df"].copy()
    tas_face = results["tas"]["face_df"].copy()
    silent_profiles = results["silent"]["profiles_df"].copy()
    silent_corr = results["silent"]["correlations_df"].copy()
    silent_comparisons = results["silent"]["comparisons_df"].copy()
    silent_strata = results["silent"]["strata_df"].copy()
    decision_df = results["decision_df"].copy()
    hypotheses_df = results["hypotheses_df"].copy()
    figures_dir = output_dir / "figures"

    lines: list[str] = []
    lines.append("# Rapport indices comportementaux v2\n")
    lines.append("## 1. Resume executif\n")
    for line in _executive_lines(results):
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 2. Bloc A — Inversion des signes\n")
    lines.append(
        "Les tests d'inversion evaluent si le pattern inverse observe en v1 peut etre explique par des signes theoriques mal adaptes au contexte BIM-VR a roles sequentiels. "
        "Le tableau ci-dessous compare la specification initiale et la variante ciblee v1-inv pour chaque indice."
    )
    lines.append("")
    lines.append(_markdown_table(signs_targeted[[
        "index_name", "variant", "feature_signs", "alpha_cronbach", "omega_total", "rho_questionnaire", "rho_score_perf", "perm_p_score_perf", "loo_min_score_perf", "loo_max_score_perf"
    ]]))
    lines.append("")
    grid_counts = (
        signs_grid.groupby("index_name")[["alpha_ge_050", "rho_score_perf_positive", "joint_support_alpha_ge_050_and_perf_positive"]]
        .sum()
        .reset_index()
        .rename(columns={
            "alpha_ge_050": "n_alpha_ge_050",
            "rho_score_perf_positive": "n_rho_perf_positive",
            "joint_support_alpha_ge_050_and_perf_positive": "n_joint_support",
        })
    )
    lines.append("### 2.1 Resume du grid search\n")
    lines.append(_markdown_table(grid_counts))
    for index_name in ["I_TMS_b", "I_TAS_b", "I_TRS_b"]:
        lines.append(f"### 2.{['I_TMS_b','I_TAS_b','I_TRS_b'].index(index_name)+2} {index_name} — meilleures et pires combinaisons\n")
        lines.append("Top 5 par rho avec Score_perf_tsk :\n")
        lines.append(_markdown_table(_grid_extremes(signs_grid, index_name, ascending=False)[["combo_id", "feature_signs", "alpha_cronbach", "rho_questionnaire", "rho_score_perf"]]))
        lines.append("Bottom 5 par rho avec Score_perf_tsk :\n")
        lines.append(_markdown_table(_grid_extremes(signs_grid, index_name, ascending=True)[["combo_id", "feature_signs", "alpha_cronbach", "rho_questionnaire", "rho_score_perf"]]))
    if results["questionnaire_heatmap_path"].exists():
        lines.append(_md_image(results["questionnaire_heatmap_path"], "Heatmap comparative v1 / v1-inv / v2 sur les correlations questionnaires principales"))
    if results["signs"]["histogram_path"].exists():
        lines.append(_md_image(results["signs"]["histogram_path"], "Distribution des correlations performance dans le grid search"))

    lines.append("## 3. Bloc B — Decomposition de I_TMS_b\n")
    lines.append(
        "Compte tenu de l'alpha negatif en v1, les trois features TMS sont traitees separement. Les corrélations sont rapportees sans imposer une aggregation a priori."
    )
    lines.append("")
    lines.append(_markdown_table(tms_df[[
        "feature_name", "analysis_family", "target", "n", "rho", "p_value", "ci95_low_boot", "ci95_high_boot", "perm_p_value", "loo_min_rho", "loo_max_rho"
    ]]))
    lines.append("### 3.1 OLS exploratoire Score_perf_tsk ~ features TMS\n")
    lines.append(_markdown_table(tms_ols_summary))
    lines.append(_markdown_table(tms_ols))
    lines.append(_markdown_table(tms_vif))

    lines.append("## 4. Bloc C — Variantes TAS\n")
    lines.append(
        "Trois variantes sont comparees : la version v1, un proxy gaze pur sur shared_obj_ratio, et une version reduite gaze+affect sans doublon metrique. "
        "La feature face_facial_synchrony est egalement analysee de maniere autonome."
    )
    lines.append("")
    lines.append(_markdown_table(tas_summary))
    lines.append("### 4.1 face_facial_synchrony comme proxy autonome\n")
    lines.append(_markdown_table(tas_face))

    lines.append("## 5. Bloc D — Division silencieuse du travail\n")
    lines.append(
        "L'hypothese H2 postule que les groupes les plus performants sont ceux qui coordonnent peu explicitement, tout en conservant un c-factor normal ou eleve. "
        "Les analyses ci-dessous portent uniquement sur le sous-echantillon predictif VR+TCI+performance (n=8)."
    )
    lines.append("")
    lines.append(_markdown_table(silent_corr))
    lines.append("### 5.1 Profils silencieux vs verbaux\n")
    lines.append(_markdown_table(silent_profiles[[
        "group_id", "I_TAS_b", "I_TRS_b", "coordination_explicite_totale", "profile_coordination", "Score_perf_tsk", "c_factor", "ratio_M1_M2"
    ]]))
    lines.append("### 5.2 Comparaisons par profil\n")
    lines.append(_markdown_table(silent_comparisons))
    lines.append("### 5.3 Stratification par c-factor\n")
    lines.append(_markdown_table(silent_strata))
    if results["silent"]["scatter_path"].exists():
        lines.append(_md_image(results["silent"]["scatter_path"], "Scatter de la coordination explicite totale vers la performance"))
    if results["silent"]["boxplot_path"].exists():
        lines.append(_md_image(results["silent"]["boxplot_path"], "Boxplot de performance par profil silencieux / verbal"))

    lines.append("## 6. Bloc E — Specification v2 retenue\n")
    lines.append(
        "La specification v2 n'est pas choisie pour maximiser une statistique unique. Elle privilegie la defensibilite theorique du mapping feature → sous-systeme, informee par les diagnostics empiriques A–D."
    )
    lines.append("")
    lines.append(_markdown_table(decision_df))

    lines.append("## 7. Articulation theorique\n")
    lines.append(
        "Pris ensemble, les resultats v2 sont compatibles avec une lecture contexte-dependante des sous-systemes transactifs : dans un environnement BIM-VR a roles sequentiels, "
        "une plus forte coordination explicite peut indexer une charge de regulation ou une friction interactionnelle plutot qu'un meilleur fonctionnement collectif. "
        "Cette lecture est coherente avec l'idee, chez Kommol et al. (2025), que les groupes nouvellement formes n'activent pas necessairement les memes signatures que des equipes matures."
    )
    lines.append("")
    lines.append("## 8. Limites et cautions\n")
    lines.append(
        "Le n reste faible (12 groupes VR pour INV+questionnaires ; 8 groupes VR+TCI+performance pour la prediction), ce qui limite toute generalisation. "
        "Le grid search sur les signes est un diagnostic exploratoire et ne doit pas etre lu comme une procedure de selection opportuniste. "
        "Enfin, les proxys comportementaux restent dependants du contexte BIM-VR et ne valent pas validation generalisable hors de cette structure de tache."
    )
    lines.append("")
    lines.append("## 9. Prochaines etapes\n")
    lines.append(
        "La suite naturelle consiste a reinjecter la specification v2 dans les regressions finales, puis a verifier sa stabilite sur un effectif plus large et, idealement, avec des donnees temporelles permettant de modeliser la recursion IMOI."
    )
    lines.append("")
    lines.append("## 10. Soutien empirique des hypotheses\n")
    lines.append(_markdown_table(hypotheses_df))

    path = output_dir / "rapport_indices_v2.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_pdf_report(results: dict[str, Any], markdown_path: Path, output_dir: Path) -> Path:
    signs_targeted = results["signs"]["targeted_df"].copy()
    signs_grid = results["signs"]["grid_df"].copy()
    tms_df = results["tms"]["decomposition_df"].copy()
    tms_ols = results["tms"]["ols_df"].copy()
    tms_ols_summary = results["tms"]["ols_summary_df"].copy()
    tms_vif = results["tms"]["vif_df"].copy()
    tas_summary = results["tas"]["summary_df"].copy()
    tas_face = results["tas"]["face_df"].copy()
    silent_profiles = results["silent"]["profiles_df"].copy()
    silent_corr = results["silent"]["correlations_df"].copy()
    silent_comparisons = results["silent"]["comparisons_df"].copy()
    silent_strata = results["silent"]["strata_df"].copy()
    decision_df = results["decision_df"].copy()
    hypotheses_df = results["hypotheses_df"].copy()
    pdf_path = output_dir / "rapport_indices_v2.pdf"

    if HAS_REPORTLAB:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="SmallV2", parent=styles["Normal"], fontSize=8, leading=10))
        elems: list[Any] = []

        elems.append(_paragraph("Rapport indices comportementaux v2", styles, "Heading1"))
        elems.append(Spacer(1, 0.15 * inch))
        for line in _executive_lines(results):
            elems.append(_paragraph(line, styles))
            elems.append(Spacer(1, 0.04 * inch))

        elems.append(PageBreak())
        elems.append(_paragraph("Bloc A — Inversion des signes", styles, "Heading2"))
        elems.append(_paragraph("Comparaison v1 / v1-inv par indice", styles))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_table_from_df(signs_targeted[[
            "index_name", "variant", "feature_signs", "alpha_cronbach", "omega_total", "rho_questionnaire", "rho_score_perf", "perm_p_score_perf"
        ]]))
        elems.append(Spacer(1, 0.08 * inch))
        grid_counts = (
            signs_grid.groupby("index_name")[["alpha_ge_050", "rho_score_perf_positive", "joint_support_alpha_ge_050_and_perf_positive"]]
            .sum()
            .reset_index()
        )
        elems.append(_table_from_df(grid_counts))
        if results["questionnaire_heatmap_path"].exists():
            elems.append(PageBreak())
            elems.append(_paragraph("Figure — Heatmap comparative v1 / v1-inv / v2", styles, "Heading3"))
            img = Image(str(results["questionnaire_heatmap_path"]), width=5.4 * inch, height=7.0 * inch)
            img.hAlign = "CENTER"
            elems.append(img)
        if results["signs"]["histogram_path"].exists():
            elems.append(PageBreak())
            elems.append(_paragraph("Figure — Distribution des correlations performance dans le grid search", styles, "Heading3"))
            img = Image(str(results["signs"]["histogram_path"]), width=6.1 * inch, height=7.4 * inch)
            img.hAlign = "CENTER"
            elems.append(img)

        elems.append(PageBreak())
        elems.append(_paragraph("Bloc B — Decomposition de I_TMS_b", styles, "Heading2"))
        elems.append(_table_from_df(tms_df, max_rows=24))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_paragraph("OLS exploratoire", styles, "Heading3"))
        elems.append(_table_from_df(tms_ols_summary))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_table_from_df(tms_ols))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_table_from_df(tms_vif))

        elems.append(PageBreak())
        elems.append(_paragraph("Bloc C — Variantes TAS", styles, "Heading2"))
        elems.append(_table_from_df(tas_summary))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_paragraph("face_facial_synchrony comme proxy autonome", styles, "Heading3"))
        elems.append(_table_from_df(tas_face))

        elems.append(PageBreak())
        elems.append(_paragraph("Bloc D — Division silencieuse", styles, "Heading2"))
        elems.append(_table_from_df(silent_corr))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_table_from_df(silent_comparisons))
        if results["silent"]["scatter_path"].exists():
            elems.append(PageBreak())
            elems.append(_paragraph("Figure — Coordination explicite totale vs performance", styles, "Heading3"))
            img = Image(str(results["silent"]["scatter_path"]), width=6.0 * inch, height=4.5 * inch)
            img.hAlign = "CENTER"
            elems.append(img)
        if results["silent"]["boxplot_path"].exists():
            elems.append(Spacer(1, 0.08 * inch))
            img = Image(str(results["silent"]["boxplot_path"]), width=5.7 * inch, height=4.1 * inch)
            img.hAlign = "CENTER"
            elems.append(img)

        elems.append(PageBreak())
        elems.append(_paragraph("Bloc E — Specification v2", styles, "Heading2"))
        elems.append(_table_from_df(decision_df))
        elems.append(Spacer(1, 0.08 * inch))
        elems.append(_paragraph("Soutien empirique des hypotheses", styles, "Heading3"))
        elems.append(_table_from_df(hypotheses_df))
        elems.append(Spacer(1, 0.10 * inch))
        elems.append(_paragraph(f"Version markdown source : {markdown_path.name}", styles, "SmallV2"))

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.build(elems)
        return pdf_path

    with PdfPages(pdf_path) as pdf:
        _pdf_text_page(pdf, "Resume executif", _executive_lines(results))
        _pdf_table_pages(pdf, "Bloc A — Inversion des signes", signs_targeted, rows_per_page=12)
        _pdf_table_pages(pdf, "Bloc B — Decomposition TMS", tms_df, rows_per_page=18)
        _pdf_table_pages(pdf, "Bloc C — Variantes TAS", tas_summary, rows_per_page=12)
        _pdf_table_pages(pdf, "Bloc C — face_facial_synchrony", tas_face, rows_per_page=12)
        _pdf_table_pages(pdf, "Bloc D — Division silencieuse", silent_comparisons, rows_per_page=12)
        _pdf_image_page(pdf, "Figure — Heatmap comparative v1 / v1-inv / v2", results["questionnaire_heatmap_path"])
        _pdf_image_page(pdf, "Figure — Coordination explicite totale vs performance", results["silent"]["scatter_path"])
        _pdf_image_page(pdf, "Figure — Performance par profil silencieux / verbal", results["silent"]["boxplot_path"])
        _pdf_image_page(pdf, "Figure — Grid search performance", results["signs"]["histogram_path"])
        _pdf_table_pages(pdf, "Bloc E — Specification v2", decision_df, rows_per_page=12)
        _pdf_table_pages(pdf, "Soutien empirique des hypotheses", hypotheses_df, rows_per_page=12)
        _pdf_text_page(pdf, "Version markdown", [markdown_path.name])
    return pdf_path


def generate_behavioral_indices_v2_report(output_dir: Path | None = None) -> dict[str, Path]:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    results = build_behavioral_indices_v2(output_dir=out)
    markdown_path = write_markdown_report(results, out)
    pdf_path = write_pdf_report(results, markdown_path, out)
    return {
        "output_dir": out,
        "markdown_path": markdown_path,
        "pdf_path": pdf_path,
        "log_path": results["log_path"],
        "figures_dir": results["figures_dir"],
        "indices_v2_path": out / "indices_v2.csv",
    }


def main() -> None:
    outputs = generate_behavioral_indices_v2_report()
    print(f"[OK] Sorties v2 : {outputs['output_dir']}")
    print(f"[OK] PDF v2     : {outputs['pdf_path']}")


if __name__ == "__main__":
    main()
