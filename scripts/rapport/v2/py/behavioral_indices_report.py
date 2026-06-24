from __future__ import annotations

import os
import sys
import textwrap
import hashlib
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

from py.behavioral_indices_analyze import INDEX_ORDER, PERFORMANCE_TARGETS, run_all_analyses
from py.behavioral_indices_build import OUTPUT_DIR, build_behavioral_indices_pipeline


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
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].apply(lambda x: "NA" if pd.isna(x) else str(int(x)))
        elif pd.api.types.is_numeric_dtype(out[col]):
            series = pd.to_numeric(out[col], errors="coerce")
            finite = series.dropna()
            should_render_as_int = (
                col_low == "n"
                or col_low.startswith("n_")
                or col_low.endswith("_n")
                or "n_group" in col_low
                or "n_feature" in col_low
                or "min_feature" in col_low
            )
            if should_render_as_int and not finite.empty and np.allclose(finite.to_numpy(), np.round(finite.to_numpy()), atol=1e-10):
                out[col] = series.apply(lambda x: "NA" if pd.isna(x) else str(int(round(float(x)))))
            else:
                out[col] = series.apply(lambda x: _fmt(x, digits=digits))
    return out


def _wrap_text_for_display(value: Any, width: int = 38) -> str:
    text = str(value) if value is not None else "NA"
    if text == "nan":
        text = "NA"
    return "<br/>".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_Aucune donnée disponible._\n"
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()
    out = _format_display_df(out).fillna("NA")
    cols = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in out.iterrows():
        vals = [str(v).replace("\n", " ") for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _paragraph(text: str, styles, style_name: str = "Normal"):
    return Paragraph(text.replace("\n", "<br/>"), styles[style_name])


def _table_from_df(df: pd.DataFrame, max_rows: int | None = None):
    if df is None or df.empty:
        df = pd.DataFrame({"Info": ["Aucune donnée disponible"]})
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()
    out = _format_display_df(out).fillna("NA")

    header_style = ParagraphStyle(
        name="CellHeader",
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8,
        alignment=1,
    )
    body_style = ParagraphStyle(
        name="CellBody",
        fontName="Helvetica",
        fontSize=7,
        leading=8,
    )

    cols = [str(c) for c in out.columns]
    wrapped_rows = []
    for _, row in out.iterrows():
        wrapped_rows.append([Paragraph(_wrap_text_for_display(v), body_style) for v in row.tolist()])
    data = [[Paragraph(_wrap_text_for_display(c, width=24), header_style) for c in cols]] + wrapped_rows

    char_weights = []
    for col in cols:
        max_len = max([len(str(col))] + [len(str(v)) for v in out[col].tolist()])
        char_weights.append(min(max(max_len, 8), 40))
    total_width = 18.0 * cm
    weight_sum = sum(char_weights) if char_weights else 1
    col_widths = [total_width * (w / weight_sum) for w in char_weights]

    table = Table(data, repeatRows=1, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
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


def _image(path: Path, width: float = 6.2 * inch, max_height: float | None = None):
    if not path.exists():
        return None
    img = Image(str(path))
    img.drawWidth = width
    img.drawHeight = img.imageHeight * (width / img.imageWidth)
    if max_height is not None and img.drawHeight > max_height:
        scale = max_height / img.drawHeight
        img.drawWidth *= scale
        img.drawHeight *= scale
    img.hAlign = "CENTER"
    return img


def _index_method_table(bundle: dict[str, Any]) -> pd.DataFrame:
    rows = []
    feature_selection = bundle["bundle"]["feature_selection_df"]
    for index_name, spec in bundle["bundle"]["resolved_specs"].items():
        retained = [f["feature_name"] for f in spec["retained_features"]]
        excluded = feature_selection[
            (feature_selection["index_name"] == index_name)
            & (feature_selection["status"] != "retained")
        ]["candidate_feature"].astype(str).tolist()
        rows.append(
            {
                "index_name": index_name,
                "description": spec["description"],
                "features_retenues": "<br/>".join(retained) if retained else "Aucune",
                "features_exclues": "<br/>".join(excluded) if excluded else "Aucune",
                "min_features": spec["min_features"],
            }
        )
    return pd.DataFrame(rows)


def _descriptives_table(bundle: dict[str, Any]) -> pd.DataFrame:
    indices = bundle["bundle"]["indices_df"].copy()
    rows = []
    for index_name in INDEX_ORDER:
        series = pd.to_numeric(indices[index_name], errors="coerce")
        rows.append(
            {
                "index_name": index_name,
                "n": int(series.notna().sum()),
                "mean": series.mean(),
                "sd": series.std(ddof=1),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _predictive_summary_table(bundle: dict[str, Any]) -> pd.DataFrame:
    corr_df = bundle["performance"]["corr_df"].copy()
    keep = corr_df[corr_df["target"].isin(PERFORMANCE_TARGETS)][
        [
            "index_name",
            "target",
            "n",
            "rho_spearman",
            "p_value",
            "p_fdr",
            "ci95_low_boot",
            "ci95_high_boot",
            "perm_p_value",
            "loo_min_rho",
            "loo_max_rho",
        ]
    ].copy()
    keep = keep.rename(
        columns={
            "rho_spearman": "rho",
            "p_value": "p",
            "ci95_low_boot": "ci95_low",
            "ci95_high_boot": "ci95_high",
            "perm_p_value": "perm_p",
        }
    )
    keep["ci95_boot"] = keep.apply(lambda r: f"[{_fmt(r['ci95_low'])}; {_fmt(r['ci95_high'])}]", axis=1)
    keep["loo_range"] = keep.apply(lambda r: f"[{_fmt(r['loo_min_rho'])}; {_fmt(r['loo_max_rho'])}]", axis=1)
    keep = keep[["index_name", "target", "n", "rho", "p", "p_fdr", "ci95_boot", "perm_p", "loo_range"]]
    return keep


def _questionnaire_corr_display_df(bundle: dict[str, Any]) -> pd.DataFrame:
    df = bundle["questionnaire"]["corr_df"].copy()
    if df.empty:
        return df
    df = df.rename(
        columns={
            "rho_spearman": "rho",
            "p_value": "p",
            "ci95_low_boot": "ci95_low",
            "ci95_high_boot": "ci95_high",
            "hypothesis": "hypothese",
        }
    )
    df["ci95_boot"] = df.apply(lambda r: f"[{_fmt(r['ci95_low'])}; {_fmt(r['ci95_high'])}]", axis=1)
    df["cible"] = df["target"]
    df["famille_test"] = df["family"]
    return df[["famille_test", "index_name", "cible", "n", "rho", "p", "p_fdr", "ci95_boot", "signif"]]


def _md_image(filename: str, width: int = 760, alt: str = "") -> str:
    caption = alt.strip() or filename
    return f"![{caption}](figures/{filename})\n\n_{caption}_\n"


def _build_executive_summary(bundle: dict[str, Any]) -> list[str]:
    internal = bundle["internal"]["summary"].set_index("index_name")
    perf_corr = bundle["performance"]["corr_df"]
    score_rows = perf_corr[perf_corr["target"] == "Score_perf_tsk"].copy()

    lines = [
        f"1. Echantillon VR comportemental utilisable : n={len(bundle['bundle']['indices_df'])} groupes ; sous-echantillon predictif VR+TCI : n={len(bundle['performance']['perf_df'])}.",
        "2. Les indices sont des proxys comportementaux pre-specifies ; deux candidates du brief ont ete retirees car absentes de inv_features_config.py (`skill_congruence_mean`, `mutual_gaze`).",
        "3. La coherence interne minimale est jugee informative si alpha/omega/inter-item convergent au moins faiblement dans la meme direction ; les valeurs doivent etre lues comme diagnostics de composition et non comme validation psychometrique stricte.",
        "4. Les corrélations indices-performance sont interpretees de maniere strictement exploratoire compte tenu du faible n et de la stabilite leave-one-out / permutation.",
        "5. Verdict factuel : "
        + "; ".join(
            [
                f"{idx} alpha={_fmt(internal.loc[idx, 'alpha_cronbach'])}, rho_perf={_fmt(score_rows.loc[score_rows['index_name'] == idx, 'rho_spearman'].iloc[0]) if not score_rows.loc[score_rows['index_name'] == idx].empty else 'NA'}"
                for idx in INDEX_ORDER
                if idx in internal.index
            ]
        ),
    ]
    return lines


def _wrap_block(text: str, width: int = 115) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _pdf_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=16, fontweight="bold", va="top")
    y = 0.92
    for raw_line in lines:
        wrapped = _wrap_block(raw_line, width=112) if len(raw_line) > 112 else [raw_line]
        for line in wrapped:
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                fig.text(0.06, 0.96, f"{title} (suite)", fontsize=16, fontweight="bold", va="top")
                y = 0.92
            fig.text(0.06, y, line, fontsize=10, va="top", family="monospace" if line.startswith("|") else None)
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
        chunk = df.iloc[start : start + rows_per_page].copy()
        chunk = chunk.fillna("NA")
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


def write_markdown_report(bundle: dict[str, Any], output_dir: Path) -> Path:
    feature_method_df = _index_method_table(bundle)
    desc_df = _descriptives_table(bundle)
    internal_df = bundle["internal"]["summary"].copy().rename(
        columns={
            "n_groups_complete": "n_groupes",
            "n_features": "n_features",
            "alpha_cronbach": "alpha",
            "omega_total": "omega",
            "mean_interitem_spearman": "r_interitem_moy",
            "icc2k": "icc2k",
            "icc2k_n_complete": "n_icc",
        }
    )
    item_total_df = bundle["internal"]["item_total"].copy().rename(
        columns={
            "item_total_rho": "rho_item_total",
            "item_total_p_value": "p_item_total",
            "index_rho_raw": "rho_indice",
        }
    )
    q_corr_df = _questionnaire_corr_display_df(bundle)
    perf_corr_df = _predictive_summary_table(bundle)
    config_df = bundle["configuration"].copy() if isinstance(bundle["configuration"], pd.DataFrame) else pd.DataFrame(bundle["configuration"])
    sensitivity_df = bundle["sensitivity"]["mean_median_df"].copy().rename(
        columns={"rho_mean_vs_median": "rho_mean_median", "p_value": "p"}
    )
    ols_df = bundle["performance"]["coef_table"].copy()
    vif_df = bundle["performance"]["vif_df"].copy()
    exec_lines = _build_executive_summary(bundle)
    figures_dir = output_dir / "figures"

    lines: list[str] = []
    lines.append("# Rapport indices comportementaux VR\n")
    lines.append("## Récapitulatif exécutif\n")
    for line in exec_lines:
        lines.append(f"- {line}")
    lines.append("")

    lines.append("## 1. Cadre théorique et spécification\n")
    lines.append(
        "Cette analyse remplace une lecture inférentielle de la PCA exploratoire par une stratégie pré-spécifiée "
        "d'indices comportementaux alignés sur le cadre tripartite Woolley & Gupta (2024) / Kommol, Riedl & Woolley (2025). "
        "Les indices `I_TMS_b`, `I_TAS_b` et `I_TRS_b` sont traités comme des proxys comportementaux transactifs et non comme des mesures directes des construits."
    )
    lines.append("")

    lines.append("## 2. Méthode\n")
    lines.append(
        "Les features candidates ont été vérifiées dans `inv_features_config.py`, puis filtrées selon leur présence effective dans les tables VR et un taux de NaN <= 20 % sur l'échantillon VR. "
        "Les z-scores sont calculés sur l'échantillon VR, puis signés selon l'hypothèse théorique. "
        "L'indice principal est la moyenne des z-scores signés ; une version médiane est calculée en sensibilité."
    )
    lines.append("")
    lines.append("### 2.1 Features retenues / exclues\n")
    lines.append(_markdown_table(feature_method_df))
    lines.append("### 2.2 Drapeau `theoretical_core`\n")
    lines.append(
        "Le flag `theoretical_core` existe dans la configuration mais n'est pas renseigné pour les candidates retenues dans l'état actuel du dépôt. "
        "Il a donc été vérifié et documenté comme limite documentaire, sans être utilisé comme critère d'exclusion strict."
    )
    lines.append("")

    lines.append("## 3. Descriptifs\n")
    lines.append(
        "Les descriptifs sont rapportés sur l'ensemble de l'échantillon VR retenu pour les indices comportementaux. "
        "Aucune stratification par scénario n'est conservée dans ce livrable, les contrastes S1/S2 n'étant pas interprétés comme des différences significatives."
    )
    lines.append("")
    lines.append(_markdown_table(desc_df))

    lines.append("## 4. Cohérence interne\n")
    lines.append(
        "Les indicateurs ci-dessous doivent être lus comme des diagnostics informatifs sur la cohérence de l'agrégation comportementale. "
        "Ils ne constituent pas une validation psychométrique stricte au sens d'une échelle multi-items classique."
    )
    lines.append("")
    lines.append(_markdown_table(internal_df))
    lines.append("")
    for index_name in INDEX_ORDER:
        sub = item_total_df[item_total_df["index_name"] == index_name].copy()
        lines.append(f"### 4.{INDEX_ORDER.index(index_name) + 1} {index_name} — item-total\n")
        lines.append(_markdown_table(sub))
        corr_path = bundle["internal"]["interitem_paths"].get(index_name)
        if corr_path is not None:
            corr_df = pd.read_csv(corr_path, index_col=0)
            lines.append("Corrélations inter-items (Spearman) :\n")
            lines.append(_markdown_table(corr_df.reset_index().rename(columns={"index": "feature"})))
        lines.append("")

    lines.append("## 5. Validité convergente (questionnaires)\n")
    lines.append(
        "Les corrélations de Spearman ci-dessous sont corrigées par FDR au niveau de la famille. "
        "Les intervalles de confiance bootstrap (1000 itérations, seed=42) sont fournis à titre exploratoire."
    )
    lines.append("")
    lines.append(_markdown_table(q_corr_df))
    if bundle["questionnaire"]["heatmap_path"].exists():
        lines.append(_md_image(bundle["questionnaire"]["heatmap_path"].name, alt="Heatmap indices questionnaires"))

    lines.append("## 6. Validité prédictive (performance)\n")
    lines.append(
        "Les analyses prédictives sont restreintes au sous-échantillon VR avec `Score_perf_tsk` et `c_score` disponibles simultanément, "
        "afin de conserver un n constant dans cette section. Toute interprétation est strictement exploratoire."
    )
    lines.append("")
    lines.append(_markdown_table(perf_corr_df))
    if not ols_df.empty:
        lines.append("### 6.1 Régression OLS complète\n")
        lines.append(_markdown_table(ols_df))
        if not vif_df.empty:
            lines.append("### 6.2 VIF\n")
            lines.append(_markdown_table(vif_df))
    if bundle["performance"]["heatmap_path"].exists():
        lines.append(_md_image(bundle["performance"]["heatmap_path"].name, alt="Heatmap indices performance"))
    if bundle["performance"]["scatter_path"].exists():
        lines.append(_md_image(bundle["performance"]["scatter_path"].name, alt="Scatter indices performance"))

    lines.append("## 7. Robustesse\n")
    lines.append("### 7.1 Sensibilité moyenne vs médiane\n")
    lines.append(_markdown_table(sensitivity_df))
    if bundle["sensitivity"]["heatmap_path"].exists():
        lines.append(_md_image(bundle["sensitivity"]["heatmap_path"].name, alt="Heatmap mean median"))
    lines.append("### 7.2 Analyse configurationnelle\n")
    lines.append(_markdown_table(config_df))

    lines.append("## 8. Discussion des patterns\n")
    lines.append(
        "Les convergences effectivement observées doivent être appréciées au regard du faible effectif VR et de la nature proxy des indices. "
        "Lorsque les corrélations convergent vers les dimensions questionnaire attendues, cela soutient une validité convergente faible à modérée ; "
        "l'absence de convergence ou l'instabilité leave-one-out doit au contraire être lue comme un signal de fragilité, pas comme une preuve négative définitive."
    )
    lines.append("")
    lines.append("## 9. Limites\n")
    lines.append("- N faible sur le sous-échantillon prédictif VR+TCI : aucune généralisation populationnelle n'est défendable.")
    lines.append("- Les indices agrègent des features comportementales et non des items psychométriques ; alpha / omega / ICC sont donc purement informatifs.")
    lines.append("- L'absence de gaze en PC empêche toute comparaison transversale robuste de l'indice TAS comportemental.")
    lines.append("- Le pipeline reste à fenêtre unique ; aucune dynamique récursive IMOI n'est modélisée ici.")
    lines.append("")
    lines.append("## 10. Perspective\n")
    lines.append(
        "Une validation factorielle plus forte exigerait un effectif substantiellement supérieur, une disponibilité complète des modalités et une modélisation temporelle des boucles transactives. "
        "Dans l'état actuel, ces indices servent surtout de pont méthodologique entre le pipeline INV et une future régression finale / écriture article IHM."
    )
    lines.append("")

    report_path = output_dir / "rapport_indices_comportementaux.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_executive_summary(bundle: dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "executive_summary.txt"
    path.write_text("\n".join(_build_executive_summary(bundle)) + "\n", encoding="utf-8")
    return path


def write_pdf_report(bundle: dict[str, Any], markdown_path: Path, output_dir: Path) -> Path:
    feature_method_df = _index_method_table(bundle)
    desc_df = _descriptives_table(bundle)
    internal_df = bundle["internal"]["summary"].copy().rename(
        columns={
            "n_groups_complete": "n_groupes",
            "n_features": "n_features",
            "alpha_cronbach": "alpha",
            "omega_total": "omega",
            "mean_interitem_spearman": "r_interitem_moy",
            "icc2k": "icc2k",
            "icc2k_n_complete": "n_icc",
        }
    )
    item_total_df = bundle["internal"]["item_total"].copy().rename(
        columns={
            "item_total_rho": "rho_item_total",
            "item_total_p_value": "p_item_total",
            "index_rho_raw": "rho_indice",
        }
    )
    q_corr_df = _questionnaire_corr_display_df(bundle)
    perf_corr_df = _predictive_summary_table(bundle)
    ols_df = bundle["performance"]["coef_table"].copy()
    vif_df = bundle["performance"]["vif_df"].copy()
    sensitivity_df = bundle["sensitivity"]["mean_median_df"].copy().rename(
        columns={"rho_mean_vs_median": "rho_mean_median", "p_value": "p"}
    )
    config_df = bundle["configuration"].copy() if isinstance(bundle["configuration"], pd.DataFrame) else pd.DataFrame(bundle["configuration"])
    figures_dir = output_dir / "figures"
    pdf_path = output_dir / "rapport_indices_comportementaux.pdf"

    if HAS_REPORTLAB:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8, leading=10))
        elements: list[Any] = []

        elements.append(_paragraph("Rapport indices comportementaux VR", styles, "Heading1"))
        elements.append(Spacer(1, 0.18 * inch))
        for line in _build_executive_summary(bundle):
            elements.append(_paragraph(line, styles))
            elements.append(Spacer(1, 0.05 * inch))

        elements.append(Spacer(1, 0.12 * inch))
        elements.append(_paragraph("1. Cadre théorique et spécification", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Cette analyse substitue à la PCA exploratoire une stratégie pré-spécifiée d'indices comportementaux alignés sur Woolley & Gupta (2024) et Kommol, Riedl & Woolley (2025). "
                "Les indices I_TMS_b, I_TAS_b et I_TRS_b sont traités comme des proxys comportementaux transactifs, et non comme des mesures directes des construits.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.10 * inch))
        elements.append(_paragraph("2. Méthode", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Les features candidates sont vérifiées dans inv_features_config.py, puis filtrées selon leur présence effective dans le dataset VR et un taux de NaN <= 20 %. "
                "Les z-scores sont calculés sur l'échantillon VR uniquement, puis signés selon l'hypothèse théorique. L'indice principal est la moyenne des z-scores signés ; une version médiane est calculée en sensibilité.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(_table_from_df(feature_method_df, max_rows=12))

        elements.append(PageBreak())
        elements.append(_paragraph("3. Descriptifs", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Les descriptifs sont rapportés sur l'ensemble de l'échantillon VR retenu pour les indices comportementaux. "
                "Aucune stratification par scénario n'est conservée dans ce livrable, les contrastes S1/S2 n'étant pas interprétés comme des différences significatives.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(_table_from_df(desc_df))

        elements.append(PageBreak())
        elements.append(_paragraph("4. Cohérence interne", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Ces indicateurs sont fournis comme diagnostics informatifs sur la cohérence de l'agrégation comportementale. Ils ne doivent pas être lus comme une validation psychométrique stricte d'échelle.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(_table_from_df(internal_df))
        for index_name in INDEX_ORDER:
            sub = item_total_df[item_total_df["index_name"] == index_name].copy()
            elements.append(Spacer(1, 0.10 * inch))
            elements.append(_paragraph(f"{index_name} — item-total", styles, "Heading3"))
            elements.append(_table_from_df(sub, max_rows=12))

        elements.append(PageBreak())
        elements.append(_paragraph("5. Validité convergente (questionnaires)", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Les corrélations de Spearman ci-dessous sont corrigées par FDR au niveau de la famille. Les intervalles de confiance bootstrap (1000 itérations, seed=42) sont fournis à titre exploratoire.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(_table_from_df(q_corr_df, max_rows=18))
        questionnaire_heatmap = bundle["questionnaire"]["heatmap_path"]
        if questionnaire_heatmap.exists():
            elements.append(PageBreak())
            elements.append(_paragraph("5.1 Figure - Heatmap indices vs questionnaires", styles, "Heading3"))
            elements.append(Spacer(1, 0.08 * inch))
            img = Image(str(questionnaire_heatmap), width=5.0 * inch, height=7.0 * inch)
            img.hAlign = "CENTER"
            elements.append(img)

        elements.append(PageBreak())
        elements.append(_paragraph("6. Validité prédictive (performance)", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Les analyses prédictives sont restreintes au sous-échantillon VR avec Score_perf_tsk et c_score disponibles simultanément afin de conserver un n constant. Toute interprétation reste strictement exploratoire.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(_table_from_df(perf_corr_df))
        if not ols_df.empty:
            elements.append(Spacer(1, 0.08 * inch))
            elements.append(_paragraph("6.1 Régression OLS complète", styles, "Heading3"))
            elements.append(_table_from_df(ols_df))
        if not vif_df.empty:
            elements.append(Spacer(1, 0.08 * inch))
            elements.append(_paragraph("6.2 VIF", styles, "Heading3"))
            elements.append(_table_from_df(vif_df))
        performance_heatmap = bundle["performance"]["heatmap_path"]
        if performance_heatmap.exists():
            elements.append(PageBreak())
            elements.append(_paragraph("6.3 Figure - Heatmap indices vs performance / c-factor", styles, "Heading3"))
            elements.append(Spacer(1, 0.08 * inch))
            img = Image(str(performance_heatmap), width=6.0 * inch, height=4.6 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        performance_scatter = bundle["performance"]["scatter_path"]
        if performance_scatter.exists():
            elements.append(PageBreak())
            elements.append(_paragraph("6.4 Figure - Scatter indices vs Score_perf_tsk", styles, "Heading3"))
            elements.append(Spacer(1, 0.08 * inch))
            img = Image(str(performance_scatter), width=4.8 * inch, height=7.4 * inch)
            img.hAlign = "CENTER"
            elements.append(img)

        elements.append(PageBreak())
        elements.append(_paragraph("7. Robustesse", styles, "Heading2"))
        elements.append(_paragraph("Sensibilité moyenne vs médiane", styles, "Heading3"))
        elements.append(_table_from_df(sensitivity_df))
        sensitivity_heatmap = bundle["sensitivity"]["heatmap_path"]
        if sensitivity_heatmap.exists():
            elements.append(Spacer(1, 0.08 * inch))
            img = Image(str(sensitivity_heatmap), width=4.8 * inch, height=4.8 * inch)
            img.hAlign = "CENTER"
            elements.append(img)
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(_paragraph("Analyse configurationnelle", styles, "Heading3"))
        elements.append(_table_from_df(config_df))

        elements.append(Spacer(1, 0.12 * inch))
        elements.append(_paragraph("8. Discussion des patterns", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Les convergences et divergences observées doivent être lues comme des signaux exploratoires, en dialogue avec Kommol et al. (2025), et non comme une validation structurelle ferme.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(_paragraph("9. Limites", styles, "Heading2"))
        elements.append(
            _paragraph(
                "N faible sur le sous-échantillon prédictif VR+TCI, proxys comportementaux plutôt qu'échelles, absence de gaze en PC et absence de dynamique temporelle récursive : ces limites bornent fortement la portée inférentielle du livrable.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(_paragraph("10. Perspective", styles, "Heading2"))
        elements.append(
            _paragraph(
                "Ces indices peuvent maintenant servir de base à une future intégration dans les régressions finales et à l'écriture article IHM, sous réserve d'un effectif plus large et d'une modélisation temporelle plus fine.",
                styles,
            )
        )
        elements.append(Spacer(1, 0.12 * inch))
        elements.append(_paragraph(f"Version markdown source : {markdown_path.name}", styles, "Small"))

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        doc.build(elements)
        return pdf_path

    with PdfPages(pdf_path) as pdf:
        _pdf_text_page(
            pdf,
            "Rapport indices comportementaux VR",
            _build_executive_summary(bundle),
        )
        _pdf_text_page(
            pdf,
            "1. Cadre théorique et spécification",
            [
                "Cette analyse substitue a la PCA exploratoire une strategie pre-specifiee d'indices comportementaux alignes sur Woolley & Gupta (2024) et Kommol, Riedl & Woolley (2025).",
                "Les indices I_TMS_b, I_TAS_b et I_TRS_b sont traites comme des proxys comportementaux transactifs et non comme des mesures directes des construits.",
            ],
        )
        _pdf_text_page(
            pdf,
            "2. Méthode",
            [
                "Les features candidates sont verifiees dans inv_features_config.py, puis filtrees selon leur presence effective dans le dataset VR et un taux de NaN <= 20%.",
                "Les z-scores sont calcules sur l'echantillon VR uniquement, puis signes selon l'hypothese theorique. L'indice primaire est la moyenne des z signes; une version mediane est calculee en sensibilite.",
                "Le flag theoretical_core a ete verifie et trace, mais non utilise comme critere d'exclusion strict car il n'est pas renseigne pour les candidates retenues dans l'etat actuel du depot.",
            ],
        )
        _pdf_table_pages(pdf, "2.1 Features retenues / exclues", feature_method_df, rows_per_page=12)
        _pdf_text_page(
            pdf,
            "3. Descriptifs des indices",
            [
                "Les descriptifs sont rapportes sur l'ensemble de l'echantillon VR retenu pour les indices comportementaux.",
                "Aucune stratification par scenario n'est conservee dans ce livrable, les contrastes S1/S2 n'etant pas interpretes comme des differences significatives.",
            ],
        )
        _pdf_table_pages(pdf, "3.1 Tableau descriptif des indices", desc_df, rows_per_page=12)
        _pdf_table_pages(pdf, "4. Cohérence interne", internal_df, rows_per_page=12)
        for index_name in INDEX_ORDER:
            sub = item_total_df[item_total_df["index_name"] == index_name].copy()
            _pdf_table_pages(pdf, f"4.x Item-total {index_name}", sub, rows_per_page=12)
            corr_path = bundle["internal"]["interitem_paths"].get(index_name)
            if corr_path is not None and corr_path.exists():
                corr_df = pd.read_csv(corr_path, index_col=0).reset_index().rename(columns={"index": "feature"})
                _pdf_table_pages(pdf, f"4.x Corrélations inter-items {index_name}", corr_df, rows_per_page=10)
        _pdf_table_pages(pdf, "5. Validité convergente", q_corr_df, rows_per_page=16)
        _pdf_image_page(pdf, "5.1 Heatmap indices vs questionnaires", bundle["questionnaire"]["heatmap_path"])
        _pdf_table_pages(pdf, "6. Validité prédictive", perf_corr_df, rows_per_page=14)
        if not ols_df.empty:
            _pdf_table_pages(pdf, "6.1 Régression OLS complète", ols_df, rows_per_page=12)
        if not vif_df.empty:
            _pdf_table_pages(pdf, "6.2 VIF", vif_df, rows_per_page=12)
        _pdf_image_page(pdf, "6.3 Heatmap indices vs performance / c-factor", bundle["performance"]["heatmap_path"])
        _pdf_image_page(pdf, "6.4 Scatter indices vs Score_perf_tsk", bundle["performance"]["scatter_path"])
        _pdf_table_pages(pdf, "7. Robustesse - moyenne vs médiane", sensitivity_df, rows_per_page=12)
        _pdf_image_page(pdf, "7.1 Heatmap moyenne vs médiane", bundle["sensitivity"]["heatmap_path"])
        _pdf_table_pages(pdf, "7.2 Analyse configurationnelle", config_df, rows_per_page=14)
        _pdf_text_page(
            pdf,
            "8. Discussion / 9. Limites / 10. Perspective",
            [
                "Les convergences et divergences observees doivent etre lues comme des signaux exploratoires, pas comme une validation structurelle ferme.",
                "Limites explicites : faible n sur le sous-echantillon predictif VR+TCI, proxys comportementaux plutot qu'echelles, absence de gaze en PC, absence de dynamique temporelle recursive.",
                "Perspective : ces indices constituent une base methodologique pour une future integration dans la regression finale et l'ecriture article IHM, sous reserve d'un echantillon plus large et d'une modelisation temporelle.",
                f"Version markdown source : {markdown_path.name}",
            ],
        )
    return pdf_path


def generate_behavioral_indices_report(output_dir: Path | None = None) -> dict[str, Path]:
    out = output_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    bundle = run_all_analyses(
        build_behavioral_indices_pipeline(output_dir=out, write_outputs=True),
        output_dir=out,
    )
    markdown_path = write_markdown_report(bundle, out)
    executive_path = write_executive_summary(bundle, out)
    pdf_path = write_pdf_report(bundle, markdown_path, out)
    log_path = out / "log_indices.txt"
    bundle["bundle"]["log"].write(log_path)
    return {
        "output_dir": out,
        "markdown_path": markdown_path,
        "pdf_path": pdf_path,
        "executive_summary_path": executive_path,
        "log_path": log_path,
        "figures_dir": out / "figures",
        "indices_csv": out / "behavioral_indices_vr.csv",
    }


def main() -> None:
    outputs = generate_behavioral_indices_report(OUTPUT_DIR)
    print(f"[OK] Rapport markdown : {outputs['markdown_path']}")
    print(f"[OK] Rapport PDF      : {outputs['pdf_path']}")
    print(f"[OK] Sorties          : {outputs['output_dir']}")


if __name__ == "__main__":
    main()
