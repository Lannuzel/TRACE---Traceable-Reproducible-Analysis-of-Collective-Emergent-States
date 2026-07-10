# -*- coding: utf-8 -*-
"""
Helpers de corrélation, de tableaux et de rendu associé pour le rapport CI.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle
from scipy.stats import spearmanr
from xml.sax.saxutils import escape as _xml_escape


N_MIN_SIG = 6
MAX_ROWS_MD = 120
MAX_ROWS_PDF = 40
P_COLOR_THRESH = [(0.01, colors.orange), (0.05, colors.yellow)]


def safe_filename(s: str, max_len: int = 120) -> str:
    s = str(s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip("._-")


# Point 4 — un |ρ| quasi parfait à très petit n est presque toujours un artefact
# de saturation de rangs (Spearman sur ≤ ~9 points), non une vraie association.
ARTEFACT_ABS_RHO_MIN = 0.95
ARTEFACT_N_MAX = 10


def flag_artefact_suspect(rho: float | None, n: int | None) -> bool:
    """True si la corrélation est un artefact suspect : |ρ| > 0.95 et n < 10."""
    try:
        return abs(float(rho)) > ARTEFACT_ABS_RHO_MIN and int(n) < ARTEFACT_N_MAX
    except (TypeError, ValueError):
        return False


def top_corr_df(
    all_corr_results: list[dict],
    top_n: int = 20,
    min_abs_rho: float = 0.6,
    max_abs_rho: float | None = None,
    max_p: float | None = None,
) -> pd.DataFrame:
    """Agrège les corrélations (top classement 3.1.9 / section 4).

    GARANTIE FDR (audit) : cette fonction est un CHEMIN D'AGRÉGATION PUR. Elle ne
    recalcule JAMAIS de correction FDR (aucun appel à bh_fdr / multipletests). La
    colonne `p_fdr` est HÉRITÉE telle quelle des DataFrames de blocs qui l'ont
    calculée par famille (VD × bloc sensoriel). Toute réintroduction d'un bh_fdr
    ici casserait l'égalité p_fdr[3.1.9] == p_fdr[3.x.5] (cf. test_fdr_coherence_319).
    """
    if not all_corr_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_corr_results).copy()
    df["abs_rho"] = df["rho"].abs()
    df = df[df["abs_rho"] >= min_abs_rho]
    if max_abs_rho is not None:
        df = df[df["abs_rho"] < max_abs_rho]

    # p_fdr hérité (jamais recalculé) ; fallback sur p brut si aucun p_fdr fourni.
    pcol = "p_fdr" if "p_fdr" in df.columns and df["p_fdr"].notna().any() else "p"
    if max_p is not None:
        df = df[df[pcol] <= max_p]

    # Point 4 : flag automatique d'artefact (|ρ|>0.95 & n<10). Ces lignes sont
    # reléguées en bas du classement pour ne pas trôner en tête (ex.
    # floor_exchange_pause→skill_congruence, n=7, ρ≈−0.99, saturation de rangs).
    df["artefact_suspect"] = [
        flag_artefact_suspect(r, nn) for r, nn in zip(df["rho"], df.get("n", pd.Series([np.nan] * len(df))))
    ]
    df = df.sort_values(["artefact_suspect", pcol, "abs_rho"], ascending=[True, True, False])

    keep = [c for c in ["block", "x", "y", "rho", "p", "p_fdr", "n", "artefact_suspect"] if c in df.columns]
    return df[keep].head(top_n).reset_index(drop=True)


def n_units(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return 0
    if "timepoint" in df.columns and df["timepoint"].notna().any():
        tmp = df[["group_id", "timepoint"]].dropna().drop_duplicates()
        return len(tmp)
    if "group_id" in df.columns:
        return int(df["group_id"].dropna().nunique())
    return len(df)


def fmt2(x) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def fmt_p(x) -> str:
    try:
        if pd.isna(x):
            return ""
        x = float(x)
        if x < 0.001:
            return "<0.001"
        return f"{x:.3f}"
    except Exception:
        return str(x)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (FDR). Retourne p_adjusted."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    out = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out

    p = pvals[ok]
    idx = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    q = p[idx] * len(p) / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[idx] = q
    out[ok] = tmp
    return out


def _spearman_ci_bootstrap(x: pd.Series, y: pd.Series, n_boot: int = 2000, seed: int = 12345) -> tuple[float, float]:
    """IC95 bootstrap percentile pour un ρ de Spearman (implémentation locale légère)."""
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    n = len(xv)
    if n < 4:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.full(n_boot, np.nan, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs, ys = xv[idx], yv[idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        rr, _ = spearmanr(xs, ys)
        boots[b] = rr
    if not np.isfinite(boots).any():
        return (np.nan, np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return (float(lo), float(hi))


def add_sig_stars(df: pd.DataFrame, p_col: str = "p", n_col: str = "n", n_min_sig: int = N_MIN_SIG) -> pd.DataFrame:
    out = df.copy()

    def stars(row):
        try:
            p = float(row[p_col])
            n = int(row[n_col])
        except Exception:
            return ""
        if n < n_min_sig:
            return ""
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    out["sig"] = out.apply(stars, axis=1)
    return out


def spearman_table(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    apply_fdr: bool = False,
    block: str | None = None,
    n_min_sig: int = N_MIN_SIG,
    fdr_family: str = "by_y",
) -> tuple[pd.DataFrame, list[dict]]:
    """Corrélations Spearman avec FDR par famille et flag artefact.

    fdr_family : "by_y" (défaut) = une famille Benjamini-Hochberg par variable
        dépendante (VD × bloc de modalité sensorielle audio/face/gaze, condition VR, schéma déclaré dans les notes) ; "table" = une
        seule famille sur toute la table (ancien comportement).
    """
    rows = []
    rows_all = []

    for x in x_cols:
        if x not in df.columns:
            continue
        for y in y_cols:
            if y not in df.columns:
                continue

            sub = df[[x, y]].dropna()
            n = int(len(sub))
            if n < 3:
                continue
            if sub[x].nunique(dropna=True) < 2 or sub[y].nunique(dropna=True) < 2:
                continue

            r, p = spearmanr(sub[x], sub[y])
            if not (np.isfinite(r) and np.isfinite(p)):
                continue

            # M5 : IC95 bootstrap percentile dans les blocs inférentiels (apply_fdr).
            if apply_fdr:
                ci_lo, ci_hi = _spearman_ci_bootstrap(sub[x], sub[y])
                ci95 = f"[{ci_lo:.2f}; {ci_hi:.2f}]" if np.isfinite(ci_lo) and np.isfinite(ci_hi) else ""
            else:
                ci95 = ""

            rec = {
                "x": x, "y": y, "rho": float(r), "p": float(p), "n": n,
                "IC95": ci95,
                "artefact_suspect": flag_artefact_suspect(float(r), n),
            }
            rows.append(rec)

            rec_all = dict(rec)
            if block is not None:
                rec_all["block"] = block
            rows_all.append(rec_all)

    if not rows:
        cols = ["x", "y", "rho", "p", "n", "sig", "artefact_suspect"]
        if apply_fdr:
            cols.insert(4, "p_fdr")
        empty = pd.DataFrame(columns=cols)
        return empty, rows_all

    out = pd.DataFrame(rows)

    if apply_fdr:
        # FDR par famille = VD × bloc de modalité sensorielle (audio/face/gaze), condition VR.
        out["p_fdr"] = np.nan
        if fdr_family == "by_y":
            for _yv, grp in out.groupby("y"):
                out.loc[grp.index, "p_fdr"] = bh_fdr(grp["p"].values)
        else:
            out["p_fdr"] = bh_fdr(out["p"].values)

    p_sort = "p_fdr" if apply_fdr else "p"
    # M3 : les artefacts suspects (|ρ|>0.95 & n<10) sont relégués en bas du tri.
    out = out.sort_values(["artefact_suspect", p_sort, "p", "rho"],
                          ascending=[True, True, True, False]).reset_index(drop=True)
    out = add_sig_stars(out, p_col=p_sort if apply_fdr else "p", n_min_sig=n_min_sig)
    # Un artefact suspect ne doit jamais porter d'étoile de significativité.
    out.loc[out["artefact_suspect"] == True, "sig"] = "⚠art"

    out["rho"] = out["rho"].round(2)
    out["p"] = out["p"].round(4)
    if apply_fdr:
        out["p_fdr"] = out["p_fdr"].round(4)

    if rows_all:
        key_to_art = {(r["x"], r["y"]): bool(r["artefact_suspect"]) for _, r in out.iterrows()}
        key_to_p = {(r["x"], r["y"]): float(r["p"]) for _, r in out.iterrows()}
        key_to_rho = {(r["x"], r["y"]): float(r["rho"]) for _, r in out.iterrows()}
        key_to_pfdr = {(r["x"], r["y"]): float(r["p_fdr"]) for _, r in out.iterrows()} if apply_fdr else {}
        for rec in rows_all:
            k = (rec["x"], rec["y"])
            rec["artefact_suspect"] = key_to_art.get(k, rec.get("artefact_suspect", False))
            rec["p"] = key_to_p.get(k, round(rec["p"], 4))
            rec["rho"] = key_to_rho.get(k, round(rec["rho"], 2))
            if apply_fdr:
                rec["p_fdr"] = key_to_pfdr.get(k, np.nan)

    # Ordre d'affichage stable : identifiants, effet, p, p_fdr, n, IC95, sig.
    _order = ["x", "y", "rho", "p", "p_fdr", "n", "IC95", "sig", "artefact_suspect"]
    out = out[[c for c in _order if c in out.columns] + [c for c in out.columns if c not in _order]]

    return out, rows_all


def md_table_highlight(df: pd.DataFrame, max_rows: int = MAX_ROWS_MD) -> str:
    if df is None or df.empty:
        return "_(vide)_\n\n"

    d = df.head(max_rows).copy()
    # Colonne interne de tri/flag : ne pas l'afficher (le flag est porté par sig=⚠art).
    d = d.drop(columns=["artefact_suspect"], errors="ignore")
    if "rho" in d.columns:
        d["rho"] = d["rho"].apply(fmt2)

    p_col = "p_fdr" if "p_fdr" in d.columns else "p"
    if p_col in d.columns:
        d[p_col] = d[p_col].apply(fmt_p)
        if "sig" in d.columns:
            d[p_col] = d[p_col] + d["sig"].apply(lambda s: f" {s}" if s else "")

    extra = "" if len(df) <= max_rows else f"\n\n_(affiché: {max_rows} / {len(df)})_\n\n"
    return d.to_markdown(index=False) + extra


_PAGE_USABLE_WIDTH = 6.5 * inch  # A4 portrait, marges 1 inch


def _auto_col_widths(d: pd.DataFrame, total_width: float = _PAGE_USABLE_WIDTH) -> list[float]:
    """Calcule des largeurs adaptées au contenu avec word-wrap intelligent."""
    ncols = len(d.columns)
    if ncols == 0:
        return []
    # Avec Paragraph + wordWrap, on peut compter sur re-layoutage du texte long
    char_w = 6.2  # points par caractère (ajusté pour word-wrap)
    min_w = 0.4 * inch
    max_w = 3.2 * inch  # Augmenté pour mieux laisser respirer le contenu
    widths = []
    for col in d.columns:
        header_len = len(str(col))
        data_len = d[col].astype(str).str.len().max() if not d.empty else 0
        # Pour colonnes très longues : cap l'estimation pour forcer word-wrap
        if data_len > 40:
            w = max(header_len, min(data_len, 35)) * char_w
        else:
            w = max(header_len, data_len) * char_w
        w = max(min_w, min(max_w, w))
        widths.append(w)
    total = sum(widths)
    if total > total_width:
        scale = total_width / total
        # Répartition plus douce : colonnes courtes conservent largeur minérales
        widths = [max(min_w, w * scale) for w in widths]
    return widths


def pdf_table_from_df(df: pd.DataFrame, max_rows: int = MAX_ROWS_PDF) -> Table:
    if df is None or df.empty:
        return Table([["(vide)"]])

    d = df.head(max_rows).copy()
    d = d.drop(columns=["artefact_suspect"], errors="ignore")
    p_col = "p_fdr" if "p_fdr" in d.columns else "p"

    if "rho" in d.columns:
        d["rho"] = d["rho"].apply(fmt2)
    if p_col in d.columns:
        d[p_col] = d[p_col].apply(fmt_p)
        if "sig" in d.columns:
            d[p_col] = d[p_col] + d["sig"].apply(lambda s: f" {s}" if s else "")

    cell_style = ParagraphStyle(
        "PdfTableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        wordWrap="CJK",
        alignment=0,  # LEFT
    )
    header_style = ParagraphStyle(
        "PdfTableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        alignment=0,
    )

    def _cell(val: Any, style: ParagraphStyle) -> Paragraph:
        txt = "" if val is None else _xml_escape(str(val))
        return Paragraph(txt, style)

    data = [[_cell(c, header_style) for c in d.columns]]
    for _, row in d.iterrows():
        data.append([_cell(row[col], cell_style) for col in d.columns])
    col_widths = _auto_col_widths(d)
    tbl = Table(data, repeatRows=1, colWidths=col_widths)

    style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),  # Align top pour word-wrap
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])

    if p_col in d.columns:
        j = list(d.columns).index(p_col)
        for i in range(len(d)):
            cell_val = d.iloc[i][p_col]
            try:
                pv = float(str(cell_val).replace("<0.001", "0.0005").split()[0])
            except Exception:
                pv = np.nan
            if np.isfinite(pv):
                for thr, col in P_COLOR_THRESH:
                    if pv < thr:
                        style.add("BACKGROUND", (j, i + 1), (j, i + 1), col)
                        break

    tbl.setStyle(style)
    return tbl


def scatter_plot_with_trend(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str) -> bool:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return False

    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return False

    plt.figure(figsize=(4.8, 3.8))
    plt.scatter(sub[x], sub[y], alpha=0.75)

    if sub[x].nunique(dropna=True) >= 2:
        coef = np.polyfit(sub[x], sub[y], 1)
        xx = np.linspace(sub[x].min(), sub[x].max(), 100)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return True


def add_two_plots_row(pdf_elems: list, paths: list[Path], styles: Any):
    if not paths:
        return

    imgs = []
    for path in paths[:2]:
        if path.exists():
            imgs.append(Image(str(path), width=3.2 * inch, height=2.4 * inch))
    if not imgs:
        return
    if len(imgs) == 1:
        pdf_elems.append(imgs[0])
    else:
        pdf_elems.append(Table([[imgs[0], imgs[1]]], colWidths=[3.3 * inch, 3.3 * inch]))


def render_performance_stats_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    results_dir: Path,
    fig_dir: Path,
    section_num: str = "1.3b",
):
    """
    Intègre dans le rapport les résultats de l'analyse statistique performance.
    """
    stats_dir = results_dir / "analyse_performance"
    report_txt = stats_dir / "rapport_analyse_performance.txt"
    emm_csv = stats_dir / "performance_emm_modalite_ajustee.csv"
    fact_csv = stats_dir / "performance_modele_factoriel.csv"

    title = f"{section_num} ANCOVA de référence sur la performance (modalité ajustée pour le scénario)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    if not stats_dir.exists():
        msg = (
            f"Répertoire analyse_performance absent ({stats_dir}). "
            "Exécuter : python analyse_performance/analyze_performance_effects.py"
        )
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    # La figure "performance_par_modalite" est volontairement omise ici :
    # elle duplique la visualisation de modalité déjà présentée plus haut
    # dans la section 1.3, tandis que 1.3b doit rester centré sur l'ANCOVA
    # de référence et les moyennes marginales ajustées.

    if fact_csv.exists():
        try:
            fact_df = pd.read_csv(fact_csv)
            fact_df = fact_df[fact_df["term"] != "Residual"].copy()
            keep_cols = [c for c in ["term", "F_stat", "p_value", "eta2_partial"] if c in fact_df.columns]
            fact_df = fact_df[keep_cols]
            for col in ["F_stat", "p_value", "eta2_partial"]:
                if col in fact_df.columns:
                    fact_df[col] = fact_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
            rename = {"term": "Facteur", "F_stat": "F", "p_value": "p", "eta2_partial": "η²p"}
            fact_df = fact_df.rename(columns={k: v for k, v in rename.items() if k in fact_df.columns})

            lines.append("**Tableau ANOVA factoriel (type II) :**\n")
            lines.append(md_table_highlight(fact_df, max_rows=10))
            pdf_elems.append(Paragraph("Tableau ANOVA factoriel (type II — modalité × scénario) :", styles["Heading4"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(pdf_table_from_df(fact_df, max_rows=10))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Tableau ANOVA non disponible : {e})", styles["Normal"]))

    if emm_csv.exists():
        try:
            emm_df = pd.read_csv(emm_csv)
            summary_txt = None
            if {"modalite", "emmean"}.issubset(emm_df.columns):
                emm_map = {
                    str(row["modalite"]).upper(): float(row["emmean"])
                    for _, row in emm_df.iterrows()
                    if pd.notna(row.get("modalite")) and pd.notna(row.get("emmean"))
                }
                if "PC" in emm_map and "VR" in emm_map:
                    diff = emm_map["PC"] - emm_map["VR"]
                    summary_txt = (
                        f"Moyennes marginales ajustées : PC = {emm_map['PC']:.1f}, "
                        f"VR = {emm_map['VR']:.1f}, écart ajusté = {diff:.1f} points."
                    )
                    lines.append(f"**{summary_txt}**\n\n")
                    pdf_elems.append(Paragraph(f"<b>{summary_txt}</b>", styles["Normal"]))
                    pdf_elems.append(Spacer(1, 0.08 * inch))
            lines.append("**Moyennes marginales estimées (ajustées pour le scénario) :**\n")
            lines.append(md_table_highlight(emm_df, max_rows=5))
            pdf_elems.append(Paragraph(
                "Moyennes marginales estimées de la performance par modalité "
                "(ajustées pour le scénario — ANCOVA additive) :",
                styles["Heading4"],
            ))
            pdf_elems.append(pdf_table_from_df(emm_df, max_rows=5))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        except Exception:
            pass

    if report_txt.exists():
        try:
            raw = report_txt.read_text(encoding="utf-8")
            keep_sections = ["--- B.", "--- C.", "--- D.", "--- F.", "LIMITES"]
            in_section = False
            selected_lines = []
            for raw_line in raw.splitlines():
                stripped = raw_line.strip()
                is_kept_header = any(stripped.startswith(k) for k in keep_sections)
                is_other_header = stripped.startswith("---") and not is_kept_header

                # Ne pas embarquer les sections intermédiaires vides/non retenues
                # (ex. "--- E. ANCOVA ---") dans le rapport final.
                if in_section and is_other_header:
                    in_section = False
                    continue

                if is_kept_header:
                    in_section = True

                if in_section:
                    selected_lines.append(raw_line)

            if selected_lines:
                pdf_elems.append(Paragraph("Résultats détaillés :", styles["Heading4"]))
                for raw_line in selected_lines:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        pdf_elems.append(Spacer(1, 0.05 * inch))
                        continue
                    if raw_line.startswith("---"):
                        clean = raw_line.replace("-", "").strip()
                        pdf_elems.append(Spacer(1, 0.08 * inch))
                        pdf_elems.append(Paragraph(f"<b>{clean}</b>", styles["Normal"]))
                    elif raw_line.startswith("→"):
                        pdf_elems.append(Paragraph(f"<i>{raw_line}</i>", styles["Normal"]))
                    else:
                        pdf_elems.append(Paragraph(raw_line, styles["Normal"]))

                lines.append("\n".join(selected_lines) + "\n\n")
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Rapport texte non chargeable : {e})", styles["Normal"]))

    pdf_elems.append(Spacer(1, 0.2 * inch))


def render_corr_block(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    title_md: str,
    title_pdf: str,
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    fig_dir: Path,
    fig_prefix: str,
    apply_fdr: bool,
    all_corr_results: list[dict] | None = None,
    top_k_plots: int = 3,
    max_rows_md: int = MAX_ROWS_MD,
    max_rows_pdf: int = MAX_ROWS_PDF,
    n_min_sig: int = N_MIN_SIG,
    descriptive_banner: bool = False,
) -> pd.DataFrame:
    lines.append(f"#### {title_md}\n")
    pdf_elems.append(Paragraph(title_pdf, styles["Heading4"]))

    # M4 : bandeau descriptif standardisé pour les blocs à très petit n (Riedl/TCI).
    if descriptive_banner:
        _banner = (
            "⚠ BLOC DESCRIPTIF (n≈7-8) — aucune inférence : pas de FDR interprétable, "
            "les étoiles de significativité ne doivent pas être lues comme des tests confirmatoires."
        )
        lines.append(f"_{_banner}_\n\n")
        pdf_elems.append(Paragraph(_banner, styles["Normal"]))

    if df is None or df.empty:
        lines.append("_(vide)_\n\n")
        pdf_elems.append(Paragraph("(vide)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return pd.DataFrame()

    x_cols = [c for c in x_cols if c in df.columns]
    y_cols = [c for c in y_cols if c in df.columns]

    if not x_cols or not y_cols:
        lines.append("_(aucune colonne exploitable)_\n\n")
        pdf_elems.append(Paragraph("(aucune colonne exploitable)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch))
        return pd.DataFrame()

    nu = n_units(df)
    lines.append(f"_(n unités disponibles pour ce bloc: {nu})_\n\n")
    pdf_elems.append(Paragraph(f"(n unités disponibles pour ce bloc: {nu})", styles["Normal"]))

    # M4/R1 (revue) : sous bandeau descriptif, aucune correction FDR ni étoile
    # de significativité ne doit être calculée — elles contrediraient le bandeau.
    # n_min_sig=10**9 désactive toute étoile (add_sig_stars exige n < n_min_sig).
    effective_apply_fdr = apply_fdr and not descriptive_banner
    effective_n_min_sig = n_min_sig if not descriptive_banner else 10**9
    tab, rows_all = spearman_table(df, x_cols, y_cols, apply_fdr=effective_apply_fdr, n_min_sig=effective_n_min_sig)
    for rec in rows_all:
        rec["block"] = title_pdf
        rec["descriptive_only"] = descriptive_banner

    if all_corr_results is not None:
        all_corr_results.extend(rows_all)

    lines.append(md_table_highlight(tab, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_from_df(tab, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.12 * inch))

    if tab is not None and not tab.empty and top_k_plots > 0:
        pcol = "p_fdr" if "p_fdr" in tab.columns else "p"
        tab2 = tab.sort_values([pcol, "p", "rho"], ascending=[True, True, False]).head(top_k_plots)

        figs = []
        for _, row in tab2.iterrows():
            x = row["x"]
            y = row["y"]
            caption = f"{x} ↔ {y} (rho={row['rho']:.2f}, {pcol}={row[pcol]})"
            outpng = fig_dir / f"{safe_filename(fig_prefix)}__{safe_filename(x)}__{safe_filename(y)}.png"
            ok = scatter_plot_with_trend(df, x, y, outpng, caption)
            if ok:
                figs.append(outpng)

        if figs:
            pdf_elems.append(Paragraph("Top corrélations (scatter)", styles["Normal"]))
            add_two_plots_row(pdf_elems, figs[:2], styles)
            if len(figs) > 2:
                add_two_plots_row(pdf_elems, figs[2:4], styles)
            pdf_elems.append(Spacer(1, 0.12 * inch))

    return tab
