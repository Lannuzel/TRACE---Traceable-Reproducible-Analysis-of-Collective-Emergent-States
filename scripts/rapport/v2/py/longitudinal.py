# -*- coding: utf-8 -*-
"""
Helpers pour les analyses longitudinales T1/T2 du rapport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer
from scipy.stats import wilcoxon

from .data import (
    add_group_base_id,
    has_real_timepoint,
    harmonize_timepoint,
    normalize_group,
    normalize_timepoint,
    performance_score_col,
)


def build_longitudinal_pairs_for_metrics(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    Construit une table appariée T1/T2 pour plusieurs métriques via group_base_id.
    """
    if df is None or df.empty or "group_id" not in df.columns:
        return pd.DataFrame()

    tmp = normalize_group(df.copy())
    tmp = add_group_base_id(tmp)
    tmp = harmonize_timepoint(
        tmp,
        session_col="session",
        raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
    )
    tmp = normalize_timepoint(tmp)

    if "group_base_id" not in tmp.columns or "timepoint" not in tmp.columns:
        return pd.DataFrame()

    keep_cols = [c for c in value_cols if c in tmp.columns]
    if not keep_cols:
        return pd.DataFrame()

    tmp = tmp[["group_base_id", "timepoint"] + keep_cols].copy()
    tmp = tmp[tmp["timepoint"].isin(["T1", "T2"])]

    for col in keep_cols:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    agg = (
        tmp.groupby(["group_base_id", "timepoint"], dropna=False)[keep_cols]
        .mean()
        .reset_index()
    )
    if agg.empty:
        return pd.DataFrame()

    wide = agg.pivot(index="group_base_id", columns="timepoint", values=keep_cols)
    if wide.empty:
        return pd.DataFrame()

    out = wide.copy()
    out.columns = [f"{var}_{tp}" for var, tp in out.columns]
    out = out.reset_index()

    valid_metrics = []
    for col in keep_cols:
        c_t1 = f"{col}_T1"
        c_t2 = f"{col}_T2"
        if c_t1 in out.columns and c_t2 in out.columns:
            valid_metrics.append(col)

    if not valid_metrics:
        return pd.DataFrame()

    cols_to_keep = ["group_base_id"]
    for col in valid_metrics:
        cols_to_keep.extend([f"{col}_T1", f"{col}_T2"])
    out = out[cols_to_keep].copy()

    for col in valid_metrics:
        c_t1 = f"{col}_T1"
        c_t2 = f"{col}_T2"
        out[f"{col}_delta_T2_minus_T1"] = out[c_t2] - out[c_t1]

    delta_cols = [f"{col}_delta_T2_minus_T1" for col in valid_metrics]
    out = out.dropna(subset=delta_cols, how="all").reset_index(drop=True)
    return out


def build_longitudinal_cscore(tci: pd.DataFrame, perf_long: pd.DataFrame) -> pd.DataFrame:
    """Récupère le C-score moyen pour les groupes longitudinaux."""
    if tci is None or tci.empty or perf_long is None or perf_long.empty:
        return pd.DataFrame()

    tmp = normalize_group(tci.copy())
    tmp = add_group_base_id(tmp)

    c_cols = [c for c in tmp.columns if c.lower() in ["c_score", "c_factor", "cscore"]]
    if not c_cols:
        return pd.DataFrame()

    c_col = c_cols[0]
    tmp[c_col] = pd.to_numeric(tmp[c_col], errors="coerce")

    c_tab = (
        tmp.groupby("group_base_id")[c_col]
        .mean()
        .reset_index()
        .rename(columns={c_col: "C_score"})
    )
    return perf_long[["group_base_id"]].merge(c_tab, on="group_base_id", how="left")


def build_perf_scenario_lookup(perf: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Prépare une table groupe / timepoint / scénario / score pour la performance."""
    if perf is None or perf.empty:
        return pd.DataFrame(), None

    score_col = performance_score_col(perf)
    if score_col is None or score_col not in perf.columns:
        return pd.DataFrame(), None

    tmp = perf.copy()
    tmp = normalize_group(tmp)
    tmp = add_group_base_id(tmp)
    tmp = harmonize_timepoint(
        tmp,
        session_col="session",
        raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
    )
    tmp = normalize_timepoint(tmp)
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")

    scenario_col = None
    for col in ["scenario", "session", "Scenario", "Session"]:
        if col in tmp.columns:
            scenario_col = col
            break

    if scenario_col is None:
        tmp["scenario"] = "UNK"
        scenario_col = "scenario"

    out = (
        tmp[["group_base_id", "group_id", "timepoint", scenario_col, score_col]]
        .rename(columns={scenario_col: "scenario", score_col: "score"})
        .dropna(subset=["group_base_id", "timepoint"])
        .groupby(["group_base_id", "group_id", "timepoint", "scenario"], dropna=False)["score"]
        .mean()
        .reset_index()
    )
    out["timepoint"] = out["timepoint"].astype(str).str.upper().str.strip()
    out = out[out["timepoint"].isin(["T1", "T2"])].copy()
    return out, "score"


def plot_longitudinal_performance(
    perf_long: pd.DataFrame,
    perf: pd.DataFrame,
    fig_dir: Path,
) -> Path | None:
    """
    Trace l'évolution T1 -> T2 de la performance en score brut et z-score.
    """
    if perf_long is None or perf_long.empty:
        return None

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "longitudinal_performance.png"

    perf_detail, _ = build_perf_scenario_lookup(perf)
    if perf_detail.empty:
        return None

    long_groups = set(perf_long["group_base_id"].dropna().astype(str).unique())
    perf_detail_long = perf_detail[
        perf_detail["group_base_id"].astype(str).isin(long_groups)
    ].copy()

    gp = (
        perf_detail_long
        .sort_values(["group_base_id", "timepoint", "scenario"])
        .groupby(["group_base_id", "timepoint"], dropna=False)
        .first()
        .reset_index()
    )

    gp_wide = gp.pivot(index="group_base_id", columns="timepoint", values=["scenario", "group_id"])
    if gp_wide.empty:
        return None

    gp_wide.columns = [f"{a}_{b}" for a, b in gp_wide.columns]
    gp_wide = gp_wide.reset_index()
    plot_df = perf_long.merge(gp_wide, on="group_base_id", how="left").copy()

    all_perf_detail, _ = build_perf_scenario_lookup(perf)
    scen_means = (
        all_perf_detail
        .groupby(["scenario"], dropna=False)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "scenario_mean"})
    )

    has_z = "T1_z" in plot_df.columns and "T2_z" in plot_df.columns
    n_panels = 2 if has_z else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8.5 * n_panels / 1.5 + 2, 5.5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    x = [0, 1]
    offsets = {"S1": -0.09, "S2": -0.03, "S3": 0.03, "S4": 0.09, "UNK": 0.00}
    colors_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0]
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        color = colors_cycle[idx % len(colors_cycle)]
        ax.plot(x, [row["T1"], row["T2"]], marker="o", linewidth=1.6, alpha=0.75, color=color)
        sc1 = row.get("scenario_T1", "UNK") if pd.notna(row.get("scenario_T1")) else "UNK"
        sc2 = row.get("scenario_T2", "UNK") if pd.notna(row.get("scenario_T2")) else "UNK"
        ax.text(-0.06, row["T1"], f'{row["group_base_id"]} ({sc1})', fontsize=8, ha="right", va="center", color=color)
        ax.text(1.06, row["T2"], f'{row["group_base_id"]} ({sc2})', fontsize=8, ha="left", va="center", color=color)

    for _, row in scen_means.iterrows():
        scenario = str(row["scenario"])
        y_mean = row["scenario_mean"]
        if pd.isna(y_mean):
            continue
        dx = offsets.get(scenario, 0.0)
        ax.plot([0 + dx, 1 + dx], [y_mean, y_mean], linestyle="--", linewidth=1.1, color="black", alpha=0.6, zorder=4)
        ax.scatter([0 + dx, 1 + dx], [y_mean, y_mean], s=34, marker="D", facecolor="white", edgecolor="black", linewidth=1.6, zorder=6)
        ax.text(1 + dx, y_mean + 0.8, f"mu {scenario}", fontsize=8, ha="center", va="bottom", weight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T1", "T2"])
    ax.set_ylabel("Score_perf_tsk (%)")
    ax.set_title("Score brut\n(comparaison directe)")
    ax.grid(alpha=0.3)

    if has_z:
        ax2 = axes[1]
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            color = colors_cycle[idx % len(colors_cycle)]
            y_z = [row["T1_z"], row["T2_z"]]
            ax2.plot(x, y_z, marker="o", linewidth=1.6, alpha=0.75, color=color)
            sc1 = row.get("scenario_T1", "UNK") if pd.notna(row.get("scenario_T1")) else "UNK"
            sc2 = row.get("scenario_T2", "UNK") if pd.notna(row.get("scenario_T2")) else "UNK"
            ax2.text(-0.06, row["T1_z"], f'{row["group_base_id"]} ({sc1})', fontsize=8, ha="right", va="center", color=color)
            ax2.text(1.06, row["T2_z"], f'{row["group_base_id"]} ({sc2})', fontsize=8, ha="left", va="center", color=color)

        ax2.axhline(0, linestyle="--", linewidth=1.2, color="black", alpha=0.5, label="Moyenne scénario (z=0)")
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["T1", "T2"])
        ax2.set_ylabel("Z-score (par scénario)")
        ax2.set_title("Z-score par scénario\n(recommandé pour comparaison longitudinale)")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc="best")

    legend_handles = [
        Line2D([0], [0], marker="o", linewidth=1.6, label="Trajectoire groupe"),
        Line2D([0], [0], marker="D", color="black", markerfacecolor="white", linewidth=1.0, linestyle="--", label="Moyenne globale scénario (score brut)"),
    ]
    axes[0].legend(handles=legend_handles, loc="best", fontsize=8)

    fig.suptitle(
        "Évolution de la performance T1 -> T2\n(score brut et z-score par scénario)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    return fig_path


def performance_longitudinal_stats(df: pd.DataFrame):
    """Statistiques descriptives et Wilcoxon sur la performance longitudinale."""
    if df is None or df.empty:
        return None

    result = {
        "n_pairs": len(df),
        "T1_mean": df["T1"].mean(),
        "T2_mean": df["T2"].mean(),
        "delta_mean": df["delta_T2_minus_T1"].mean(),
        "delta_median": df["delta_T2_minus_T1"].median(),
        "p_wilcoxon": None,
    }

    if len(df) >= 3:
        try:
            result["p_wilcoxon"] = wilcoxon(df["T2"], df["T1"]).pvalue
        except Exception:
            pass

    return result


def paired_change_table(df: pd.DataFrame, value_cols: list[str], label: str) -> pd.DataFrame:
    """Résumé T1/T2 variable par variable pour un bloc longitudinal."""
    if df is None or df.empty:
        return pd.DataFrame()

    if "group_id" not in df.columns or not has_real_timepoint(df):
        return pd.DataFrame()

    sub = df.copy()
    sub = sub.dropna(subset=["timepoint"])
    sub["timepoint"] = sub["timepoint"].astype(str).str.upper().str.strip()
    sub = sub[sub["timepoint"].isin(["T1", "T2"])]

    rows = []
    for col in value_cols:
        if col not in sub.columns or not pd.api.types.is_numeric_dtype(sub[col]):
            continue

        wide = (
            sub[["group_id", "timepoint", col]]
            .dropna()
            .pivot_table(index="group_id", columns="timepoint", values=col, aggfunc="mean")
            .reset_index()
        )

        if not {"T1", "T2"}.issubset(wide.columns):
            continue

        wide = wide.dropna(subset=["T1", "T2"]).copy()
        n_pairs = len(wide)
        if n_pairs < 2:
            continue

        delta = wide["T2"] - wide["T1"]
        pval = np.nan
        if n_pairs >= 5:
            try:
                pval = wilcoxon(wide["T2"], wide["T1"], zero_method="wilcox").pvalue
            except Exception:
                pval = np.nan

        rows.append({
            "bloc": label,
            "variable": col,
            "n_pairs": n_pairs,
            "T1_mean": wide["T1"].mean(),
            "T2_mean": wide["T2"].mean(),
            "delta_T2_minus_T1_mean": delta.mean(),
            "delta_median": delta.median(),
            "p_wilcoxon": pval,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["abs_delta"] = out["delta_T2_minus_T1_mean"].abs()
    out = out.sort_values(["p_wilcoxon", "abs_delta"], ascending=[True, False], na_position="last").reset_index(drop=True)
    return out.drop(columns="abs_delta")


def render_paired_change_block(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    title_md: str,
    title_pdf: str,
    df: pd.DataFrame,
    value_cols: list[str],
    md_table_fn: Callable[..., str],
    pdf_table_fn: Callable[..., Any],
    max_rows_md: int = 120,
    max_rows_pdf: int = 40,
):
    """Rendu générique d'un tableau de changements appariés T1/T2."""
    lines.append(f"### {title_md}\n")
    pdf_elems.append(Paragraph(title_pdf, styles["Heading3"]))

    tab = paired_change_table(df, value_cols, title_pdf)
    if tab is None or tab.empty:
        lines.append("_(pas assez de paires T1/T2 exploitables)_\n\n")
        pdf_elems.append(Paragraph("(pas assez de paires T1/T2 exploitables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    for col in ["T1_mean", "T2_mean", "delta_T2_minus_T1_mean", "delta_median"]:
        if col in tab.columns:
            tab[col] = tab[col].round(3)
    if "p_wilcoxon" in tab.columns:
        tab["p_wilcoxon"] = tab["p_wilcoxon"].round(4)

    lines.append(md_table_fn(tab, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_fn(tab, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.2 * inch))


def build_performance_longitudinal(perf: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la table T1/T2 de performance et ajoute un z-score par scénario.
    """
    if perf is None or perf.empty:
        return pd.DataFrame()

    perf = normalize_group(perf)
    perf = add_group_base_id(perf)
    perf = harmonize_timepoint(
        perf,
        raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
    )
    perf = normalize_timepoint(perf)

    score_col = performance_score_col(perf)
    if score_col is None or not {"group_base_id", "timepoint"} <= set(perf.columns):
        return pd.DataFrame()

    cols = ["group_id", "group_base_id", "timepoint", score_col]
    if "scenario" in perf.columns:
        cols.append("scenario")
    sub = perf[cols].copy()
    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce")
    sub = sub.dropna(subset=[score_col, "group_base_id", "timepoint"])
    sub = sub[sub["timepoint"].isin(["T1", "T2"])]

    if "scenario" in sub.columns:
        sub["z_score"] = sub.groupby("scenario")[score_col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 1e-9 else 0.0
        )
    else:
        sub["z_score"] = (sub[score_col] - sub[score_col].mean()) / sub[score_col].std() if sub[score_col].std() > 1e-9 else 0.0

    wide_raw = (
        sub.pivot_table(index="group_base_id", columns="timepoint", values=score_col, aggfunc="mean")
        .reset_index()
    )
    wide_z = (
        sub.pivot_table(index="group_base_id", columns="timepoint", values="z_score", aggfunc="mean")
        .reset_index()
        .rename(columns={"T1": "T1_z", "T2": "T2_z"})
    )

    if not {"T1", "T2"}.issubset(wide_raw.columns):
        return pd.DataFrame()

    wide = wide_raw.dropna(subset=["T1", "T2"]).copy()
    wide["delta_T2_minus_T1"] = wide["T2"] - wide["T1"]
    wide = wide.merge(wide_z, on="group_base_id", how="left")
    if "T1_z" in wide.columns and "T2_z" in wide.columns:
        wide["delta_z"] = wide["T2_z"] - wide["T1_z"]
    return wide


def _build_pair_availability_table(
    inv_face: pd.DataFrame | None,
    inv_speech: pd.DataFrame | None,
    inv_gaze_all: pd.DataFrame | None,
    hl: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compte la disponibilité T1/T2 par bloc de données."""
    pair_rows = []
    for label, df_src in [
        ("Face", inv_face),
        ("Speech", inv_speech),
        ("Gaze", inv_gaze_all),
        ("High-level", hl),
    ]:
        if df_src is None or df_src.empty or "group_id" not in df_src.columns:
            pair_rows.append({"bloc": label, "n_T1": 0, "n_T2": 0, "n_pairs_T1T2": 0})
            continue

        tmp = normalize_group(df_src.copy())
        tmp = add_group_base_id(tmp)
        tmp = harmonize_timepoint(
            tmp,
            session_col="session",
            raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
        )
        tmp = normalize_timepoint(tmp)

        if "group_base_id" not in tmp.columns or not has_real_timepoint(tmp):
            pair_rows.append({"bloc": label, "n_T1": 0, "n_T2": 0, "n_pairs_T1T2": 0})
            continue

        uu = tmp[["group_base_id", "timepoint"]].dropna().drop_duplicates().copy()
        uu["timepoint"] = uu["timepoint"].astype(str).str.upper().str.strip()
        uu = uu[uu["timepoint"].isin(["T1", "T2"])]

        n_t1 = int(uu.loc[uu["timepoint"] == "T1", "group_base_id"].nunique())
        n_t2 = int(uu.loc[uu["timepoint"] == "T2", "group_base_id"].nunique())

        if uu.empty:
            n_pairs = 0
        else:
            wide_pairs = (
                uu.assign(v=1)
                .pivot_table(index="group_base_id", columns="timepoint", values="v", aggfunc="max")
                .fillna(0)
            )
            for col in ["T1", "T2"]:
                if col not in wide_pairs.columns:
                    wide_pairs[col] = 0
            n_pairs = int(((wide_pairs["T1"] == 1) & (wide_pairs["T2"] == 1)).sum())

        pair_rows.append({"bloc": label, "n_T1": n_t1, "n_T2": n_t2, "n_pairs_T1T2": n_pairs})

    return pd.DataFrame(pair_rows)


def render_longitudinal_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    fig_dir: Path,
    perf: pd.DataFrame | None,
    tci: pd.DataFrame | None,
    hl: pd.DataFrame | None,
    inv_face: pd.DataFrame | None,
    inv_speech: pd.DataFrame | None,
    inv_gaze_all: pd.DataFrame | None,
    core_hl: list[str],
    md_table_fn: Callable[..., str],
    pdf_table_fn: Callable[..., Any],
    max_rows_md: int = 120,
    max_rows_pdf: int = 40,
    section_num: str = "6",
):
    """Rend la section complète des analyses longitudinales T1/T2."""
    lines.append(f"## {section_num}. Analyses longitudinales (T1 ↔ T2)\n")
    pdf_elems.append(Paragraph(f"{section_num}. Analyses longitudinales (T1 ↔ T2)", styles["Heading2"]))

    perf_long = build_performance_longitudinal(perf)
    cscore_long = build_longitudinal_cscore(tci, perf_long)
    fig_path = plot_longitudinal_performance(perf_long, perf, fig_dir)

    if cscore_long is not None and not cscore_long.empty:
        lines.append("#### C-score des groupes longitudinaux\n")
        tab = perf_long.merge(cscore_long, on="group_base_id", how="left")

        z_cols = [c for c in ["T1_z", "T2_z", "delta_z"] if c in tab.columns]
        base_cols = ["group_base_id", "T1", "T2", "delta_T2_minus_T1"] + z_cols + ["C_score"]
        tab = tab[[c for c in base_cols if c in tab.columns]]
        tab = tab.rename(columns={
            "group_base_id": "group",
            "T1": "perf_T1",
            "T2": "perf_T2",
            "delta_T2_minus_T1": "delta_perf",
        })
        for col in tab.select_dtypes(include="number").columns:
            tab[col] = tab[col].round(2)

        lines.append(tab.to_markdown(index=False) + "\n\n")
        pdf_elems.append(pdf_table_fn(tab))
        pdf_elems.append(Spacer(1, 0.2 * inch))

    if fig_path is not None:
        lines.append("#### Evolution individuelle des groupes\n")
        lines.append(f"![]({fig_path.name})\n\n")
        pdf_elems.append(Paragraph("Evolution individuelle des groupes (score brut + z-score)", styles["Heading4"]))
        pdf_elems.append(Image(str(fig_path), width=6.5 * inch, height=3.5 * inch))
        pdf_elems.append(Spacer(1, 0.2 * inch))

    lines.append(f"### {section_num}.1 Disponibilité des paires T1/T2\n")
    pdf_elems.append(Paragraph(f"{section_num}.1 Disponibilité des paires T1/T2", styles["Heading3"]))

    pair_df = _build_pair_availability_table(
        inv_face=inv_face,
        inv_speech=inv_speech,
        inv_gaze_all=inv_gaze_all,
        hl=hl,
    )
    lines.append(md_table_fn(pair_df, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_fn(pair_df, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.2 * inch))

    lines.append(f"### {section_num}.2 Évolution de la performance entre T1 et T2\n")
    pdf_elems.append(Paragraph(f"{section_num}.2 Évolution de la performance entre T1 et T2", styles["Heading3"]))
    pdf_elems.append(Paragraph(
        "Note : le z-score est la métrique recommandée pour la comparaison longitudinale. "
        "Les groupes réalisent souvent des scénarios différents à T1 et T2. "
        "Le z-score normalise la performance relative aux autres groupes du même scénario.",
        styles["Normal"],
    ))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    if perf_long is None or perf_long.empty:
        lines.append("_(pas assez de paires T1/T2 exploitables)_\n\n")
        pdf_elems.append(Paragraph("(pas assez de paires T1/T2 exploitables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
    else:
        stats = performance_longitudinal_stats(perf_long) or {}
        perf_tab = pd.DataFrame([{
            "n_pairs": stats.get("n_pairs", len(perf_long)),
            "T1_mean": stats.get("T1_mean", perf_long["T1"].mean()),
            "T2_mean": stats.get("T2_mean", perf_long["T2"].mean()),
            "delta_T2_minus_T1_mean": stats.get("delta_mean", perf_long["delta_T2_minus_T1"].mean()),
            "delta_median": stats.get("delta_median", perf_long["delta_T2_minus_T1"].median()),
            "p_wilcoxon": stats.get("p_wilcoxon", np.nan),
        }])
        for col in ["T1_mean", "T2_mean", "delta_T2_minus_T1_mean", "delta_median", "p_wilcoxon"]:
            if col in perf_tab.columns:
                perf_tab[col] = perf_tab[col].round(3)

        lines.append("**Score brut :**\n")
        lines.append(perf_tab.to_markdown(index=False) + "\n\n")
        pdf_elems.append(Paragraph("Score brut (Score_perf_tsk) :", styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))
        pdf_elems.append(pdf_table_fn(perf_tab, max_rows=max_rows_pdf))
        pdf_elems.append(Spacer(1, 0.1 * inch))

        if "T1_z" in perf_long.columns and "T2_z" in perf_long.columns:
            z_tab = pd.DataFrame([{
                "n_pairs": len(perf_long),
                "T1_z_mean": perf_long["T1_z"].mean(),
                "T2_z_mean": perf_long["T2_z"].mean(),
                "delta_z_mean": perf_long["delta_z"].mean() if "delta_z" in perf_long.columns else np.nan,
                "delta_z_median": perf_long["delta_z"].median() if "delta_z" in perf_long.columns else np.nan,
            }])
            for col in z_tab.columns:
                if col != "n_pairs":
                    z_tab[col] = z_tab[col].round(3)

            lines.append("**Z-score par scénario (recommandé pour comparaison longitudinale) :**\n")
            lines.append(z_tab.to_markdown(index=False) + "\n\n")
            pdf_elems.append(Paragraph("Z-score par scénario (recommandé pour comparaison longitudinale) :", styles["Heading4"]))
            pdf_elems.append(pdf_table_fn(z_tab, max_rows=max_rows_pdf))
            pdf_elems.append(Spacer(1, 0.1 * inch))

            ind_cols = ["group_base_id", "T1", "T2", "delta_T2_minus_T1", "T1_z", "T2_z", "delta_z"]
            ind_tab = perf_long[[c for c in ind_cols if c in perf_long.columns]].copy()
            for col in ["T1", "T2", "delta_T2_minus_T1", "T1_z", "T2_z", "delta_z"]:
                if col in ind_tab.columns:
                    ind_tab[col] = ind_tab[col].round(2)

            lines.append("**Détail par groupe :**\n")
            lines.append(ind_tab.to_markdown(index=False) + "\n\n")
            pdf_elems.append(Paragraph("Détail par groupe (score brut + z-score) :", styles["Heading4"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(pdf_table_fn(ind_tab, max_rows=max_rows_pdf))
            pdf_elems.append(Spacer(1, 0.2 * inch))

    lines.append(f"### {section_num}.3 Évolution des high-level features entre T1 et T2\n")
    pdf_elems.append(Paragraph(f"{section_num}.3 Évolution des high-level features entre T1 et T2", styles["Heading3"]))

    hl_cols_long = [c for c in core_hl if hl is not None and c in hl.columns]
    hl_long = build_longitudinal_pairs_for_metrics(hl, hl_cols_long)
    if hl_long is not None and not hl_long.empty:
        hl_cols_long = [
            c for c in hl_cols_long
            if f"{c}_T1" in hl_long.columns and f"{c}_T2" in hl_long.columns
        ]
    else:
        hl_cols_long = []

    if hl_long is None or hl_long.empty:
        lines.append("_(pas assez de paires T1/T2 exploitables)_\n\n")
        pdf_elems.append(Paragraph("(pas assez de paires T1/T2 exploitables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    rows = []
    for col in hl_cols_long:
        c_t1 = f"{col}_T1"
        c_t2 = f"{col}_T2"
        c_d = f"{col}_delta_T2_minus_T1"
        if not {c_t1, c_t2, c_d}.issubset(hl_long.columns):
            continue

        sub = hl_long[[c_t1, c_t2, c_d]].dropna()
        if sub.empty:
            continue

        rows.append({
            "metric": col,
            "n_pairs": len(sub),
            "T1_mean": sub[c_t1].mean(),
            "T2_mean": sub[c_t2].mean(),
            "delta_T2_minus_T1_mean": sub[c_d].mean(),
            "delta_median": sub[c_d].median(),
        })

    if not rows:
        lines.append("_(pas assez de paires T1/T2 exploitables)_\n\n")
        pdf_elems.append(Paragraph("(pas assez de paires T1/T2 exploitables)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.15 * inch))
        return

    hl_tab = pd.DataFrame(rows)
    for col in ["T1_mean", "T2_mean", "delta_T2_minus_T1_mean", "delta_median"]:
        hl_tab[col] = hl_tab[col].round(3)

    lines.append(md_table_fn(hl_tab, max_rows=max_rows_md))
    pdf_elems.append(pdf_table_fn(hl_tab, max_rows=max_rows_pdf))
    pdf_elems.append(Spacer(1, 0.2 * inch))
