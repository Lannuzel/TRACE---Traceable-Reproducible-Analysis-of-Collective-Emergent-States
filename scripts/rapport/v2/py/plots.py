# -*- coding: utf-8 -*-
"""
Helpers de visualisation pour le rapport CI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import normalize_group


def _harmonize_modalite_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    ren = {}
    if "Modalite" in out.columns and "modalite" not in out.columns:
        ren["Modalite"] = "modalite"
    if ren:
        out = out.rename(columns=ren)
    if "modalite" in out.columns:
        out["modalite"] = out["modalite"].astype(str).str.upper().str.strip()
    return out


def n_groups(df):
    if df is None or df.empty:
        return 0
    if "group_id" not in df.columns:
        return 0
    return df["group_id"].astype(str).str.lower().nunique()


def n_groups_total(df: pd.DataFrame) -> int:
    if df is None or df.empty or "group_id" not in df.columns:
        return 0
    tmp = normalize_group(df.copy())
    return int(tmp["group_id"].dropna().nunique())


def n_groups_by_modalite(df: pd.DataFrame, modalite: str) -> int:
    if df is None or df.empty:
        return 0
    tmp = _harmonize_modalite_col(df)
    tmp = normalize_group(tmp)
    if "group_id" not in tmp.columns or "modalite" not in tmp.columns:
        return 0
    sub = tmp[tmp["modalite"] == str(modalite).upper()]
    return int(sub["group_id"].dropna().nunique())


def plot_data_flow_schema(
    out_png: Path,
    tci_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    q_df: pd.DataFrame,
    title: str = "Répartition des données dans le pipeline expérimental",
):
    n_tci = n_groups_total(tci_df)
    n_perf_pc = n_groups_by_modalite(perf_df, "PC")
    n_perf_vr = n_groups_by_modalite(perf_df, "VR")
    n_q_pc = n_groups_by_modalite(q_df, "PC")
    n_q_vr = n_groups_by_modalite(q_df, "VR")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fontsize=11):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5))

    w, h = 0.22, 0.16
    box(0.39, 0.76, w, h, f"TCI\nn groupes = {n_tci}")
    box(0.14, 0.44, w, h, f"Tâche PC\nn groupes = {n_perf_pc}")
    box(0.64, 0.44, w, h, f"Tâche VR\nn groupes = {n_perf_vr}")
    box(0.14, 0.12, w, h, f"Questionnaire PC\nn groupes = {n_q_pc}")
    box(0.64, 0.12, w, h, f"Questionnaire VR\nn groupes = {n_q_vr}")

    arrow(0.50, 0.76, 0.25, 0.60)
    arrow(0.50, 0.76, 0.75, 0.60)
    arrow(0.25, 0.44, 0.25, 0.28)
    arrow(0.75, 0.44, 0.75, 0.28)

    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def plot_modalities_availability(
    out_png: Path,
    counts: dict[str, int],
    title: str = "Disponibilité des unités par modalité",
):
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.ylabel("Nombre d'unités")
    plt.title(title)
    plt.ylim(0, max(values) + 3 if values else 1)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()


def plot_performance_modalite_scenario(perf_df: pd.DataFrame, out_path: Path):
    if perf_df is None or perf_df.empty:
        return False

    df = perf_df.copy()
    if "Score_perf_tsk" not in df.columns:
        if "Score_perf_tsk_mean" in df.columns:
            df["Score_perf_tsk"] = df["Score_perf_tsk_mean"]
        else:
            return False

    df["modalite"] = df["modalite"].astype(str).str.upper()
    df["scenario"] = df["scenario"].astype(str).str.upper()
    df["cond"] = df["modalite"] + " " + df["scenario"]

    order = ["PC S1", "PC S2", "VR S1", "VR S2"]
    colors = {
        "PC S1": "#1f77b4",
        "PC S2": "#ff7f0e",
        "VR S1": "#2ca02c",
        "VR S2": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, cond in enumerate(order):
        sub = df[df["cond"] == cond]
        if sub.empty:
            continue

        y = sub["Score_perf_tsk"].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, color=colors[cond], alpha=0.8, s=15)

        mean = np.mean(y)
        sd = np.std(y)
        ax.errorbar(i, mean, yerr=sd, fmt="o", color=colors[cond], capsize=6, markersize=9)
        ax.text(i, mean + sd + 1, f"{mean:.1f}", ha="center", fontsize=10, fontweight="bold")

    labels = []
    for cond in order:
        n = len(df[df["cond"] == cond])
        labels.append(f"{cond}\n(n={n})")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score_perf_tsk")
    ax.set_title("Performance (Score_perf_tsk) par modalité × scénario")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_performance_by_modalite(perf_df: pd.DataFrame, out_path: Path):
    if perf_df is None or perf_df.empty:
        return False

    df = perf_df.copy()
    if "Score_perf_tsk" not in df.columns:
        return False
    if "modalite" not in df.columns:
        return False

    df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
    base_order = ["PC", "VR"]
    palette = {"PC": "#4e79a7", "VR": "#e15759"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    plot_order = []
    data = []
    ns = []
    for mod in base_order:
        vals = pd.to_numeric(df.loc[df["modalite"] == mod, "Score_perf_tsk"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue
        plot_order.append(mod)
        data.append(vals)
        ns.append(len(vals))

    if not data:
        plt.close(fig)
        return False

    positions = np.arange(len(plot_order))
    # Les sous-rapports VR_only / PC_only ne contiennent parfois qu'une seule
    # modalite. On evite alors de passer des tableaux vides (ou trop petits)
    # au violin plot de matplotlib.
    use_violin = all(len(vals) >= 2 and np.nanstd(vals) > 0 for vals in data)
    if use_violin:
        violin = ax.violinplot(data, positions=positions, widths=0.7, showmeans=False, showmedians=True)
        for body, mod in zip(violin["bodies"], plot_order):
            body.set_facecolor(palette.get(mod, "#4e79a7"))
            body.set_alpha(0.25)
    else:
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.2},
        )
        for patch, mod in zip(box["boxes"], plot_order):
            patch.set_facecolor(palette.get(mod, "#4e79a7"))
            patch.set_alpha(0.22)
            patch.set_edgecolor(palette.get(mod, "#4e79a7"))

    rng = np.random.default_rng(42)
    all_vals = np.concatenate(data) if data else np.array([], dtype=float)
    if len(all_vals) > 0:
        y_span = float(np.nanmax(all_vals) - np.nanmin(all_vals))
        label_offset = max(1.0, 0.04 * y_span)
    else:
        label_offset = 1.0

    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        x = np.full(len(vals), i) + rng.normal(0, 0.05, size=len(vals))
        mod = plot_order[i]
        ax.scatter(x, vals, s=18, alpha=0.75, color=palette.get(mod, "#4e79a7"))
        ax.text(i, np.nanmean(vals) + label_offset, f"{np.nanmean(vals):.1f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{mod}\n(n={n})" for mod, n in zip(plot_order, ns)])
    ax.set_ylabel("Score_perf_tsk")
    if len(plot_order) == 1:
        ax.set_title(f"Performance finale — {plot_order[0]} uniquement")
    else:
        ax.set_title("Performance finale par modalité")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_performance_components(perf: pd.DataFrame, out_path: Path) -> bool:
    """
    Stacked bar chart showing M1–M4 sub-component decomposition per condition.
    """
    if perf is None or perf.empty:
        return False

    m_cols = ["M1_consignes_%", "M2_nombre_%", "M3_precision_%", "M4_temps_%"]
    avail = [c for c in m_cols if c in perf.columns]
    if not avail:
        return False

    df = perf.copy()
    for c in avail:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.upper()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.upper()
    df["cond"] = df.get("modalite", "?").astype(str) + " " + df.get("scenario", "?").astype(str)

    order = ["PC S1", "PC S2", "VR S1", "VR S2"]
    weights = {"M1_consignes_%": 0.50, "M2_nombre_%": 0.00, "M3_precision_%": 0.50, "M4_temps_%": 0.00}
    labels = {
        "M1_consignes_%": "M1 Consignes (50%)",
        "M2_nombre_%": "M2 Nombre (0%)",
        "M3_precision_%": "M3 Précision (50%)",
        "M4_temps_%": "M4 Temps (0%)",
    }
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_x = np.arange(len(order))
    bottom = np.zeros(len(order))

    for idx, mc in enumerate(avail):
        vals = []
        for cond in order:
            sub = df[df["cond"] == cond]
            if sub.empty or mc not in sub.columns:
                vals.append(0)
            else:
                raw = sub[mc].mean()
                vals.append(raw * weights.get(mc, 0.25))
        vals = np.array(vals)
        ax.bar(
            bar_x,
            vals,
            bottom=bottom,
            label=labels.get(mc, mc),
            color=colors[idx % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    for i, cond in enumerate(order):
        sub = df[df["cond"] == cond]
        if not sub.empty and "Score_perf_tsk" in sub.columns:
            tot = pd.to_numeric(sub["Score_perf_tsk"], errors="coerce").mean()
            ax.text(i, bottom[i] + 0.5, f"{tot:.1f}%", ha="center", fontweight="bold", fontsize=9)

    x_labels = []
    for cond in order:
        n = len(df[df["cond"] == cond])
        x_labels.append(f"{cond}\n(n={n})")

    ax.set_xticks(bar_x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Contribution pondérée (%)")
    ax.set_title("Décomposition M1–M4 de la performance v1 par condition")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True
