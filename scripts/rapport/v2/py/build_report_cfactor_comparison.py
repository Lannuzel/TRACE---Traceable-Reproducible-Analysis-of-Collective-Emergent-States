#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_report_cfactor_comparison.py

Rapport comparatif c_factor_sample vs c_factor_pop — sections 2-4 du rapport principal
PC+VR, répliquées en parallèle pour les deux variantes, plus section 5 diff.

Répertoire de sortie : results/rapport_v2/PC_VR/cfactor_comparison/
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_V2_DIR = _HERE.parents[1]
_SCRIPTS_DIR = _HERE.parents[3]
_PROJECT_ROOT = _HERE.parents[4]

for _p in [str(_V2_DIR), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from py.corr import bh_fdr  # noqa: E402
from py.network import (  # noqa: E402
    prepare_global_correlation_network_data,
    plot_global_correlation_network,
    compute_network_metrics,
    _plot_family_network,
    classify_variable,
)

RESULTS_DIR = _PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "rapport_v2" / "PC_VR" / "cfactor_comparison"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"

EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}

# Seuils réseau identiques au rapport principal
RHO_THRESHOLD = 0.55
P_THRESHOLD = 0.1

RIEDL_VARS = [
    "skill_mean", "strategy_ratio_mean", "strategy_norm", "contribution_mean",
    "effort_task_sum", "effort_task_norm", "skill_congruence_mean", "skill_max",
]
RME_VARS = ["rme_mean", "rme_max", "rme_min"]
QUEST_VARS = ["CRE", "SPE", "COR", "COM", "SOC", "TSK", "Cohesion_questionnaire_score"]

CFACTOR_VARIANTS = ["c_factor_sample", "c_factor_pop"]


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (merged_df, pop_df)."""
    merged_path = RESULTS_DIR / "merged_dataset" / "with_pruning" / "merged_dataset_complete_all.csv"
    pop_path = RESULTS_DIR / "TCI" / "c_scores_allowed_pop.csv"

    merged = pd.read_csv(merged_path, low_memory=False)
    pop = pd.read_csv(pop_path, low_memory=False)

    # Exclure groupes documentés comme invalides
    if "group_id" in merged.columns:
        merged = merged[~merged["group_id"].isin(EXCLUDED_GROUPS)].reset_index(drop=True)
    if "group_id" in pop.columns:
        pop = pop[~pop["group_id"].isin(EXCLUDED_GROUPS)].reset_index(drop=True)

    # c_score dans merged == c_factor_sample dans pop (vérifié: delta < 1e-5)
    # Ajouter c_factor_pop au merged via join sur group_id
    if "c_factor_pop" not in merged.columns:
        pop_cols = ["group_id", "c_factor_pop", "c_factor_sample"]
        available = [c for c in pop_cols if c in pop.columns]
        merged = merged.merge(pop[available], on="group_id", how="left")

    # Alias: c_factor_sample = c_score (local PCA)
    if "c_factor_sample" not in merged.columns and "c_score" in merged.columns:
        merged["c_factor_sample"] = merged["c_score"]

    return merged, pop


def add_cohesion_global(df: pd.DataFrame) -> pd.DataFrame:
    coh_cols = [c for c in ["COM", "SOC", "TSK"] if c in df.columns]
    if coh_cols:
        df["Cohesion_questionnaire_score"] = df[coh_cols].mean(axis=1)
    return df


def sanity_log(merged: pd.DataFrame, pop: pd.DataFrame) -> list[str]:
    logs = []
    logs.append(f"N groupes merged (après exclusions) : {merged['group_id'].nunique()}")
    logs.append(f"N groupes pop allowed : {pop['group_id'].nunique()}")

    vr = merged[merged.get("modalite", pd.Series()).str.upper() == "VR"] if "modalite" in merged.columns else pd.DataFrame()
    pc = merged[merged.get("modalite", pd.Series()).str.upper() == "PC"] if "modalite" in merged.columns else pd.DataFrame()
    logs.append(f"  VR : {vr['group_id'].nunique() if not vr.empty else 0} groupes")
    logs.append(f"  PC : {pc['group_id'].nunique() if not pc.empty else 0} groupes")

    for variant in CFACTOR_VARIANTS:
        if variant in merged.columns:
            n_valid = merged[variant].notna().sum()
            logs.append(f"  {variant} : {n_valid} valeurs non-NaN sur {len(merged)} lignes")
        else:
            logs.append(f"  {variant} : COLONNE ABSENTE")

    if "c_factor_pop" in merged.columns:
        n_pop_missing = int(merged["c_factor_pop"].isna().sum())
        pct_missing = n_pop_missing / len(merged) * 100
        logs.append(f"  c_factor_pop manquants : {n_pop_missing} ({pct_missing:.1f}%)")
    else:
        logs.append("  c_factor_pop : COLONNE ABSENTE")

    # Corrélation Spearman pop vs sample
    if all(c in merged.columns for c in ["c_factor_sample", "c_factor_pop"]):
        sub = merged[["c_factor_sample", "c_factor_pop"]].dropna()
        if len(sub) >= 5:
            rho, p = spearmanr(sub["c_factor_sample"], sub["c_factor_pop"])
            logs.append(f"  Spearman c_factor_sample × c_factor_pop (global) : rho={rho:.3f}, p={p:.4f}, n={len(sub)}")

    return logs


# ---------------------------------------------------------------------------
# Calcul des corrélations de Spearman
# ---------------------------------------------------------------------------

def spearman_pair(df: pd.DataFrame, x: str, y: str, block: str) -> dict | None:
    sub = df[[x, y]].dropna()
    if len(sub) < 5:
        return None
    rho, p = spearmanr(sub[x], sub[y])
    return {"block": block, "x": x, "y": y, "rho": round(float(rho), 6),
            "p": round(float(p), 6), "n": len(sub)}


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001 **"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


def sig_label(p: float) -> str:
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Bloc 2.2 : TCI ↔ Performance
# ---------------------------------------------------------------------------

def compute_block_22(df: pd.DataFrame, variant: str) -> list[dict]:
    results = []
    target = "Score_perf_tsk"
    if target not in df.columns:
        return results

    for x in [variant] + RME_VARS:
        if x not in df.columns:
            continue
        row = spearman_pair(df, x, target, "2.2 TCI ↔ Performance (Score final)")
        if row:
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Bloc 2.3 : Riedl ↔ TCI
# ---------------------------------------------------------------------------

def compute_block_23(df: pd.DataFrame, variant: str) -> list[dict]:
    results = []
    tci_vars = [variant] + RME_VARS
    for x in RIEDL_VARS:
        if x not in df.columns:
            continue
        for y in tci_vars:
            if y not in df.columns:
                continue
            row = spearman_pair(df, x, y, "2.3 Riedl ↔ TCI (tous groupes)")
            if row:
                results.append(row)
    return results


# ---------------------------------------------------------------------------
# Bloc 2.1 : Riedl ↔ Performance (inchangé)
# ---------------------------------------------------------------------------

def compute_block_21(df: pd.DataFrame) -> list[dict]:
    results = []
    target = "Score_perf_tsk"
    if target not in df.columns:
        return results
    for x in RIEDL_VARS:
        if x not in df.columns:
            continue
        row = spearman_pair(df, x, target, "2.1 Riedl ↔ Performance (Score final)")
        if row:
            results.append(row)
    return results


# ---------------------------------------------------------------------------
# Bloc 2.4 : Riedl ↔ Questionnaire (inchangé)
# ---------------------------------------------------------------------------

def compute_block_24(df: pd.DataFrame) -> list[dict]:
    results = []
    for x in RIEDL_VARS:
        if x not in df.columns:
            continue
        for y in QUEST_VARS:
            if y not in df.columns:
                continue
            row = spearman_pair(df, x, y, "2.4 Riedl ↔ Questionnaire")
            if row:
                results.append(row)
    return results


# ---------------------------------------------------------------------------
# Rendu Markdown des tableaux
# ---------------------------------------------------------------------------

def md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(aucun résultat)_\n"
    df = pd.DataFrame(rows)[cols]
    return df.to_markdown(index=False) + "\n"


def corr_to_md_rows(corrs: list[dict]) -> list[dict]:
    out = []
    for r in sorted(corrs, key=lambda x: -abs(x.get("rho", 0))):
        out.append({
            "x": r["x"],
            "y": r["y"],
            "rho": round(r["rho"], 2),
            "p": fmt_p(r["p"]),
            "n": r["n"],
            "sig": sig_label(r["p"]),
        })
    return out


# ---------------------------------------------------------------------------
# Section 5 — Diff table
# ---------------------------------------------------------------------------

def compute_diff_table(df: pd.DataFrame) -> pd.DataFrame:
    """Paires (Riedl × cfactor) avec ρ_sample, ρ_pop, Δρ, sign_change."""
    rows = []
    x_vars = RIEDL_VARS + QUEST_VARS

    for x in x_vars:
        if x not in df.columns:
            continue
        r_s = spearman_pair(df, x, "c_factor_sample", "diff")
        r_p = spearman_pair(df, x, "c_factor_pop", "diff")
        if r_s is None or r_p is None:
            continue
        sign_change = (r_s["p"] <= 0.05) != (r_p["p"] <= 0.05)
        rows.append({
            "x": x,
            "rho_sample": round(r_s["rho"], 4),
            "p_sample": round(r_s["p"], 4),
            "rho_pop": round(r_p["rho"], 4),
            "p_pop": round(r_p["p"], 4),
            "delta_rho": round(r_p["rho"] - r_s["rho"], 4),
            "sign_change": sign_change,
            "n": r_s["n"],
        })
    return pd.DataFrame(rows)


def compute_significance_switches(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Paires qui changent de significativité entre sample et pop."""
    if diff_df.empty:
        return pd.DataFrame()
    sub = diff_df[diff_df["sign_change"] == True].copy()
    sub["sig_sample"] = sub["p_sample"] <= 0.05
    sub["sig_pop"] = sub["p_pop"] <= 0.05
    sub["direction"] = sub.apply(
        lambda r: "sample→sig, pop→non-sig" if r["sig_sample"] else "pop→sig, sample→non-sig",
        axis=1
    )
    return sub[["x", "rho_sample", "p_sample", "rho_pop", "p_pop", "direction"]]


# ---------------------------------------------------------------------------
# Figures réseau
# ---------------------------------------------------------------------------

def build_and_plot_network(
    all_corrs: list[dict],
    variant: str,
    out_dir: Path,
    label: str,
) -> dict | None:
    """Construit et sauvegarde le réseau pour une variante donnée."""
    data = prepare_global_correlation_network_data(
        all_corrs,
        rho_threshold=RHO_THRESHOLD,
        p_threshold=P_THRESHOLD,
    )
    if data is None:
        print(f"  [SKIP réseau {variant}] aucune arête")
        return None

    import networkx as nx

    G = data["graph"]
    edges_df = data["edges_df"]
    node_families = data["node_families"]
    nodes_df = data["nodes_df"]

    from py.network import _FAMILY_COLORS

    weighted_degree = {r["node"]: float(r["weighted_degree"]) for _, r in nodes_df.iterrows()}

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_axis_off()
    pos = nx.spring_layout(G, seed=42, k=2.5 / max(len(G.nodes()) ** 0.5, 1))
    node_colors = [_FAMILY_COLORS.get(node_families[n], "#bab0ac") for n in G.nodes()]

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] == "positive"]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] == "negative"]
    pos_widths = [G[u][v]["weight"] * 5 for u, v in pos_edges]
    neg_widths = [G[u][v]["weight"] * 5 for u, v in neg_edges]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=pos_widths,
                           edge_color="#e15759", alpha=0.65, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=neg_widths,
                           edge_color="#4e79a7", alpha=0.65, ax=ax, style="dashed")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=[300 + weighted_degree.get(n, 0) * 120 for n in G.nodes()],
                           alpha=0.90, ax=ax)
    labels = {n: (n[:22] + "..") if len(n) > 24 else n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    family_handles = [
        Patch(facecolor=col, label=fam.capitalize())
        for fam, col in _FAMILY_COLORS.items()
        if fam in set(node_families.values())
    ]
    edge_handles = [
        Line2D([0], [0], color="#e15759", linewidth=2, label="Corrélation positive"),
        Line2D([0], [0], color="#4e79a7", linewidth=2, linestyle="dashed",
               label="Corrélation négative"),
    ]
    ax.legend(handles=family_handles + edge_handles, loc="lower left", fontsize=8,
              framealpha=0.85, title="Familles de variables", title_fontsize=8)
    ax.set_title(
        f"Réseau des corrélations fortes — {label}\n"
        f"(|rho| >= {RHO_THRESHOLD}, p <= {P_THRESHOLD})\n"
        f"{len(G.nodes())} variables, {len(G.edges())} associations",
        fontsize=11, fontweight="bold", pad=10
    )
    plt.tight_layout()
    png_path = out_dir / f"global_correlation_network_{variant}.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {png_path.name} ({len(G.nodes())} nœuds, {len(G.edges())} arêtes)")

    # Réseau par famille
    fam_path = out_dir / f"global_correlation_family_network_{variant}.png"
    _save_family_network(edges_df, fam_path, variant)

    # CSV support
    edges_df.to_csv(out_dir / f"network_edges_{variant}.csv", index=False)
    nodes_df.to_csv(out_dir / f"network_nodes_{variant}.csv", index=False)

    return data


def _save_family_network(edges_df: pd.DataFrame, path: Path, variant: str):
    try:
        import networkx as nx
        from py.network import _FAMILY_COLORS
        family_edges: dict[tuple, list[float]] = {}
        for _, row in edges_df.iterrows():
            fx, fy = row["family_x"], row["family_y"]
            if fx == fy:
                continue
            key = tuple(sorted([fx, fy]))
            family_edges.setdefault(key, []).append(abs(row["rho"]))
        if not family_edges:
            return
        G2 = nx.Graph()
        for (fa, fb), rhos in family_edges.items():
            G2.add_edge(fa, fb, weight=len(rhos), mean_rho=round(float(np.mean(rhos)), 3))
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.set_axis_off()
        pos2 = nx.spring_layout(G2, seed=42, k=3.0)
        node_colors2 = [_FAMILY_COLORS.get(n, "#bab0ac") for n in G2.nodes()]
        edge_widths2 = [G2[u][v]["weight"] * 0.8 for u, v in G2.edges()]
        nx.draw_networkx_nodes(G2, pos2, node_color=node_colors2, node_size=1800, alpha=0.92, ax=ax)
        nx.draw_networkx_edges(G2, pos2, width=edge_widths2, alpha=0.65, edge_color="#555555", ax=ax)
        nx.draw_networkx_labels(G2, pos2, font_size=10, font_weight="bold", ax=ax)
        edge_labels2 = {(u, v): f"n={G2[u][v]['weight']}\nrho={G2[u][v]['mean_rho']:.2f}"
                        for u, v in G2.edges()}
        nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2, font_size=7, ax=ax)
        ax.set_title(
            f"Réseau agrégé par famille — {variant}\n"
            "(épaisseur = nb associations fortes, annotation = rho moyen)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [OK] {path.name}")
    except Exception as exc:
        print(f"  [WARN] family network {variant}: {exc}")


# ---------------------------------------------------------------------------
# Analyse structurelle (sections 4.3.1 – 4.3.7)
# ---------------------------------------------------------------------------

def _top10_df(centrality: dict, name: str, node_families: dict) -> str:
    rows = sorted(centrality.items(), key=lambda x: -x[1])[:10]
    df = pd.DataFrame([
        {"node": n, "family": node_families.get(n, "?"), name: round(v, 4)}
        for n, v in rows
    ])
    return df.to_markdown(index=False) + "\n"


def structural_analysis_md(data: dict, variant: str) -> str:
    import networkx as nx

    G = data["graph"]
    edges_df = data["edges_df"]
    node_families = data["node_families"]
    nodes_df = data["nodes_df"]

    # compute_network_metrics retourne les clés directement (degree_centrality, etc.)
    metrics = compute_network_metrics(data)

    lines = []

    # 4.3.1
    gi = metrics.get("global_info", {})
    lines.append(f"\n##### 4.3.1.{variant} — Informations globales\n")
    gm_rows = [
        ("Nombre de nœuds", gi.get("n_nodes", G.number_of_nodes())),
        ("Nombre d'arêtes", gi.get("n_edges", G.number_of_edges())),
        ("Densité du réseau", round(float(gi.get("density", nx.density(G) if G.number_of_nodes() >= 2 else 0)), 3)),
        ("Nombre de composantes connexes", gi.get("n_components", len(list(nx.connected_components(G))))),
        ("Taille de la plus grande composante", gi.get("largest_component_size", 0)),
        ("Part de la plus grande composante", round(float(gi.get("largest_component_ratio", 0) or 0) * 100, 1)),
        ("Arêtes positives", gi.get("positive_edges", 0)),
        ("Arêtes négatives", gi.get("negative_edges", 0)),
        ("Part d'arêtes positives (%)", round(float(gi.get("positive_ratio", 0) or 0) * 100, 1)),
        ("Part d'arêtes négatives (%)", round(float(gi.get("negative_ratio", 0) or 0) * 100, 1)),
    ]
    gm_df = pd.DataFrame(gm_rows, columns=["Métrique", "Valeur"])
    lines.append(gm_df.to_markdown(index=False) + "\n")

    # 4.3.2
    lines.append(f"\n##### 4.3.2.{variant} — Centralités des nœuds\n")
    for cname in ["degree", "betweenness", "eigenvector", "closeness"]:
        cent_map = metrics.get(f"{cname}_centrality", {})
        if cent_map:
            lines.append(f"**Top {cname} centrality**\n")
            lines.append(_top10_df(cent_map, f"{cname}_centrality", node_families))

    eig_scope = metrics.get("eigenvector_scope", "")
    if eig_scope:
        lines.append(f"_{eig_scope}_\n")

    # 4.3.3
    lines.append(f"\n##### 4.3.3.{variant} — Cohésion locale\n")
    clust_map = metrics.get("clustering_by_node", {})
    avg_clust = round(float(metrics.get("avg_clustering", 0)), 3)
    lines.append(f"Le coefficient de clustering moyen du réseau est de {avg_clust}.\n")
    if clust_map:
        clust_df = pd.DataFrame([
            {"node": n, "family": node_families.get(n, "?"), "clustering_coefficient": round(v, 4)}
            for n, v in sorted(clust_map.items(), key=lambda x: -x[1])[:10]
        ])
        lines.append(clust_df.to_markdown(index=False) + "\n")

    # 4.3.4 Communautés
    lines.append(f"\n##### 4.3.4.{variant} — Détection de communautés\n")
    communities = metrics.get("communities", [])
    if communities:
        comm_rows = []
        for i, comm in enumerate(communities, 1):
            families_count: dict[str, int] = {}
            for n in comm:
                f = node_families.get(n, "other")
                families_count[f] = families_count.get(f, 0) + 1
            families_str = ", ".join(f"{f}:{c}" for f, c in sorted(families_count.items()))
            comm_rows.append({
                "community": i,
                "n_nodes": len(comm),
                "main_variables": ", ".join(sorted(comm)[:6]),
                "families": families_str,
            })
        comm_df = pd.DataFrame(comm_rows)
        lines.append(comm_df.to_markdown(index=False) + "\n")
    else:
        lines.append("_(Aucune communauté détectée)_\n")

    # 4.3.5 Variables bridge
    lines.append(f"\n##### 4.3.5.{variant} — Variables bridge / pivot\n")
    bridge_rows_raw = metrics.get("bridge_rows", [])
    bridge_rows_filtered = [r for r in bridge_rows_raw if r.get("external_families", 0) > 0]
    if bridge_rows_filtered:
        bridge_df = pd.DataFrame(bridge_rows_filtered).sort_values("cross_family_weight", ascending=False)
        for col in ["cross_family_weight"]:
            if col in bridge_df.columns:
                bridge_df[col] = bridge_df[col].round(3)
        lines.append(bridge_df.to_markdown(index=False) + "\n")
    else:
        lines.append("_(Aucune variable bridge)_\n")

    # 4.3.6 Distance au nœud central c_factor_*
    central_node = variant
    lines.append(f"\n##### 4.3.6.{variant} — Distance au nœud central `{central_node}`\n")
    if central_node in G.nodes():
        try:
            lengths = nx.single_source_shortest_path_length(G, central_node)
            dist_df = pd.DataFrame([
                {"node": n, "family": node_families.get(n, "?"), f"distance_to_{central_node}": d}
                for n, d in sorted(lengths.items(), key=lambda x: x[1])
            ])
            lines.append(dist_df.to_markdown(index=False) + "\n")
        except Exception as exc:
            lines.append(f"_(non calculable : {exc})_\n")
    else:
        lines.append(f"_(`{central_node}` absent du réseau)_\n")

    # 4.3.7 Signe des relations
    lines.append(f"\n##### 4.3.7.{variant} — Analyse du signe des relations\n")
    pos_edges_df = metrics.get("positive_edges_df", pd.DataFrame())
    neg_edges_df = metrics.get("negative_edges_df", pd.DataFrame())

    if not pos_edges_df.empty:
        lines.append("**Principales arêtes positives**\n")
        top_pos = pos_edges_df.head(10)
        show_cols = [c for c in ["x", "y", "rho", "edge_weight", "family_x", "family_y", "block"] if c in top_pos.columns]
        lines.append(top_pos[show_cols].round(3).to_markdown(index=False) + "\n")
    else:
        lines.append("_(aucune arête positive retenue)_\n")

    if not neg_edges_df.empty:
        lines.append("**Principales arêtes négatives**\n")
        top_neg = neg_edges_df.head(5)
        show_cols = [c for c in ["x", "y", "rho", "edge_weight", "family_x", "family_y", "block"] if c in top_neg.columns]
        lines.append(top_neg[show_cols].round(3).to_markdown(index=False) + "\n")
    else:
        lines.append("_(aucune arête négative retenue)_\n")

    notes = metrics.get("notes", [])
    if notes:
        lines.append("\n**Notes de calcul**\n\n")
        for note in notes:
            lines.append(f"- {note}\n")

    return "".join(lines)


def structural_analysis_synthesis(data_s: dict | None, data_p: dict | None) -> str:
    """Section 4.3.7 synthèse comparative."""
    import networkx as nx

    def hub_summary(data: dict, variant: str) -> str:
        if data is None:
            return f"  [{variant}] : réseau absent"
        G = data["graph"]
        node_families = data["node_families"]
        metrics = compute_network_metrics(data)
        eig = metrics.get("eigenvector_centrality", {})
        eig_val = round(eig.get(variant, float("nan")), 4) if eig else float("nan")
        top5_deg = sorted(nx.degree_centrality(G).items(), key=lambda x: -x[1])[:5]
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        dens = round(nx.density(G), 3) if n_nodes >= 2 else 0
        n_comm = len(metrics.get("communities", []))
        n_comp = len(list(nx.connected_components(G)))
        return (
            f"  [{variant}] : {n_nodes} nœuds, {n_edges} arêtes, densité={dens}, "
            f"{n_comp} composante(s), {n_comm} communauté(s) ; "
            f"centralité eigenvector de `{variant}` = {eig_val} ; "
            f"top-5 degree = {[n for n, _ in top5_deg]}"
        )

    lines = ["\n#### Synthèse comparative c_factor_sample vs c_factor_pop\n\n"]
    lines.append(hub_summary(data_s, "c_factor_sample") + "\n")
    lines.append(hub_summary(data_p, "c_factor_pop") + "\n\n")
    lines.append(
        "Synthèse prudente : la comparaison des deux réseaux permet d'évaluer dans quelle mesure "
        "la substitution du c_factor_sample par le c_factor_pop modifie la structure des associations "
        "globales. À ce stade, ces résultats doivent rester descriptifs : ils suggèrent des proximités "
        "structurelles entre TCI / c-factor, variables Riedl, INV et questionnaires, sans établir de causalité "
        "(Woolley & Gupta, 2024 ; Kommol et al., 2025 ; Riedl et al., 2021 ; Sassier-Roublin et al., 2025).\n"
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Génération du rapport Markdown complet
# ---------------------------------------------------------------------------

def generate_markdown(merged: pd.DataFrame, sanity_lines: list[str],
                      precomputed_network_data: dict | None = None) -> str:
    lines: list[str] = []

    lines.append("# Rapport comparatif C-factor — sections 2–4 (c_factor_sample vs c_factor_pop)\n\n")
    lines.append(
        "Ce rapport réplique les sections 2 à 4 du `rapport_principal_PC_VR.md` en parallèle "
        "pour deux variantes du c-factor : `c_factor_sample` (projection PCA locale, identique à "
        "`c_score` dans le rapport original) et `c_factor_pop` (projection sur les loadings de la "
        "population parente Riedl et al., 2021). L'objectif est d'évaluer empiriquement si la "
        "substitution modifie la structure des corrélations globales et du réseau.\n\n"
        "Références : Woolley & Gupta (2024) ; Kommol et al. (2025) ; Riedl et al. (2021) ; "
        "Sassier-Roublin et al. (2025).\n"
    )

    # Sanity checks
    lines.append("\n## Sanity checks\n\n")
    for l in sanity_lines:
        lines.append(f"- {l}\n")
    lines.append("\n")

    # -------------------------------------------------------------------------
    # Section 2 — Corrélations globales
    # -------------------------------------------------------------------------
    lines.append("## 2. Corrélations globales (group-level)\n\n")

    # 2.1 Riedl ↔ Performance (inchangé)
    b21 = compute_block_21(merged)
    n21 = len({r.get("n") for r in b21 if r})
    n21_val = b21[0]["n"] if b21 else 0
    lines.append("#### 2.1 Riedl ↔ Performance (Score final)\n\n")
    lines.append(f"_(n unités disponibles pour ce bloc: {merged['group_id'].nunique()})_\n\n")
    if b21:
        rows_21 = corr_to_md_rows(b21)
        lines.append(md_table(rows_21, ["x", "y", "rho", "p", "n", "sig"]))
    else:
        lines.append("_(aucun résultat)_\n")
    lines.append(
        "_Note : cette section ne contient pas de c-factor et est bit-identique au rapport original._\n\n"
    )

    # 2.2 TCI ↔ Performance — double métrique
    lines.append("#### 2.2 TCI ↔ Performance (Score final)\n\n")
    lines.append(f"_(n unités disponibles pour ce bloc: {merged['group_id'].nunique()})_\n\n")

    for variant in CFACTOR_VARIANTS:
        if variant not in merged.columns:
            lines.append(f"_({variant} : colonne absente)_\n\n")
            continue
        b22 = compute_block_22(merged, variant)
        lines.append(f"**Variante : `{variant}`**\n\n")
        if b22:
            rows_22 = corr_to_md_rows(b22)
            lines.append(md_table(rows_22, ["x", "y", "rho", "p", "n", "sig"]))
        else:
            lines.append("_(aucun résultat)_\n")

    # 2.3 Riedl ↔ TCI — double métrique
    lines.append("#### 2.3 Riedl ↔ TCI (tous groupes)\n\n")
    lines.append(f"_(n unités disponibles pour ce bloc: {merged['group_id'].nunique()})_\n\n")

    for variant in CFACTOR_VARIANTS:
        if variant not in merged.columns:
            lines.append(f"_({variant} : colonne absente)_\n\n")
            continue
        b23 = compute_block_23(merged, variant)
        lines.append(f"**Variante : `{variant}`**\n\n")
        if b23:
            rows_23 = corr_to_md_rows(b23)
            lines.append(md_table(rows_23, ["x", "y", "rho", "p", "n", "sig"]))
        else:
            lines.append("_(aucun résultat)_\n")

    # 2.4 Riedl ↔ Questionnaire (inchangé)
    b24 = compute_block_24(merged)
    lines.append("#### 2.4 Riedl ↔ Questionnaire\n\n")
    lines.append(f"_(n unités disponibles pour ce bloc: {merged['group_id'].nunique()})_\n\n")
    if b24:
        rows_24 = corr_to_md_rows(b24)
        lines.append(md_table(rows_24, ["x", "y", "rho", "p", "n", "sig"]))
    else:
        lines.append("_(aucun résultat)_\n")
    lines.append(
        "_Note : cette section ne contient pas de c-factor et est bit-identique au rapport original._\n\n"
    )

    # -------------------------------------------------------------------------
    # Section 3 — INV
    # -------------------------------------------------------------------------
    lines.append("## 3. Indices non verbaux (INV) et features haut niveau\n\n")
    lines.append(
        "Dans la version v2, les analyses INV détaillées sont externalisées dans un rapport séparé "
        "pour le sous-ensemble VR (`rapport_INV_VR.pdf`). Le rapport comparatif c-factor ne reproduit "
        "pas cette section, qui ne contient pas de c-factor. Se référer au rapport `rapport_INV_VR.pdf`.\n\n"
    )

    # -------------------------------------------------------------------------
    # Section 4 — Corrélations les plus fortes
    # -------------------------------------------------------------------------
    lines.append("## 4. Corrélations les plus fortes\n\n")
    lines.append(
        "Cette section distingue les associations les plus fortes (`|rho| >= 0.60`) et des associations "
        "significatives supplémentaires, plus modestes en taille d'effet (`0.50 <= |rho| < 0.60`, `p <= 0.05`). "
        "Chaque tableau est produit en deux versions selon la variante c-factor utilisée. Une colonne "
        "`cfactor_variant` indique quelle variante intervient dans la paire.\n\n"
    )

    all_corrs_by_variant: dict[str, list[dict]] = {}
    for variant in CFACTOR_VARIANTS:
        b22 = compute_block_22(merged, variant)
        b23 = compute_block_23(merged, variant)
        all_corrs = compute_block_21(merged) + b22 + b23 + compute_block_24(merged)
        all_corrs_by_variant[variant] = all_corrs

    # Tableau 4.0 fortes (|rho| >= 0.60), toutes variantes fusionnées avec flag
    def cfactor_variant_tag(x: str, y: str) -> str:
        if x in CFACTOR_VARIANTS:
            return x.replace("c_factor_", "")
        if y in CFACTOR_VARIANTS:
            return y.replace("c_factor_", "")
        return "none"

    all_corrs_combined: list[dict] = []
    for variant, corrs in all_corrs_by_variant.items():
        for r in corrs:
            rc = dict(r)
            rc["cfactor_variant"] = cfactor_variant_tag(rc.get("x", ""), rc.get("y", ""))
            all_corrs_combined.append(rc)

    # Dédupliquer les paires sans c-factor (bloc 2.1, 2.4)
    seen_no_cfactor: set[tuple] = set()
    deduped: list[dict] = []
    for r in all_corrs_combined:
        if r["cfactor_variant"] == "none":
            key = (r["block"], r["x"], r["y"])
            if key in seen_no_cfactor:
                continue
            seen_no_cfactor.add(key)
        deduped.append(r)

    strong_mask = [abs(r["rho"]) >= 0.60 for r in deduped]
    strong_corrs = [r for r, m in zip(deduped, strong_mask) if m]
    strong_corrs_sorted = sorted(strong_corrs, key=lambda x: (-abs(x["rho"]), x.get("p", 1)))

    lines.append("### 4.0 Corrélations les plus fortes (`|rho| >= 0.60`)\n\n")
    if strong_corrs_sorted:
        strong_df = pd.DataFrame(strong_corrs_sorted)
        show_cols = [c for c in ["block", "x", "y", "rho", "p", "n", "cfactor_variant"] if c in strong_df.columns]
        lines.append(strong_df[show_cols].to_markdown(index=False) + "\n\n")
    else:
        lines.append("_(aucune corrélation `|rho| >= 0.60`)_\n\n")

    # 4.1 Supplémentaires
    supp_mask = [(0.50 <= abs(r["rho"]) < 0.60 and r.get("p", 1) <= 0.05) for r in deduped]
    supp_corrs = [r for r, m in zip(deduped, supp_mask) if m]
    lines.append("### 4.1 Corrélations significatives supplémentaires (0.50 ≤ |rho| < 0.60)\n\n")
    if supp_corrs:
        supp_df = pd.DataFrame(sorted(supp_corrs, key=lambda x: (-abs(x["rho"]), x.get("p", 1))))
        show_cols = [c for c in ["block", "x", "y", "rho", "p", "n", "cfactor_variant"] if c in supp_df.columns]
        lines.append(supp_df[show_cols].to_markdown(index=False) + "\n\n")
    else:
        lines.append("_(aucune corrélation significative supplémentaire pour `0.50 <= |rho| < 0.60`)_\n\n")

    # 4.2 Réseaux
    lines.append("### 4.2 Réseau des corrélations fortes\n\n")
    lines.append(
        "Deux réseaux sont produits en parallèle selon la variante c-factor. "
        f"Convention identique au rapport principal : |rho| >= {RHO_THRESHOLD}, p <= {P_THRESHOLD}. "
        "Les arêtes rouges indiquent une corrélation positive, les arêtes bleues (pointillés) une "
        "corrélation négative. La taille des nœuds reflète leur degré pondéré.\n\n"
    )
    for variant in CFACTOR_VARIANTS:
        corrs = all_corrs_by_variant[variant]
        lines.append(f"**Réseau `{variant}`**\n\n")
        lines.append(f"![](figures/global_correlation_network_{variant}.png)\n")
        lines.append(f"_Réseau des corrélations globales fortes — {variant} (|rho|>={RHO_THRESHOLD}, p<={P_THRESHOLD})_\n\n")

    # 4.3 Analyse structurelle
    lines.append("### 4.3 Analyse structurelle du réseau\n\n")
    lines.append(
        "Cette section réplique les sous-sections 4.3.1 à 4.3.7 du rapport principal pour chacune "
        "des deux variantes c-factor. Les métriques de réseau sont calculées indépendamment pour "
        "chaque variante. Chaque sous-section est suffixée avec la variante concernée.\n\n"
    )

    if precomputed_network_data is not None:
        network_data = precomputed_network_data
    else:
        network_data = {}
        for variant in CFACTOR_VARIANTS:
            corrs = all_corrs_by_variant[variant]
            data = build_and_plot_network(corrs, variant, FIGURES_DIR, variant)
            network_data[variant] = data

    for variant in CFACTOR_VARIANTS:
        data = network_data.get(variant)
        if data:
            lines.append(f"\n#### Analyse structurelle — `{variant}`\n")
            lines.append(structural_analysis_md(data, variant))
        else:
            lines.append(f"\n#### Analyse structurelle — `{variant}`\n\n_(Réseau vide ou non calculable)_\n")

    lines.append(structural_analysis_synthesis(
        network_data.get("c_factor_sample"),
        network_data.get("c_factor_pop"),
    ))

    lines.append("\n**Réseaux agrégés par famille**\n\n")
    for variant in CFACTOR_VARIANTS:
        lines.append(f"![](figures/global_correlation_family_network_{variant}.png)\n")
        lines.append(f"_Réseau agrégé par famille — {variant}_\n\n")

    # -------------------------------------------------------------------------
    # Section 5 — Diff table
    # -------------------------------------------------------------------------
    lines.append("## 5. Tableau de différences : c_factor_sample vs c_factor_pop\n\n")
    lines.append(
        "Cette section répond directement à la question empirique principale : la substitution "
        "de `c_factor_sample` par `c_factor_pop` change-t-elle significativement les résultats ?\n\n"
    )

    diff_df = compute_diff_table(merged)
    switches_df = compute_significance_switches(diff_df)

    lines.append("### 5.1 Tableau complet des différences\n\n")
    if not diff_df.empty:
        lines.append("_Colonnes : ρ_sample, p_sample, ρ_pop, p_pop, Δρ (=ρ_pop − ρ_sample), sign_change (bascule de significativité p≤0.05)_\n\n")
        lines.append(diff_df.to_markdown(index=False) + "\n\n")
    else:
        lines.append("_(données insuffisantes)_\n\n")

    lines.append("### 5.2 Bascules de significativité\n\n")
    lines.append("Paires qui changent de statut de significativité (p ≤ 0.05) entre les deux variantes.\n\n")
    if not switches_df.empty:
        lines.append(switches_df.to_markdown(index=False) + "\n\n")
    else:
        lines.append("_(aucune bascule de significativité détectée)_\n\n")

    # 5.3 Métriques globales réseau comparées
    lines.append("### 5.3 Métriques globales du réseau\n\n")
    import networkx as nx
    net_metrics_rows = []
    for variant in CFACTOR_VARIANTS:
        data = network_data.get(variant)
        if data is None:
            net_metrics_rows.append({
                "version": variant, "n_noeuds": 0, "n_aretes": 0,
                "densite": "N/A", "n_composantes": "N/A",
                "plus_grande_composante": "N/A", "clustering_moyen": "N/A",
                "n_communautes": "N/A"
            })
            continue
        G = data["graph"]
        metrics = compute_network_metrics(data)
        gi = metrics.get("global_info", {})
        avg_clust = round(float(metrics.get("avg_clustering", 0)), 3)
        n_comm = len(metrics.get("communities", []))
        net_metrics_rows.append({
            "version": variant,
            "n_noeuds": gi.get("n_nodes", G.number_of_nodes()),
            "n_aretes": gi.get("n_edges", G.number_of_edges()),
            "densite": round(float(gi.get("density", 0)), 3),
            "n_composantes": gi.get("n_components", 0),
            "plus_grande_composante": gi.get("largest_component_size", 0),
            "clustering_moyen": avg_clust,
            "n_communautes": n_comm,
        })
    net_metrics_df = pd.DataFrame(net_metrics_rows)
    net_metrics_df.to_csv(TABLES_DIR / "network_metrics_comparison.csv", index=False, encoding="utf-8")
    lines.append(net_metrics_df.to_markdown(index=False) + "\n\n")

    # 5.4 Top 5 hubs
    lines.append("### 5.4 Top 5 hubs (degree + eigenvector) par version\n\n")
    top5_by_variant: dict[str, dict] = {}
    for variant in CFACTOR_VARIANTS:
        data = network_data.get(variant)
        if data is None:
            top5_by_variant[variant] = {}
            continue
        metrics = compute_network_metrics(data)
        deg = metrics.get("degree_centrality", {})
        eig = metrics.get("eigenvector_centrality", {})
        top5_deg = sorted(deg.items(), key=lambda x: -x[1])[:5]
        top5_eig = sorted(eig.items(), key=lambda x: -x[1])[:5] if eig else []
        top5_by_variant[variant] = {"degree": [n for n, _ in top5_deg], "eigenvector": [n for n, _ in top5_eig]}

    hub_rows = []
    for variant in CFACTOR_VARIANTS:
        t = top5_by_variant.get(variant, {})
        hub_rows.append({
            "version": variant,
            "top5_degree": ", ".join(t.get("degree", [])),
            "top5_eigenvector": ", ".join(t.get("eigenvector", [])),
        })
    hub_df = pd.DataFrame(hub_rows)
    lines.append(hub_df.to_markdown(index=False) + "\n\n")

    # Notes
    lines.append("---\n\n")
    lines.append(
        "_Note méthodologique_ : Le c_factor_pop est calculé en projetant les scores de tâches locaux "
        "sur les loadings PC1 de la population parente Riedl et al. (2021), sans re-fit local. "
        "Le c_factor_sample est la projection sur la PCA fittée sur l'échantillon courant (N=12 groupes allowed). "
        "La divergence entre les deux variantes (ρ ≈ -0.014 sur allowed, 0.171 sur all) reflète la différence "
        "de structure des tâches entre l'échantillon local et la population parente.\n"
    )

    return "".join(lines)


# ---------------------------------------------------------------------------
# Export CSV support
# ---------------------------------------------------------------------------

def export_csv_support(merged: pd.DataFrame):
    # correlations_block_2_with_pop.csv
    rows_all = []
    for variant in CFACTOR_VARIANTS:
        if variant not in merged.columns:
            continue
        b22 = compute_block_22(merged, variant)
        b23 = compute_block_23(merged, variant)
        for r in b22 + b23:
            r["cfactor_variant"] = variant
            rows_all.append(r)
    b21 = compute_block_21(merged)
    for r in b21:
        r["cfactor_variant"] = "none"
        rows_all.append(r)
    b24 = compute_block_24(merged)
    for r in b24:
        r["cfactor_variant"] = "none"
        rows_all.append(r)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(TABLES_DIR / "correlations_block_2_with_pop.csv", index=False, encoding="utf-8")

    # correlations_block_4_with_pop.csv — fortes
    strong_rows = [r for r in rows_all if abs(r.get("rho", 0)) >= 0.50]
    if strong_rows:
        pd.DataFrame(strong_rows).to_csv(TABLES_DIR / "correlations_block_4_with_pop.csv", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# Génération PDF natif ReportLab
# ---------------------------------------------------------------------------

def _rl_styles():
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    styles = getSampleStyleSheet()
    extra = {
        "H1":    styles["Heading1"],
        "H2":    styles["Heading2"],
        "H3":    styles["Heading3"],
        "Body":  styles["BodyText"],
        "Small": ParagraphStyle("RSmall", parent=styles["BodyText"],
                                fontSize=8, leading=10, spaceAfter=2),
        "Code":  ParagraphStyle("RCode", parent=styles["BodyText"],
                                fontName="Courier", fontSize=8, leading=10),
        "Box":   ParagraphStyle("RBox", parent=styles["BodyText"],
                                fontSize=9, leading=11,
                                backColor=colors.HexColor("#eef3fb"),
                                borderPad=4, spaceAfter=6),
        "Warn":  ParagraphStyle("RWarn", parent=styles["BodyText"],
                                fontSize=9, leading=11,
                                backColor=colors.HexColor("#fff8e1"),
                                borderPad=4, spaceAfter=6),
        "Title": styles["Title"],
    }
    return extra


def _rl_table(df: pd.DataFrame, highlight_col: str | None = None,
              col_widths=None, font_size: int = 7):
    """Convertit un DataFrame en Table ReportLab."""
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

    if df.empty:
        return None

    def _fmt(v):
        if pd.isna(v):
            return ""
        s = str(v)
        return s[:55] + ".." if len(s) > 57 else s

    header = list(df.columns)
    data = [header] + [[_fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]

    # Colonne highlight_col → fond coloré
    highlight_col_idx = header.index(highlight_col) if highlight_col and highlight_col in header else None

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#4e79a7")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), font_size),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f7fb")]),
        ("GRID",         (0, 0), (-1, -1), 0.35, colors.HexColor("#c0c8d8")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight_col_idx is not None:
        style_cmds.append(
            ("BACKGROUND", (highlight_col_idx, 1),
             (highlight_col_idx, -1), colors.HexColor("#fff3cd"))
        )
    t.setStyle(TableStyle(style_cmds))
    return t


def _para(text: str, styles, key: str = "Body"):
    from reportlab.platypus import Paragraph
    # Échapper les caractères XML problématiques mais laisser les balises HTML simples
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Rétablir balises ReportLab autorisées
    for tag in ["b", "i", "br", "/b", "/i", "/br"]:
        safe = safe.replace(f"&lt;{tag}&gt;", f"<{tag}>")
    try:
        return Paragraph(safe, styles[key])
    except Exception:
        return Paragraph(safe[:200], styles["Body"])


def _sp(h: float = 0.1):
    from reportlab.platypus import Spacer
    from reportlab.lib.units import inch
    return Spacer(1, h * inch)


def _sig_color(p: float):
    """Retourne une chaine HTML de couleur selon la p-value."""
    if p < 0.01:
        return "#c0392b"
    if p < 0.05:
        return "#e67e22"
    return "#555555"


def _corr_display_df(corrs: list[dict], variant_label: str) -> pd.DataFrame:
    """Prépare un DataFrame d'affichage pour un bloc de corrélations."""
    rows = []
    for r in sorted(corrs, key=lambda x: -abs(x.get("rho", 0))):
        p = r.get("p", 1.0)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        rows.append({
            "x": r["x"],
            "y": r["y"],
            "rho": f"{r['rho']:+.3f}",
            "p": f"{p:.3f}",
            "n": r["n"],
            "sig": sig,
        })
    return pd.DataFrame(rows)


def generate_pdf_native(
    pdf_path: Path,
    pop: pd.DataFrame,
    merged: pd.DataFrame,
    sanity_lines: list[str],
    diff_df: pd.DataFrame,
    network_data: dict,
) -> None:
    """Génère le PDF complet directement via ReportLab (pas via Markdown)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, HRFlowable, KeepTogether,
    )
    from reportlab.lib import colors

    styles = _rl_styles()
    elems: list = []

    W = 17 * cm  # largeur utile

    def h1(txt): return _para(txt, styles, "H1")
    def h2(txt): return _para(txt, styles, "H2")
    def h3(txt): return _para(txt, styles, "H3")
    def body(txt): return _para(txt, styles, "Body")
    def small(txt): return _para(txt, styles, "Small")
    def box(txt): return _para(txt, styles, "Box")
    def warn(txt): return _para(txt, styles, "Warn")
    def sp(h=0.08): return _sp(h)
    def hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=4)

    # =========================================================================
    # Titre
    # =========================================================================
    elems += [
        _para("Rapport comparatif C-factor", styles, "Title"),
        _para("c_factor_sample vs c_factor_pop — Sections 2 à 4", styles, "H2"),
        sp(0.05),
        body(
            "Réplication des sections 2 à 4 du rapport principal PC+VR avec deux variantes du "
            "c-factor en parallèle. c_factor_sample = projection PCA locale (identique à c_score du "
            "rapport original). c_factor_pop = projection sur les loadings PC1 de la population "
            "parente (Riedl et al., 2021)."
        ),
        sp(0.05),
        small("Références : Woolley & Gupta (2024) ; Kommol et al. (2025) ; "
              "Riedl et al. (2021) ; Sassier-Roublin et al. (2025)."),
        hr(), sp(0.04),
    ]

    # =========================================================================
    # TABLEAU 0 — C-factor avant / après par groupe
    # =========================================================================
    elems += [h1("0. C-factor par groupe : sample vs pop"), sp(0.06)]
    elems += [
        box(
            "Ce tableau présente les valeurs individuelles de c_factor_sample et c_factor_pop "
            "pour chacun des 12 groupes allowed (VR uniquement), ainsi que leur rang respectif "
            "et le delta de rang (rang_pop - rang_sample). Un delta négatif indique que le groupe "
            "est mieux classé selon c_factor_sample que selon c_factor_pop."
        ),
        sp(0.06),
    ]

    # Construire le tableau comparatif
    cf_cols = [c for c in ["group_id", "c_factor_sample", "c_factor_pop",
                            "rank_sample", "rank_pop", "rank_delta"] if c in pop.columns]
    cf_df = pop[cf_cols].copy().sort_values("rank_sample")
    cf_df["c_factor_sample"] = cf_df["c_factor_sample"].round(3)
    cf_df["c_factor_pop"] = cf_df["c_factor_pop"].round(3)
    cf_df["rank_delta"] = cf_df["rank_delta"].astype(int) if "rank_delta" in cf_df.columns else cf_df.get("rank_delta", "")
    cf_df.columns = [c.replace("_", " ") for c in cf_df.columns]

    t = _rl_table(cf_df, highlight_col="c factor pop",
                  col_widths=[3*cm, 3.2*cm, 3.2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
    if t:
        elems += [t, sp(0.06)]

    # Corrélation Spearman pop vs sample
    sub_cf = pop[["c_factor_sample", "c_factor_pop"]].dropna()
    if len(sub_cf) >= 5:
        rho_cf, p_cf = spearmanr(sub_cf["c_factor_sample"], sub_cf["c_factor_pop"])
        elems += [
            warn(
                f"Spearman c_factor_sample x c_factor_pop sur N={len(sub_cf)} groupes : "
                f"rho = {rho_cf:.3f}, p = {p_cf:.4f}. "
                "La corrélation quasi-nulle indique que les deux métriques ordonnent "
                "les groupes de manière quasi-indépendante. Interpréter les résultats "
                "de c_factor_pop avec prudence."
            ),
            sp(0.06),
        ]

    elems.append(PageBreak())

    # =========================================================================
    # Sanity checks
    # =========================================================================
    elems += [h2("Sanity checks"), sp(0.04)]
    for line in sanity_lines:
        elems.append(small(f"• {line}"))
    elems += [sp(0.08), hr()]

    # =========================================================================
    # Section 2 — Corrélations globales
    # =========================================================================
    elems += [h1("2. Corrélations globales (group-level)"), sp(0.06)]

    # --- 2.1 Riedl ↔ Performance ---
    elems += [h2("2.1 Riedl ↔ Performance (Score final)"), sp(0.04)]
    elems += [
        small(f"(N groupes = {merged['group_id'].nunique()} | inchangé par rapport au rapport original)"),
        sp(0.04),
    ]
    b21 = compute_block_21(merged)
    if b21:
        t = _rl_table(_corr_display_df(b21, ""), font_size=7,
                      col_widths=[4.5*cm, 4*cm, 2*cm, 2.5*cm, 1.5*cm, 1.5*cm])
        if t:
            elems += [t, sp(0.04)]
    elems += [small("Note : cette section ne contient pas de c-factor — bit-identique au rapport original."), sp(0.08), hr()]

    # --- 2.2 TCI ↔ Performance ---
    elems += [h2("2.2 TCI ↔ Performance (Score final)"), sp(0.04)]
    elems += [small(f"(N groupes = {merged['group_id'].nunique()})"), sp(0.04)]

    for variant in CFACTOR_VARIANTS:
        if variant not in merged.columns:
            continue
        b22 = compute_block_22(merged, variant)
        elems += [h3(f"Variante : {variant}"), sp(0.03)]
        if b22:
            t = _rl_table(_corr_display_df(b22, variant), font_size=7,
                          col_widths=[4*cm, 4*cm, 2*cm, 2.5*cm, 1.5*cm, 1.5*cm])
            if t:
                elems += [t, sp(0.04)]
    elems += [sp(0.04), hr()]

    # --- 2.3 Riedl ↔ TCI ---
    elems += [h2("2.3 Riedl ↔ TCI (tous groupes)"), sp(0.04)]
    elems += [small(f"(N groupes = {merged['group_id'].nunique()})"), sp(0.04)]

    for variant in CFACTOR_VARIANTS:
        if variant not in merged.columns:
            continue
        b23 = compute_block_23(merged, variant)
        elems += [h3(f"Variante : {variant}"), sp(0.03)]
        if b23:
            t = _rl_table(_corr_display_df(b23, variant), font_size=7,
                          col_widths=[4.5*cm, 3.5*cm, 2*cm, 2.5*cm, 1.5*cm, 1.5*cm])
            if t:
                elems += [t, sp(0.04)]
    elems += [sp(0.04), hr()]

    # --- 2.4 Riedl ↔ Questionnaire ---
    elems += [h2("2.4 Riedl ↔ Questionnaire"), sp(0.04)]
    elems += [small(f"(N groupes = {merged['group_id'].nunique()} | inchangé par rapport au rapport original)"), sp(0.04)]
    b24 = compute_block_24(merged)
    if b24:
        t = _rl_table(_corr_display_df(b24, ""), font_size=7,
                      col_widths=[4.5*cm, 4*cm, 2*cm, 2.5*cm, 1.5*cm, 1.5*cm])
        if t:
            elems += [t, sp(0.04)]
    elems += [small("Note : cette section ne contient pas de c-factor — bit-identique au rapport original."), sp(0.08)]

    elems.append(PageBreak())

    # =========================================================================
    # Section 3 — INV
    # =========================================================================
    elems += [
        h1("3. Indices non verbaux (INV)"), sp(0.06),
        body(
            "Dans la version v2, les analyses INV sont externalisées dans rapport_INV_VR.pdf. "
            "Cette section ne contient pas de c-factor et n'est pas reproduite ici."
        ),
        sp(0.08), hr(),
    ]

    # =========================================================================
    # Section 4 — Corrélations les plus fortes
    # =========================================================================
    elems += [h1("4. Corrélations les plus fortes"), sp(0.06)]
    elems += [
        body(
            "Les tableaux 4.0 et 4.1 sont produits pour les deux variantes c-factor. "
            "Colonne cfactor_variant : 'sample', 'pop' ou 'none' (paires sans c-factor)."
        ),
        sp(0.06),
    ]

    # Recalculer toutes les corrélations
    all_corrs_by_variant: dict[str, list[dict]] = {}
    for variant in CFACTOR_VARIANTS:
        b22 = compute_block_22(merged, variant)
        b23 = compute_block_23(merged, variant)
        all_corrs_by_variant[variant] = compute_block_21(merged) + b22 + b23 + compute_block_24(merged)

    def cfactor_tag(x, y):
        if x in CFACTOR_VARIANTS: return x.replace("c_factor_", "")
        if y in CFACTOR_VARIANTS: return y.replace("c_factor_", "")
        return "none"

    # Fusionner + dédupliquer paires sans c-factor
    all_combined: list[dict] = []
    seen_none: set = set()
    for variant, corrs in all_corrs_by_variant.items():
        for r in corrs:
            rc = dict(r)
            rc["cfactor_variant"] = cfactor_tag(rc.get("x",""), rc.get("y",""))
            if rc["cfactor_variant"] == "none":
                key = (rc["block"], rc["x"], rc["y"])
                if key in seen_none:
                    continue
                seen_none.add(key)
            all_combined.append(rc)

    strong = [r for r in all_combined if abs(r.get("rho", 0)) >= 0.60]
    strong_sorted = sorted(strong, key=lambda x: -abs(x.get("rho", 0)))

    # 4.0
    elems += [h2("4.0 Corrélations les plus fortes (|rho| >= 0.60)"), sp(0.04)]
    if strong_sorted:
        df40 = pd.DataFrame([{
            "bloc": r["block"].replace("(tous groupes)", "").strip(),
            "x": r["x"], "y": r["y"],
            "rho": f"{r['rho']:+.3f}",
            "p": f"{r['p']:.4f}",
            "n": r["n"],
            "variante": r["cfactor_variant"],
        } for r in strong_sorted])
        t = _rl_table(df40, highlight_col="variante", font_size=7,
                      col_widths=[4.2*cm, 3.8*cm, 3.8*cm, 1.8*cm, 2*cm, 1*cm, 1.8*cm])
        if t:
            elems += [t, sp(0.06)]
    else:
        elems += [body("_(aucune corrélation |rho| >= 0.60)_"), sp(0.06)]

    # 4.1
    supp = [r for r in all_combined if 0.50 <= abs(r.get("rho", 0)) < 0.60 and r.get("p", 1) <= 0.05]
    elems += [h2("4.1 Corrélations significatives supplémentaires (0.50 ≤ |rho| < 0.60, p ≤ 0.05)"), sp(0.04)]
    if supp:
        df41 = pd.DataFrame([{
            "bloc": r["block"].replace("(tous groupes)", "").strip(),
            "x": r["x"], "y": r["y"],
            "rho": f"{r['rho']:+.3f}",
            "p": f"{r['p']:.4f}",
            "n": r["n"],
            "variante": r["cfactor_variant"],
        } for r in sorted(supp, key=lambda x: -abs(x.get("rho",0)))])
        t = _rl_table(df41, highlight_col="variante", font_size=7,
                      col_widths=[4.2*cm, 3.8*cm, 3.8*cm, 1.8*cm, 2*cm, 1*cm, 1.8*cm])
        if t:
            elems += [t, sp(0.06)]
    else:
        elems += [body("_(aucune corrélation significative supplémentaire)_"), sp(0.06)]

    # 4.2 — Figures réseau
    elems += [PageBreak(), h2("4.2 Réseau des corrélations fortes"), sp(0.06)]
    elems += [
        body(
            f"Seuil : |rho| >= {RHO_THRESHOLD}, p <= {P_THRESHOLD}. "
            "Arêtes rouges = corrélation positive, bleues pointillés = négative. "
            "Taille des nœuds proportionnelle au degré pondéré."
        ),
        sp(0.06),
    ]

    for variant in CFACTOR_VARIANTS:
        elems += [h3(f"Réseau — {variant}"), sp(0.04)]
        net_png = FIGURES_DIR / f"global_correlation_network_{variant}.png"
        fam_png = FIGURES_DIR / f"global_correlation_family_network_{variant}.png"
        if net_png.exists():
            elems.append(Image(str(net_png), width=W, height=W * 0.71))
            elems += [
                small(f"Réseau des corrélations fortes — {variant} (|rho|>={RHO_THRESHOLD}, p<={P_THRESHOLD})"),
                sp(0.06),
            ]
        if fam_png.exists():
            elems.append(Image(str(fam_png), width=W * 0.75, height=W * 0.6))
            elems += [small(f"Réseau agrégé par famille — {variant}"), sp(0.08)]

    # 4.3 — Analyse structurelle comparée
    elems += [PageBreak(), h2("4.3 Analyse structurelle comparée"), sp(0.06)]

    # Tableau comparatif global
    if network_data:
        import networkx as nx
        net_rows = []
        for variant in CFACTOR_VARIANTS:
            data = network_data.get(variant)
            if data is None:
                net_rows.append({"version": variant, "nœuds": 0, "arêtes": 0,
                                  "densité": "N/A", "composantes": "N/A",
                                  "clust. moy.": "N/A", "communautés": "N/A"})
                continue
            G = data["graph"]
            metrics = compute_network_metrics(data)
            gi = metrics.get("global_info", {})
            net_rows.append({
                "version": variant,
                "nœuds": gi.get("n_nodes", G.number_of_nodes()),
                "arêtes": gi.get("n_edges", G.number_of_edges()),
                "densité": round(float(gi.get("density", 0)), 3),
                "composantes": gi.get("n_components", 0),
                "clust. moy.": round(float(metrics.get("avg_clustering", 0)), 3),
                "communautés": len(metrics.get("communities", [])),
            })
        t = _rl_table(pd.DataFrame(net_rows), font_size=8,
                      col_widths=[4.5*cm, 2*cm, 2*cm, 2.2*cm, 3*cm, 2.5*cm, 2.8*cm])
        if t:
            elems += [t, sp(0.06)]

    # Centralités par variante
    for variant in CFACTOR_VARIANTS:
        data = network_data.get(variant)
        if data is None:
            continue
        metrics = compute_network_metrics(data)
        nf = data["node_families"]
        elems += [h3(f"Centralités — {variant}"), sp(0.04)]

        for cname, col_label in [
            ("degree_centrality", "degree"),
            ("eigenvector_centrality", "eigenvector"),
            ("betweenness_centrality", "betweenness"),
        ]:
            cent = metrics.get(cname, {})
            if not cent:
                continue
            top = sorted(cent.items(), key=lambda x: -x[1])[:8]
            df_c = pd.DataFrame([
                {"variable": n, "famille": nf.get(n, "?"), col_label: round(v, 4)}
                for n, v in top
            ])
            elems += [small(f"Top {col_label} centrality"), sp(0.02)]
            t = _rl_table(df_c, font_size=7, col_widths=[5.5*cm, 3*cm, 3*cm])
            if t:
                elems += [t, sp(0.04)]

        # Distance au nœud central
        G = data["graph"]
        if variant in G.nodes():
            try:
                lengths = nx.single_source_shortest_path_length(G, variant)
                dist_df = pd.DataFrame([
                    {"variable": n, "famille": nf.get(n, "?"), f"dist. à {variant[-6:]}": d}
                    for n, d in sorted(lengths.items(), key=lambda x: x[1])
                ])
                elems += [small(f"Distance topologique depuis {variant}"), sp(0.02)]
                t = _rl_table(dist_df, font_size=7, col_widths=[5.5*cm, 3*cm, 4*cm])
                if t:
                    elems += [t, sp(0.04)]
            except Exception:
                pass

        # Communautés
        communities = metrics.get("communities", [])
        if communities:
            comm_rows = []
            for i, comm in enumerate(communities, 1):
                fc: dict[str, int] = {}
                for n in comm:
                    f = nf.get(n, "other")
                    fc[f] = fc.get(f, 0) + 1
                comm_rows.append({
                    "comm.": i,
                    "n": len(comm),
                    "variables principales": ", ".join(sorted(comm)[:5]),
                    "familles": ", ".join(f"{f}:{c}" for f, c in sorted(fc.items())),
                })
            elems += [small("Communautés détectées (greedy modularity)"), sp(0.02)]
            t = _rl_table(pd.DataFrame(comm_rows), font_size=7,
                          col_widths=[1.5*cm, 1*cm, 6*cm, 5.5*cm])
            if t:
                elems += [t, sp(0.04)]
        elems += [hr(), sp(0.04)]

    elems.append(PageBreak())

    # =========================================================================
    # Section 5 — Diff table
    # =========================================================================
    elems += [h1("5. Tableau de différences : c_factor_sample vs c_factor_pop"), sp(0.06)]
    elems += [
        body(
            "Cette section répond directement à la question empirique principale : "
            "la substitution de c_factor_sample par c_factor_pop change-t-elle "
            "significativement les résultats ? "
            "Colonnes : rho_sample, p_sample, rho_pop, p_pop, "
            "delta_rho (= rho_pop - rho_sample), sign_change (bascule p <= 0.05)."
        ),
        sp(0.06),
    ]

    # 5.1
    elems += [h2("5.1 Tableau complet des différences"), sp(0.04)]
    if not diff_df.empty:
        df51 = diff_df.copy()
        for col in ["rho_sample", "p_sample", "rho_pop", "p_pop", "delta_rho"]:
            if col in df51.columns:
                df51[col] = df51[col].round(3)
        t = _rl_table(df51, highlight_col="delta_rho", font_size=7,
                      col_widths=[3.5*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2.3*cm, 2*cm, 1*cm])
        if t:
            elems += [t, sp(0.06)]
    else:
        elems += [body("_(données insuffisantes)_"), sp(0.06)]

    # 5.2 Bascules
    elems += [h2("5.2 Bascules de significativité"), sp(0.04)]
    switches_df = compute_significance_switches(diff_df)
    if not switches_df.empty:
        t = _rl_table(switches_df, font_size=7)
        if t:
            elems += [t, sp(0.06)]
    else:
        elems += [body("Aucune bascule de significativité entre les deux variantes."), sp(0.06)]

    # 5.3 Métriques réseau comparées
    if "net_metrics_df" in dir():
        pass  # déjà dans 4.3
    # (déjà produit dans la section 4.3)

    # 5.4 Top 5 hubs comparés
    elems += [h2("5.4 Top 5 hubs par version"), sp(0.04)]
    hub_rows = []
    for variant in CFACTOR_VARIANTS:
        data = network_data.get(variant)
        if data is None:
            hub_rows.append({"version": variant, "top5 degree": "N/A", "top5 eigenvector": "N/A"})
            continue
        metrics = compute_network_metrics(data)
        deg = metrics.get("degree_centrality", {})
        eig = metrics.get("eigenvector_centrality", {})
        top5_deg = [n for n, _ in sorted(deg.items(), key=lambda x: -x[1])[:5]]
        top5_eig = [n for n, _ in sorted(eig.items(), key=lambda x: -x[1])[:5]] if eig else []
        hub_rows.append({
            "version": variant,
            "top5 degree": ", ".join(top5_deg),
            "top5 eigenvector": ", ".join(top5_eig),
        })
    t = _rl_table(pd.DataFrame(hub_rows), font_size=8,
                  col_widths=[4.5*cm, 6.5*cm, 6*cm])
    if t:
        elems += [t, sp(0.06)]

    # Note méthodologique
    elems += [
        hr(), sp(0.04),
        small(
            "Note méthodologique : c_factor_pop est projeté sur les loadings PC1 de la population "
            "parente Riedl et al. (2021), sans re-fit local. c_factor_sample est fité sur l'échantillon "
            "courant (N=12 groupes allowed). La divergence (rho ≈ -0.014 sur allowed) reflète la "
            "différence de structure des tâches entre l'échantillon local et la population parente. "
            "Échelle Likert 9 points."
        ),
    ]

    # =========================================================================
    # Build
    # =========================================================================
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Rapport comparatif C-factor",
    )
    doc.build(elems)
    print(f"  [OK] PDF natif ReportLab : {pdf_path.name}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Chargement des données ===")
    merged, pop = load_data()
    merged = add_cohesion_global(merged)

    print("\n=== Sanity checks ===")
    sanity_lines = sanity_log(merged, pop)
    for l in sanity_lines:
        print(f"  {l}")

    # Vérification critique : c_factor_pop manquant > 20 % parmi les groupes avec c_score
    if "c_factor_pop" not in merged.columns:
        print("\n[BLOCAGE] colonne c_factor_pop introuvable après merge. Arrêt.")
        sys.exit(1)
    # Limiter la vérification aux groupes qui ont un c_score (groupes allowed)
    has_cscore = merged["c_score"].notna() if "c_score" in merged.columns else merged["c_factor_sample"].notna()
    n_allowed = has_cscore.sum()
    n_missing_among_allowed = merged.loc[has_cscore, "c_factor_pop"].isna().sum()
    pct_among_allowed = n_missing_among_allowed / max(n_allowed, 1) * 100
    if pct_among_allowed > 20:
        print(f"\n[BLOCAGE] c_factor_pop manquant pour {pct_among_allowed:.1f}% des groupes allowed (> 20%). Arrêt.")
        sys.exit(1)
    print(f"  [OK] c_factor_pop présent pour {int(n_allowed - n_missing_among_allowed)}/{int(n_allowed)} groupes allowed.")

    # Vérification cohérence Spearman VR
    if "modalite" in merged.columns:
        vr_df = merged[merged["modalite"].str.upper() == "VR"]
        if all(c in vr_df.columns for c in ["c_factor_sample", "c_factor_pop"]):
            sub = vr_df[["c_factor_sample", "c_factor_pop"]].dropna()
            if len(sub) >= 5:
                rho_vr, p_vr = spearmanr(sub["c_factor_sample"], sub["c_factor_pop"])
                print(f"  Spearman sample×pop sur VR uniquement : rho={rho_vr:.3f}, p={p_vr:.4f}, n={len(sub)}")
                if abs(rho_vr) < 0.1 and len(sub) >= 8:
                    print("  [AVERTISSEMENT] Corrélation VR très faible — interpréter c_factor_pop avec précaution.")

    print("\n=== Export CSV support ===")
    export_csv_support(merged)
    diff_df = compute_diff_table(merged)
    if not diff_df.empty:
        diff_df.to_csv(TABLES_DIR / "diff_table_cfactor_sample_vs_pop.csv", index=False, encoding="utf-8")
        print(f"  diff_table : {len(diff_df)} paires")

    print("\n=== Calcul des réseaux (nécessaire pour PDF + Markdown) ===")
    all_corrs_by_variant_main: dict[str, list[dict]] = {}
    network_data_main: dict[str, dict | None] = {}
    for variant in CFACTOR_VARIANTS:
        b22 = compute_block_22(merged, variant)
        b23 = compute_block_23(merged, variant)
        all_corrs_by_variant_main[variant] = compute_block_21(merged) + b22 + b23 + compute_block_24(merged)
        data = build_and_plot_network(all_corrs_by_variant_main[variant], variant, FIGURES_DIR, variant)
        network_data_main[variant] = data

    diff_df_main = compute_diff_table(merged)
    if not diff_df_main.empty:
        diff_df_main.to_csv(TABLES_DIR / "diff_table_cfactor_sample_vs_pop.csv", index=False, encoding="utf-8")

    print("\n=== Génération du rapport Markdown ===")
    md_content = generate_markdown(merged, sanity_lines,
                                   precomputed_network_data=network_data_main)
    md_path = OUT_DIR / "rapport_principal_PC_VR_cfactor_comparison.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] {md_path.name} ({len(md_content)} chars)")

    print("\n=== Génération du PDF natif ReportLab ===")
    pdf_path = OUT_DIR / "rapport_principal_PC_VR_cfactor_comparison.pdf"
    generate_pdf_native(pdf_path, pop, merged, sanity_lines, diff_df_main, network_data_main)

    print("\n=== Résumé des sorties ===")
    for f in sorted(OUT_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT_DIR)}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
