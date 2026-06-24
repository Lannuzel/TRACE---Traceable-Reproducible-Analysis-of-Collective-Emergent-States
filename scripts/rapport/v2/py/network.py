# -*- coding: utf-8 -*-
"""
Helpers pour la visualisation et l'analyse structurelle du réseau global
de corrélations fortes du rapport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.platypus import Paragraph, Spacer

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


_FAMILY_RULES: list[tuple[str, list[str]]] = [
    ("performance",  ["score_perf", "score_final", "perf_tsk", "performance"]),
    ("tci",          ["c_score", "rme", "c_factor"]),
    ("riedl",        ["skill", "effort", "strategy", "congruance", "congruence", "contribution"]),
    ("questionnaire",["cre", "soc", "spe", "tsk", "cor", "com"]),
    ("audio",        ["audio_", "mean_turn", "turn_", "pause_", "overlap_", "speaking_"]),
    ("gaze",         ["gaze", "shared_obj", "visual_attention", "mutual_gaze"]),
    ("face",         ["face_", "smile_", "negative_affect",
                      "joy_", "sad_", "affect_", "pos_neg"]),
    ("high_level",   ["alignment", "synchrony", "sync_", "coordination"]),
]

_FAMILY_COLORS: dict[str, str] = {
    "performance":  "#e15759",
    "tci":          "#4e79a7",
    "riedl":        "#f28e2b",
    "questionnaire":"#59a14f",
    "audio":        "#b07aa1",
    "gaze":         "#76b7b2",
    "face":         "#ff9da7",
    "high_level":   "#9c755f",
    "other":        "#bab0ac",
}


def classify_variable(var: str) -> str:
    """Classe une variable dans une famille selon son nom."""
    v = var.lower()
    for family, patterns in _FAMILY_RULES:
        if any(p.lower() in v for p in patterns):
            return family
    return "other"


def build_global_correlation_graph(edges_df: pd.DataFrame):
    """
    Construit le graphe réseau à partir d'un tableau d'arêtes filtrées.
    """
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        weight = float(abs(row["rho"]))
        G.add_edge(
            row["x"],
            row["y"],
            rho=float(row["rho"]),
            weight=weight,
            distance=(1.0 / max(weight, 1e-6)),
            sign=row["edge_sign"],
        )

    node_families = {n: classify_variable(n) for n in G.nodes()}
    nx.set_node_attributes(G, node_families, "family")

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    nodes_df = pd.DataFrame([
        {
            "node": n,
            "family": node_families[n],
            "degree": degree[n],
            "weighted_degree": round(float(weighted_degree[n]), 3),
        }
        for n in G.nodes()
    ]).sort_values("weighted_degree", ascending=False).reset_index(drop=True)

    return G, node_families, nodes_df


def prepare_global_correlation_network_data(
    corr_list: list[dict],
    rho_threshold: float = 0.55,
    p_threshold: float = 0.05,
    min_n: int = 6,
) -> dict[str, object] | None:
    """
    Prépare les données du réseau global à partir de la liste de corrélations.
    """
    if not HAS_NX or not corr_list:
        return None

    df = pd.DataFrame(corr_list).copy()
    if not {"x", "y", "rho", "p"}.issubset(df.columns):
        return None

    pcol = "p_fdr" if "p_fdr" in df.columns and df["p_fdr"].notna().any() else "p"
    mask = (df["rho"].abs() >= rho_threshold) & (df[pcol] <= p_threshold)
    if "n" in df.columns:
        mask &= df["n"] >= min_n
    edges_df = df[mask].copy().reset_index(drop=True)
    if edges_df.empty:
        return None

    edges_df["family_x"] = edges_df["x"].apply(classify_variable)
    edges_df["family_y"] = edges_df["y"].apply(classify_variable)
    edges_df["edge_sign"] = edges_df["rho"].apply(lambda r: "positive" if r >= 0 else "negative")
    edges_df["edge_weight"] = edges_df["rho"].abs().round(3)

    G, node_families, nodes_df = build_global_correlation_graph(edges_df)
    return {
        "graph": G,
        "edges_df": edges_df,
        "nodes_df": nodes_df,
        "node_families": node_families,
        "rho_threshold": rho_threshold,
        "p_threshold": p_threshold,
        "min_n": min_n,
        "pcol": pcol,
    }


def plot_global_correlation_network(
    corr_list: list[dict],
    out_dir: Path,
    rho_threshold: float = 0.55,
    p_threshold: float = 0.05,
    min_n: int = 6,
    network_data: dict[str, object] | None = None,
) -> Path | None:
    """
    Génère un réseau de corrélations globales fortes entre variables de
    différentes familles théoriques.
    """
    if not HAS_NX:
        print("  [SKIP] networkx non disponible")
        return None

    if network_data is None:
        network_data = prepare_global_correlation_network_data(
            corr_list,
            rho_threshold=rho_threshold,
            p_threshold=p_threshold,
            min_n=min_n,
        )

    if network_data is None:
        print("  [SKIP] Aucune corrélation disponible")
        return None

    edges_df: pd.DataFrame = network_data["edges_df"]  # type: ignore[assignment]
    G = network_data["graph"]
    node_families = network_data["node_families"]
    nodes_df: pd.DataFrame = network_data["nodes_df"]  # type: ignore[assignment]

    print(f"  {len(edges_df)} aretes retenues (|rho|>={rho_threshold})")

    keep_cols = [c for c in ["x", "y", "rho", "p", "p_fdr", "n",
                              "family_x", "family_y", "edge_sign", "edge_weight", "block"]
                 if c in edges_df.columns]
    edges_df[keep_cols].to_csv(out_dir / "global_correlation_network_edges.csv", index=False)
    nodes_df.to_csv(out_dir / "global_correlation_network_nodes.csv", index=False)
    weighted_degree = {r["node"]: float(r["weighted_degree"]) for _, r in nodes_df.iterrows()}

    print("\n  Top variables les plus connectees (weighted degree) :")
    for _, r in nodes_df.head(8).iterrows():
        print(f"    {r['node']:<50} [{r['family']}] deg={r['degree']}  w={r['weighted_degree']:.2f}")

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
                           node_size=[300 + weighted_degree[n] * 120 for n in G.nodes()],
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
        Line2D([0], [0], color="#e15759", linewidth=2, label="Correlation positive"),
        Line2D([0], [0], color="#4e79a7", linewidth=2, linestyle="dashed",
               label="Correlation negative"),
    ]
    ax.legend(handles=family_handles + edge_handles,
              loc="lower left", fontsize=8, framealpha=0.85,
              title="Familles de variables", title_fontsize=8)

    ax.set_title(
        f"Reseau des correlations fortes (|rho| >= {rho_threshold}, p <= {p_threshold})\n"
        f"{len(G.nodes())} variables, {len(G.edges())} associations",
        fontsize=11, fontweight="bold", pad=10
    )

    plt.tight_layout()
    png_path = out_dir / "global_correlation_network.png"
    pdf_path = out_dir / "global_correlation_network.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    try:
        plt.savefig(pdf_path, bbox_inches="tight")
    except Exception:
        pass
    plt.close()
    print(f"  [OK] global_correlation_network.png ({len(G.nodes())} noeuds, {len(G.edges())} aretes)")

    _plot_family_network(edges_df, out_dir)
    return png_path


def _plot_family_network(edges_df: pd.DataFrame, out_dir: Path):
    """Réseau agrégé au niveau des familles de variables."""
    if not HAS_NX:
        return

    family_edges: dict[tuple, list[float]] = {}
    for _, row in edges_df.iterrows():
        fx, fy = row["family_x"], row["family_y"]
        if fx == fy:
            continue
        key = tuple(sorted([fx, fy]))
        family_edges.setdefault(key, []).append(abs(row["rho"]))

    if not family_edges:
        return

    fam_edge_rows = []
    G2 = nx.Graph()
    for (fa, fb), rhos in family_edges.items():
        n_links = len(rhos)
        mean_rho = round(float(np.mean(rhos)), 3)
        G2.add_edge(fa, fb, weight=n_links, mean_rho=mean_rho)
        fam_edge_rows.append({"family_x": fa, "family_y": fb,
                               "n_associations": n_links, "mean_abs_rho": mean_rho})

    pd.DataFrame(fam_edge_rows).sort_values(
        "n_associations", ascending=False
    ).to_csv(out_dir / "global_correlation_family_edges.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_axis_off()

    pos2 = nx.spring_layout(G2, seed=42, k=3.0)
    node_colors2 = [_FAMILY_COLORS.get(n, "#bab0ac") for n in G2.nodes()]
    edge_widths2 = [G2[u][v]["weight"] * 0.8 for u, v in G2.edges()]

    nx.draw_networkx_nodes(G2, pos2, node_color=node_colors2,
                           node_size=1800, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(G2, pos2, width=edge_widths2, alpha=0.65,
                           edge_color="#555555", ax=ax)
    nx.draw_networkx_labels(G2, pos2, font_size=10, font_weight="bold", ax=ax)

    edge_labels2 = {(u, v): f"n={G2[u][v]['weight']}\nrho={G2[u][v]['mean_rho']:.2f}"
                    for u, v in G2.edges()}
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2,
                                 font_size=7, ax=ax)

    ax.set_title(
        "Reseau agrege par famille de variables\n"
        "(epaisseur = nombre d'associations fortes, annotation = rho moyen)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "global_correlation_family_network.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  [OK] global_correlation_family_network.png")


def compute_network_metrics(network_data: dict[str, object], top_n: int = 10) -> dict[str, object]:
    """
    Calcule un ensemble de métriques structurelles pour un réseau de corrélations.
    """
    G = network_data["graph"]
    edges_df: pd.DataFrame = network_data["edges_df"]  # type: ignore[assignment]
    node_families: dict[str, str] = network_data["node_families"]  # type: ignore[assignment]

    notes: list[str] = []
    components = list(nx.connected_components(G))
    largest_component_nodes = max(components, key=len) if components else set()
    G_lcc = G.subgraph(largest_component_nodes).copy() if largest_component_nodes else G.copy()

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G) if n_nodes >= 2 else 0.0
    n_components = len(components)
    largest_component_size = len(largest_component_nodes)

    pos_count = int((edges_df["edge_sign"] == "positive").sum()) if "edge_sign" in edges_df.columns else 0
    neg_count = int((edges_df["edge_sign"] == "negative").sum()) if "edge_sign" in edges_df.columns else 0
    pos_ratio = (pos_count / n_edges) if n_edges else np.nan
    neg_ratio = (neg_count / n_edges) if n_edges else np.nan

    degree_centrality = nx.degree_centrality(G) if n_nodes >= 2 else {n: 0.0 for n in G.nodes()}
    betweenness_centrality = (
        nx.betweenness_centrality(G, weight="distance", normalized=True)
        if n_nodes >= 3 and n_edges >= 1 else {n: 0.0 for n in G.nodes()}
    )
    closeness_centrality = (
        nx.closeness_centrality(G, distance="distance")
        if n_nodes >= 2 and n_edges >= 1 else {n: 0.0 for n in G.nodes()}
    )

    eigenvector_centrality: dict[str, float] = {}
    eigenvector_scope = "graph complet"
    if n_nodes >= 2 and n_edges >= 1:
        eig_graph = G if n_components <= 1 else G_lcc
        if n_components > 1:
            eigenvector_scope = f"plus grande composante connexe (n={eig_graph.number_of_nodes()})"
            notes.append(
                "Eigenvector centrality calculée sur la plus grande composante connexe "
                "car le réseau global n'est pas connexe."
            )
        try:
            eigenvector_centrality = nx.eigenvector_centrality(
                eig_graph,
                weight="weight",
                max_iter=1000,
            )
        except Exception as exc:
            notes.append(f"Eigenvector centrality non calculable ({exc}).")
            eigenvector_centrality = {}

    clustering_by_node = nx.clustering(G, weight="weight") if n_nodes >= 2 else {n: 0.0 for n in G.nodes()}
    avg_clustering = float(np.mean(list(clustering_by_node.values()))) if clustering_by_node else 0.0

    communities: list[set[str]] = []
    community_method = "greedy modularity (NetworkX)"
    if n_nodes >= 2 and n_edges >= 1:
        try:
            communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
        except Exception as exc:
            notes.append(f"Détection de communautés non calculable ({exc}).")
            communities = []

    bridge_rows: list[dict[str, object]] = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        own_family = node_families.get(node, "other")
        neighbor_families = {node_families.get(nb, "other") for nb in neighbors}
        external_families = {fam for fam in neighbor_families if fam != own_family}
        cross_family_edges = sum(1 for nb in neighbors if node_families.get(nb, "other") != own_family)
        cross_family_weight = sum(
            float(G[node][nb].get("weight", 0.0))
            for nb in neighbors
            if node_families.get(nb, "other") != own_family
        )
        bridge_rows.append({
            "node": node,
            "family": own_family,
            "neighbor_families": len(neighbor_families),
            "external_families": len(external_families),
            "cross_family_edges": cross_family_edges,
            "cross_family_weight": cross_family_weight,
        })

    c_score_distances: list[dict[str, object]] = []
    if "c_score" in G.nodes():
        try:
            dist_map = nx.single_source_shortest_path_length(G, "c_score")
            c_score_distances = [
                {"node": node, "family": node_families.get(node, "other"), "distance_to_c_score": dist}
                for node, dist in dist_map.items()
            ]
        except Exception as exc:
            notes.append(f"Distances à c_score non calculables ({exc}).")
    else:
        notes.append("Le nœud `c_score` est absent du réseau global.")

    return {
        "global_info": {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "n_components": n_components,
            "largest_component_size": largest_component_size,
            "largest_component_ratio": (largest_component_size / n_nodes) if n_nodes else np.nan,
            "positive_edges": pos_count,
            "negative_edges": neg_count,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
        },
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "eigenvector_centrality": eigenvector_centrality,
        "eigenvector_scope": eigenvector_scope,
        "closeness_centrality": closeness_centrality,
        "clustering_by_node": clustering_by_node,
        "avg_clustering": avg_clustering,
        "communities": communities,
        "community_method": community_method,
        "bridge_rows": bridge_rows,
        "c_score_distances": c_score_distances,
        "positive_edges_df": edges_df[edges_df["edge_sign"] == "positive"].copy().sort_values("edge_weight", ascending=False),
        "negative_edges_df": edges_df[edges_df["edge_sign"] == "negative"].copy().sort_values("edge_weight", ascending=False),
        "notes": notes,
        "top_n": top_n,
        "graph": G,
        "node_families": node_families,
    }


def build_network_metrics_tables(metrics: dict[str, object], top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Transforme les métriques réseau en tableaux prêts pour le rapport."""
    G = metrics["graph"]
    node_families: dict[str, str] = metrics["node_families"]  # type: ignore[assignment]
    global_info: dict[str, float] = metrics["global_info"]  # type: ignore[assignment]
    communities: list[set[str]] = metrics["communities"]  # type: ignore[assignment]
    positive_edges_df: pd.DataFrame = metrics["positive_edges_df"]  # type: ignore[assignment]
    negative_edges_df: pd.DataFrame = metrics["negative_edges_df"]  # type: ignore[assignment]

    tables: dict[str, pd.DataFrame] = {}

    tables["global_info"] = pd.DataFrame([
        {"Métrique": "Nombre de nœuds", "Valeur": int(global_info["n_nodes"])},
        {"Métrique": "Nombre d'arêtes", "Valeur": int(global_info["n_edges"])},
        {"Métrique": "Densité du réseau", "Valeur": round(float(global_info["density"]), 3)},
        {"Métrique": "Nombre de composantes connexes", "Valeur": int(global_info["n_components"])},
        {"Métrique": "Taille de la plus grande composante", "Valeur": int(global_info["largest_component_size"])},
        {
            "Métrique": "Part de la plus grande composante",
            "Valeur": round(float(global_info["largest_component_ratio"]) * 100, 1)
            if pd.notna(global_info["largest_component_ratio"]) else np.nan,
        },
        {"Métrique": "Arêtes positives", "Valeur": int(global_info["positive_edges"])},
        {"Métrique": "Arêtes négatives", "Valeur": int(global_info["negative_edges"])},
        {
            "Métrique": "Part d'arêtes positives (%)",
            "Valeur": round(float(global_info["positive_ratio"]) * 100, 1)
            if pd.notna(global_info["positive_ratio"]) else np.nan,
        },
        {
            "Métrique": "Part d'arêtes négatives (%)",
            "Valeur": round(float(global_info["negative_ratio"]) * 100, 1)
            if pd.notna(global_info["negative_ratio"]) else np.nan,
        },
    ])

    def _centrality_table(name: str, values: dict[str, float]) -> pd.DataFrame:
        if not values:
            return pd.DataFrame(columns=["node", "family", name])
        rows = [
            {"node": node, "family": node_families.get(node, "other"), name: round(float(score), 4)}
            for node, score in values.items()
        ]
        return pd.DataFrame(rows).sort_values(name, ascending=False).head(top_n).reset_index(drop=True)

    tables["degree"] = _centrality_table("degree_centrality", metrics["degree_centrality"])  # type: ignore[arg-type]
    tables["betweenness"] = _centrality_table("betweenness_centrality", metrics["betweenness_centrality"])  # type: ignore[arg-type]
    tables["eigenvector"] = _centrality_table("eigenvector_centrality", metrics["eigenvector_centrality"])  # type: ignore[arg-type]
    tables["closeness"] = _centrality_table("closeness_centrality", metrics["closeness_centrality"])  # type: ignore[arg-type]

    clustering_values: dict[str, float] = metrics["clustering_by_node"]  # type: ignore[assignment]
    if clustering_values:
        clustering_rows = [
            {"node": node, "family": node_families.get(node, "other"), "clustering_coefficient": round(float(score), 4)}
            for node, score in clustering_values.items()
        ]
        tables["clustering"] = pd.DataFrame(clustering_rows).sort_values(
            "clustering_coefficient",
            ascending=False,
        ).head(top_n).reset_index(drop=True)
    else:
        tables["clustering"] = pd.DataFrame(columns=["node", "family", "clustering_coefficient"])

    community_rows: list[dict[str, object]] = []
    weighted_degree = dict(G.degree(weight="weight"))
    for idx, community in enumerate(sorted(communities, key=len, reverse=True), start=1):
        sub_nodes = list(community)
        top_nodes = sorted(sub_nodes, key=lambda n: weighted_degree.get(n, 0.0), reverse=True)[:6]
        family_counts = (
            pd.Series([node_families.get(n, "other") for n in sub_nodes])
            .value_counts()
            .sort_values(ascending=False)
        )
        community_rows.append({
            "community": idx,
            "n_nodes": len(sub_nodes),
            "main_variables": ", ".join(top_nodes),
            "families": ", ".join(f"{fam}:{cnt}" for fam, cnt in family_counts.items()),
        })
    tables["communities"] = pd.DataFrame(community_rows)

    bridge_df = pd.DataFrame(metrics["bridge_rows"]).copy()  # type: ignore[arg-type]
    if not bridge_df.empty:
        bridge_df["cross_family_weight"] = bridge_df["cross_family_weight"].astype(float).round(3)
        tables["bridges"] = bridge_df.sort_values(
            ["external_families", "cross_family_edges", "cross_family_weight"],
            ascending=[False, False, False],
        ).head(top_n).reset_index(drop=True)
    else:
        tables["bridges"] = pd.DataFrame(
            columns=["node", "family", "neighbor_families", "external_families", "cross_family_edges", "cross_family_weight"]
        )

    c_score_df = pd.DataFrame(metrics["c_score_distances"]).copy()  # type: ignore[arg-type]
    if not c_score_df.empty:
        tables["c_score_distances"] = c_score_df.sort_values(
            ["distance_to_c_score", "family", "node"],
            ascending=[True, True, True],
        ).head(max(top_n, 15)).reset_index(drop=True)
    else:
        tables["c_score_distances"] = pd.DataFrame(columns=["node", "family", "distance_to_c_score"])

    def _edge_table(df_edges: pd.DataFrame) -> pd.DataFrame:
        if df_edges is None or df_edges.empty:
            return pd.DataFrame(columns=["x", "y", "rho", "family_x", "family_y", "block"])
        keep_cols = [c for c in ["x", "y", "rho", "family_x", "family_y", "block"] if c in df_edges.columns]
        d = df_edges[keep_cols].copy()
        d["rho_abs"] = df_edges["rho"].abs().round(3)
        d["rho"] = d["rho"].round(3)
        cols = ["x", "y", "rho", "rho_abs"] + [c for c in ["family_x", "family_y", "block"] if c in d.columns]
        return d[cols].head(top_n).reset_index(drop=True)

    tables["positive_edges"] = _edge_table(positive_edges_df)
    tables["negative_edges"] = _edge_table(negative_edges_df)
    return tables


def render_network_metrics_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    network_data: dict[str, object] | None,
    md_table_fn: Callable[[pd.DataFrame], str],
    pdf_table_fn: Callable[..., Any],
    fmt2_fn: Callable[[Any], str],
    inch_unit: float,
    section_num: str = "4.2",
    top_n: int = 10,
):
    """
    Rend une section de lecture structurelle du réseau global de corrélations.
    """
    title = f"{section_num} Analyse structurelle du réseau"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading3"]))

    if network_data is None:
        msg = "Analyse structurelle non disponible : réseau global absent ou trop pauvre pour être interprété."
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch_unit))
        return

    metrics = compute_network_metrics(network_data, top_n=top_n)
    tables = build_network_metrics_tables(metrics, top_n=top_n)
    global_info: dict[str, float] = metrics["global_info"]  # type: ignore[assignment]
    notes: list[str] = metrics["notes"]  # type: ignore[assignment]
    community_method = metrics["community_method"]
    eigenvector_scope = metrics["eigenvector_scope"]

    intro = (
        "Cette section décrit la structure interne du réseau de corrélations fortes afin d'identifier "
        "les variables centrales, les sous-systèmes cohérents et les variables pivot entre blocs théoriques. "
        "Dans le cadre du projet, ces résultats servent surtout à discuter la place potentielle du c-factor / TCI, "
        "des indicateurs Riedl, des états questionnaire et des INV comme médiateurs dynamiques dans une logique IMOI / TMS."
    )
    lines.append(intro + "\n\n")
    pdf_elems.append(Paragraph(intro, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch_unit))

    def _add_subsection(title_md: str, title_pdf: str, explanation: str, table: pd.DataFrame | None = None):
        lines.append(f"#### {title_md}\n")
        pdf_elems.append(Paragraph(title_pdf, styles["Heading4"]))
        lines.append(explanation + "\n\n")
        pdf_elems.append(Paragraph(explanation, styles["Normal"]))
        if table is not None and not table.empty:
            lines.append(md_table_fn(table))
            lines.append("\n")
            pdf_elems.append(pdf_table_fn(table))
        else:
            lines.append("_(vide ou non calculable)_\n\n")
            pdf_elems.append(Paragraph("(vide ou non calculable)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch_unit))

    _add_subsection(
        f"{section_num}.1 Informations globales sur le réseau",
        f"{section_num}.1 Informations globales sur le réseau",
        (
            "Ces indicateurs résument l'architecture générale du réseau : taille, densité, fragmentation en composantes "
            "et équilibre entre liens positifs et négatifs. Dans une lecture projet, un réseau dense et peu fragmenté "
            "suggère une organisation plus intégrée entre niveaux théoriques ; un réseau plus fragmenté suggère des blocs "
            "plus spécialisés ou plus périphériques."
        ),
        tables["global_info"],
    )

    _add_subsection(
        f"{section_num}.2 Centralités des nœuds",
        f"{section_num}.2 Centralités des nœuds",
        (
            "Les centralités aident à repérer les variables les plus structurantes. "
            "Le degree centrality repère les hubs locaux ; la betweenness centrality repère les variables médiatrices ou ponts ; "
            f"l'eigenvector centrality capture l'influence globale via des voisins eux-mêmes importants ({eigenvector_scope}) ; "
            "la closeness centrality repère les variables rapidement connectées au reste du réseau."
        ),
        None,
    )

    for label, table in [
        ("Top degree centrality", tables["degree"]),
        ("Top betweenness centrality", tables["betweenness"]),
        ("Top eigenvector centrality", tables["eigenvector"]),
        ("Top closeness centrality", tables["closeness"]),
    ]:
        lines.append(f"**{label}**\n\n")
        pdf_elems.append(Paragraph(label, styles["Normal"]))
        if table is not None and not table.empty:
            lines.append(md_table_fn(table))
            lines.append("\n")
            pdf_elems.append(pdf_table_fn(table))
        else:
            lines.append("_(non calculable)_\n\n")
            pdf_elems.append(Paragraph("(non calculable)", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.08 * inch_unit))

    _add_subsection(
        f"{section_num}.3 Cohésion locale / organisation locale",
        f"{section_num}.3 Cohésion locale / organisation locale",
        (
            f"Le coefficient de clustering moyen du réseau est de {fmt2_fn(metrics['avg_clustering'])}. "
            "Cette métrique indique dans quelle mesure les voisins d'une variable sont eux-mêmes interconnectés. "
            "Un clustering élevé est compatible avec des sous-systèmes cohérents, par exemple des paquets INV ou des blocs "
            "questionnaire / performance fortement couplés."
        ),
        tables["clustering"],
    )

    _add_subsection(
        f"{section_num}.4 Détection de communautés / clusters",
        f"{section_num}.4 Détection de communautés / clusters",
        (
            f"Les communautés ont été détectées via la méthode {community_method}. "
            "Elles correspondent à des sous-ensembles de variables plus fortement liées entre elles qu'au reste du réseau. "
            "Dans le cadre du projet, cela peut refléter des blocs fonctionnels autour de la CI, des processus Riedl, "
            "des médiateurs INV ou des états questionnaire."
        ),
        tables["communities"],
    )

    _add_subsection(
        f"{section_num}.5 Variables bridge / variables pivot",
        f"{section_num}.5 Variables bridge / variables pivot",
        (
            "Le tableau ci-dessous repère les nœuds reliés à plusieurs familles théoriques distinctes. "
            "Ces variables sont particulièrement intéressantes dans une lecture IMOI / TMS car elles peuvent jouer "
            "un rôle de médiation entre inputs fonctionnels, dynamiques interactionnelles et outputs émergents."
        ),
        tables["bridges"],
    )

    _add_subsection(
        f"{section_num}.6 Distance au nœud central théorique `c_score`",
        f"{section_num}.6 Distance au nœud central théorique `c_score`",
        (
            "Lorsque `c_score` est présent, la distance topologique permet de repérer les variables les plus proches du noyau CI. "
            "Cela aide à discuter quels indicateurs semblent structurellement proches du c-factor dans le réseau global."
        ),
        tables["c_score_distances"],
    )

    _add_subsection(
        f"{section_num}.7 Analyse du signe des relations",
        f"{section_num}.7 Analyse du signe des relations",
        (
            "Les arêtes positives suggèrent des co-variations convergentes entre variables, tandis que les arêtes négatives "
            "peuvent signaler des tensions, oppositions fonctionnelles ou trade-offs. "
            "Ces oppositions sont particulièrement utiles pour discuter des équilibres entre coordination, performance, "
            "charge interactionnelle et états émergents."
        ),
        None,
    )

    lines.append("**Principales arêtes positives**\n\n")
    pdf_elems.append(Paragraph("Principales arêtes positives", styles["Normal"]))
    if not tables["positive_edges"].empty:
        lines.append(md_table_fn(tables["positive_edges"]))
        lines.append("\n")
        pdf_elems.append(pdf_table_fn(tables["positive_edges"]))
    else:
        lines.append("_(aucune arête positive retenue)_\n\n")
        pdf_elems.append(Paragraph("(aucune arête positive retenue)", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch_unit))

    lines.append("**Principales arêtes négatives**\n\n")
    pdf_elems.append(Paragraph("Principales arêtes négatives", styles["Normal"]))
    if not tables["negative_edges"].empty:
        lines.append(md_table_fn(tables["negative_edges"]))
        lines.append("\n")
        pdf_elems.append(pdf_table_fn(tables["negative_edges"]))
    else:
        lines.append("_(aucune arête négative retenue)_\n\n")
        pdf_elems.append(Paragraph("(aucune arête négative retenue)", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch_unit))

    if notes:
        lines.append("**Notes de calcul**\n\n")
        pdf_elems.append(Paragraph("Notes de calcul", styles["Heading4"]))
        for note in notes:
            lines.append(f"- {note}\n")
            pdf_elems.append(Paragraph(f"- {note}", styles["Normal"]))
        lines.append("\n")
        pdf_elems.append(Spacer(1, 0.08 * inch_unit))

    top_degree_df = tables["degree"]
    top_bridge_df = tables["bridges"]
    top_comm_df = tables["communities"]
    synthesis_bits: list[str] = []
    if not top_degree_df.empty:
        synthesis_bits.append(f"un noyau central autour de `{top_degree_df.iloc[0]['node']}`")
    if not top_bridge_df.empty:
        synthesis_bits.append(f"des variables pivot telles que `{top_bridge_df.iloc[0]['node']}`")
    if not top_comm_df.empty:
        synthesis_bits.append(f"{len(top_comm_df)} communauté(s) détectée(s)")
    if int(global_info["n_components"]) > 1:
        synthesis_bits.append(f"un réseau fragmenté en {int(global_info['n_components'])} composantes")

    if synthesis_bits:
        summary = (
            "Synthèse prudente : le réseau met en évidence "
            + ", ".join(synthesis_bits)
            + ". "
            "À ce stade, ces résultats doivent rester descriptifs : ils suggèrent des proximités structurelles entre "
            "TCI / c-factor, variables Riedl, INV et questionnaires, sans établir de causalité."
        )
        lines.append("#### Synthèse\n\n")
        lines.append(summary + "\n\n")
        pdf_elems.append(Paragraph("Synthèse", styles["Heading4"]))
        pdf_elems.append(Paragraph(summary, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.12 * inch_unit))
