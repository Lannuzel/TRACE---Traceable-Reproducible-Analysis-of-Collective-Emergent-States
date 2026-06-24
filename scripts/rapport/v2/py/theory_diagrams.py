# -*- coding: utf-8 -*-
"""
theory_diagrams.py — Diagrammes théorie / empirique et modèle computationnel.

Cette version remplace l'ancien rendu "réseau libre" par un layout
semi-structuré en blocs :
    Input -> Mediators -> Outputs

Le nœud "CI / C-factor" fusionne :
    - l'idée de potentiel collectif initial,
    - sa formalisation via le c-factor / CI.

Il conserve un double rôle théorique :
    - vers les médiateurs,
    - et vers la performance.

La vue empirique superpose ensuite les liens soutenus par les corrélations
agrégées au niveau construit, ainsi que les liens latent-dimension validés
par alpha lorsque la cohérence interne est suffisante.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Protocol

import numpy as np
import pandas as pd

try:
    import networkx as nx

    HAS_NX = True
except ImportError:
    HAS_NX = False
    nx = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


class _MiniDiGraph:
    """Fallback minimaliste quand networkx n'est pas disponible.

    Le layout du diagramme n'utilise plus les algorithmes de networkx ; nous
    avons seulement besoin d'un conteneur orienté avec copie et accès aux
    métadonnées d'arêtes.
    """

    def __init__(self) -> None:
        self._nodes: list[str] = []
        self._adj: dict[str, dict[str, dict]] = {}

    def add_node(self, node: str) -> None:
        if node not in self._adj:
            self._adj[node] = {}
            self._nodes.append(node)

    def add_edge(self, src: str, dst: str, **attrs) -> None:
        self.add_node(src)
        self.add_node(dst)
        self._adj[src][dst] = dict(attrs)

    def has_edge(self, src: str, dst: str) -> bool:
        return src in self._adj and dst in self._adj[src]

    def copy(self) -> "_MiniDiGraph":
        other = _MiniDiGraph()
        other._nodes = list(self._nodes)
        other._adj = {
            src: {dst: dict(attrs) for dst, attrs in dst_map.items()}
            for src, dst_map in self._adj.items()
        }
        return other

    def edges(self, data: bool = False):
        rows = []
        for src in self._nodes:
            for dst, attrs in self._adj.get(src, {}).items():
                rows.append((src, dst, attrs) if data else (src, dst))
        return rows

    def nodes(self):
        return list(self._nodes)

    def __getitem__(self, key: str):
        return self._adj[key]


class GraphLike(Protocol):
    """Interface minimale commune a networkx.DiGraph et au fallback local."""

    def add_node(self, node: str) -> None: ...

    def add_edge(self, src: str, dst: str, **attrs: Any) -> None: ...

    def has_edge(self, src: str, dst: str) -> bool: ...

    def copy(self) -> "GraphLike": ...

    def edges(self, data: bool = False): ...

    def nodes(self): ...

    def __getitem__(self, key: str): ...


# ============================================================================
# SEUILS
# ============================================================================
P_THRESHOLD = 0.05
RHO_THRESHOLD = 0.30
LATENT_ALPHA_THRESHOLD = 0.70
EMPIRICAL_LABEL_LIMIT = 8


# ============================================================================
# MAPPING VARIABLE -> CONSTRUIT THEORIQUE
# ============================================================================
THEORETICAL_MAPPING: Dict[str, str] = {
    # Performance
    "Score_perf_tsk": "Performance",
    "Score_perf_tsk_z": "Performance",
    "Score_perf_tsk_mean": "Performance",

    # CI / c-factor (potentiel collectif initial)
    "c_score": "CI_Cfactor_Input",
    "rme_mean": "CI_Cfactor_Input",
    "rme_min": "CI_Cfactor_Input",
    "rme_max": "CI_Cfactor_Input",

    # Processus Riedl
    "skill_mean": "Skill_Congruence",
    "skill_max": "Skill_Congruence",
    "skill_congruence_mean": "Skill_Congruence",
    "strategy_ratio_mean": "Strategies",
    "strategy_norm": "Strategies",
    "effort_task_sum": "Effort",
    "effort_task_norm": "Effort",

    # Questionnaire
    "COR": "Coordination",
    "CRE": "Credibility",
    "SPE": "Specialization",
    "SOC": "Social_Cohesion",
    "TSK": "Task_Cohesion",
    "COM": "Communication",
    "Cohesion_questionnaire_score": "Cohesion",
    "COHESION_score": "Cohesion",

    # Audio INV
    "mean_turn_s": "Audio_INV",
    "pause_ratio": "Audio_INV",
    "overlap_ratio": "Audio_INV",
    "pairwise_overlap_ratio": "Audio_INV",
    "pairwise_overlap_s": "Audio_INV",
    "audio_avg_speaking_turn_duration_s": "Audio_INV",
    "audio_overlap_speaking_ratio": "Audio_INV",
    "audio_total_speaking_turns": "Audio_INV",
    "audio_backchannel_rate_per_min": "Audio_INV",
    "audio_floor_exchange_pause_mean_s": "Audio_INV",
    "floor_exchange_pause_mean_s": "Audio_INV",
    "n_floor_exchanges": "Audio_INV",
    "interruptions_rate_per_min": "Audio_INV",
    "successful_interruption_ratio": "Audio_INV",
    "audio_successful_interruption_ratio": "Audio_INV",
    "n_attempted_interruptions": "Audio_INV",
    "n_successful_interruptions": "Audio_INV",
    "rapid_floor_takeovers_total": "Audio_INV",

    # Face INV
    "joy_mean_mean": "Face_INV",
    "sad_mean_mean": "Face_INV",
    "face_negative_affect_ratio": "Face_INV",
    "face_facial_synchrony": "Face_INV",
    "affect_alignment_idx": "Face_INV",
    "face_smile_ratio": "Face_INV",

    # Gaze INV — nouvelles colonnes directionnelles
    "gaze_convergence_ratio": "Gaze_INV",
    "gaze_convergence_dur_total_s": "Gaze_INV",
    "gaze_convergence_mean_angle_deg": "Gaze_INV",
    "gaze_convergence_n_episodes": "Gaze_INV",
    "mutual_gaze_ratio": "Gaze_INV",
    "gaze_entropy_dir_mean": "Gaze_INV",
    "gaze_entropy": "Gaze_INV",
    "gaze_attention_coordination_idx": "Gaze_INV",
    "gaze_shared_visual_attention_ratio": "Gaze_INV",
    "gaze_mutual_gaze_ratio": "Gaze_INV",
    # Legacy (conservées si présentes)
    "shared_obj_ratio": "Gaze_INV",
    "shared_obj_dur_mean_s": "Gaze_INV",
    "gaze_entropy_mean_participants": "Gaze_INV",
    "gaze_speaker_coupling_idx": "Gaze_INV",

    # High-level TMS-oriented composites -> dimensions conceptuelles
}


# ============================================================================
# LIENS THEORIQUES
# ============================================================================
THEORETICAL_EDGES = [
    # Inputs -> potentiel collectif
    {"source": "Team_Composition", "target": "CI_Cfactor_Input", "weight": 3},
    {"source": "Context_Environment", "target": "CI_Cfactor_Input", "weight": 2},
    {"source": "Task_Characteristics", "target": "CI_Cfactor_Input", "weight": 2},

    # Potentiel collectif -> dynamiques observées et performance
    {"source": "CI_Cfactor_Input", "target": "Skill_Congruence", "weight": 2},
    {"source": "CI_Cfactor_Input", "target": "Strategies", "weight": 2},
    {"source": "CI_Cfactor_Input", "target": "Effort", "weight": 2},
    {"source": "CI_Cfactor_Input", "target": "Performance", "weight": 3},

    # Sous-systèmes transactifs -> dimensions observées
    # {"source": "TMS", "target": "Specialization", "weight": 3},
    # {"source": "TMS", "target": "Credibility", "weight": 3},
    # {"source": "TAS", "target": "Coordination", "weight": 3},
    # {"source": "TAS", "target": "Strategies", "weight": 3},
    # {"source": "TRS", "target": "Effort", "weight": 3},
    # {"source": "TRS", "target": "Cohesion", "weight": 3},

    # TMS questionnaire latent -> dimensions du TMS perçu
    {"source": "TMS", "target": "Specialization", "weight": 2},
    {"source": "TMS", "target": "Coordination", "weight": 2},
    {"source": "TMS", "target": "Credibility", "weight": 2},

    # Mesure du construit cohésion
    {"source": "Cohesion", "target": "Social_Cohesion", "weight": 3},
    {"source": "Cohesion", "target": "Task_Cohesion", "weight": 3},
    {"source": "Cohesion", "target": "Communication", "weight": 3},

    # Skill congruence comme nœud transversal
    # {"source": "Skill_Congruence", "target": "TMS", "weight": 2},
    # {"source": "Skill_Congruence", "target": "Performance", "weight": 2},

    # Mesures INV -> médiateurs
    # {"source": "Audio_INV", "target": "TMS", "weight": 2},
    # {"source": "Audio_INV", "target": "TAS", "weight": 3},
    # {"source": "Audio_INV", "target": "TRS", "weight": 2},
    # {"source": "Face_INV", "target": "TRS", "weight": 3},
    # {"source": "Gaze_INV", "target": "TMS", "weight": 2},
    # {"source": "Gaze_INV", "target": "TAS", "weight": 3},

    # Dimensions -> performance
    {"source": "Specialization", "target": "Performance", "weight": 2},
    {"source": "Credibility", "target": "Performance", "weight": 2},
    {"source": "Coordination", "target": "Performance", "weight": 3},
    {"source": "Strategies", "target": "Performance", "weight": 2},
    {"source": "Effort", "target": "Performance", "weight": 2},
    {"source": "Cohesion", "target": "Performance", "weight": 2},


]


# ============================================================================
# LIENS LATENT-DIMENSION VALIDES PAR ALPHA
# ============================================================================
LATENT_MEASUREMENT_EDGES: Dict[str, list[str]] = {
    "TMS": ["Coordination", "Credibility", "Specialization"],
    "Cohesion": ["Social_Cohesion", "Task_Cohesion", "Communication"],
}

# Certains nœuds servent encore à la logique interne/empirique, mais ne doivent
# plus être affichés comme blocs séparés dans la figure section 7.
HIDDEN_NODES = {"TMS_Q", "Cohesion"}


# ============================================================================
# GEOMETRIE DES BLOCS ET DES NOEUDS
# Géométries ancrées sur les fichiers draw.io de référence.
# ============================================================================
DRAWIO_MODEL_PAGE = (1169.0, 827.0)
DRAWIO_PIPELINE_PAGE = (1169.0, 827.0)


def _drawio_axes_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    page: tuple[float, float] = DRAWIO_MODEL_PAGE,
) -> dict[str, tuple[float, float]]:
    page_w, page_h = page
    return {
        "xy": (x / page_w, 1.0 - (y + h) / page_h),
        "wh": (w / page_w, h / page_h),
    }


def _drawio_block(
    x: float,
    y: float,
    w: float,
    h: float,
    header: str,
    page: tuple[float, float] = DRAWIO_MODEL_PAGE,
) -> dict[str, object]:
    spec = _drawio_axes_rect(x, y, w, h, page=page)
    spec["header"] = header
    return spec


def _drawio_node(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fc: str,
    ec: str,
    text_color: str = "#111827",
    fontsize: float = 10.0,
    page: tuple[float, float] = DRAWIO_MODEL_PAGE,
) -> dict[str, object]:
    rect = _drawio_axes_rect(x, y, w, h, page=page)
    rx, ry = rect["xy"]
    rw, rh = rect["wh"]
    return {
        "xy": (rx + rw / 2, ry + rh / 2),
        "wh": (rw, rh),
        "label": label,
        "fc": fc,
        "ec": ec,
        "text_color": text_color,
        "fontsize": fontsize,
    }


BLOCK_LAYOUT = {
    "input": _drawio_block(26, 165, 240, 155, "Input"),
    "mediators": _drawio_block(248, 72, 680, 360, "Mediators"),
    "output": _drawio_block(942, 217, 186, 69, "Outputs"),
    "measures": _drawio_block(320, 500, 481, 76, "Nonverbal cues / INV measures"),
}

NODE_LAYOUT = {
    "Team_Composition": _drawio_node(36, 196, 220, 24, "Team composition", "#27c5c3", "#0f172a", text_color="white", fontsize=9.8),
    "Context_Environment": _drawio_node(36, 232, 220, 24, "Context / Environment", "#27c5c3", "#0f172a", text_color="white", fontsize=9.7),
    "Task_Characteristics": _drawio_node(36, 268, 220, 24, "Task characteristics", "#27c5c3", "#0f172a", text_color="white", fontsize=9.7),
    "CI_Cfactor_Input": _drawio_node(92, 368, 92, 37, "CI / C-factor", "#c8d8f0", "#0f172a", fontsize=10.6),
    "TMS": _drawio_node(369, 178, 149, 42, "Transactive Memory\nSystem (TMS)", "#f6c7c4", "#d97777", fontsize=9.1),
    "TAS": _drawio_node(550, 177, 153, 44, "Transactive Attention\nSystem (TAS)", "#d7efc3", "#85b86a", fontsize=9.0),
    "TRS": _drawio_node(729, 177, 159, 44, "Transactive Reasoning\nSystem (TRS)", "#eadff5", "#9e7ac7", fontsize=9.0),
    "Skill_Congruence": _drawio_node(392, 274, 150, 38, "Skill congruence", "#cad4ff", "#748ffc", fontsize=9.6),
    "TMS_Q": _drawio_node(498, 274, 86, 36, "TMS (Q)", "#ffe6b8", "#d99a00", fontsize=9.0),
    "Specialization": _drawio_node(357, 356, 90, 37, "Specialization", "#fde6e5", "#d97777", fontsize=9.6),
    "Credibility": _drawio_node(461, 356, 90, 37, "Credibility", "#fde6e5", "#d97777", fontsize=9.6),
    "Coordination": _drawio_node(584.5, 353, 90.5, 40, "Coordination", "#e7f4da", "#85b86a", fontsize=9.6),
    "Strategies": _drawio_node(584.5, 275.5, 85, 35, "Strategies", "#e7f4da", "#85b86a", fontsize=9.5),
    "Effort": _drawio_node(760, 273, 90, 40, "Collective\neffort", "#f1e9f9", "#9e7ac7", fontsize=9.4),
    "Cohesion": _drawio_node(758.75, 353.5, 92.5, 40, "Cohesion", "#f1e9f9", "#9e7ac7", fontsize=9.6),
    "Social_Cohesion": _drawio_node(676, 354, 68, 36, "Social cohesion", "#eef7e7", "#85b86a", fontsize=7.8),
    "Task_Cohesion": _drawio_node(752, 354, 68, 36, "Task cohesion", "#eef7e7", "#85b86a", fontsize=7.8),
    "Communication": _drawio_node(828, 354, 68, 36, "Communication", "#eef7e7", "#85b86a", fontsize=7.8),
    "Audio_INV": _drawio_node(515, 530, 77.91, 35.55, "Audio INV", "#b8e0f2", "#3c8db7", fontsize=8.9),
    "Face_INV": _drawio_node(403, 530, 85.65, 36.44, "Face INV", "#ffd68a", "#d99a00", fontsize=8.9),
    "Gaze_INV": _drawio_node(628, 530, 74.00, 35.55, "Gaze INV", "#cfe7a6", "#6e9f35", fontsize=8.9),
    "Performance": _drawio_node(942, 247, 186, 39, "Objective / subjective\nresults\nPerformance", "#d8dee4", "#6b7280", fontsize=10.2),
}

EDGE_CURVATURE = {
    ("CI_Cfactor_Input", "Performance"): -0.22,
    ("CI_Cfactor_Input", "Skill_Congruence"): 0.10,
    ("CI_Cfactor_Input", "Strategies"): 0.10,
    ("CI_Cfactor_Input", "Effort"): -0.10,
    ("Skill_Congruence", "Performance"): 0.20,
    # ("Audio_INV", "TMS"): 0.16,
    # ("Audio_INV", "TAS"): 0.08,
    # ("Audio_INV", "TRS"): -0.16,
    # ("Face_INV", "TRS"): 0.10,
    # ("Gaze_INV", "TMS"): 0.18,
    # ("Gaze_INV", "TAS"): -0.12,
    ("TMS", "Specialization"): 0.06,
    ("TMS", "Coordination"): -0.02,
    ("TMS", "Credibility"): -0.08,
    ("Cohesion", "Social_Cohesion"): -0.04,
    ("Cohesion", "Task_Cohesion"): 0.04,
    ("Cohesion", "Communication"): -0.14,
}


# ============================================================================
# UTILITAIRES
# ============================================================================
def _edge_passes_threshold(
    rho: float,
    p: float,
    rho_threshold: float = RHO_THRESHOLD,
    p_threshold: float = P_THRESHOLD,
) -> bool:
    if pd.isna(rho) or pd.isna(p):
        return False
    return (abs(rho) >= rho_threshold) and (p <= p_threshold)


def _normalize_width(
    value: float,
    vmin: float = 0.30,
    vmax: float = 0.90,
    out_min: float = 1.6,
    out_max: float = 6.2,
) -> float:
    value = abs(value)
    value = max(vmin, min(vmax, value))
    ratio = (value - vmin) / (vmax - vmin + 1e-12)
    return out_min + ratio * (out_max - out_min)


def _normalize_alpha_width(
    alpha: float,
    vmin: float = LATENT_ALPHA_THRESHOLD,
    vmax: float = 0.95,
    out_min: float = 2.0,
    out_max: float = 5.2,
) -> float:
    alpha = max(vmin, min(vmax, alpha))
    ratio = (alpha - vmin) / (vmax - vmin + 1e-12)
    return out_min + ratio * (out_max - out_min)


def _node_center(node: str) -> tuple[float, float]:
    spec = NODE_LAYOUT[node]
    return spec["xy"]


def _aggregate_empirical(
    corr_rows: list[dict],
    mapping: Dict[str, str],
    rho_threshold: float = RHO_THRESHOLD,
    p_threshold: float = P_THRESHOLD,
    modality_filter: Iterable[str] | None = None,
) -> pd.DataFrame:
    rows = []
    allowed_modalities = None
    if modality_filter:
        allowed_modalities = {str(m).strip() for m in modality_filter if str(m).strip()}
    for rec in corr_rows:
        x, y = rec.get("x", ""), rec.get("y", "")
        rho = rec.get("rho", np.nan)
        p = rec.get("p", np.nan)
        if x not in mapping or y not in mapping:
            continue
        src, dst = mapping[x], mapping[y]
        if src == dst:
            continue
        if allowed_modalities and (
            "Audio_INV" in allowed_modalities
            or "Face_INV" in allowed_modalities
            or "Gaze_INV" in allowed_modalities
        ):
            if (src not in allowed_modalities) and (dst not in allowed_modalities):
                continue
        rows.append({"source": src, "target": dst, "rho": rho, "p": p, "x": x, "y": y})

    if not rows:
        return pd.DataFrame(columns=["source", "target", "rho", "p", "n_pairs", "is_kept", "detail"])

    df = pd.DataFrame(rows)
    df["edge_key"] = df.apply(lambda r: tuple(sorted([r["source"], r["target"]])), axis=1)
    df["source"] = df["edge_key"].apply(lambda t: t[0])
    df["target"] = df["edge_key"].apply(lambda t: t[1])
    df.drop(columns="edge_key", inplace=True)

    agg_rows = []
    for (src, dst), g in df.groupby(["source", "target"]):
        idx = g["rho"].abs().idxmax()
        best = g.loc[idx]
        detail = " | ".join(f"{r.x}~{r.y} (rho={r.rho:.2f})" for r in g.itertuples())
        agg_rows.append(
            {
                "source": src,
                "target": dst,
                "rho": float(best["rho"]),
                "p": float(best["p"]),
                "n_pairs": len(g),
                "is_kept": _edge_passes_threshold(
                    float(best["rho"]),
                    float(best["p"]),
                    rho_threshold=rho_threshold,
                    p_threshold=p_threshold,
                ),
                "detail": detail,
            }
        )

    return pd.DataFrame(agg_rows)


def _new_graph() -> GraphLike:
    if HAS_NX and nx is not None:
        return nx.DiGraph()
    return _MiniDiGraph()


def _build_theory_graph() -> GraphLike:
    G = _new_graph()
    for node in NODE_LAYOUT:
        G.add_node(node)
    for e in THEORETICAL_EDGES:
        G.add_edge(e["source"], e["target"], theory_weight=e["weight"])
    return G


def _add_empirical(G: GraphLike, df_agg: pd.DataFrame) -> GraphLike:
    H = G.copy()
    for _, row in df_agg.iterrows():
        src, dst = row["source"], row["target"]
        rho, p = float(row["rho"]), float(row["p"])
        kept = bool(row["is_kept"])

        if H.has_edge(src, dst):
            edge = H[src][dst]
        elif H.has_edge(dst, src):
            edge = H[dst][src]
        else:
            H.add_edge(
                src,
                dst,
                theory_weight=0,
                empirical_rho=rho,
                empirical_p=p,
                empirical_kept=kept,
                exploratory=True,
            )
            continue

        edge["empirical_rho"] = rho
        edge["empirical_p"] = p
        edge["empirical_kept"] = kept
    return H


def _normalize_construct_alphas(construct_alphas: dict[str, float] | None) -> dict[str, float]:
    if not construct_alphas:
        return {}
    alias = {
        "TMS": "TMS",
        "TMS_Q": "TMS",
        "TMSQ": "TMS",
        "COHESION": "Cohesion",
        "COHESION_Q": "Cohesion",
        "Cohesion_Q": "Cohesion",
        "Cohesion": "Cohesion",
    }
    normalized: dict[str, float] = {}
    for key, value in construct_alphas.items():
        canonical = alias.get(str(key).strip(), str(key).strip())
        normalized[canonical] = value
    return normalized


def _apply_latent_measurement_support(
    G: GraphLike,
    construct_alphas: dict[str, float] | None,
    alpha_threshold: float = LATENT_ALPHA_THRESHOLD,
) -> GraphLike:
    H = G.copy()
    normalized = _normalize_construct_alphas(construct_alphas)
    for latent, dims in LATENT_MEASUREMENT_EDGES.items():
        alpha = normalized.get(latent, np.nan)
        is_valid = not pd.isna(alpha) and float(alpha) >= alpha_threshold
        for dim in dims:
            if not H.has_edge(latent, dim):
                continue
            edge = H[latent][dim]
            edge["latent_alpha"] = float(alpha) if not pd.isna(alpha) else np.nan
            edge["latent_alpha_valid"] = is_valid
            edge["measurement_supported"] = is_valid
    return H


def _draw_block(ax, xy, wh, header: str, header_color: str = "#157a8a", body_color: str = "#f8fafc") -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=body_color, edgecolor="#0f172a", lw=1.6)
    )
    header_h = max(0.028, min(0.045, h * 0.18))
    ax.add_patch(
        Rectangle((x, y + h - header_h), w, header_h, transform=ax.transAxes, facecolor=header_color, edgecolor="#0f172a", lw=1.6)
    )
    ax.text(
        x + w / 2,
        y + h - header_h / 2,
        header,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        fontstyle="italic",
        fontweight="bold",
        color="white",
    )


def _draw_background(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _draw_block(ax, **BLOCK_LAYOUT["input"], header_color="#1ca6a3", body_color="#f7fbfb")
    _draw_block(ax, **BLOCK_LAYOUT["mediators"], header_color="#1f7aa8", body_color="#fbfbfc")
    _draw_block(ax, **BLOCK_LAYOUT["output"], header_color="#0e7c7b", body_color="#f7fbfb")
    _draw_block(ax, **BLOCK_LAYOUT["measures"], header_color="#1f3d7a", body_color="#f8fafc")

    # Calage interne sur le schéma draw.io du modèle théorique, mais mis à
    # l'échelle du bloc médiateur courant pour pouvoir l'agrandir sans casser
    # les sous-zones.
    def _rect_in_mediator(child_x, child_y, child_w, child_h):
        base_x, base_y, base_w, base_h = 272.0, 88.0, 637.0, 332.0
        med = BLOCK_LAYOUT["mediators"]
        bx, by = med["xy"]
        bw, bh = med["wh"]
        rel_x = (child_x - base_x) / base_w
        rel_y = (child_y - base_y) / base_h
        rel_w = child_w / base_w
        rel_h = child_h / base_h
        return {
            "xy": (bx + rel_x * bw, by + (1.0 - rel_y - rel_h) * bh),
            "wh": (rel_w * bw, rel_h * bh),
        }

    prod = _rect_in_mediator(342, 127, 377, 283)
    maint = _rect_in_mediator(719, 127, 178, 283)
    sub_row = _rect_in_mediator(282, 127, 615, 114)
    dyn_row = _rect_in_mediator(282, 241, 615, 90)
    beh_row = _rect_in_mediator(282, 331, 615, 80)

    for rect in (prod, maint):
        ax.add_patch(
            Rectangle(
                rect["xy"],
                rect["wh"][0],
                rect["wh"][1],
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor="#d1d5db",
                lw=0.8,
                alpha=0.55,
            )
        )

    for rect in (sub_row, dyn_row, beh_row):
        ax.add_patch(
            Rectangle(
                rect["xy"],
                rect["wh"][0],
                rect["wh"][1],
                transform=ax.transAxes,
                facecolor="none",
                edgecolor="#cbd5e1",
                lw=0.9,
            )
        )

    ax.text(
        prod["xy"][0] + prod["wh"][0] / 2,
        prod["xy"][1] + prod["wh"][1] - 0.02,
        "Production",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#334155",
    )
    ax.text(
        maint["xy"][0] + maint["wh"][0] / 2,
        maint["xy"][1] + maint["wh"][1] - 0.02,
        "Maintenance",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#334155",
    )

    row_labels = [
        ("Transactive\nsub-system", sub_row),
        ("Dynamic\nindicators", dyn_row),
        ("Behavioral\nprocess", beh_row),
    ]
    for label, rect in row_labels:
        ax.text(
            rect["xy"][0] + 0.028,
            rect["xy"][1] + rect["wh"][1] / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.5,
            fontweight="bold",
            color="#334155",
        )

    med_bottom = BLOCK_LAYOUT["mediators"]["xy"][1]
    measures_top = BLOCK_LAYOUT["measures"]["xy"][1] + BLOCK_LAYOUT["measures"]["wh"][1]
    emergence_y = med_bottom - (med_bottom - measures_top) * 0.45
    ax.text(
        0.505,
        emergence_y,
        "Emergence levels measured by multimodal nonverbal cues",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#64748b",
    )


def _trim_edge_labels(labels: list[dict], max_labels: int = EMPIRICAL_LABEL_LIMIT) -> list[dict]:
    if len(labels) <= max_labels:
        return labels
    labels_sorted = sorted(
        labels,
        key=lambda item: (
            item.get("priority", 0),
            item.get("strength", 0.0),
        ),
        reverse=True,
    )
    return labels_sorted[:max_labels]


def _draw_node_box(ax, node: str) -> FancyBboxPatch:
    spec = NODE_LAYOUT[node]
    x, y = spec["xy"]
    w, h = spec["wh"]
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor=spec["fc"],
        edgecolor=spec["ec"],
        linewidth=1.2,
        mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        spec["label"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=spec["fontsize"],
        fontweight="bold",
        color=spec["text_color"],
    )
    return patch


def _draw_nodes(ax, G: GraphLike) -> dict[str, FancyBboxPatch]:
    patches: dict[str, FancyBboxPatch] = {}
    for node in G.nodes():
        if node in HIDDEN_NODES:
            continue
        patches[node] = _draw_node_box(ax, node)
    return patches


def _label_position(src: str, dst: str, offset: float = 0.012) -> tuple[float, float]:
    x1, y1 = _node_center(src)
    x2, y2 = _node_center(dst)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    norm = max((dx**2 + dy**2) ** 0.5, 1e-9)
    nxp = -dy / norm
    nyp = dx / norm
    return mx + nxp * offset, my + nyp * offset


def _draw_edge(
    ax,
    patches: dict[str, FancyBboxPatch],
    src: str,
    dst: str,
    color: str,
    lw: float,
    linestyle: str,
    alpha: float = 1.0,
    mutation_scale: int = 16,
    zorder: int = 2,
) -> None:
    if src in HIDDEN_NODES or dst in HIDDEN_NODES:
        return
    rad = EDGE_CURVATURE.get((src, dst), 0.0)
    arrow = FancyArrowPatch(
        _node_center(src),
        _node_center(dst),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        alpha=alpha,
        patchA=patches.get(src),
        patchB=patches.get(dst),
        connectionstyle=f"arc3,rad={rad:.3f}",
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _add_auto_legend(ax, mode: str = "combined") -> None:
    handles = [
        Line2D([0], [0], color="#9aa1aa", lw=2.0, linestyle="dashed", label="Lien théorique"),
    ]
    if mode in {"combined", "empirical"}:
        handles.extend(
            [
                Line2D([0], [0], color="#d62828", lw=2.8, linestyle="solid", label="Empirique positif"),
                Line2D([0], [0], color="#1d4ed8", lw=2.8, linestyle="solid", label="Empirique négatif"),
                Line2D([0], [0], color="#2a9d8f", lw=2.8, linestyle="solid", label="Lien latent-dimension validé par α"),
            ]
        )
    if mode == "combined":
        handles.append(Line2D([0], [0], color="#6a4c93", lw=2.2, linestyle="dashdot", label="Empirique exploratoire"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=True, framealpha=0.96, fontsize=8)


def _draw_theory_only(ax, G: GraphLike, title: str = "Modèle théorique") -> None:
    _draw_background(ax)
    patches = _draw_nodes(ax, G)
    for u, v, d in G.edges(data=True):
        if u in HIDDEN_NODES or v in HIDDEN_NODES:
            continue
        if d.get("exploratory", False):
            continue
        _draw_edge(
            ax,
            patches,
            u,
            v,
            color="#9aa1aa",
            lw=0.9 + 0.75 * d.get("theory_weight", 1),
            linestyle="dashed",
            alpha=0.92,
            mutation_scale=14,
            zorder=1,
        )
    _add_auto_legend(ax, mode="theory")
    ax.set_title(title, fontsize=13, fontweight="bold")


def _draw_theory_plus_empirical(
    ax,
    G: GraphLike,
    title: str = "Théorie + soutien empirique",
    show_r_labels: bool = False,
) -> None:
    _draw_background(ax)
    patches = _draw_nodes(ax, G)
    labels = []

    for u, v, d in G.edges(data=True):
        if u in HIDDEN_NODES or v in HIDDEN_NODES:
            continue
        if d.get("exploratory", False):
            continue
        _draw_edge(
            ax,
            patches,
            u,
            v,
            color="#c0c4ca",
            lw=0.8 + 0.65 * d.get("theory_weight", 1),
            linestyle="dashed",
            alpha=0.78,
            mutation_scale=14,
            zorder=1,
        )

    for u, v, d in G.edges(data=True):
        if u in HIDDEN_NODES or v in HIDDEN_NODES:
            continue
        if d.get("measurement_supported", False):
            alpha_value = float(d.get("latent_alpha", np.nan))
            if pd.notna(alpha_value):
                _draw_edge(
                    ax,
                    patches,
                    u,
                    v,
                    color="#2a9d8f",
                    lw=_normalize_alpha_width(alpha_value),
                    linestyle="solid",
                    alpha=0.95,
                    mutation_scale=16,
                    zorder=3,
                )
                if show_r_labels:
                    labels.append(
                        {
                            "pos": _label_position(u, v, offset=0.014),
                            "text": f"α={alpha_value:.2f}",
                            "priority": 3,
                            "strength": float(alpha_value),
                        }
                    )

        rho = d.get("empirical_rho", np.nan)
        if not d.get("empirical_kept", False) or pd.isna(rho):
            continue

        is_expl = bool(d.get("exploratory", False))
        if is_expl:
            color = "#6a4c93" if rho > 0 else "#1982c4"
            style = "dashdot"
        else:
            color = "#d62828" if rho > 0 else "#1d4ed8"
            style = "solid"
        _draw_edge(
            ax,
            patches,
            u,
            v,
            color=color,
            lw=_normalize_width(float(rho), out_min=1.4 if is_expl else 1.8, out_max=5.4 if is_expl else 6.4),
            linestyle=style,
            alpha=0.96,
            mutation_scale=18,
            zorder=4 if is_expl else 5,
        )
        if show_r_labels:
            prefix = "exp " if is_expl else ""
            labels.append(
                {
                    "pos": _label_position(u, v),
                    "text": f"{prefix}{float(rho):.2f}",
                    "priority": 1 if is_expl else 2,
                    "strength": abs(float(rho)),
                }
            )

    if show_r_labels:
        for item in _trim_edge_labels(labels):
            lx, ly = item["pos"]
            text = item["text"]
            ax.text(
                lx,
                ly,
                text,
                transform=ax.transAxes,
                fontsize=7,
                ha="center",
                va="center",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82),
                zorder=10,
            )

    _add_auto_legend(ax, mode="combined")
    ax.set_title(title, fontsize=13, fontweight="bold")


def _draw_empirical_only(
    ax,
    G: GraphLike,
    title: str = "Liens empiriques observés",
    show_r_labels: bool = False,
) -> None:
    _draw_background(ax)
    patches = _draw_nodes(ax, G)
    labels = []

    for u, v, d in G.edges(data=True):
        if u in HIDDEN_NODES or v in HIDDEN_NODES:
            continue
        if d.get("measurement_supported", False):
            alpha_value = float(d.get("latent_alpha", np.nan))
            if pd.isna(alpha_value):
                continue
            _draw_edge(
                ax,
                patches,
                u,
                v,
                color="#2a9d8f",
                lw=_normalize_alpha_width(alpha_value),
                linestyle="solid",
                alpha=0.95,
                mutation_scale=16,
                zorder=3,
            )
            if show_r_labels:
                labels.append(
                    {
                        "pos": _label_position(u, v, offset=0.014),
                        "text": f"α={alpha_value:.2f}",
                        "priority": 3,
                        "strength": float(alpha_value),
                    }
                )
            continue

        rho = d.get("empirical_rho", np.nan)
        if not d.get("empirical_kept", False) or pd.isna(rho):
            continue

        is_expl = bool(d.get("exploratory", False))
        if is_expl:
            color = "#6a4c93" if rho > 0 else "#1982c4"
            style = "dashdot"
        else:
            color = "#d62828" if rho > 0 else "#1d4ed8"
            style = "solid"
        _draw_edge(
            ax,
            patches,
            u,
            v,
            color=color,
            lw=_normalize_width(float(rho)),
            linestyle=style,
            alpha=0.96,
            mutation_scale=18,
            zorder=4,
        )
        if show_r_labels:
            labels.append(
                {
                    "pos": _label_position(u, v),
                    "text": f"{float(rho):.2f}",
                    "priority": 1 if is_expl else 2,
                    "strength": abs(float(rho)),
                }
            )

    if show_r_labels:
        for item in _trim_edge_labels(labels):
            lx, ly = item["pos"]
            text = item["text"]
            ax.text(
                lx,
                ly,
                text,
                transform=ax.transAxes,
                fontsize=7,
                ha="center",
                va="center",
                color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82),
                zorder=10,
            )

    _add_auto_legend(ax, mode="empirical")
    ax.set_title(title, fontsize=13, fontweight="bold")


def _save_png_pdf(fig, png_path: Path, dpi: int) -> None:
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")


def _normalize_modality_filter(modality_filter: Iterable[str] | None) -> list[str] | None:
    if modality_filter is None:
        return None
    alias = {
        "audio": "Audio_INV",
        "audio_inv": "Audio_INV",
        "face": "Face_INV",
        "face_inv": "Face_INV",
        "gaze": "Gaze_INV",
        "gaze_inv": "Gaze_INV",
    }
    normalized = []
    for raw in modality_filter:
        key = str(raw).strip()
        if not key:
            continue
        normalized.append(alias.get(key.lower(), key))
    return normalized or None


def _draw_computational_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def add_box(
        x,
        y,
        w,
        h,
        text,
        fc,
        ec="#0f172a",
        tc="white",
        fs=11,
        italic=False,
        lw=1.8,
        boxstyle="square,pad=0.0",
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            boxstyle=boxstyle,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fs,
            color=tc,
            fontstyle="italic" if italic else "normal",
            fontweight="bold" if not italic else "normal",
        )
        return patch

    def add_arrow(
        p1,
        p2,
        color="black",
        style="solid",
        lw=1.8,
        rad=0.0,
        mutation_scale=18,
    ):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=mutation_scale,
                linewidth=lw,
                linestyle=style,
                color=color,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    def rect(x, y, w, h):
        return _drawio_axes_rect(x, y, w, h, page=DRAWIO_PIPELINE_PAGE)

    input_outer = rect(100.5, 31, 215, 121)
    output_outer = rect(413.75, 48, 196.25, 93)
    c_factor = rect(34, 214, 184, 68)
    signal_outer = rect(87.875, 367, 252.5, 115)
    feature_outer = rect(46, 514, 342, 78)
    mediator_outer = rect(214.25, 205.57, 355.75, 124)
    mediator_header = rect(215.70, 205.57, 354.30, 63.00)
    tms_row = rect(215.70, 235.57, 354.30, 33.00)
    tas_row = rect(215.21, 268.57, 354.79, 30.00)
    trs_row = rect(215.45, 298.57, 354.55, 30.00)

    input_patch = add_box(
        input_outer["xy"][0],
        input_outer["xy"][1],
        input_outer["wh"][0],
        input_outer["wh"][1],
        "Input layer\n\nTeam composition\nEnvironment / Context\nTask characteristics",
        "#1ca6a3",
        fs=11,
        italic=True,
        lw=2.0,
    )
    c_factor_patch = add_box(
        c_factor["xy"][0],
        c_factor["xy"][1],
        c_factor["wh"][0],
        c_factor["wh"][1],
        "CI / C-factor\n(initial collective potential)",
        "#c8d8f0",
        tc="#111827",
        fs=9.2,
        lw=2.2,
    )

    ax.add_patch(
        Rectangle(
            (c_factor["xy"][0] - 0.01, c_factor["xy"][1] - 0.008),
            c_factor["wh"][0] + 0.02,
            c_factor["wh"][1] + 0.016,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#d62828",
            linewidth=2.0,
        )
    )

    signal_patch = add_box(
        signal_outer["xy"][0],
        signal_outer["xy"][1],
        signal_outer["wh"][0],
        signal_outer["wh"][1],
        "Signal layer\n\nGaze dynamics\nVocal / paraverbal activity\nFacial expressions",
        "#132f7a",
        fs=10.8,
        italic=True,
        lw=2.0,
    )
    feature_patch = add_box(
        feature_outer["xy"][0],
        feature_outer["xy"][1],
        feature_outer["wh"][0],
        feature_outer["wh"][1],
        "Feature & indicator layer\n\nparticipation balance, overlap / turn-taking,\ngaze coupling, shared attention,\nemotional congruence, facial synchrony",
        "#f8fafc",
        tc="#111827",
        fs=8.6,
        lw=1.8,
    )

    ax.add_patch(
        Rectangle(
            mediator_outer["xy"],
            mediator_outer["wh"][0],
            mediator_outer["wh"][1],
            transform=ax.transAxes,
            fill=False,
            edgecolor="#d62828",
            linewidth=2.2,
        )
    )
    mediator_patch = add_box(
        mediator_header["xy"][0],
        mediator_header["xy"][1],
        mediator_header["wh"][0],
        mediator_header["wh"][1],
        "Mediator dynamics",
        "#1f7aa8",
        fs=12.7,
        italic=True,
        lw=2.0,
    )
    add_box(
        tms_row["xy"][0],
        tms_row["xy"][1],
        tms_row["wh"][0],
        tms_row["wh"][1],
        "TMS -> specialization / credibility",
        "#2f89c2",
        fs=9.8,
        lw=1.5,
    )
    add_box(
        tas_row["xy"][0],
        tas_row["xy"][1],
        tas_row["wh"][0],
        tas_row["wh"][1],
        "TAS -> coordination / strategies",
        "#2f89c2",
        fs=9.8,
        lw=1.5,
    )
    add_box(
        trs_row["xy"][0],
        trs_row["xy"][1],
        trs_row["wh"][0],
        trs_row["wh"][1],
        "TRS -> effort / cohesion",
        "#2f89c2",
        fs=9.8,
        lw=1.5,
    )

    output_patch = add_box(
        output_outer["xy"][0],
        output_outer["xy"][1],
        output_outer["wh"][0],
        output_outer["wh"][1],
        "Outputs\n\nObjective metrics\nSubjective metrics",
        "#0e7c7b",
        fs=10.8,
        italic=True,
        lw=2.0,
    )

    add_arrow((0.31, 0.80), (0.43, 0.64), lw=2.1)
    add_arrow((0.17, 0.67), (0.17, 0.47), lw=1.9)
    add_arrow((0.22, 0.57), (0.43, 0.60), lw=1.8)
    add_arrow((0.21, 0.27), (0.21, 0.09), lw=1.8)
    add_arrow((0.36, 0.12), (0.43, 0.50), lw=2.0)
    add_arrow((0.58, 0.60), (0.80, 0.74), lw=2.1)
    add_arrow((0.80, 0.72), (0.26, 0.80), color="#7c8798", style="dashed", lw=1.6, rad=0.08)
    add_arrow((0.79, 0.70), (0.53, 0.73), color="#7c8798", style="dashed", lw=1.5, rad=-0.05)

    ax.text(0.33, 0.81, "task execution", transform=ax.transAxes, fontsize=9.4, color="#64748b")
    ax.text(0.23, 0.41, "time-synchronized multimodal capture", transform=ax.transAxes, fontsize=8.8, color="#64748b", fontstyle="italic")
    ax.text(0.68, 0.60, "g(I_TMS, I_TAS, I_TRS)", transform=ax.transAxes, fontsize=9.6, color="#111827", fontstyle="italic")
    ax.text(0.73, 0.46, "episodic cycles", transform=ax.transAxes, fontsize=8.4, color="#7c8798")


# ============================================================================
# API PUBLIQUE
# ============================================================================
def generate_theory_diagrams(
    corr_rows: list[dict],
    out_dir: Path,
    dpi: int = 200,
    p_threshold: float = P_THRESHOLD,
    rho_threshold: float = RHO_THRESHOLD,
    modality_filter: Iterable[str] | None = None,
    show_r_labels: bool = False,
    latent_construct_alphas: dict[str, float] | None = None,
    latent_alpha_threshold: float = LATENT_ALPHA_THRESHOLD,
) -> dict[str, Path | None]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(modality_filter, str):
        modality_filter = [m.strip() for m in modality_filter.split(",") if m.strip()]
    modality_filter = _normalize_modality_filter(modality_filter)

    df_agg = _aggregate_empirical(
        corr_rows,
        THEORETICAL_MAPPING,
        rho_threshold=rho_threshold,
        p_threshold=p_threshold,
        modality_filter=modality_filter,
    )
    if not df_agg.empty:
        df_agg.to_csv(out_dir / "theory_empirical_edges.csv", index=False, encoding="utf-8-sig")

    G_theory = _build_theory_graph()
    G_full = _add_empirical(G_theory, df_agg) if not df_agg.empty else G_theory.copy()
    G_full = _apply_latent_measurement_support(
        G_full,
        latent_construct_alphas,
        alpha_threshold=latent_alpha_threshold,
    )

    results: dict[str, Path | None] = {}
    figsize_single = (14.5, 8.6)

    p1 = out_dir / "theory_model_only.png"
    fig, ax = plt.subplots(figsize=figsize_single)
    _draw_theory_only(ax, G_theory, title="A. Modèle théorique")
    plt.tight_layout()
    _save_png_pdf(fig, p1, dpi=dpi)
    plt.close(fig)
    results["theory"] = p1
    results["theory_pdf"] = p1.with_suffix(".pdf")

    p2 = out_dir / "theory_plus_empirical.png"
    fig, ax = plt.subplots(figsize=figsize_single)
    _draw_theory_plus_empirical(ax, G_full, title="B. Théorie + soutien empirique", show_r_labels=show_r_labels)
    plt.tight_layout()
    _save_png_pdf(fig, p2, dpi=dpi)
    plt.close(fig)
    results["theory_empirical"] = p2
    results["theory_empirical_pdf"] = p2.with_suffix(".pdf")

    p3 = out_dir / "theory_empirical_only.png"
    fig, ax = plt.subplots(figsize=figsize_single)
    _draw_empirical_only(ax, G_full, title="C. Liens empiriques observés", show_r_labels=show_r_labels)
    plt.tight_layout()
    _save_png_pdf(fig, p3, dpi=dpi)
    plt.close(fig)
    results["empirical"] = p3
    results["empirical_pdf"] = p3.with_suffix(".pdf")

    p4 = out_dir / "theory_comparison.png"
    # Empilement vertical pour améliorer la lisibilité sur page portrait et
    # éviter l'écrasement des blocs quand les deux panneaux sont côte à côte.
    # Taille légèrement augmentée pour préserver la netteté après insertion PDF.
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 16.6))
    _draw_theory_only(axes[0], G_theory, title="A. Modèle théorique")
    _draw_theory_plus_empirical(axes[1], G_full, title="B. Théorie + soutien empirique", show_r_labels=show_r_labels)
    plt.tight_layout()
    _save_png_pdf(fig, p4, dpi=dpi)
    plt.close(fig)
    results["comparison"] = p4
    results["comparison_pdf"] = p4.with_suffix(".pdf")

    n_theory = len(THEORETICAL_EDGES)
    n_supported = sum(1 for _, _, d in G_full.edges(data=True) if d.get("empirical_kept") and not d.get("exploratory"))
    n_measurement = sum(1 for _, _, d in G_full.edges(data=True) if d.get("measurement_supported"))
    n_explo = sum(1 for _, _, d in G_full.edges(data=True) if d.get("empirical_kept") and d.get("exploratory"))
    print(
        f"  [OK] Diagrammes théoriques : {n_supported}/{n_theory} liens soutenus, "
        f"{n_measurement} lien(s) latent-dimension validé(s) par alpha, {n_explo} exploratoire(s)"
    )

    return results


def generate_computational_diagram(
    out_dir: Path,
    dpi: int = 200,
) -> dict[str, Path | None]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "computational_model_articulation.png"
    fig, ax = plt.subplots(figsize=(15.5, 9.2))
    _draw_computational_panel(ax)
    ax.set_title("Modèle computationnel et chaîne d'opérationnalisation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_png_pdf(fig, png_path, dpi=dpi)
    plt.close(fig)
    return {
        "computation": png_path,
        "computation_pdf": png_path.with_suffix(".pdf"),
    }
