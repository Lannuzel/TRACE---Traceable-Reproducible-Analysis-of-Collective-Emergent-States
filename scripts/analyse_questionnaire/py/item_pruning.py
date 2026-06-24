#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
item_pruning.py — Détection items faibles (r.drop) + données épurées.

Deux niveaux de résultat :
  - Signalement : TOUS les items avec r.drop < seuil sont listés, quelle que
    soit la dimension (colonne `supprime` = False si non applicable).
  - Suppression : effective uniquement si la dimension vérifie les deux conditions :
      1. remain ≥ 2 items après retrait (évite de vider une dimension)
      2. alpha_initial < alpha_threshold (inutile de toucher une dimension déjà acceptable)
"""

from __future__ import annotations

import pandas as pd

from .config import RDROP_THRESHOLD, ALPHA_ACCEPTABILITY_THRESHOLD
from .reliability import item_stats_by_dimension, _cronbach_alpha


def prune_by_rdrop(
    df_long: pd.DataFrame,
    rdrop_threshold: float = RDROP_THRESHOLD,
    alpha_threshold: float = ALPHA_ACCEPTABILITY_THRESHOLD,
) -> dict:
    """
    Identifie les items faibles (r.drop < rdrop_threshold) et les liste tous.
    La suppression effective n'est appliquée que pour les dimensions dont :
      - l'alpha initial est < alpha_threshold (dimension à problème)
      - il resterait ≥ 2 items après retrait

    Les items des dimensions "acceptables" (α ≥ threshold) sont signalés
    mais marqués supprime=False.

    Retourne:
        {
            "all_flagged":   DataFrame de TOUS les items signalés (r.drop < seuil),
                             avec colonne `supprime` (bool) et `raison_non_supprime`
            "items_to_drop": sous-ensemble réellement supprimé (supprime=True)
            "pruned_long":   DataFrame long épuré
            "dims_all_bad":  dimensions où aucun retrait n'est possible
            "dim_info":      statistiques par dimension
        }
    """
    item_table = item_stats_by_dimension(df_long)

    # ── Calcul de l'alpha initial par dimension ──────────────────────────────
    dims_alpha: dict[str, float | None] = {}
    for dim in item_table["dimension"].unique():
        sub = df_long.loc[
            (df_long["dimension"] == dim) & df_long["reponse_num"].notna(),
            ["Participant", "code", "reponse_num"],
        ]
        wide = sub.pivot_table(index="Participant", columns="code", values="reponse_num")
        valid = wide.var()[wide.var() > 0].index.tolist()
        if len(valid) < 2:
            dims_alpha[dim] = None
        else:
            dims_alpha[dim] = _cronbach_alpha(wide[valid].values)

    # ── Signalement : tous les items sous le seuil r.drop ────────────────────
    bad = item_table[
        item_table["r.drop"].isna() | (item_table["r.drop"] < rdrop_threshold)
    ].copy()

    if bad.empty:
        # Aucun item sous le seuil → rien à signaler ni supprimer
        return {
            "all_flagged":  bad,
            "items_to_drop": bad,
            "pruned_long":  df_long.copy(),
            "dims_all_bad": [],
            "dim_info":     pd.DataFrame(),
        }

    # Nombre d'items par dimension
    n_items_dim = item_table.groupby("dimension")["code"].nunique().rename("n_items")
    n_bad_dim   = bad.groupby("dimension")["code"].nunique().rename("n_bad")
    dim_info = pd.DataFrame({"n_items": n_items_dim, "n_bad": n_bad_dim}).fillna(0)
    dim_info["remain"] = dim_info["n_items"] - dim_info["n_bad"]
    dim_info["alpha_initial"] = dim_info.index.map(dims_alpha)

    dims_all_bad = dim_info[dim_info["remain"] < 2].index.tolist()

    # ── Décision de suppression par item ─────────────────────────────────────
    def _decide(row) -> tuple[bool, str]:
        dim = row["dimension"]
        info = dim_info.loc[dim] if dim in dim_info.index else None
        if info is None:
            return False, "dimension inconnue"
        alpha_init = dims_alpha.get(dim)
        if alpha_init is not None and alpha_init >= alpha_threshold:
            return False, f"alpha acceptable (α={alpha_init:.3f} ≥ {alpha_threshold})"
        if info["remain"] < 2:
            return False, "pas assez d'items restants (< 2)"
        return True, ""

    # Evite result_type="expand" dont le comportement varie selon les versions pandas
    decisions = [_decide(row) for _, row in bad.iterrows()]
    bad["supprime"] = [d[0] for d in decisions]
    bad["raison_non_supprime"] = [d[1] for d in decisions]
    bad = bad.sort_values(["dimension", "r.drop"])

    # ── Application de la suppression ────────────────────────────────────────
    items_to_drop = bad[bad["supprime"]].copy()
    codes_to_drop = set(items_to_drop["code"])
    pruned = df_long[~df_long["code"].isin(codes_to_drop)].copy()

    return {
        "all_flagged":  bad,
        "items_to_drop": items_to_drop,
        "pruned_long":  pruned,
        "dims_all_bad": dims_all_bad,
        "dim_info":     dim_info,
    }
