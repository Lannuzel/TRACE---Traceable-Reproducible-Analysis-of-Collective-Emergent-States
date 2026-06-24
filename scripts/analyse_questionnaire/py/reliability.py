#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reliability.py — Alpha de Cronbach + statistiques item (r.drop, alpha_if_deleted).

Implémentation pure Python/numpy — pas de dépendance R ou psych.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIMENSION_LABELS


def _cronbach_alpha(data: np.ndarray) -> float | None:
    """
    Calcule l'alpha de Cronbach pour une matrice (n_subjects × k_items).
    Retourne None si le calcul est impossible.
    """
    k = data.shape[1]
    if k < 2:
        return None
    item_vars = np.nanvar(data, axis=0, ddof=1)
    total_var = np.nanvar(np.nansum(data, axis=1), ddof=1)
    if total_var == 0:
        return None
    return float(k / (k - 1) * (1 - np.nansum(item_vars) / total_var))


def _item_total_correlation(data: np.ndarray, col_idx: int) -> float | None:
    """
    Corrected item-total correlation (r.drop) pour l'item col_idx.
    = corrélation entre l'item et le total des autres items.
    """
    item = data[:, col_idx]
    rest = np.nansum(np.delete(data, col_idx, axis=1), axis=1)
    # Enlevons les lignes avec NaN
    mask = ~(np.isnan(item) | np.isnan(rest))
    if mask.sum() < 3:
        return None
    r = np.corrcoef(item[mask], rest[mask])[0, 1]
    return float(r) if np.isfinite(r) else None


def cronbach_by_dimension(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'alpha de Cronbach pour chaque dimension.

    Retourne un DataFrame avec colonnes: dimension, label, n_items, alpha.
    """
    dims = sorted(df_long["dimension"].dropna().unique())
    rows = []
    for dim in dims:
        sub = df_long.loc[
            (df_long["dimension"] == dim) & df_long["reponse_num"].notna(),
            ["Participant", "code", "reponse_num"],
        ]
        wide = sub.pivot_table(index="Participant", columns="code", values="reponse_num")
        # Supprimer colonnes avec variance nulle
        variances = wide.var()
        valid_cols = variances[variances > 0].index.tolist()
        n_items = len(valid_cols)
        if n_items < 2:
            rows.append({"dimension": dim, "n_items": n_items, "alpha": None})
            continue
        data = wide[valid_cols].values
        alpha = _cronbach_alpha(data)
        rows.append({
            "dimension": dim,
            "n_items": n_items,
            "alpha": round(alpha, 3) if alpha is not None else None,
        })

    result = pd.DataFrame(rows)
    result["label"] = result["dimension"].map(DIMENSION_LABELS)
    return result[["dimension", "label", "n_items", "alpha"]]


def item_stats_by_dimension(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques par item au sein de chaque dimension :
    n, mean, sd, r.drop (corrected item-total), alpha_if_deleted.
    """
    dims = sorted(df_long["dimension"].dropna().unique())
    all_rows = []

    for dim in dims:
        sub = df_long.loc[
            (df_long["dimension"] == dim) & df_long["reponse_num"].notna(),
            ["Participant", "code", "reponse_num"],
        ]
        wide = sub.pivot_table(index="Participant", columns="code", values="reponse_num")

        variances = wide.var()
        valid_cols = variances[variances > 0].index.tolist()
        if len(valid_cols) < 2:
            continue

        data = wide[valid_cols].values
        col_names = valid_cols

        for idx, col in enumerate(col_names):
            item_vals = data[:, idx]
            mask = ~np.isnan(item_vals)
            n = int(mask.sum())
            mean_val = float(np.nanmean(item_vals)) if n > 0 else None
            sd_val = float(np.nanstd(item_vals, ddof=1)) if n > 1 else None
            rdrop = _item_total_correlation(data, idx)

            # Alpha if deleted
            remaining = np.delete(data, idx, axis=1)
            alpha_del = _cronbach_alpha(remaining) if remaining.shape[1] >= 2 else None

            # Code short
            from .transform import extract_item_short
            code_short = extract_item_short(col)

            all_rows.append({
                "code": col,
                "dimension": dim,
                "label_dim": DIMENSION_LABELS.get(dim, dim),
                "code_short": code_short,
                "n": n,
                "mean": round(mean_val, 3) if mean_val is not None else None,
                "sd": round(sd_val, 3) if sd_val is not None else None,
                "r.drop": round(rdrop, 3) if rdrop is not None else None,
                "alpha_if_deleted": round(alpha_del, 3) if alpha_del is not None else None,
            })

    result = pd.DataFrame(all_rows)
    if not result.empty:
        result = result.sort_values(["dimension", "r.drop"])
    return result
