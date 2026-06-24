#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
role_tests.py — Synthèse par rôle + tests inter-rôles (ANOVA / Kruskal-Wallis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import DIMENSION_LABELS
from .descriptives import descriptives_by_dimension


def _pval_roles(df_dim: pd.DataFrame) -> float | None:
    """
    Teste si les scores diffèrent significativement entre rôles.
    Shapiro → ANOVA (si normal) ou Kruskal-Wallis (sinon).
    """
    sub = df_dim[
        df_dim["Role"].notna()
        & ~df_dim["Role"].isin(["nan", "<NA>", "None", ""])
    ].dropna(subset=["reponse_num"]).copy()
    groups = {r: g["reponse_num"].values for r, g in sub.groupby("Role")}
    if len(groups) < 2 or any(len(v) < 2 for v in groups.values()):
        return None

    # Test de normalité par groupe
    normal = True
    for vals in groups.values():
        if len(vals) < 3:
            normal = False
            break
        _, p_shap = sp_stats.shapiro(vals)
        if p_shap <= 0.05:
            normal = False
            break

    group_arrays = list(groups.values())
    if normal:
        _, p = sp_stats.f_oneway(*group_arrays)
    else:
        _, p = sp_stats.kruskal(*group_arrays)

    return float(p) if np.isfinite(p) else None


def summary_by_role(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Table récapitulative par dimension :
    descriptives + moyennes par rôle + p-value du test inter-rôles.
    """
    # Moyennes par rôle
    # Filtre les rôles manquants ou résidus de conversion str
    sub = df_long[
        df_long["reponse_num"].notna()
        & df_long["Role"].notna()
        & ~df_long["Role"].isin(["nan", "<NA>", "None", ""])
    ].copy()
    mean_role = (
        sub.groupby(["dimension", "Role"])["reponse_num"]
        .mean()
        .round(2)
        .reset_index()
    )
    mean_role["txt"] = mean_role["Role"] + " = " + mean_role["reponse_num"].astype(str)
    mean_by_role = (
        mean_role.groupby("dimension")["txt"]
        .apply(lambda x: "; ".join(x))
        .reset_index()
        .rename(columns={"txt": "mean_by_role"})
    )

    # P-values
    dims = sorted(df_long["dimension"].dropna().unique())
    pvals = []
    for dim in dims:
        df_dim = df_long[df_long["dimension"] == dim]
        p = _pval_roles(df_dim)
        pvals.append({"dimension": dim, "p_value": p})
    pvals_df = pd.DataFrame(pvals)

    # Merge
    desc = descriptives_by_dimension(df_long)
    result = desc.merge(mean_by_role, on="dimension", how="left")
    result = result.merge(pvals_df, on="dimension", how="left")
    result = result.sort_values("p_value")

    return result
