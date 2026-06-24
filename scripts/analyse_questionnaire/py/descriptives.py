#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descriptives.py — Statistiques descriptives par dimension.
"""

import pandas as pd

from .config import DIMENSION_LABELS


def descriptives_by_dimension(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques descriptives agrégées par dimension :
    n_responses, n_participants, n_items, mean, sd, median, min, max.
    """
    sub = df_long.loc[df_long["reponse_num"].notna()].copy()

    stats = (
        sub.groupby("dimension")
        .agg(
            n_responses=("reponse_num", "count"),
            n_participants=("Participant", "nunique"),
            n_items=("code", "nunique"),
            mean=("reponse_num", "mean"),
            sd=("reponse_num", "std"),
            median=("reponse_num", "median"),
            min=("reponse_num", "min"),
            max=("reponse_num", "max"),
        )
        .reset_index()
    )

    stats["mean"] = stats["mean"].round(2)
    stats["sd"] = stats["sd"].round(2)
    stats["label"] = stats["dimension"].map(DIMENSION_LABELS)

    cols = ["dimension", "label", "n_responses", "n_participants", "n_items",
            "mean", "sd", "median", "min", "max"]
    return stats[cols]
