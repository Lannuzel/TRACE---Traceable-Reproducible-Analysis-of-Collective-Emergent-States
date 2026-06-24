#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform.py — Transformation wide → long + recodage Likert + inversion.
"""

from __future__ import annotations

import re
import textwrap

import pandas as pd

from .config import ALL_LEVELS, CODE_MAP, INVERT_SHORT
from .io_read import get_meta_cols, get_role_df, get_question_cols


def extract_dimension(colname: str) -> str | None:
    """Extrait le code dimension (COR, CRE, ...) depuis un nom de colonne."""
    # Format make.names : G1Q00001.COR03.question...
    m = re.search(r"(G1Q00001|G2Q00001)\.([A-Z]{3})", colname)
    if m:
        return m.group(2)
    # Format brut Excel : G1Q00001[COR01]
    m = re.search(r"(G1Q00001|G2Q00001)\[([A-Z]{3})\d{2}\]", colname)
    if m:
        return m.group(2)
    return None


def extract_item_short(colname: str) -> str | None:
    """
    Extrait le code court de l'item (ex: 'G1Q00001.COR03').
    Supporte le format make.names et le format brut [COR03].
    """
    # Format make.names
    m = re.search(r"(G1Q00001|G2Q00001)\.([A-Z]{3}\d{2})", colname)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    # Format brut Excel
    m = re.search(r"(G1Q00001|G2Q00001)\[([A-Z]{3}\d{2})\]", colname)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return None


def extract_question(colname: str) -> str | None:
    """Extrait le texte de la question depuis le nom de colonne."""
    if colname.startswith("G3Q0000"):
        q = re.sub(r"^G3Q0000\d+\.?\s*", "", colname)
        return q.strip() or None
    parts = colname.split(".")
    if len(parts) > 3:
        return " ".join(parts[3:]).strip()
    # Format brut : texte après ]. 
    m = re.search(r"\]\.\s*(.+)$", colname)
    if m:
        return m.group(1).strip()
    return None


def extract_after_opinion(text: str | None, width: int = 40) -> str | None:
    """Extrait le texte après 'opinion' pour un affichage court."""
    if text is None:
        return None
    m = re.search(r"(?i)opinion.+?[.\s]*(.+)$", text)
    if m:
        return textwrap.fill(m.group(1).strip(), width=width)
    return textwrap.fill(text, width=width) if len(text) > width else text


def _recode_response(val) -> float | None:
    """Convertit une réponse Likert (texte ou chiffre) en valeur numérique 1–9."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Déjà numérique 1–9
    if re.fullmatch(r"[1-9]", s):
        return float(s)
    # Texte Likert
    if s in CODE_MAP:
        return float(CODE_MAP[s])
    return None


def make_long_survey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le dataframe wide en format long avec recodage et inversion.

    Colonnes résultantes :
        Participant, Groupe, code, reponse, reponse_num, dimension,
        question, question_wrapped, code_short, Role
    """
    role_df = get_role_df(df)
    meta_map = get_meta_cols(df)
    question_cols = get_question_cols(df)

    # Colonnes d'identification à conserver dans l'export long.
    id_cols = ["Participant"]
    for std_name, real_col in meta_map.items():
        if real_col is not None and real_col not in id_cols:
            id_cols.append(real_col)

    keep = [c for c in id_cols if c in df.columns] + question_cols
    sub = df[keep].copy()
    sub["row_id"] = range(len(sub))

    # Pivot long
    long = sub.melt(
        id_vars=["row_id"] + [c for c in id_cols if c in sub.columns],
        var_name="code",
        value_name="reponse",
    )

    # Harmonise les noms des métadonnées au plus tôt pour les exports.
    rename_map = {
        real_col: std_name
        for std_name, real_col in meta_map.items()
        if real_col is not None and real_col in long.columns and real_col != std_name
    }
    if rename_map:
        long = long.rename(columns=rename_map)

    long["dimension"] = long["code"].apply(extract_dimension)
    long["question"] = long["code"].apply(extract_question)
    long["question_wrapped"] = long["question"].apply(extract_after_opinion)
    long["reponse_num_brut"] = long["reponse"].apply(_recode_response)

    # Code court + inversion
    long["code_short"] = long["code"].apply(extract_item_short)
    long["item_inverse"] = long["code_short"].isin(INVERT_SHORT)
    long["reponse_num"] = long["reponse_num_brut"]
    long.loc[
        long["item_inverse"] & long["reponse_num"].notna(),
        "reponse_num",
    ] = 10 - long.loc[
        long["item_inverse"] & long["reponse_num"].notna(),
        "reponse_num",
    ]

    # Remplace la réponse texte pour les numériques bruts
    mask_num = long["reponse"].astype(str).str.fullmatch(r"[1-9]")
    long.loc[mask_num, "reponse"] = long.loc[mask_num, "reponse"].astype(int).map(
        lambda i: ALL_LEVELS[i - 1] if 1 <= i <= 9 else None
    )

    # Jointure rôle
    long = long.merge(role_df, on="Participant", how="left")

    long.drop(columns=["row_id"], inplace=True)
    return long
