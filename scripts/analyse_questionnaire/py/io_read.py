#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_read.py — Lecture du fichier questionnaire (Excel ou CSV).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def read_survey(path: str | Path) -> pd.DataFrame:
    """
    Lit le fichier questionnaire (xlsx ou csv) et normalise la colonne Participant.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        # Essaie plusieurs encodages
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(path, sep=";", encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"Impossible de lire {path} avec les encodages connus.")

    # Renomme la colonne ID
    id_col = [c for c in df.columns if "ID de la réponse" in str(c)]
    if id_col:
        df = df.rename(columns={id_col[0]: "Participant"})
    df["Participant"] = df["Participant"].astype(str)

    return df


def get_role_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait le rôle de chaque participant.
    Cherche la colonne contenant "Lors de l'activité en VR, vous étiez".
    """
    role_col = [c for c in df.columns if "activité en VR" in str(c) and "étiez" in str(c)]
    if not role_col:
        # Fallback : G3Q00008
        role_col = [c for c in df.columns if c.startswith("G3Q00008")]

    if role_col:
        out = df[["Participant", role_col[0]]].copy()
        out.columns = ["Participant", "Role"]
        # Convertit en str pour uniformiser les valeurs, puis remplace les
        # représentations textuelles de NaN générées par astype(str) en pd.NA réel,
        # afin que dropna() les filtre correctement dans les étapes suivantes.
        out["Role"] = out["Role"].astype(str).replace(["nan", "<NA>", "None", ""], pd.NA)
    else:
        out = df[["Participant"]].copy()
        out["Role"] = pd.NA
    out["Participant"] = out["Participant"].astype(str)
    return out


def get_question_cols(df: pd.DataFrame) -> list[str]:
    """Retourne les colonnes correspondant aux items TMS (G1/G2) et divers (G3)."""
    pattern = re.compile(r"G1Q00001|G2Q00001|G3Q0000")
    return [c for c in df.columns if pattern.search(c)]


def get_item_cols(df: pd.DataFrame) -> list[str]:
    """Retourne uniquement les colonnes items G1Q00001[...] et G2Q00001[...]."""
    pattern = re.compile(r"^G[12]Q00001\[")
    return [c for c in df.columns if pattern.match(c)]


def get_meta_cols(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Détecte les colonnes métadonnées (Session, Modalité, Scénario, Groupe).
    Retourne un dict {nom_standard: nom_réel_dans_df}.
    """
    mapping = {}
    for target, keywords in [
        ("Session", ["Session"]),
        ("Modalite", ["Modalité", "Modalite"]),
        ("Scenario", ["Scénario", "Scenario"]),
        ("Groupe", ["Groupe"]),
    ]:
        found = [c for c in df.columns if any(k in str(c) for k in keywords)]
        mapping[target] = found[0] if found else None
    return mapping
