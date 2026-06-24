#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g3_context.py -- Extraction et synthese des questions G3 (profil participant
+ commentaires libres) du questionnaire.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from .io_read import get_meta_cols, get_role_df


FAMILIARITY_LEVELS = [
    "Pas du tout familier(e)",
    "Peu familier(e)",
    "Modérément familier(e)",
    "Assez familier(e)",
    "Très familier(e)",
]
FAMILIARITY_MAP = {label: idx + 1 for idx, label in enumerate(FAMILIARITY_LEVELS)}

GENDER_NORMALIZATION = {
    "un homme": "Homme",
    "une femme": "Femme",
    "non-binaire": "Non-binaire",
    "non binaire": "Non-binaire",
}

TRIVIAL_COMMENT_MARKERS = {
    "ras",
    "r a s",
    "r.a.s",
    "non",
    "non rien a signaler",
    "rien a signaler",
    "aucun",
    "aucune remarque",
    "rien",
}

COMMENT_THEME_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "task": {
        "Aucune remarque": ("ras", "rien a signaler", "aucune remarque", "non rien a signaler"),
        "Clarte/consignes": ("clair", "claire", "consigne", "comprendre", "compréhens", "explication"),
        "Difficulte/charge": ("diffic", "dur", "complique", "compliqué", "hard", "pas evident", "premiere pratique"),
        "Communication/coordination": ("communication", "coord", "diriger", "ecoute", "écoute"),
        "Role/technique": ("modelis", "modélis", "calcul", "lecteur", "prise en main"),
    },
    "group": {
        "Aucune remarque": ("ras", "aucune remarque", "non", "rien"),
        "Bonne collaboration": ("bonne collaboration", "bonne entente", "bonne dynamique", "sympa", "bien collabor", "reactif", "réactif"),
        "Ecoute/communication": ("ecoute", "écoute", "communication", "a l ecoute", "à l'écoute", "reactif"),
        "Difficulte relationnelle": ("patience", "pas ecoute", "pas écout", "bacle", "bâcle", "mauvaise", "difficile"),
    },
}


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))


def _normalize_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = _strip_accents(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _clean_comment(text: Any) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_group(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _normalize_timepoint(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    if not s:
        return ""
    m = re.search(r"(\d+)", s)
    if m:
        return f"T{int(m.group(1))}"
    return s


def _normalize_modalite(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper().replace(" ", "")
    if "VR" in s:
        return "VR"
    if "PC" in s:
        return "PC"
    return s


def _normalize_scenario(value: Any) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper().replace(" ", "")
    if "S1" in s:
        return "S1"
    if "S2" in s:
        return "S2"
    return s


def _find_first_col(df: pd.DataFrame, prefix: str) -> str | None:
    matches = [c for c in df.columns if str(c).startswith(prefix)]
    return matches[0] if matches else None


def _round_to_familiarity_label(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    idx = int(np.clip(np.rint(float(value)), 1, len(FAMILIARITY_LEVELS))) - 1
    return FAMILIARITY_LEVELS[idx]


def extract_participant_profile(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les questions G3 au niveau participant avec métadonnées normalisées.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    meta_map = get_meta_cols(df_raw)
    role_df = get_role_df(df_raw)
    out = pd.DataFrame()
    out["Participant"] = df_raw["Participant"].astype(str)

    group_col = meta_map.get("Groupe")
    session_col = meta_map.get("Session")
    modalite_col = meta_map.get("Modalite")
    scenario_col = meta_map.get("Scenario")

    out["group_id"] = df_raw[group_col].apply(_normalize_group) if group_col else ""
    out["timepoint"] = df_raw[session_col].apply(_normalize_timepoint) if session_col else ""
    out["modalite"] = df_raw[modalite_col].apply(_normalize_modalite) if modalite_col else ""
    out["scenario"] = df_raw[scenario_col].apply(_normalize_scenario) if scenario_col else ""

    gender_col = _find_first_col(df_raw, "G3Q00001")
    age_col = _find_first_col(df_raw, "G3Q00002")
    vr_col = _find_first_col(df_raw, "G3Q00003")
    fam1_col = _find_first_col(df_raw, "G3Q00004")
    fam2_col = _find_first_col(df_raw, "G3Q00005")
    task_comment_col = _find_first_col(df_raw, "G3Q00006")
    group_comment_col = _find_first_col(df_raw, "G3Q00007")

    out["gender_raw"] = df_raw[gender_col].astype(str).str.strip() if gender_col else ""
    out["gender"] = (
        out["gender_raw"]
        .map(lambda x: GENDER_NORMALIZATION.get(_normalize_text(x), str(x).strip() if str(x).strip() else np.nan))
    )
    out["age"] = pd.to_numeric(df_raw[age_col], errors="coerce") if age_col else np.nan

    out["vr_familiarity"] = df_raw[vr_col].astype(str).str.strip() if vr_col else ""
    out["vr_familiarity_score"] = out["vr_familiarity"].map(FAMILIARITY_MAP)

    out["team_familiarity_member1"] = df_raw[fam1_col].astype(str).str.strip() if fam1_col else ""
    out["team_familiarity_member2"] = df_raw[fam2_col].astype(str).str.strip() if fam2_col else ""
    out["team_familiarity_member1_score"] = out["team_familiarity_member1"].map(FAMILIARITY_MAP)
    out["team_familiarity_member2_score"] = out["team_familiarity_member2"].map(FAMILIARITY_MAP)
    out["team_familiarity_mean_score"] = out[
        ["team_familiarity_member1_score", "team_familiarity_member2_score"]
    ].mean(axis=1, skipna=True)
    out["team_familiarity_mean_label"] = out["team_familiarity_mean_score"].apply(_round_to_familiarity_label)

    out["comment_task"] = df_raw[task_comment_col].apply(_clean_comment) if task_comment_col else ""
    out["comment_group"] = df_raw[group_comment_col].apply(_clean_comment) if group_comment_col else ""

    out = out.merge(role_df[["Participant", "Role"]], on="Participant", how="left")
    out["Role"] = out["Role"].astype(str).replace({"nan": "", "<NA>": ""}).str.strip()

    out = out.drop_duplicates().reset_index(drop=True)
    return out


def summarize_participant_profile(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produit :
    - un tableau de résumé global
    - un tableau counts/pourcentages pour genre, familiarité VR et familiarité équipe
    """
    if profile_df is None or profile_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = profile_df.copy()
    age_vals = pd.to_numeric(work["age"], errors="coerce").dropna() if "age" in work.columns else pd.Series(dtype=float)
    team_vals = pd.to_numeric(work["team_familiarity_mean_score"], errors="coerce").dropna() if "team_familiarity_mean_score" in work.columns else pd.Series(dtype=float)

    summary = pd.DataFrame([{
        "n_participants": int(len(work)),
        "n_groups": int(work["group_id"].replace("", np.nan).dropna().nunique()) if "group_id" in work.columns else 0,
        "age_mean": float(age_vals.mean()) if not age_vals.empty else np.nan,
        "age_sd": float(age_vals.std(ddof=1)) if len(age_vals) >= 2 else np.nan,
        "age_median": float(age_vals.median()) if not age_vals.empty else np.nan,
        "age_min": float(age_vals.min()) if not age_vals.empty else np.nan,
        "age_max": float(age_vals.max()) if not age_vals.empty else np.nan,
        "team_familiarity_mean_score": float(team_vals.mean()) if not team_vals.empty else np.nan,
        "team_familiarity_median_score": float(team_vals.median()) if not team_vals.empty else np.nan,
    }])

    category_rows: list[dict[str, Any]] = []
    category_specs = [
        ("gender", "gender", None),
        ("vr_familiarity", "vr_familiarity", FAMILIARITY_LEVELS),
        ("team_familiarity", "team_familiarity_mean_label", FAMILIARITY_LEVELS),
    ]

    for variable, col, order in category_specs:
        if col not in work.columns:
            continue
        vals = work[col].replace("", np.nan).dropna()
        if vals.empty:
            continue
        counts = vals.value_counts(dropna=False)
        levels = order if order is not None else list(counts.index)
        for level in levels:
            n = int(counts.get(level, 0))
            if n == 0:
                continue
            category_rows.append({
                "variable": variable,
                "level": level,
                "n": n,
                "pct": round(100 * n / len(vals), 1),
            })

    return summary, pd.DataFrame(category_rows)


def _comment_themes(text: str, comment_type: str) -> list[str]:
    norm = _normalize_text(text)
    if not norm:
        return []
    if norm in TRIVIAL_COMMENT_MARKERS:
        return ["Aucune remarque"]

    themes: list[str] = []
    for label, keywords in COMMENT_THEME_RULES.get(comment_type, {}).items():
        if label == "Aucune remarque":
            continue
        if any(keyword in norm for keyword in keywords):
            themes.append(label)
    if not themes:
        themes.append("Autre")
    return themes


def build_free_comments(profile_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit :
    - un tableau long des commentaires non vides
    - un résumé des thèmes détectés
    """
    if profile_df is None or profile_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    comment_specs = [
        ("task", "comment_task"),
        ("group", "comment_group"),
    ]

    for _, row in profile_df.iterrows():
        base = {
            "Participant": row.get("Participant", ""),
            "group_id": row.get("group_id", ""),
            "timepoint": row.get("timepoint", ""),
            "modalite": row.get("modalite", ""),
            "scenario": row.get("scenario", ""),
            "Role": row.get("Role", ""),
        }
        for comment_type, col in comment_specs:
            text = _clean_comment(row.get(col, ""))
            if not text:
                continue
            norm = _normalize_text(text)
            is_trivial = norm in TRIVIAL_COMMENT_MARKERS
            themes = _comment_themes(text, comment_type)
            excerpt = text[:180] + ("..." if len(text) > 180 else "")
            rows.append({
                **base,
                "comment_type": comment_type,
                "text": text,
                "excerpt": excerpt,
                "is_trivial": bool(is_trivial),
                "themes": "; ".join(themes),
            })

    comments_df = pd.DataFrame(rows)
    if comments_df.empty:
        return comments_df, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for comment_type, sub in comments_df.groupby("comment_type", dropna=False):
        n_comments = len(sub)
        theme_counts: dict[str, int] = {}
        for themes in sub["themes"].astype(str):
            for theme in [t.strip() for t in themes.split(";") if t.strip()]:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        for theme, count in sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            summary_rows.append({
                "comment_type": comment_type,
                "theme": theme,
                "n_comments": int(count),
                "pct_comments": round(100 * count / n_comments, 1) if n_comments else np.nan,
            })

    return comments_df.reset_index(drop=True), pd.DataFrame(summary_rows)
