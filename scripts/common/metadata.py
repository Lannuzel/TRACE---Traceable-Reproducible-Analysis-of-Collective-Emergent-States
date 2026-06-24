"""
metadata.py — Extract experimental metadata from directory paths.

All functions work on directory paths structured like:
    .../T1_BSI_A1/VR/S1/bim066/...
    .../T2_FISA_A5/PC/S2/bim074/...
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def extract_condition(path) -> Optional[str]:
    """Extract condition (VR / PC) from directory path."""
    parts = [p.upper() for p in Path(path).parts]
    if "VR" in parts:
        return "VR"
    if "PC" in parts:
        return "PC"
    return None


def extract_scenario(path) -> Optional[str]:
    """Extract scenario (S1 / S2 / ...) from directory path."""
    for p in Path(path).parts:
        if re.fullmatch(r"(?i)S\d+", p):
            return p.upper()
    return None


def extract_timepoint(path) -> str:
    """
    Extract timepoint (T1 / T2) from directory path.

    Detects folders like T1_BSI_A1, T2_FISA_A5, or bare T1/T2.
    """
    for p in Path(path).parts:
        u = p.upper()
        if u.startswith("T1_") or u == "T1":
            return "T1"
        if u.startswith("T2_") or u == "T2":
            return "T2"
    return "UNK"


def extract_all_metadata(path) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extract (condition, scenario, timepoint) from a directory path.

    Returns:
        (condition, scenario, timepoint) — any can be None/"UNK" if not found.
    """
    return extract_condition(path), extract_scenario(path), extract_timepoint(path)


def extract_group_base_id(group_id: str) -> str:
    """
    Strip trailing _N suffix: bim066_2 -> bim066.
    """
    return re.sub(r"_\d+$", "", group_id.strip())


def add_group_base_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'group_base_id' column derived from 'group_id'."""
    df = df.copy()
    if "group_id" not in df.columns:
        return df
    df["group_base_id"] = df["group_id"].astype(str).apply(extract_group_base_id)
    return df


def is_excluded_group(group_id: str, excluded: set) -> bool:
    """Check if a group_id should be excluded (exact match only)."""
    return group_id.strip() in excluded


def exclude_invalid_groups(
    df: pd.DataFrame,
    excluded: Optional[set] = None,
) -> pd.DataFrame:
    """
    Remove rows for excluded groups.

    Default: excludes bim065 (not bim065_2), bim002, bim032, bim075.
    """
    from .constants import EXCLUDED_GROUPS

    if excluded is None:
        excluded = EXCLUDED_GROUPS
    df = df.copy()
    if "group_id" not in df.columns:
        return df
    return df.loc[~df["group_id"].astype(str).str.strip().isin(excluded)].copy()


def upsert_meta_cols(
    df: pd.DataFrame,
    group_id: str,
    condition: Optional[str],
    scenario: Optional[str],
    timepoint: str = "UNK",
) -> pd.DataFrame:
    """Insert or update metadata columns (group_id, condition, scenario, timepoint)."""
    if df is None or df.empty:
        return df
    df = df.copy()

    def _upsert(col, val, pos):
        if col in df.columns:
            df[col] = val
        else:
            df.insert(min(pos, len(df.columns)), col, val)

    _upsert("group_id", group_id, 0)
    _upsert("condition", condition, 1)
    _upsert("scenario", scenario, 2)
    _upsert("timepoint", timepoint, 3)
    return df


def ensure_key_cols(
    df: pd.DataFrame,
    group_col_candidates: List[str],
    condition_col_candidates: List[str],
    scenario_col_candidates: List[str],
    timepoint_col_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Standardize key columns (group_id, condition, scenario, timepoint)
    by mapping from multiple candidate column names.
    """
    df = df.copy()
    cols = set(df.columns)

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    gcol = pick(group_col_candidates)
    ccol = pick(condition_col_candidates)
    scol = pick(scenario_col_candidates)

    if gcol is None:
        raise ValueError(
            f"Could not find group column among: {group_col_candidates}. "
            f"Found: {sorted(cols)[:60]}"
        )
    if ccol is None:
        raise ValueError(
            f"Could not find condition column among: {condition_col_candidates}. "
            f"Found: {sorted(cols)[:60]}"
        )

    df["group_id"] = df[gcol].astype(str).str.strip()
    df["condition"] = df[ccol].astype(str).str.strip().str.upper()
    df["scenario"] = (
        df[scol].astype(str).str.strip().str.upper() if scol else pd.NA
    )

    tcol = None
    if timepoint_col_candidates:
        tcol = pick(timepoint_col_candidates)

    if tcol is None:
        df["timepoint"] = "UNK"
    else:
        df["timepoint"] = (
            df[tcol]
            .astype(str)
            .replace({"nan": "UNK", "None": "UNK", "": "UNK"})
            .fillna("UNK")
            .str.upper()
        )

    return df
