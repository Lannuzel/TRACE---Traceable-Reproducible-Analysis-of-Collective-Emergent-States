"""
stats.py — Statistical utility functions.

Z-scoring, entropy, coefficient of variation, safe division,
Pearson correlation with NaN handling, etc.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def safe_div(a: float, b: float) -> float:
    """Return a/b or NaN when b is zero/inf/nan."""
    if b is None or b == 0 or not np.isfinite(b):
        return np.nan
    return float(a / b)


def coef_var(x) -> float:
    """Coefficient of variation (CV = std/mean). Returns NaN if ill-defined."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    mu = x.mean()
    if mu == 0:
        return np.nan
    return float(x.std(ddof=0) / mu)


def shannon_entropy(x) -> float:
    """Shannon entropy in nats (base-e). x is a 1-D array of counts or probabilities.
    NOTE: malgré le nom historique, cette fonction utilise log naturel (nats), pas log2 (bits).
    Pour des bits, utiliser shannon_entropy_bits(). Pour une valeur normalisée [0,1], utiliser
    shannon_entropy_bits(x, normalize=True).
    Utilisée par participation_entropy_from_row() — valeurs typiques entre 0 et log(3)≈1.099.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    p = x / x.sum()
    return float(-(p * np.log(p)).sum())


def shannon_entropy_bits(x, normalize: bool = False) -> float:
    """Shannon entropy en log2 (bits).
    normalize=False : valeur brute en bits, max = log2(N).
    normalize=True  : valeur normalisée dans [0, 1] par division par log2(N).
    Pour N=3 participants : max brut = log2(3)≈1.585 bits ; max normalisé = 1.0.
    Préférer normalize=True pour des comparaisons inter-groupes de taille différente.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return np.nan
    p = x / x.sum()
    H = float(-(p * np.log2(p)).sum())
    if normalize and x.size > 1:
        H /= np.log2(float(x.size))
    return H


def zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a Series, returning NaN where undefined."""
    s = pd.to_numeric(s, errors="coerce")
    if not s.notna().any():
        return pd.Series(np.nan, index=s.index, dtype=float)

    mu = float(np.nanmean(s.values))
    sd = float(np.nanstd(s.values, ddof=0))

    if not np.isfinite(sd) or sd == 0:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s.notna()] = 0.0
        return out

    return (s - mu) / sd


def zscore_df(
    df: pd.DataFrame,
    cols: List[str],
    by: Optional[List[str]] = None,
    prefix: str = "z_",
) -> pd.DataFrame:
    """
    Z-score selected columns, optionally within groups defined by *by*.

    New columns are named {prefix}{original_column}.
    """
    df = df.copy()
    if by is None or len(by) == 0:
        for c in cols:
            if c in df.columns:
                df[prefix + c] = zscore_series(df[c])
        return df

    for c in cols:
        if c in df.columns:
            df[prefix + c] = df.groupby(by, dropna=False)[c].transform(zscore_series)
    return df


def nanmean_rows(arr: np.ndarray) -> np.ndarray:
    """Row-wise nanmean, returning NaN for all-NaN slices (no warning)."""
    with np.errstate(all="ignore"):
        m = np.nanmean(arr, axis=0)
    all_nan = np.all(np.isnan(arr), axis=0)
    m[all_nan] = np.nan
    return m


def clip01(s: pd.Series) -> pd.Series:
    """Clip a Series to [0, 1] after coercing to numeric."""
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def ratio(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Element-wise a / (b + eps), coerced to numeric."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / (b + eps)


def coalesce(*series_list: pd.Series) -> pd.Series:
    """Return the first non-NaN value across multiple Series (element-wise)."""
    out = None
    for s in series_list:
        if s is None:
            continue
        s = pd.to_numeric(s, errors="coerce")
        if out is None:
            out = s.copy()
        else:
            out = out.fillna(s)
    return out if out is not None else pd.Series(dtype=float)


def first_valid_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Return the first column from *candidates* that has at least one non-NaN value."""
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def first_valid_source(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Return a string Series naming which candidate column supplied
    the value for each row (provenance tracking).
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for c in candidates:
        if c in df.columns:
            mask = out.isna() & pd.to_numeric(df[c], errors="coerce").notna()
            out.loc[mask] = c
    return out
