"""
io_utils.py — Robust CSV loading and column normalization.

Handles EU-format (semicolon separator, comma decimal) and
auto-detection of separator styles.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import ROLES, PC_ROLE_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric, setting errors to NaN."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_csv_eu(path, **kwargs) -> pd.DataFrame:
    """Read EU-format CSV (semicolon separator, comma decimal)."""
    return pd.read_csv(path, sep=";", decimal=",", engine="python", **kwargs)


def read_csv_smart(path, sep: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Read CSV with auto-detection of separator.

    Tries semicolon first, falls back to comma if only 1 column.
    If *sep* is given, uses it directly.
    """
    if sep is not None:
        return pd.read_csv(path, sep=sep, engine="python", **kwargs)
    try:
        df = pd.read_csv(path, sep=";", engine="python", **kwargs)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=",", engine="python", **kwargs)


def read_csv_safe(path, **kwargs) -> pd.DataFrame:
    """
    Read CSV with fallback for broken headers / quoting issues.
    """
    try:
        return pd.read_csv(path, quoting=csv.QUOTE_MINIMAL, **kwargs)
    except Exception:
        logger.warning("Fallback reader (QUOTE_NONE) for %s", path)
        return pd.read_csv(
            path, quoting=csv.QUOTE_NONE, on_bad_lines="skip", **kwargs
        )


# ---------------------------------------------------------------------------
# Role-based file discovery
# ---------------------------------------------------------------------------

def find_pc_role_wavs(processed_dir: Path) -> Dict[str, Path]:
    """
    Locate per-role WAV files in a processed_openface directory.

    Naming convention: <role>__<hash>__audio.wav
    Maps modelisateur_c -> modelisateur.
    """
    wavs = list(processed_dir.glob("*__audio.wav"))
    if not wavs:
        raise FileNotFoundError(f"No '*__audio.wav' found in {processed_dir}")

    by_role: Dict[str, list] = {r: [] for r in ROLES}

    for p in wavs:
        prefix = p.name.lower().split("__", 1)[0]
        role = PC_ROLE_MAP.get(prefix)
        if role and role in by_role:
            by_role[role].append(p)

    missing = [r for r in ROLES if not by_role[r]]
    if missing:
        raise FileNotFoundError(
            f"Missing role(s) in {processed_dir}: {missing}"
        )

    return {r: max(by_role[r], key=lambda x: x.stat().st_mtime) for r in ROLES}


def find_pc_facs_files(group_dir: Path) -> Dict[str, Path]:
    """
    Locate per-role FACS CSV files under processed_openface/facs/.

    Maps modelisateur_c -> modelisateur.
    """
    facs_dir = group_dir / "processed_openface" / "facs"
    if not facs_dir.is_dir():
        raise FileNotFoundError(f"FACS directory not found: {facs_dir}")

    files = list(facs_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No FACS CSV found in {facs_dir}")

    by_role: Dict[str, list] = {r: [] for r in ROLES}

    for f in files:
        prefix = f.name.lower().split("__", 1)[0]
        role = PC_ROLE_MAP.get(prefix)
        if role and role in by_role:
            by_role[role].append(f)

    missing = [r for r in ROLES if not by_role[r]]
    if missing:
        raise FileNotFoundError(
            f"Missing FACS role(s) for {group_dir.name}: {missing}"
        )

    return {r: max(by_role[r], key=lambda x: x.stat().st_mtime) for r in ROLES}


def find_vr_role_file(group_dir: Path, role: str, pattern: str) -> Path:
    """
    Locate a single file matching *pattern* under group_dir/role/.

    Raises RuntimeError if 0 or >1 candidates found.
    """
    folder = group_dir / role
    cands = list(folder.glob(pattern))
    if len(cands) != 1:
        raise RuntimeError(
            f"{folder}: expected 1 '{pattern}', found {len(cands)}"
        )
    return cands[0]


def find_wav(folder: Path) -> Path:
    """Find the most recently modified WAV file in *folder*."""
    wavs = list(folder.glob("*.wav")) or list(folder.glob("**/*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV found in {folder}")
    return max(wavs, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Group directory discovery
# ---------------------------------------------------------------------------

def is_vr_group(path: Path) -> bool:
    """True if *path* contains sub-folders for all 3 roles."""
    return all((path / r).is_dir() for r in ROLES)


def is_pc_group(path: Path, file_pattern: str = "*__audio.wav") -> bool:
    """True if *path* contains processed_openface/ with matching files."""
    processed = path / "processed_openface"
    return processed.is_dir() and any(processed.glob(file_pattern))


def find_groups(root: Path, file_pattern: str = "*__audio.wav") -> List[Path]:
    """
    Recursively discover group directories under *root*.

    A group is detected as:
      - VR: contains calculateur/, modelisateur/, lecteur/ subfolders
      - PC: contains processed_openface/ with files matching *file_pattern*
    """
    groups = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if is_vr_group(path) or is_pc_group(path, file_pattern):
            groups.append(path)
    return sorted(set(groups))


def find_gaze_groups(root: Path) -> List[Path]:
    """
    Discover gaze group directories (contain *_EyeTrackingData.csv
    in at least one sub-folder).
    """
    import glob as _glob

    groups = []
    for cur, dirs, _ in root.walk() if hasattr(root, 'walk') else _walk_compat(root):
        cur_path = Path(cur) if not isinstance(cur, Path) else cur
        if _is_gaze_group(cur_path):
            groups.append(cur_path)
            if isinstance(dirs, list):
                dirs.clear()
    return sorted(groups)


def _is_gaze_group(path: Path) -> bool:
    import glob as _glob

    if not path.is_dir():
        return False
    for name in path.iterdir():
        if name.is_dir() and list(name.glob("*_EyeTrackingData.csv")):
            return True
    return False


def _walk_compat(root: Path):
    """os.walk-compatible wrapper for Path."""
    import os
    for cur, dirs, files in os.walk(root):
        yield Path(cur), dirs, files
