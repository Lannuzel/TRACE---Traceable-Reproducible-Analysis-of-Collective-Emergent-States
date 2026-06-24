# -*- coding: utf-8 -*-
"""
Riedl-inspired operationalization of collective intelligence indicators.

Constructs measured at the individual level (per member, per task):
  - skill_i         : contrafactual score -- what the member would have scored alone
  - contribution_i  : effective contribution -- items in the final group product
                      attributable to this member (correct items authored)
  - effort_task_i   : activity directly related to task manipulation (TASK_ATTRIBUTE)
  - effort_comm_i   : communication activity (COMMUNICATION_MESSAGE)
  - effort_total_i  : combined effort (task + comm)

Constructs at the group x task level:
  - skill_congruence        : alignment between skill_i and effort (parameterisable)
  - contribution_congruence : alignment between contribution_i and effort
  - strategy metrics        : coverage_any, coverage_correct_final, redundancy_rate

Constructs at the group level (summary):
  - aggregated means / sums of the above
  - distribution metrics    : effort_cv, effort_gini, contribution_cv, contribution_gini
  - top_member_effort_share, top_member_contribution_share

This script is methodologically closer to Riedl et al. (2021) than a pure
event-count pipeline, while remaining compatible with imperfect experimental logs.

Main methodological choices:
  - Effort is split into task vs. communication, then combined
  - Contribution is derived from snapshot "Last subject author" + correctness
  - Brainstorming and typing are treated explicitly as typing-oriented tasks
  - Typing-oriented tasks default to skill_congruence = 1.0
  - Skill congruence can use effort_task, effort_comm, or effort_total

Outputs (backward-compatible):
  riedl_indiv.csv
  riedl_group_task.csv
  riedl_group_summary.csv
  riedl_cfactor_merge.csv  (optional, if --c-scores provided)
  riedl_correlations.csv   (optional, if --c-scores provided)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Optional project import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from common.constants import EXCLUDED_GROUPS
except Exception:
    EXCLUDED_GROUPS = set()


# =========================================================
# Configuration
# =========================================================

CSV_COLUMNS = [
    "StudyPrefix", "SessionSuffix", "SessionStartDate", "SessionId",
    "TaskName", "SoloTask", "EventType", "EventContent", "Timestamp",
    "SenderSubjectExternalId", "ReceiverSubjectExternalId", "SummaryDescription"
]

EFFORT_EVENT_TYPES = {"TASK_ATTRIBUTE", "COMMUNICATION_MESSAGE"}
IGNORED_TASKS = {"RMET"}

# Only true typing tasks get congruence = 1.0 automatically
TASK_KEYS = {
    "brain_eq": ["Brainstorming_equation", "Brainstrorming_equation"],
    "brain_words": ["Brainstorming_word_P_N", "Brainstrorming_word_P_N"],
    "raven": ["MatrixSolvingN1_FR", "MatrixSolvingN1", "Matrices", "Raven"],
    "mem3": ["MemoryGrid1_FR", "MemoryGrid 3x3", "MemoryGrid1"],
    "mem5": ["MemoryGrid2_FR", "MemoryGrid 5x5", "MemoryGrid2"],
    "mem7": ["MemoryGrid3_FR", "MemoryGrid 7x7", "MemoryGrid3"],
    "sudoku": ["Sudoku_FR", "Sudoku"],
    "typing": ["TypingText_FR", "TypingText", "Dactylographie"],
}

# Typologie explicite des tâches pour audit méthodologique.
# Les tâches brainstorming sont rangées dans la même famille que la dactylographie
# car elles ne reposent pas sur un score individuel de justesse comparable aux
# tâches "correct-answer" et leur congruence est donc forcée à 1.0.
TASK_TYPE = {
    "brain_eq": "typing_oriented",
    "brain_words": "typing_oriented",
    "typing": "typing_oriented",
    "raven": "correct_answer",
    "mem3": "correct_answer",
    "mem5": "correct_answer",
    "mem7": "correct_answer",
    "sudoku": "correct_answer",
}
TYPING_TASK_KEYS = {key for key, kind in TASK_TYPE.items() if kind == "typing_oriented"}

DEFAULT_TOTALS = {
    "raven": 18,
    "mem3": 9,
    "mem5": 25,
    "mem7": 49,
}

RAVEN_NUM_TO_LETTER = {str(i): chr(ord("A") + i - 1) for i in range(1, 9)}


# =========================================================
# Parameters
# =========================================================

@dataclass
class AnalysisParams:
    effort_mode: str = "text_length"            # event_count | text_length
    congruence_mode: str = "clip_zero"         # raw | clip_zero | rescale01
    congruence_effort_scope: str = "effort_task"  # effort_task | effort_comm | effort_total
    sudoku_denominator: str = "editable"       # editable | full_grid
    open_task_strategy_mode: str = "undefined" # undefined | proxy_unique_tokens
    min_members_for_congruence: int = 2
    debug: bool = False


# =========================================================
# Helpers
# =========================================================

def log(msg: str, enabled: bool = True) -> None:
    if enabled:
        print(msg)


def safe_div(a: float, b: float) -> float:
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b)


def normalize_group_id(x: str) -> str:
    s = str(x).strip().lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return s
    return f"bim{int(m.group(1)):03d}"


def task_key_from_name(name: str) -> str:
    n = str(name)
    for k, aliases in TASK_KEYS.items():
        for a in aliases:
            if a.lower() in n.lower():
                return k
    return n


def sanitize(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s)).strip("_")


def safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_ignored_task(name: str) -> bool:
    n = str(name).lower()
    if n in {x.lower() for x in IGNORED_TASKS}:
        return True
    if "introduction" in n or "discussion" in n:
        return True
    return False


def _lin_to_rc(k: int) -> tuple[int, int]:
    k = int(k)
    r = (k - 1) // 9 + 1
    c = (k - 1) % 9 + 1
    return r, c


def normalize_text_token(s: str) -> str:
    s = str(s).strip().lower()
    s = (
        s.replace("\u00e0", "a").replace("\u00e2", "a").replace("\u00e4", "a")
         .replace("\u00e9", "e").replace("\u00e8", "e").replace("\u00ea", "e").replace("\u00eb", "e")
         .replace("\u00ee", "i").replace("\u00ef", "i")
         .replace("\u00f4", "o").replace("\u00f6", "o")
         .replace("\u00f9", "u").replace("\u00fb", "u").replace("\u00fc", "u")
         .replace("\u00e7", "c")
    )
    return s


def split_text_units(text: str) -> list[str]:
    """
    Découpe un texte libre en unités lexicales simples.

    Utilisé pour les tâches ouvertes (`typing_oriented`) afin d'obtenir une
    mesure de couverture exploitable sans réintroduire un score de justesse
    individuel artificiel.
    """
    if text is None:
        return []
    raw = str(text).replace("¶", " ").replace("|", " ").replace("\n", " ")
    units = re.findall(r"\w+", raw, flags=re.UNICODE)
    return [normalize_text_token(unit) for unit in units if str(unit).strip()]


def get_task_type(task_key: str) -> str:
    """Retourne la famille analytique d'une tâche TCI."""
    return TASK_TYPE.get(task_key, "correct_answer")


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = perfect equality, 1 = perfect inequality)."""
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or v.sum() == 0:
        return np.nan
    v = np.sort(v)
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def coeff_variation(values: np.ndarray) -> float:
    """CV = std / mean.  NaN if mean == 0 or < 2 values."""
    v = np.array(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan
    m = np.mean(v)
    if m == 0:
        return np.nan
    return float(np.std(v, ddof=1) / m)


# =========================================================
# CSV reading
# =========================================================

def read_eventlog(path: Path) -> pd.DataFrame:
    read_kwargs = dict(
        sep=";",
        skiprows=1,
        header=0,
        names=CSV_COLUMNS,
        dtype=str,
        encoding="utf-8",
        engine="python",
    )

    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception:
        df = pd.read_csv(
            path,
            **read_kwargs,
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )

    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    for c in ["SessionId", "TaskName", "EventType", "Timestamp",
              "SenderSubjectExternalId", "SummaryDescription", "EventContent", "SessionSuffix"]:
        df[c] = df[c].fillna("").astype(str)

    df["SessionId"] = df["SessionId"].str.extract(r"(\d+)", expand=False).fillna(df["SessionId"])
    df = df[(df["SessionId"] != "") & (df["TaskName"] != "")].copy()
    return df


def _read_any_csv(path: Path) -> pd.DataFrame:
    try:
        with open(path, "rb") as f:
            first = f.readline().decode("utf-8", errors="ignore").strip().lower()
        if first.startswith("sep="):
            delim = first.split("=", 1)[1].strip() or ";"
            return pd.read_csv(path, sep=delim, skiprows=1, encoding="utf-8", engine="python")
    except Exception:
        pass

    seps = [",", ";", "\t", "|"]
    encs = ["utf-8", "utf-8-sig", "latin1"]

    best_df = None
    best_cols = -1
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python")
                if df.shape[1] > best_cols:
                    best_df = df
                    best_cols = df.shape[1]
            except Exception:
                continue

    if best_df is not None:
        return best_df

    return pd.read_csv(path, engine="python")


def load_snapshot_dir(d: Optional[Path]) -> Dict[str, pd.DataFrame]:
    snaps: Dict[str, pd.DataFrame] = {}
    if not d or not d.exists():
        return snaps
    for f in d.glob("*.csv"):
        snaps[f.stem] = _read_any_csv(f)
    return snaps


def get_first_nonempty_snapshot(snaps: Dict[str, pd.DataFrame], keys: List[str]) -> Optional[pd.DataFrame]:
    for k in keys:
        if k in snaps:
            s = snaps[k]
            if isinstance(s, pd.DataFrame) and not s.empty:
                return s
    return None


# =========================================================
# Regex / extractors
# =========================================================

RE_FIELD = re.compile(r"(?:field|item|question|q)\s*(\d+)", re.I)
RE_LABEL_LETTER = re.compile(r"[:=]\s*([A-H])\.?\s*$", re.I)
RE_LABEL_DIGIT = re.compile(r"[:=]\s*([1-8])\s*\.?\s*$", re.I)
RE_CELL_A = re.compile(r"\br\s*(\d+)\s*c\s*(\d+)\b", re.I)
RE_CELL_B = re.compile(r"\b(\d+)[_\-x,;:\s]+(\d+)\b", re.I)
RE_NUM = re.compile(r"[:=]\s*([+-]?\d+)\s*$")
RE_CELL_LINEAR = re.compile(r"cell\s*(\d+)\s*[:=]\s*([+-]?\d+)\s*$", re.I)
RE_FIELD_LINEAR = re.compile(r"\bfield\s*(\d+)\b", re.I)


def extract_field_idx(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = RE_FIELD.search(s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(\d{1,3})\b", s)
    return int(m2.group(1)) if m2 else None


def extract_label_letter(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()

    m = RE_LABEL_LETTER.search(s)
    if m:
        return m.group(1).upper()

    m = RE_LABEL_DIGIT.search(s)
    if m:
        return RAVEN_NUM_TO_LETTER.get(m.group(1))

    tokens = re.findall(r"\b([A-Ha-h])\.?\b", s)
    if tokens:
        return tokens[-1].upper()

    digits = re.findall(r"\b([1-8])\b", s)
    if digits:
        return RAVEN_NUM_TO_LETTER.get(digits[-1])

    return None


def extract_cell_id(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()

    m = RE_FIELD_LINEAR.search(s)
    if m:
        k = int(m.group(1))
        if 1 <= k <= 81:
            r, c = _lin_to_rc(k)
            return f"r{r}c{c}"

    m = RE_CELL_LINEAR.search(s)
    if m:
        k = int(m.group(1))
        r, c = _lin_to_rc(k)
        return f"r{r}c{c}"

    m = RE_CELL_A.search(s)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        if 1 <= r <= 9 and 1 <= c <= 9:
            return f"r{r}c{c}"

    m = RE_CELL_B.search(s)
    if m:
        r, c = int(m.group(1)), int(m.group(2))
        if 1 <= r <= 9 and 1 <= c <= 9:
            return f"r{r}c{c}"

    return None


def extract_trailing_number(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = RE_NUM.search(s)
    if m:
        return int(m.group(1))
    toks = re.findall(r"([+-]?\d+)", s)
    return int(toks[-1]) if toks else None


def extract_trailing_token(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    if ":" in s:
        return s.split(":")[-1].strip()
    toks = re.split(r"[\s/]+", s.strip())
    return toks[-1] if toks else None


def _extract_cell_index_from_summary(s: str) -> Optional[int]:
    """Extract cell index from Typed in cell 26 :4 or Answered for cell 4 :paume."""
    if not isinstance(s, str):
        return None
    m = re.search(r"\bcell\s+(\d+)", s, re.I)
    return int(m.group(1)) if m else None


# =========================================================
# Effort (split: task vs. communication)
# =========================================================

def estimate_text_effort(row: pd.Series) -> int:
    """Approximate keystroke volume from textual content."""
    txt = row.get("EventContent", "") or row.get("SummaryDescription", "") or ""
    txt = str(txt)
    return len(txt)


def compute_effort(df_task: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    """
    Compute effort split into effort_task and effort_comm per subject.

    Returns DataFrame with columns:
      SubjectId, effort_task, effort_comm, effort_total,
      comm_n_messages, comm_text_length, effort_metric
    """
    g = df_task[df_task["EventType"].isin(EFFORT_EVENT_TYPES)].copy()
    if g.empty:
        return pd.DataFrame(columns=[
            "SubjectId", "effort_task", "effort_comm", "effort_total",
            "comm_n_messages", "comm_text_length", "effort_metric",
            "effort_value",
        ])

    g_task = g[g["EventType"] == "TASK_ATTRIBUTE"].copy()
    g_comm = g[g["EventType"] == "COMMUNICATION_MESSAGE"].copy()

    if params.effort_mode == "event_count":
        eff_task = (
            g_task.groupby("SenderSubjectExternalId")["EventType"]
            .size()
            .reset_index(name="effort_task")
        ) if not g_task.empty else pd.DataFrame(columns=["SenderSubjectExternalId", "effort_task"])

        eff_comm = (
            g_comm.groupby("SenderSubjectExternalId")["EventType"]
            .size()
            .reset_index(name="effort_comm")
        ) if not g_comm.empty else pd.DataFrame(columns=["SenderSubjectExternalId", "effort_comm"])

    elif params.effort_mode == "text_length":
        if not g_task.empty:
            g_task["_eff"] = g_task.apply(estimate_text_effort, axis=1)
            eff_task = (
                g_task.groupby("SenderSubjectExternalId")["_eff"]
                .sum()
                .reset_index(name="effort_task")
            )
        else:
            eff_task = pd.DataFrame(columns=["SenderSubjectExternalId", "effort_task"])

        if not g_comm.empty:
            g_comm["_eff"] = g_comm.apply(estimate_text_effort, axis=1)
            eff_comm = (
                g_comm.groupby("SenderSubjectExternalId")["_eff"]
                .sum()
                .reset_index(name="effort_comm")
            )
        else:
            eff_comm = pd.DataFrame(columns=["SenderSubjectExternalId", "effort_comm"])

    else:
        raise ValueError(f"Unknown effort_mode: {params.effort_mode}")

    # Communication proxies: n_messages and total text length
    if not g_comm.empty:
        comm_stats = g_comm.groupby("SenderSubjectExternalId").agg(
            comm_n_messages=("EventType", "size"),
            comm_text_length=("SummaryDescription", lambda s: sum(len(str(x)) for x in s)),
        ).reset_index()
    else:
        comm_stats = pd.DataFrame(columns=["SenderSubjectExternalId", "comm_n_messages", "comm_text_length"])

    # Get all unique subjects
    all_subjects = g["SenderSubjectExternalId"].unique()
    out = pd.DataFrame({"SenderSubjectExternalId": all_subjects})

    out = out.merge(eff_task, on="SenderSubjectExternalId", how="left")
    out = out.merge(eff_comm, on="SenderSubjectExternalId", how="left")
    out = out.merge(comm_stats, on="SenderSubjectExternalId", how="left")

    for col, dtype in [("effort_task", float), ("effort_comm", float),
                       ("comm_n_messages", int), ("comm_text_length", int)]:
        out[col] = out[col].infer_objects(copy=False).fillna(0).astype(dtype)
    out["effort_total"] = out["effort_task"] + out["effort_comm"]
    out["effort_metric"] = params.effort_mode

    # Backward-compatible column
    out["effort_value"] = out["effort_total"]

    return out.rename(columns={"SenderSubjectExternalId": "SubjectId"})


# =========================================================
# Scoring helpers
# =========================================================

def strategy_ratio(count_covered: int, total: Optional[int]) -> float:
    return safe_div(count_covered, total)


def build_group_last_actions(
    df_task: pd.DataFrame,
    key_cols: List[str],
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return last action per actor x key."""
    g = df_task.copy()
    g["_order"] = np.arange(len(g))
    if sort_cols is None:
        sort_cols = ["Timestamp", "_order"]
    sort_cols = [c for c in sort_cols if c in g.columns] + ["_order"]
    group_cols = ["SenderSubjectExternalId"] + key_cols
    g = g.sort_values(group_cols + sort_cols)
    return g.groupby(group_cols, as_index=False).tail(1)


def _find_snap_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find column name in snapshot, case-insensitive, strip spaces."""
    low = {}
    for c in df.columns:
        low[str(c).strip().lower()] = c
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


# =========================================================
# Contribution from Snapshot
# =========================================================

def compute_contribution_from_snapshot(
    snap: pd.DataFrame,
    task_key: str,
    expected_total: int,
) -> Optional[pd.DataFrame]:
    """
    Derive contribution_i from the task snapshot.

    The snapshot records the final state of each item/cell with:
      - "Last subject author": who made the final answer
      - "Answer" / "Ground Truth": whether it is correct

    contribution_i = number of correct final items authored by this member / total

    Returns DataFrame with SubjectId, contribution_correct, contribution_total,
    contribution_i, or None if snapshot lacks required columns.
    """
    if snap is None or snap.empty:
        return None

    author_col = _find_snap_col(snap, ["Last subject author", "last_subject_author", "LastAuthor"])
    answer_col = _find_snap_col(snap, ["Answer", "answer", "response"])
    gt_col = _find_snap_col(snap, ["Ground Truth", "ground_truth", "correct", "solution", "expected"])

    if author_col is None:
        return None

    df = snap.copy()
    df["_author"] = df[author_col].astype(str).str.strip()

    # Filter out rows with no author
    df = df[df["_author"].ne("") & df["_author"].ne("nan") & df["_author"].ne("NaN")].copy()
    if df.empty:
        return None

    if gt_col is not None and answer_col is not None:
        # Compare answer to ground truth
        ans = df[answer_col].astype(str).str.strip().str.lower()
        gt = df[gt_col].astype(str).str.strip().str.lower()

        if task_key in ("mem3", "mem5", "mem7"):
            ans = ans.map(normalize_text_token)
            gt = gt.map(normalize_text_token)

        df["_correct"] = (ans == gt).astype(int)

        # Mark NaN answers as incorrect
        df.loc[ans.isin(["", "nan", "none"]), "_correct"] = 0
    else:
        # No ground truth: count all final items as contribution
        df["_correct"] = 1

    denom = expected_total if expected_total > 0 else max(1, len(df))

    per_sub = (
        df.groupby("_author")
        .agg(
            contribution_correct=("_correct", "sum"),
            contribution_total=("_correct", "count"),
        )
        .reset_index()
        .rename(columns={"_author": "SubjectId"})
    )
    per_sub["contribution_i"] = per_sub["contribution_correct"] / denom

    return per_sub[["SubjectId", "contribution_correct", "contribution_total", "contribution_i"]]


# =========================================================
# Strategy (enhanced)
# =========================================================

def compute_enhanced_strategy(
    df_task: pd.DataFrame,
    key_col: str,
    correct_set: Optional[set],
    expected_total: Optional[int],
    final_correct: Optional[set] = None,
) -> Dict:
    """
    Compute enhanced strategy metrics:
      - coverage_any:            fraction of elements touched at least once
      - coverage_correct_final:  fraction of elements correctly resolved at final state
      - redundancy_rate:         proportion of actions on already-touched elements
      - strategy_ratio:          (legacy) = coverage_any
    """
    g = df_task.copy()
    g = g.dropna(subset=[key_col])

    if g.empty or expected_total is None or expected_total <= 0:
        return {
            "strategy_defined": 1 if expected_total and expected_total > 0 else 0,
            "strategy_elements_covered": 0,
            "strategy_total": expected_total,
            "strategy_ratio": 0.0 if expected_total and expected_total > 0 else np.nan,
            "coverage_any": 0.0 if expected_total and expected_total > 0 else np.nan,
            "coverage_correct_final": np.nan,
            "redundancy_rate": np.nan,
        }

    all_keys = g[key_col].astype(str)
    unique_touched = set(all_keys.unique())
    n_touched = len(unique_touched)
    n_total_actions = len(all_keys)

    coverage_any = n_touched / expected_total if expected_total > 0 else np.nan

    if final_correct is not None:
        coverage_correct_final = len(final_correct) / expected_total if expected_total > 0 else np.nan
    else:
        coverage_correct_final = np.nan

    # Redundancy: how many actions are on elements that were already touched
    if n_total_actions > 0:
        seen = set()
        redundant = 0
        for k in all_keys:
            if k in seen:
                redundant += 1
            seen.add(k)
        redundancy_rate = redundant / n_total_actions if n_total_actions > 0 else np.nan
    else:
        redundancy_rate = np.nan

    return {
        "strategy_defined": 1,
        "strategy_elements_covered": n_touched,
        "strategy_total": expected_total,
        "strategy_ratio": strategy_ratio(n_touched, expected_total),
        "coverage_any": coverage_any,
        "coverage_correct_final": coverage_correct_final,
        "redundancy_rate": redundancy_rate,
    }


# =========================================================
# Sudoku
# =========================================================

def sudoku_from_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    cell_col = None
    for cand in ["cell index", "cell_index", "cell", "cellid", "cell id"]:
        if cand in low:
            cell_col = low[cand]
            break

    gt_col = None
    for cand in ["ground truth", "ground_truth", "correct_value", "correct", "solution", "expected"]:
        if cand in low:
            gt_col = low[cand]
            break

    editable_col = None
    for cand in ["is_editable", "editable", "can_edit", "modifiable"]:
        if cand in low:
            editable_col = low[cand]
            break

    if cell_col is None or gt_col is None:
        raise ValueError(f"Sudoku snapshot missing required columns. Found: {list(df.columns)}")

    out = df[[cell_col, gt_col]].rename(columns={cell_col: "cell_index", gt_col: "correct_value"}).copy()
    out["cell_index"] = pd.to_numeric(out["cell_index"], errors="coerce")
    out["correct_value"] = pd.to_numeric(out["correct_value"], errors="coerce")
    out = out.dropna(subset=["cell_index", "correct_value"]).copy()

    out["cell_index"] = out["cell_index"].astype(int)
    is_zero_based = (out["cell_index"].min() == 0) or (out["cell_index"].max() <= 80)
    lin = (out["cell_index"] + 1) if is_zero_based else out["cell_index"]

    out["cell_id"] = lin.map(lambda k: f"r{_lin_to_rc(k)[0]}c{_lin_to_rc(k)[1]}")
    out["correct_value"] = out["correct_value"].astype(int)

    if editable_col is not None:
        tmp = pd.to_numeric(df.loc[out.index, editable_col], errors="coerce").fillna(0)
        out["is_editable"] = tmp.astype(int).clip(0, 1)
    else:
        out["is_editable"] = 1

    return out[["cell_id", "correct_value", "is_editable"]]


def score_sudoku(
    df_task: pd.DataFrame,
    snap: pd.DataFrame,
    params: AnalysisParams,
) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    """Returns (per_sub_skill, strategy_dict, contribution_df)."""
    key = sudoku_from_snapshot(snap)
    editable_cells = set(key.loc[key["is_editable"] == 1, "cell_id"].astype(str))

    denom = 81
    if params.sudoku_denominator == "editable":
        denom = int(len(editable_cells)) if editable_cells else 81

    g = df_task[df_task["EventType"] == "TASK_ATTRIBUTE"].copy()
    g["_cell"] = g["SummaryDescription"].map(extract_cell_id)
    g["_val"] = g["SummaryDescription"].map(extract_trailing_number)
    g = g.dropna(subset=["_cell", "_val"]).copy()

    if g.empty:
        per_sub = pd.DataFrame(columns=["SubjectId", "correct", "total", "skill_i"])
        strat = compute_enhanced_strategy(g, "_cell", None, denom)
        contrib = compute_contribution_from_snapshot(snap, "sudoku", denom)
        return per_sub, strat, contrib

    last = build_group_last_actions(g, ["_cell"])
    key_map = dict(zip(key["cell_id"].astype(str), key["correct_value"].astype(int)))
    last["_cell"] = last["_cell"].astype(str)
    last["_editable"] = last["_cell"].isin(editable_cells)
    last["_is_correct"] = last.apply(
        lambda r: int(r["_editable"] and int(r["_val"]) == int(key_map.get(r["_cell"], 1e9))),
        axis=1
    )

    per_sub = (
        last[last["_editable"]]
        .groupby("SenderSubjectExternalId")["_is_correct"]
        .agg(correct="sum", total="count")
        .reset_index()
        .rename(columns={"SenderSubjectExternalId": "SubjectId"})
    )
    per_sub["skill_i"] = per_sub["correct"] / denom if denom > 0 else np.nan

    # Final correct cells for enhanced strategy
    final_correct = set(
        last.loc[last["_editable"] & (last["_is_correct"] == 1), "_cell"].astype(str)
    )

    strat = compute_enhanced_strategy(g, "_cell", editable_cells, denom, final_correct)

    # Contribution from snapshot
    contrib = compute_contribution_from_snapshot(snap, "sudoku", denom)

    return per_sub, strat, contrib


# =========================================================
# Memory
# =========================================================

def memory_truth_from_snapshot(snap: pd.DataFrame) -> set[str]:
    gt_col = _find_snap_col(snap, ["Ground Truth", "ground_truth", "correct"])
    if gt_col is None:
        col = next((c for c in snap.columns if snap[c].notna().sum() > 0), snap.columns[0])
        return {normalize_text_token(w) for w in snap[col].dropna().astype(str)}
    return {normalize_text_token(w) for w in snap[gt_col].dropna().astype(str)}


def score_memory(
    df_task: pd.DataFrame,
    snap: pd.DataFrame,
    expected_total: int,
) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    """Returns (per_sub_skill, strategy_dict, contribution_df)."""
    truth = memory_truth_from_snapshot(snap)

    g = df_task[df_task["EventType"] == "TASK_ATTRIBUTE"].copy()
    g["_word"] = g["SummaryDescription"].map(extract_trailing_token)
    g["_word"] = g["_word"].astype(str).map(normalize_text_token)
    g = g[g["_word"].str.len() > 0].copy()

    if g.empty:
        per_sub = pd.DataFrame(columns=["SubjectId", "correct", "total", "skill_i"])
        strat = compute_enhanced_strategy(g.assign(_word=[]), "_word", truth, expected_total)
        contrib = compute_contribution_from_snapshot(snap, "mem", expected_total)
        return per_sub, strat, contrib

    correct_unique = g[g["_word"].isin(truth)].groupby("SenderSubjectExternalId")["_word"].nunique().reset_index()
    total_unique = g.groupby("SenderSubjectExternalId")["_word"].nunique().reset_index()

    per_sub = total_unique.merge(correct_unique, on="SenderSubjectExternalId", how="left", suffixes=("_total", "_correct"))
    per_sub["correct"] = per_sub["_word_correct"].fillna(0).astype(int)
    per_sub["total"] = per_sub["_word_total"].fillna(0).astype(int)
    per_sub["skill_i"] = per_sub["correct"] / expected_total if expected_total > 0 else np.nan
    per_sub = per_sub.rename(columns={"SenderSubjectExternalId": "SubjectId"})
    per_sub = per_sub[["SubjectId", "correct", "total", "skill_i"]]

    # Contribution from snapshot
    contrib = compute_contribution_from_snapshot(snap, "mem", expected_total)

    # Determine final correct set for strategy
    answer_col = _find_snap_col(snap, ["Answer", "answer"])
    gt_col = _find_snap_col(snap, ["Ground Truth", "ground_truth"])
    final_correct = None
    if answer_col and gt_col:
        ans_vals = snap[answer_col].astype(str).map(normalize_text_token)
        gt_vals = snap[gt_col].astype(str).map(normalize_text_token)
        correct_mask = ans_vals == gt_vals
        final_correct = set(gt_vals[correct_mask])

    strat = compute_enhanced_strategy(g, "_word", truth, expected_total, final_correct)

    return per_sub, strat, contrib


# =========================================================
# Raven / Matrix
# =========================================================

def matrix_from_snapshot(snap: pd.DataFrame) -> pd.DataFrame:
    df = snap.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}

    item_col = None
    for cand in ["field index", "item", "field", "question", "q", "index", "id"]:
        if cand in low:
            item_col = low[cand]
            break
    if item_col is None:
        item_col = df.columns[0]

    corr_col = None
    for cand in ["ground truth", "correct_label", "correct", "answer", "solution", "expected", "target"]:
        if cand in low:
            corr_col = low[cand]
            break
    if corr_col is None:
        corr_col = df.columns[1] if len(df.columns) >= 2 else df.columns[0]

    out = df[[item_col, corr_col]].rename(columns={item_col: "item_raw", corr_col: "correct_raw"}).copy()
    out["item"] = out["item_raw"].astype(str).str.extract(r"(\d+)", expand=False)
    out["item"] = pd.to_numeric(out["item"], errors="coerce").astype("Int64")

    cr = out["correct_raw"].astype(str).str.strip().str.upper().str.rstrip(".")
    cr_digit = cr.str.extract(r"^([1-8])$", expand=False)
    cr = cr.where(cr_digit.isna(), cr_digit.map(RAVEN_NUM_TO_LETTER))
    cr_letter = cr.str.extract(r"\b([A-H])\b", expand=False)
    cr = cr.where(cr_letter.isna(), cr_letter)

    out["correct_label"] = cr
    out = out.dropna(subset=["item", "correct_label"]).copy()
    out["correct_label"] = out["correct_label"].astype(str).str.strip().str.upper().str.rstrip(".")
    out = out[out["correct_label"].str.match(r"^[A-H]$")]
    return out[["item", "correct_label"]]


def score_matrix(
    df_task: pd.DataFrame,
    snap: pd.DataFrame,
    expected_total: int,
) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    """Returns (per_sub_skill, strategy_dict, contribution_df)."""
    key = matrix_from_snapshot(snap)
    key_map = dict(zip(key["item"].astype(int), key["correct_label"].astype(str)))

    g = df_task[df_task["EventType"] == "TASK_ATTRIBUTE"].copy()
    g["_item"] = g["SummaryDescription"].map(extract_field_idx)
    g["_label"] = g["SummaryDescription"].map(extract_label_letter)
    g = g.dropna(subset=["_item", "_label"]).copy()

    if g.empty:
        per_sub = pd.DataFrame(columns=["SubjectId", "correct", "total", "skill_i"])
        strat = compute_enhanced_strategy(g, "_item", None, expected_total)
        contrib = compute_contribution_from_snapshot(snap, "raven", expected_total)
        return per_sub, strat, contrib

    last = build_group_last_actions(g, ["_item"])
    last["_item"] = pd.to_numeric(last["_item"], errors="coerce").astype("Int64")
    last = last[last["_item"].between(1, expected_total * 2)].copy()
    last["_label"] = last["_label"].astype(str).str.strip().str.upper().str.rstrip(".")
    last["_is_correct"] = last.apply(
        lambda r: int(str(r["_label"]) == str(key_map.get(int(r["_item"]), ""))) if pd.notna(r["_item"]) else np.nan,
        axis=1
    )

    per_sub = (
        last.groupby("SenderSubjectExternalId")["_is_correct"]
        .agg(correct="sum", total="count")
        .reset_index()
        .rename(columns={"SenderSubjectExternalId": "SubjectId"})
    )
    per_sub["skill_i"] = per_sub["correct"] / expected_total if expected_total > 0 else np.nan

    # Final correct items for strategy
    final_correct = set(
        last.loc[last["_is_correct"] == 1, "_item"].astype(str)
    )
    g["_item"] = g["_item"].astype(str)
    strat = compute_enhanced_strategy(g, "_item", set(str(i) for i in key_map.keys()), expected_total, final_correct)

    # Contribution from snapshot
    contrib = compute_contribution_from_snapshot(snap, "raven", expected_total)

    return per_sub, strat, contrib


# =========================================================
# Open tasks: Brainstorming (NOT typing-oriented)
# =========================================================

def score_brainstorming(
    df_task: pd.DataFrame,
    task_key: str,
    params: AnalysisParams,
    snap: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    """
    Brainstorming scoring (typing-oriented).

    Conformément à l'adaptation Riedl demandée :
    - skill_i reste indéfini (NaN) ;
    - l'effort individuel sera réinterprété en aval comme nombre d'idées uniques ;
    - la stratégie de groupe repose sur la couverture finale d'idées/éléments,
      puis sera normalisée sur le maximum observé dans l'échantillon pour la tâche.
    """
    g = df_task[df_task["EventType"] == "TASK_ATTRIBUTE"].copy()
    g["_idea"] = g["SummaryDescription"].map(extract_trailing_token).astype(str).map(normalize_text_token)
    g = g[g["_idea"].str.len() > 1].copy()

    per_sub = (
        g.groupby("SenderSubjectExternalId")["_idea"]
        .nunique()
        .reset_index(name="total")
        .rename(columns={"SenderSubjectExternalId": "SubjectId"})
    )
    per_sub["correct"] = np.nan
    per_sub["skill_i"] = np.nan

    strategy_elements_covered = np.nan
    strategy_reference = "sample_max_by_task_pending"
    if snap is not None and not snap.empty:
        low = {str(c).strip().lower(): c for c in snap.columns}
        if task_key == "brain_eq":
            answer_col = low.get("answer")
            if answer_col is not None:
                answers = (
                    snap[answer_col]
                    .dropna()
                    .astype(str)
                    .map(normalize_text_token)
                )
                strategy_elements_covered = float((answers.str.len() > 0).sum())
                strategy_reference = "snapshot_final_equations"
        else:
            answer_col = low.get("answer")
            if answer_col is not None:
                units: set[str] = set()
                for value in snap[answer_col].dropna().astype(str):
                    units.update({unit for unit in split_text_units(value) if unit})
                strategy_elements_covered = float(len(units))
                strategy_reference = "snapshot_final_ideas"

    if pd.isna(strategy_elements_covered) and params.open_task_strategy_mode == "proxy_unique_tokens":
        strategy_elements_covered = float(g["_idea"].nunique())
        strategy_reference = "eventlog_unique_tokens"

    if pd.notna(strategy_elements_covered):
        strat = {
            "strategy_defined": 1,
            "strategy_elements_covered": strategy_elements_covered,
            "strategy_total": np.nan,
            "strategy_ratio": np.nan,
            "coverage_any": np.nan,
            "coverage_correct_final": np.nan,
            "redundancy_rate": np.nan,
            "strategy_reference": strategy_reference,
            "task_quantity_total": strategy_elements_covered,
            "task_quantity_unit": "n_ideas",
        }
    else:
        strat = {
            "strategy_defined": 0,
            "strategy_elements_covered": np.nan,
            "strategy_total": np.nan,
            "strategy_ratio": np.nan,
            "coverage_any": np.nan,
            "coverage_correct_final": np.nan,
            "redundancy_rate": np.nan,
            "strategy_reference": "undefined",
            "task_quantity_total": np.nan,
            "task_quantity_unit": "n_ideas",
        }

    # No contribution for brainstorming (no ground truth)
    return per_sub[["SubjectId", "correct", "total", "skill_i"]], strat, None


# =========================================================
# Typing
# =========================================================

def score_typing(
    df_task: pd.DataFrame,
    params: AnalysisParams,
    snap: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    g = df_task[df_task["EventType"] == "TASK_ATTRIBUTE"].copy()
    per_sub = (
        g.groupby("SenderSubjectExternalId")["EventType"]
        .size()
        .reset_index(name="total")
        .rename(columns={"SenderSubjectExternalId": "SubjectId"})
    )
    per_sub["correct"] = np.nan
    per_sub["skill_i"] = np.nan

    strategy_elements_covered = np.nan
    strategy_total = np.nan
    quantity_total = np.nan
    strategy_reference = "sample_max_by_task_pending"
    if snap is not None and not snap.empty:
        low = {str(c).strip().lower(): c for c in snap.columns}
        answer_col = low.get("answer")
        if answer_col is not None and snap[answer_col].notna().any():
            answer_text = " ".join(snap[answer_col].dropna().astype(str).tolist())
            answer_norm = answer_text.replace("¶", "").replace("|", "").replace("\n", "").strip()
            quantity_total = float(len(answer_norm))
            strategy_elements_covered = quantity_total
            strategy_reference = "snapshot_answer_char_count"
        gt_col = low.get("ground truth")
        if gt_col is not None and snap[gt_col].notna().any():
            gt_text = " ".join(snap[gt_col].dropna().astype(str).tolist())
            gt_norm = gt_text.replace("¶", "").replace("|", "").replace("\n", "").strip()
            strategy_total = float(len(gt_norm))
    if pd.isna(strategy_elements_covered):
        strategy_elements_covered = float(len(g))
        quantity_total = strategy_elements_covered
        strategy_reference = "eventlog_action_count"

    strat = {
        "strategy_defined": 1,
        "strategy_elements_covered": strategy_elements_covered,
        "strategy_total": strategy_total,
        "strategy_ratio": strategy_ratio(int(strategy_elements_covered), strategy_total) if pd.notna(strategy_total) else np.nan,
        "coverage_any": strategy_ratio(int(strategy_elements_covered), strategy_total) if pd.notna(strategy_total) else np.nan,
        "coverage_correct_final": np.nan,
        "redundancy_rate": np.nan,
        "strategy_reference": strategy_reference,
        "task_quantity_total": quantity_total,
        "task_quantity_unit": "n_chars",
    }
    return per_sub[["SubjectId", "correct", "total", "skill_i"]], strat, None


# =========================================================
# Merge + congruence
# =========================================================

def merge_effort_skill(
    effort_df: pd.DataFrame,
    per_sub_skill: Optional[pd.DataFrame],
    contribution_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge effort, skill, and contribution data per subject."""
    per_sub = effort_df.copy()

    if per_sub_skill is not None and not per_sub_skill.empty:
        per_sub = per_sub.merge(per_sub_skill, on="SubjectId", how="left")

    if contribution_df is not None and not contribution_df.empty:
        per_sub = per_sub.merge(contribution_df, on="SubjectId", how="left")

    for col in ("correct", "total", "skill_i"):
        if col not in per_sub.columns:
            per_sub[col] = np.nan

    for col in ("contribution_correct", "contribution_total", "contribution_i"):
        if col not in per_sub.columns:
            per_sub[col] = np.nan

    numeric_cols = [
        "effort_task", "effort_comm", "effort_total", "effort_value",
        "correct", "total", "skill_i",
        "contribution_correct", "contribution_total", "contribution_i",
    ]
    for col in numeric_cols:
        if col in per_sub.columns:
            per_sub[col] = pd.to_numeric(per_sub[col], errors="coerce")

    # Compute share columns
    total_effort_task = per_sub["effort_task"].sum()
    total_effort_total = per_sub["effort_total"].sum()
    total_contrib = per_sub["contribution_correct"].sum()

    per_sub["effort_share_task"] = per_sub["effort_task"] / total_effort_task if total_effort_task > 0 else np.nan
    per_sub["effort_share_total"] = per_sub["effort_total"] / total_effort_total if total_effort_total > 0 else np.nan
    per_sub["contribution_share"] = per_sub["contribution_correct"] / total_contrib if pd.notna(total_contrib) and total_contrib > 0 else np.nan

    return per_sub


def postprocess_congruence_value(r: float, mode: str) -> float:
    if pd.isna(r):
        return np.nan
    if mode == "raw":
        return float(r)
    if mode == "clip_zero":
        return float(max(0.0, r))
    if mode == "rescale01":
        return float((r + 1.0) / 2.0)
    raise ValueError(f"Unknown congruence mode: {mode}")


def _compute_pearson_congruence(
    x: pd.Series,
    y: pd.Series,
    params: AnalysisParams,
) -> float:
    """Pearson correlation between x and y, with postprocessing."""
    mask = x.notna() & y.notna()
    xm = x[mask].astype(float)
    ym = y[mask].astype(float)

    if len(xm) < params.min_members_for_congruence:
        return np.nan

    if xm.nunique() <= 1 and ym.nunique() <= 1:
        return 1.0
    if xm.nunique() <= 1 or ym.nunique() <= 1:
        return np.nan

    try:
        r, _ = pearsonr(xm, ym)
        return postprocess_congruence_value(float(r), params.congruence_mode)
    except Exception:
        return np.nan


def compute_skill_congruence(indiv_df: pd.DataFrame, task_key: str, params: AnalysisParams) -> float:
    """
    Task-wise alignment between skill_i and the selected effort scope.

    For typing-oriented tasks, congruence defaults to 1.0.
    """
    if task_key in TYPING_TASK_KEYS:
        return 1.0

    if indiv_df is None or indiv_df.empty:
        return np.nan

    effort_col = params.congruence_effort_scope
    if effort_col not in indiv_df.columns:
        effort_col = "effort_total"

    x = pd.to_numeric(indiv_df.get("skill_i"), errors="coerce")
    y = pd.to_numeric(indiv_df.get(effort_col), errors="coerce")

    return _compute_pearson_congruence(x, y, params)


def compute_contribution_congruence(indiv_df: pd.DataFrame, task_key: str, params: AnalysisParams) -> float:
    """Alignment between contribution_i and the selected effort scope."""
    if task_key in TYPING_TASK_KEYS:
        return 1.0

    if indiv_df is None or indiv_df.empty:
        return np.nan

    effort_col = params.congruence_effort_scope
    if effort_col not in indiv_df.columns:
        effort_col = "effort_total"

    x = pd.to_numeric(indiv_df.get("contribution_i"), errors="coerce")
    y = pd.to_numeric(indiv_df.get(effort_col), errors="coerce")

    return _compute_pearson_congruence(x, y, params)


# =========================================================
# Distribution metrics (group x task level)
# =========================================================

def compute_distribution_metrics(indiv_df: pd.DataFrame) -> Dict:
    """Compute effort and contribution distribution metrics for one task."""
    result = {}

    for prefix, col in [("effort_task", "effort_task"), ("effort_total", "effort_total"),
                         ("contribution", "contribution_correct")]:
        vals = pd.to_numeric(indiv_df.get(col), errors="coerce").dropna().values
        result[f"{prefix}_cv"] = coeff_variation(vals)
        result[f"{prefix}_gini"] = gini_coefficient(vals)

        if len(vals) > 0 and vals.sum() > 0:
            shares = vals / vals.sum()
            result[f"top_member_{prefix}_share"] = float(shares.max())
        else:
            result[f"top_member_{prefix}_share"] = np.nan

    return result


def apply_typing_oriented_individual_overrides(indiv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajuste l'effort individuel pour les tâches ouvertes.

    Pour les tâches `typing_oriented`, on distingue explicitement :
    - les cas où une quantité individuelle est observable dans les traces ;
    - les cas où seule la quantité de groupe est observable via le snapshot final.

    Cela permet de rester fidèle à Riedl et al. (2021) : on n'encode pas
    artificiellement un "zéro effort" lorsqu'en réalité la contribution
    individuelle n'est simplement pas mesurable dans les logs.
    """
    if indiv_df is None or indiv_df.empty or "TaskKey" not in indiv_df.columns:
        return indiv_df

    out = indiv_df.copy()
    if "typing_quantity_individual_observed" not in out.columns:
        out["typing_quantity_individual_observed"] = np.nan

    brainstorm_mask = out["TaskKey"].isin(["brain_eq", "brain_words"])
    observed_brain_mask = brainstorm_mask & out["total"].notna()
    unobserved_brain_mask = brainstorm_mask & ~out["total"].notna()

    if observed_brain_mask.any():
        unique_ideas = pd.to_numeric(out.loc[observed_brain_mask, "total"], errors="coerce").fillna(0.0)
        out.loc[observed_brain_mask, "effort_task"] = unique_ideas
        out.loc[observed_brain_mask, "effort_total"] = (
            unique_ideas
            + pd.to_numeric(out.loc[observed_brain_mask, "effort_comm"], errors="coerce").fillna(0.0)
        )
        out.loc[observed_brain_mask, "effort_value"] = out.loc[observed_brain_mask, "effort_total"]
        out.loc[observed_brain_mask, "effort_metric"] = "unique_ideas_plus_comm"
        out.loc[observed_brain_mask, "typing_quantity_individual_observed"] = 1

        total_effort_task = pd.to_numeric(out.loc[observed_brain_mask, "effort_task"], errors="coerce").sum()
        total_effort_total = pd.to_numeric(out.loc[observed_brain_mask, "effort_total"], errors="coerce").sum()
        out.loc[observed_brain_mask, "effort_share_task"] = (
            pd.to_numeric(out.loc[observed_brain_mask, "effort_task"], errors="coerce") / total_effort_task
            if total_effort_task > 0 else np.nan
        )
        out.loc[observed_brain_mask, "effort_share_total"] = (
            pd.to_numeric(out.loc[observed_brain_mask, "effort_total"], errors="coerce") / total_effort_total
            if total_effort_total > 0 else np.nan
        )

    # `brain_words` et `typing` n'ont pas de quantité individuelle fiable dans les
    # logs actuels ; on l'explicite comme non observée au lieu de laisser 0.0.
    unobserved_mask = unobserved_brain_mask | out["TaskKey"].eq("typing")
    if unobserved_mask.any():
        comm_only = pd.to_numeric(out.loc[unobserved_mask, "effort_comm"], errors="coerce").fillna(0.0)
        out.loc[unobserved_mask, "effort_task"] = np.nan
        out.loc[unobserved_mask, "effort_total"] = comm_only
        out.loc[unobserved_mask, "effort_value"] = comm_only
        out.loc[unobserved_mask, "effort_metric"] = "comm_only_task_quantity_unobserved"
        out.loc[unobserved_mask, "effort_share_task"] = np.nan
        total_effort_total = pd.to_numeric(out.loc[unobserved_mask, "effort_total"], errors="coerce").sum()
        out.loc[unobserved_mask, "effort_share_total"] = (
            pd.to_numeric(out.loc[unobserved_mask, "effort_total"], errors="coerce") / total_effort_total
            if total_effort_total > 0 else np.nan
        )
        out.loc[unobserved_mask, "typing_quantity_individual_observed"] = 0

    return out


def normalize_typing_oriented_strategy(group_task: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise la stratégie des tâches `typing_oriented` sur le maximum observé
    dans l'échantillon, tâche par tâche.
    """
    if group_task is None or group_task.empty or "TaskKey" not in group_task.columns:
        return group_task

    out = group_task.copy()
    if "task_type" not in out.columns:
        out["task_type"] = out["TaskKey"].map(get_task_type)
    if "strategy_reference" not in out.columns:
        out["strategy_reference"] = np.nan

    mask = out["task_type"] == "typing_oriented"
    if not mask.any():
        return out

    for task_key, idx in out.loc[mask].groupby("TaskKey").groups.items():
        raw = pd.to_numeric(out.loc[idx, "strategy_elements_covered"], errors="coerce")
        max_raw = raw.max()
        existing_total = pd.to_numeric(out.loc[idx, "strategy_total"], errors="coerce")
        sample_idx = existing_total[existing_total.isna()].index
        out.loc[sample_idx, "strategy_total"] = max_raw
        if pd.notna(max_raw) and max_raw > 0:
            out.loc[sample_idx, "strategy_ratio"] = raw.loc[sample_idx] / max_raw
            out.loc[sample_idx, "coverage_any"] = raw.loc[sample_idx] / max_raw
            out.loc[idx, "strategy_defined"] = 1
            out.loc[idx, "strategy_reference"] = (
                out.loc[idx, "strategy_reference"].fillna("").astype(str) + "|sample_max_by_task"
            ).str.strip("|")
        else:
            out.loc[sample_idx, "strategy_ratio"] = np.nan
            out.loc[sample_idx, "coverage_any"] = np.nan

    return out


def add_parent_population_norms(group_summary: pd.DataFrame, reference_full_csv: Path | None) -> pd.DataFrame:
    """
    Ajoute des variantes de normalisation basées sur la population parente Riedl.

    - `effort_norm_pop` n'utilise `TeamEffort` de `full.csv` que si l'echelle
      apparait compatible avec un total d'effort brut. Si `TeamEffort` est deja
      sur une echelle reduite 0-1 (cas observe dans `full.csv` courant), on
      conserve le fallback echantillon afin d'eviter une pseudo-normalisation
      non interpretable.
    - `strategy_norm_pop` s'appuie sur `TeamCoverage` de `full.csv` quand disponible,
      sinon sur le maximum échantillon de `strategy_ratio_mean`.
    """
    if group_summary is None or group_summary.empty:
        return group_summary

    out = group_summary.copy()
    out["effort_norm_pop_source"] = "sample"
    out["strategy_norm_pop_source"] = "sample"

    sample_eff_max = pd.to_numeric(out["effort_task_sum"], errors="coerce").max()
    sample_strat_max = pd.to_numeric(out["strategy_ratio_mean"], errors="coerce").max()
    eff_ref = sample_eff_max
    strat_ref = sample_strat_max

    if reference_full_csv is not None and reference_full_csv.exists():
        try:
            ref = pd.read_csv(reference_full_csv, encoding="utf-8")
            ref_eff = pd.to_numeric(ref.get("TeamEffort"), errors="coerce").max()
            ref_strat = pd.to_numeric(ref.get("TeamCoverage"), errors="coerce").max()
            # `TeamEffort` du full.csv Riedl courant est deja sur une echelle
            # reduite (~0-0.5), donc non comparable a `effort_task_sum` brut.
            # On n'active la reference parentale que si l'echelle est compatible
            # avec un total brut (> 1).
            if pd.notna(ref_eff) and ref_eff > 1:
                eff_ref = ref_eff
                out["effort_norm_pop_source"] = "Riedl2021"
            elif pd.notna(ref_eff) and ref_eff > 0:
                out["effort_norm_pop_source"] = "sample_parent_scale_incompatible"
            if pd.notna(ref_strat) and ref_strat > 0:
                strat_ref = ref_strat
                out["strategy_norm_pop_source"] = "Riedl2021"
        except Exception:
            pass

    out["effort_norm_pop"] = (
        pd.to_numeric(out["effort_task_sum"], errors="coerce") / eff_ref
        if pd.notna(eff_ref) and eff_ref > 0 else np.nan
    )
    out["strategy_norm_pop"] = (
        pd.to_numeric(out["strategy_ratio_mean"], errors="coerce") / strat_ref
        if pd.notna(strat_ref) and strat_ref > 0 else np.nan
    )
    return out


# =========================================================
# One task / group processing
# =========================================================

def process_one_task(
    group_id: str,
    session_id: str,
    task_name: str,
    df_task: pd.DataFrame,
    snaps: Dict[str, pd.DataFrame],
    params: AnalysisParams,
) -> Tuple[pd.DataFrame, Dict]:
    tkey = task_key_from_name(task_name)
    task_type = get_task_type(tkey)
    effort = compute_effort(df_task, params)

    per_sub_skill: Optional[pd.DataFrame] = None
    contribution_df: Optional[pd.DataFrame] = None
    strat = {
        "strategy_defined": 0,
        "strategy_elements_covered": np.nan,
        "strategy_total": np.nan,
        "strategy_ratio": np.nan,
        "coverage_any": np.nan,
        "coverage_correct_final": np.nan,
        "redundancy_rate": np.nan,
    }

    if tkey == "sudoku":
        snap = get_first_nonempty_snapshot(snaps, ["Sudoku_FR"])
        if snap is not None:
            per_sub_skill, strat, contribution_df = score_sudoku(df_task, snap, params)

    elif tkey in ("mem3", "mem5", "mem7"):
        snap_name = {"mem3": "MemoryGrid1_FR", "mem5": "MemoryGrid2_FR", "mem7": "MemoryGrid3_FR"}[tkey]
        snap = get_first_nonempty_snapshot(snaps, [snap_name])
        if snap is not None:
            per_sub_skill, strat, contribution_df = score_memory(df_task, snap, DEFAULT_TOTALS[tkey])

    elif tkey == "raven":
        snap = get_first_nonempty_snapshot(snaps, ["MatrixSolvingN1_FR", "MatrixSolvingN1"])
        if snap is not None:
            per_sub_skill, strat, contribution_df = score_matrix(df_task, snap, DEFAULT_TOTALS["raven"])

    elif tkey in ("brain_eq", "brain_words"):
        snap = get_first_nonempty_snapshot(snaps, [task_name])
        per_sub_skill, strat, contribution_df = score_brainstorming(df_task, tkey, params, snap=snap)

    elif tkey == "typing":
        snap = get_first_nonempty_snapshot(snaps, [task_name, "TypingText_FR", "TypingText"])
        per_sub_skill, strat, contribution_df = score_typing(df_task, params, snap=snap)

    per_sub = merge_effort_skill(effort, per_sub_skill, contribution_df)
    per_sub.insert(0, "TaskKey", tkey)
    per_sub.insert(0, "task_type", task_type)
    per_sub.insert(0, "TaskName", task_name)
    per_sub.insert(0, "SessionId", session_id)
    per_sub.insert(0, "GroupID", group_id)
    per_sub = apply_typing_oriented_individual_overrides(per_sub)

    skill_cong = compute_skill_congruence(per_sub, tkey, params)
    contrib_cong = compute_contribution_congruence(per_sub, tkey, params)
    dist_metrics = compute_distribution_metrics(per_sub)

    effort_task_total = float(per_sub["effort_task"].sum()) if not per_sub.empty else 0.0
    effort_comm_total = float(per_sub["effort_comm"].sum()) if not per_sub.empty else 0.0
    effort_total = float(per_sub["effort_total"].sum()) if not per_sub.empty else 0.0

    # Pour les tâches typing-oriented, la littérature Riedl indique que le
    # quantity of contribution peut servir de base aux mesures de processus.
    # Quand les traces individuelles ne permettent pas une répartition fiable,
    # on conserve l'agrégation groupe à partir de la quantité observable.
    task_quantity_total = pd.to_numeric(pd.Series([strat.get("task_quantity_total", np.nan)]), errors="coerce").iloc[0]
    task_quantity_unit = strat.get("task_quantity_unit", "")
    if task_type == "typing_oriented" and pd.notna(task_quantity_total) and task_quantity_total > 0:
        if effort_task_total <= 0:
            effort_task_total = float(task_quantity_total)
            effort_total = effort_task_total + effort_comm_total

    gsum = {
        "GroupID": group_id,
        "SessionId": session_id,
        "TaskName": task_name,
        "TaskKey": tkey,
        "task_type": task_type,
        # Strategy (original + enhanced)
        "strategy_defined": strat["strategy_defined"],
        "strategy_elements_covered": strat["strategy_elements_covered"],
        "strategy_total": strat["strategy_total"],
        "strategy_ratio": strat["strategy_ratio"],
        "coverage_any": strat.get("coverage_any", np.nan),
        "coverage_correct_final": strat.get("coverage_correct_final", np.nan),
        "redundancy_rate": strat.get("redundancy_rate", np.nan),
        "strategy_reference": strat.get("strategy_reference", np.nan),
        "task_quantity_total": task_quantity_total,
        "task_quantity_unit": task_quantity_unit,
        # Effort (split)
        "effort_task_total": effort_task_total,
        "effort_comm_total": effort_comm_total,
        "effort_total": effort_total,
        "effort_metric": params.effort_mode,
        "comm_n_messages": int(per_sub["comm_n_messages"].sum()) if "comm_n_messages" in per_sub.columns else 0,
        "comm_text_length": int(per_sub["comm_text_length"].sum()) if "comm_text_length" in per_sub.columns else 0,
        # Congruence
        "skill_congruence": skill_cong,
        "contribution_congruence": contrib_cong,
        "congruence_effort_scope": params.congruence_effort_scope,
        "skill_congruence_forced": int(tkey in TYPING_TASK_KEYS),
        # Skill
        "skill_mean_task": pd.to_numeric(per_sub["skill_i"], errors="coerce").mean(),
        "skill_max_task": pd.to_numeric(per_sub["skill_i"], errors="coerce").max(),
        # Contribution
        "contribution_mean_task": pd.to_numeric(per_sub["contribution_i"], errors="coerce").mean(),
        "contribution_sum_task": pd.to_numeric(per_sub["contribution_correct"], errors="coerce").sum(),
        # Counts
        "n_members_effort": int(per_sub["effort_total"].notna().sum()),
        "n_members_skill": int(per_sub["skill_i"].notna().sum()),
        "n_members_contribution": int(per_sub["contribution_i"].notna().sum()),
    }
    # Add distribution metrics
    gsum.update(dist_metrics)

    return per_sub, gsum


def process_one_group(group_dir: Path, params: AnalysisParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ev_candidates = sorted(group_dir.rglob("EventLog*.csv"), key=lambda p: p.stat().st_size)
    if not ev_candidates:
        raise FileNotFoundError(f"No EventLog*.csv found in {group_dir}")
    eventlog_path = ev_candidates[-1]

    snap_dirs = [p for p in group_dir.rglob("TaskSnapshot_*") if p.is_dir()]
    snapshot_dir = snap_dirs[0] if snap_dirs else None
    snaps = load_snapshot_dir(snapshot_dir)

    df = read_eventlog(eventlog_path)

    if "SessionSuffix" in df.columns and df["SessionSuffix"].astype(str).str.strip().ne("").any():
        suffix = df.loc[df["SessionSuffix"].astype(str).str.strip().ne(""), "SessionSuffix"].iloc[0]
        group_id = normalize_group_id(suffix)
    else:
        group_id = normalize_group_id(group_dir.name)

    sessions = sorted(df["SessionId"].astype(str).unique())

    all_indiv: List[pd.DataFrame] = []
    all_group: List[Dict] = []

    for sid in sessions:
        df_s = df[(df["SessionId"].astype(str) == str(sid)) & (~df["TaskName"].apply(_is_ignored_task))].copy()
        if df_s.empty:
            continue

        for tname, g_task in df_s.groupby("TaskName"):
            per_sub, gsum = process_one_task(group_id, str(sid), tname, g_task, snaps, params)
            all_indiv.append(per_sub)
            all_group.append(gsum)

    indiv_cols = [
        "GroupID", "SessionId", "TaskName", "TaskKey", "SubjectId",
        "effort_task", "effort_comm", "effort_total", "effort_value", "effort_metric",
        "typing_quantity_individual_observed",
        "comm_n_messages", "comm_text_length",
        "correct", "total", "skill_i",
        "contribution_correct", "contribution_total", "contribution_i",
        "effort_share_task", "effort_share_total", "contribution_share",
    ]
    indiv = (
        pd.concat(all_indiv, ignore_index=True)
        if all_indiv else
        pd.DataFrame(columns=indiv_cols)
    )
    group = pd.DataFrame(all_group)
    return indiv, group


# =========================================================
# Group summary
# =========================================================

def aggregate_member_skill(df_indiv: pd.DataFrame) -> pd.DataFrame:
    """Member-level aggregation of skill across tasks."""
    if df_indiv.empty:
        return pd.DataFrame(columns=["GroupID", "SubjectId", "member_skill_mean", "member_skill_max", "n_skill_tasks"])

    d = df_indiv.copy()
    d["skill_i"] = pd.to_numeric(d["skill_i"], errors="coerce")
    d = d[d["skill_i"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["GroupID", "SubjectId", "member_skill_mean", "member_skill_max", "n_skill_tasks"])

    out = (
        d.groupby(["GroupID", "SubjectId"])["skill_i"]
        .agg(member_skill_mean="mean", member_skill_max="max", n_skill_tasks="count")
        .reset_index()
    )
    return out


def aggregate_member_contribution(df_indiv: pd.DataFrame) -> pd.DataFrame:
    """Member-level aggregation of contribution across tasks."""
    if df_indiv.empty or "contribution_i" not in df_indiv.columns:
        return pd.DataFrame(columns=["GroupID", "SubjectId", "member_contribution_mean", "member_contribution_max"])

    d = df_indiv.copy()
    d["contribution_i"] = pd.to_numeric(d["contribution_i"], errors="coerce")
    d = d[d["contribution_i"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["GroupID", "SubjectId", "member_contribution_mean", "member_contribution_max"])

    out = (
        d.groupby(["GroupID", "SubjectId"])["contribution_i"]
        .agg(member_contribution_mean="mean", member_contribution_max="max")
        .reset_index()
    )
    return out


def aggregate_group_summary(group_task: pd.DataFrame, indiv_df: pd.DataFrame) -> pd.DataFrame:
    if group_task.empty:
        return pd.DataFrame(columns=[
            "GroupID",
            "effort_total_sum", "effort_total_mean",
            "effort_task_sum", "effort_task_mean",
            "effort_comm_sum", "effort_comm_mean",
            "strategy_ratio_mean", "strategy_covered_sum",
            "coverage_any_mean", "coverage_correct_final_mean",
            "redundancy_rate_mean",
            "skill_congruence_mean", "skill_congruence_mean_core", "contribution_congruence_mean",
            "n_tasks", "n_tasks_strategy_defined",
            "effort_norm", "strategy_norm",
            "skill_mean", "skill_max",
            "contribution_mean", "contribution_max",
            "n_members_with_skill",
        ])

    num_safe = lambda s: pd.to_numeric(s, errors="coerce").dropna().mean()
    num_sum  = lambda s: pd.to_numeric(s, errors="coerce").dropna().sum()

    agg = (
        group_task.groupby("GroupID")
        .agg(
            effort_total_sum=("effort_total", "sum"),
            effort_total_mean=("effort_total", "mean"),
            effort_task_sum=("effort_task_total", "sum"),
            effort_task_mean=("effort_task_total", "mean"),
            effort_comm_sum=("effort_comm_total", "sum"),
            effort_comm_mean=("effort_comm_total", "mean"),
            comm_n_messages_sum=("comm_n_messages", "sum"),
            comm_text_length_sum=("comm_text_length", "sum"),
            strategy_ratio_mean=("strategy_ratio", num_safe),
            strategy_covered_sum=("strategy_elements_covered", num_sum),
            coverage_any_mean=("coverage_any", num_safe),
            coverage_correct_final_mean=("coverage_correct_final", num_safe),
            redundancy_rate_mean=("redundancy_rate", num_safe),
            skill_congruence_mean=("skill_congruence", num_safe),
            contribution_congruence_mean=("contribution_congruence", num_safe),
            contribution_mean_task_avg=("contribution_mean_task", num_safe),
            contribution_sum_task_total=("contribution_sum_task", num_sum),
            n_tasks=("TaskName", "nunique"),
            n_tasks_strategy_defined=("strategy_defined", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            # Distribution metrics: average across tasks
            effort_task_cv=("effort_task_cv", num_safe),
            effort_task_gini=("effort_task_gini", num_safe),
            effort_total_cv=("effort_total_cv", num_safe),
            effort_total_gini=("effort_total_gini", num_safe),
            contribution_cv=("contribution_cv", num_safe),
            contribution_gini=("contribution_gini", num_safe),
            top_member_effort_task_share=("top_member_effort_task_share", num_safe),
            top_member_effort_total_share=("top_member_effort_total_share", num_safe),
            top_member_contribution_share=("top_member_contribution_share", num_safe),
        )
        .reset_index()
    )

    # Variante "core" : moyenne de congruence de skill restreinte aux tâches
    # à réponse correcte, en excluant explicitement les tâches typing-oriented.
    core_mask = ~group_task["TaskKey"].isin(TYPING_TASK_KEYS)
    core_congruence = (
        group_task.loc[core_mask]
        .groupby("GroupID")["skill_congruence"]
        .agg(
            skill_congruence_mean_core=num_safe,
            n_tasks_skill_congruence_core=lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()),
        )
        .reset_index()
    )
    agg = agg.merge(core_congruence, on="GroupID", how="left")

    # Member-level skill aggregation
    member_skill = aggregate_member_skill(indiv_df)
    if not member_skill.empty:
        skill_group = (
            member_skill.groupby("GroupID")
            .agg(
                skill_mean=("member_skill_mean", "mean"),
                skill_max=("member_skill_max", "max"),
                n_members_with_skill=("SubjectId", "nunique"),
            )
            .reset_index()
        )
        agg = agg.merge(skill_group, on="GroupID", how="left")
    else:
        agg["skill_mean"] = np.nan
        agg["skill_max"] = np.nan
        agg["n_members_with_skill"] = 0

    # Member-level contribution aggregation
    member_contrib = aggregate_member_contribution(indiv_df)
    if not member_contrib.empty:
        contrib_group = (
            member_contrib.groupby("GroupID")
            .agg(
                contribution_mean=("member_contribution_mean", "mean"),
                contribution_max=("member_contribution_max", "max"),
            )
            .reset_index()
        )
        agg = agg.merge(contrib_group, on="GroupID", how="left")
    else:
        agg["contribution_mean"] = np.nan
        agg["contribution_max"] = np.nan

    # Normalization
    max_eff = pd.to_numeric(agg["effort_total_sum"], errors="coerce").max()
    agg["effort_norm"] = agg["effort_total_sum"] / max_eff if pd.notna(max_eff) and max_eff > 0 else np.nan

    max_eff_task = pd.to_numeric(agg["effort_task_sum"], errors="coerce").max()
    agg["effort_task_norm"] = agg["effort_task_sum"] / max_eff_task if pd.notna(max_eff_task) and max_eff_task > 0 else np.nan

    max_strat = pd.to_numeric(agg["strategy_ratio_mean"], errors="coerce").max()
    agg["strategy_norm"] = agg["strategy_ratio_mean"] / max_strat if pd.notna(max_strat) and max_strat > 0 else np.nan

    return agg


# =========================================================
# C-factor
# =========================================================

ID_CANDS = ["SessionId", "session_id", "GroupID", "group_id", "Session", "session"]
C_CANDS = ["C_factor", "c_factor", "C", "c", "c_score", "cfactor"]

def find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in df.columns:
            return k
        if k.lower() in low:
            return low[k.lower()]
    return None


def load_c_scores(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, encoding="utf-8")
    idcol = find_col(c, ID_CANDS)
    ccol = find_col(c, C_CANDS)
    if idcol is None or ccol is None:
        raise ValueError("c_scores.csv must contain a group/session id column and a c-factor column.")

    out = c[[idcol, ccol]].copy()
    out.columns = ["GroupID", "C_factor"]
    out["GroupID"] = out["GroupID"].apply(normalize_group_id)
    out["C_factor"] = pd.to_numeric(out["C_factor"], errors="coerce")
    return out


def export_correlations_from_merged(merged: pd.DataFrame, out_dir: Path) -> None:
    predictors = [
        "skill_mean",
        "skill_max",
        "contribution_mean",
        "contribution_max",
        "effort_total_sum",
        "effort_task_sum",
        "effort_comm_sum",
        "strategy_ratio_mean",
        "coverage_any_mean",
        "coverage_correct_final_mean",
        "skill_congruence_mean",
        "contribution_congruence_mean",
        "effort_norm",
        "effort_task_norm",
        "strategy_norm",
        "effort_task_gini",
        "effort_total_gini",
        "contribution_gini",
        "top_member_effort_task_share",
        "top_member_contribution_share",
    ]
    rows = []
    for x in predictors:
        if x not in merged.columns:
            continue
        xv = pd.to_numeric(merged[x], errors="coerce")
        yv = pd.to_numeric(merged["C_factor"], errors="coerce")
        mask = xv.notna() & yv.notna()
        n = int(mask.sum())

        if n >= 3 and xv[mask].nunique() > 1 and yv[mask].nunique() > 1:
            pr, pp = pearsonr(xv[mask], yv[mask])
            sr, sp = spearmanr(xv[mask], yv[mask], nan_policy="omit")
        else:
            pr = pp = sr = sp = np.nan

        rows.append({
            "predictor": x,
            "n": n,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp,
        })

    pd.DataFrame(rows).to_csv(out_dir / "riedl_correlations.csv", index=False, encoding="utf-8")


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute Riedl-inspired skill, effort, strategy, and congruence.")
    ap.add_argument("--groups", nargs="+", required=True, help="Group folders or a parent folder.")
    ap.add_argument("--c-scores", default=None, help="Path to c_scores.csv.")
    ap.add_argument("--out-dir", default="out_riedl", help="Output directory.")
    ap.add_argument(
        "--reference-full-csv",
        default=None,
        help="Chemin vers le `full.csv` de la population parente Riedl 2021 pour calculer les variantes *_pop.",
    )
    ap.add_argument("--effort-mode", choices=["event_count", "text_length"], default="text_length")
    ap.add_argument("--congruence-mode", choices=["raw", "clip_zero", "rescale01"], default="clip_zero")
    ap.add_argument("--congruence-effort-scope", choices=["effort_task", "effort_comm", "effort_total"], default="effort_task",
                     help="Which effort component to use for congruence computation.")
    ap.add_argument("--sudoku-denominator", choices=["editable", "full_grid"], default="editable")
    ap.add_argument("--open-task-strategy-mode", choices=["undefined", "proxy_unique_tokens"], default="undefined")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = safe_mkdir(Path(args.out_dir))

    params = AnalysisParams(
        effort_mode=args.effort_mode,
        congruence_mode=args.congruence_mode,
        congruence_effort_scope=args.congruence_effort_scope,
        sudoku_denominator=args.sudoku_denominator,
        open_task_strategy_mode=args.open_task_strategy_mode,
        debug=args.debug,
    )

    def has_eventlog(p: Path) -> bool:
        return p.is_dir() and any(p.rglob("EventLog*.csv"))

    def _is_extraction_dir(p: Path) -> bool:
        n = p.name
        return n.startswith("Event_Log_") or n.startswith("TaskSnapshot_")

    expanded: List[Path] = []
    for g in args.groups:
        p = Path(g)
        if not p.exists() or p.is_file():
            continue

        # Look for child group directories, excluding extraction directories
        sub_groups = [
            sub for sub in p.iterdir()
            if sub.is_dir() and not _is_extraction_dir(sub) and has_eventlog(sub)
        ]
        if sub_groups:
            expanded.extend(sorted(sub_groups))
        elif has_eventlog(p):
            expanded.append(p)

    expanded = sorted(set(expanded))
    if not expanded:
        raise FileNotFoundError(f"No group folders with EventLog*.csv found under: {args.groups}")

    all_indiv: List[pd.DataFrame] = []
    all_group: List[pd.DataFrame] = []

    excluded_norm = {normalize_group_id(x) for x in EXCLUDED_GROUPS} if EXCLUDED_GROUPS else set()

    for gdir in expanded:
        gid = normalize_group_id(gdir.name)
        if gid in excluded_norm:
            log(f"[INFO] Excluded group: {gid}", params.debug)
            continue

        try:
            indiv, group = process_one_group(gdir, params)
            if not indiv.empty:
                all_indiv.append(indiv)
            if not group.empty:
                all_group.append(group)
        except Exception as e:
            print(f"[WARN] Skipping group {gdir.name} due to {type(e).__name__}: {e}")

    df_indiv = pd.concat(all_indiv, ignore_index=True) if all_indiv else pd.DataFrame()
    df_group_task = pd.concat(all_group, ignore_index=True) if all_group else pd.DataFrame()
    df_group_task = normalize_typing_oriented_strategy(df_group_task)

    df_group_summary = aggregate_group_summary(df_group_task, df_indiv)
    ref_full_csv = Path(args.reference_full_csv) if args.reference_full_csv else None
    df_group_summary = add_parent_population_norms(df_group_summary, ref_full_csv)

    # Trace methodological settings in exports
    if not df_indiv.empty:
        df_indiv["analysis_effort_mode"] = params.effort_mode
        df_indiv["analysis_congruence_mode"] = params.congruence_mode
        df_indiv["analysis_congruence_effort_scope"] = params.congruence_effort_scope
        df_indiv["analysis_sudoku_denominator"] = params.sudoku_denominator

    if not df_group_task.empty:
        df_group_task["analysis_effort_mode"] = params.effort_mode
        df_group_task["analysis_congruence_mode"] = params.congruence_mode
        df_group_task["analysis_congruence_effort_scope"] = params.congruence_effort_scope
        df_group_task["analysis_sudoku_denominator"] = params.sudoku_denominator
        df_group_task["analysis_reference_full_csv"] = str(ref_full_csv) if ref_full_csv else ""

    if not df_group_summary.empty:
        df_group_summary["analysis_effort_mode"] = params.effort_mode
        df_group_summary["analysis_congruence_mode"] = params.congruence_mode
        df_group_summary["analysis_congruence_effort_scope"] = params.congruence_effort_scope
        df_group_summary["analysis_sudoku_denominator"] = params.sudoku_denominator
        df_group_summary["analysis_reference_full_csv"] = str(ref_full_csv) if ref_full_csv else ""

    df_indiv.to_csv(out_dir / "riedl_indiv.csv", index=False, encoding="utf-8")
    df_group_task.to_csv(out_dir / "riedl_group_task.csv", index=False, encoding="utf-8")
    df_group_summary.to_csv(out_dir / "riedl_group_summary.csv", index=False, encoding="utf-8")

    if args.c_scores:
        cdf = load_c_scores(Path(args.c_scores))
        merged = cdf.merge(df_group_summary, on="GroupID", how="inner")
        merged.to_csv(out_dir / "riedl_cfactor_merge.csv", index=False, encoding="utf-8")
        export_correlations_from_merged(merged, out_dir)

    print("[OK] Exports at:", out_dir)


if __name__ == "__main__":
    main()
