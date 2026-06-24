#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_emotions_group.py

Analyse triadique d'émotions dérivées de FACS/AUs (OpenFace ou Meta Quest Pro):
- Positif (Happiness/Joy) = AU6 + AU12
- Négatif frame-level (sad_intensity) = mean(AU1, AU4, AU15)   ← variable intermédiaire
  → face_negative_affect_ratio (variable finale HLF) = au15_au17_coactive_pct_mean (AU15+AU17 co-actifs)

Deux définitions de synchronie :
1) Co-activation : overlap sur signaux binaires (Jaccard + nb épisodes synchrones)
2) Co-fluctuation : corrélation de Pearson sur séries fenêtrées (windows synchronisées)

Robustesse VR vs PC :
- Seuillage d'activation par quantile (défaut) / zscore / absolu (CLI)
- Fenêtrage identique pour les trois rôles sur l'intervalle temporel commun

Sortie :
- Un CSV global (une ligne par groupe) avec métadonnées + métriques.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---- common package imports ------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.constants import ROLES
from common.metadata import extract_condition, extract_scenario, extract_timepoint
from common.io_utils import read_csv_eu, normalize_columns, find_pc_facs_files, find_groups
from common.temporal import sliding_windows


# -------------------------
# Loading & preprocessing
# -------------------------

def _detect_marker_rows(df: pd.DataFrame, time_col: str) -> pd.Series:
    other_cols = [c for c in df.columns if c != time_col]
    if not other_cols:
        return pd.Series(False, index=df.index)
    return df[other_cols].apply(
        lambda s: s.astype(str).str.contains(r"(?i)marker", na=False)
    ).any(axis=1)


def preprocess_time_align(df_raw: pd.DataFrame, time_col: str = "Timestamp") -> pd.DataFrame:
    """
    - Convertit Timestamp en float
    - Aligne t=0 sur MARKER si présent, sinon t=min
    - Supprime lignes marker
    - Set index=Timestamp (sec), tri, unique
    - Convertit colonnes numériques, interpole
    """
    df = df_raw.copy()

    if time_col not in df.columns:
        raise ValueError(f"Colonne temps '{time_col}' introuvable. Colonnes: {list(df.columns)[:20]}...")

    df[time_col] = pd.to_numeric(df[time_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    marker_mask = _detect_marker_rows(df, time_col)
    if marker_mask.any():
        marker_ts = df.loc[marker_mask, time_col].min()
        if pd.notna(marker_ts):
            df[time_col] = df[time_col] - float(marker_ts)
        df = df.loc[~marker_mask].copy()
    else:
        t0 = df[time_col].min()
        if pd.notna(t0):
            df[time_col] = df[time_col] - float(t0)

    df = df.dropna(subset=[time_col]).copy()
    df = df.sort_values(time_col)
    df = df.drop_duplicates(subset=[time_col], keep="first")
    df = df.set_index(time_col)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    df = df.interpolate(method="linear", limit_direction="both")
    df = df[~df.index.isna()].sort_index()
    return df


# -------------------------
# AU composites: Joy & Sadness
# -------------------------

def build_emotion_intensities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit:
    - AU1 = InnerBrowRaiser (L/R moyen si dispo)
    - AU4 = BrowLowerer
    - AU6 = CheekRaiser
    - AU12 = LipCornerPuller
    - AU15 = LipCornerDepressor
    - AU17 = ChinRaiser

    puis :
    - joy_intensity = mean(AU6, AU12)
    - sad_intensity = mean(AU1, AU4, AU15)
    """
    out = df.copy()

    def avg_lr(base: str) -> pd.Series:
        left_right = [c for c in (f"{base}L", f"{base}R") if c in out.columns]
        top_bottom = [c for c in (f"{base}T", f"{base}B") if c in out.columns]

        if len(left_right) == 2:
            return pd.concat(
                [pd.to_numeric(out[c], errors="coerce") for c in left_right],
                axis=1,
            ).mean(axis=1, skipna=True)
        if len(top_bottom) == 2:
            return pd.concat(
                [pd.to_numeric(out[c], errors="coerce") for c in top_bottom],
                axis=1,
            ).mean(axis=1, skipna=True)
        if base in out.columns:
            return pd.to_numeric(out[base], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    out["AU1"] = avg_lr("InnerBrowRaiser")
    out["AU4"] = avg_lr("BrowLowerer")
    out["AU6"] = avg_lr("CheekRaiser")
    out["AU12"] = avg_lr("LipCornerPuller")
    out["AU15"] = avg_lr("LipCornerDepressor")
    out["AU17"] = avg_lr("ChinRaiser")

    out["joy_intensity"] = pd.concat(
        [out["AU6"].clip(lower=0), out["AU12"].clip(lower=0)],
        axis=1,
    ).mean(axis=1, skipna=True)
    out["sad_intensity"] = pd.concat(
        [out["AU1"].clip(lower=0), out["AU4"].clip(lower=0), out["AU15"].clip(lower=0)],
        axis=1,
    ).mean(axis=1, skipna=True)
    return out


# -------------------------
# Thresholding (robust VR vs PC)
# -------------------------

def compute_threshold(x: pd.Series, mode: str, thr_abs: float, q: float, z_k: float) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan

    mode = mode.lower()
    if mode == "absolute":
        return float(thr_abs)
    if mode == "quantile":
        return float(x.quantile(q))
    if mode == "zscore":
        mu = float(x.mean())
        sd = float(x.std(ddof=1))
        if sd < 1e-12:
            return np.inf
        return float(mu + z_k * sd)

    raise ValueError(f"mode seuil inconnu: {mode}")


def add_binary_states(
    df: pd.DataFrame,
    thr_mode: str,
    joy_thr_abs: float,
    sad_thr_abs: float,
    q: float,
    z_k: float
) -> pd.DataFrame:
    out = df.copy()
    joy_thr = compute_threshold(out["joy_intensity"], thr_mode, joy_thr_abs, q, z_k)
    sad_thr = compute_threshold(out["sad_intensity"], thr_mode, sad_thr_abs, q, z_k)

    def binary_from_threshold(series: pd.Series, thr: float) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if not np.isfinite(thr) or s.notna().sum() == 0:
            return pd.Series(np.nan, index=s.index, dtype=float)
        return pd.Series(
            np.where(s.notna(), (s > thr).astype(float), np.nan),
            index=s.index,
            dtype=float,
        )

    def coactive_binary(a: pd.Series, b: pd.Series) -> pd.Series:
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        missing = a_num.isna() | b_num.isna()
        values = np.where(missing, np.nan, np.where((a_num == 1.0) & (b_num == 1.0), 1.0, 0.0))
        return pd.Series(values, index=a.index, dtype=float)

    out["joy_thr_used"] = joy_thr
    out["sad_thr_used"] = sad_thr

    out["joy_active"] = (out["joy_intensity"] > joy_thr).astype(int) if np.isfinite(joy_thr) else 0
    out["sad_active"] = (out["sad_intensity"] > sad_thr).astype(int) if np.isfinite(sad_thr) else 0

    au6_thr = compute_threshold(out["AU6"], thr_mode, joy_thr_abs, q, z_k)
    au12_thr = compute_threshold(out["AU12"], thr_mode, joy_thr_abs, q, z_k)
    au1_thr = compute_threshold(out["AU1"], thr_mode, sad_thr_abs, q, z_k)
    au4_thr = compute_threshold(out["AU4"], thr_mode, sad_thr_abs, q, z_k)
    au15_thr = compute_threshold(out["AU15"], thr_mode, sad_thr_abs, q, z_k)
    au17_thr = compute_threshold(out["AU17"], thr_mode, sad_thr_abs, q, z_k)

    out["au6_thr_used"] = au6_thr
    out["au12_thr_used"] = au12_thr
    out["au1_thr_used"] = au1_thr
    out["au4_thr_used"] = au4_thr
    out["au15_thr_used"] = au15_thr
    out["au17_thr_used"] = au17_thr

    out["au6_active"] = binary_from_threshold(out["AU6"], au6_thr)
    out["au12_active"] = binary_from_threshold(out["AU12"], au12_thr)
    out["au1_active"] = binary_from_threshold(out["AU1"], au1_thr)
    out["au4_active"] = binary_from_threshold(out["AU4"], au4_thr)
    out["au15_active"] = binary_from_threshold(out["AU15"], au15_thr)
    out["au17_active"] = binary_from_threshold(out["AU17"], au17_thr)
    out["au6_au12_coactive"] = coactive_binary(out["au6_active"], out["au12_active"])
    out["au4_au15_coactive"] = coactive_binary(out["au4_active"], out["au15_active"])
    out["au15_au17_coactive"] = coactive_binary(out["au15_active"], out["au17_active"])

    out["success"] = 1.0
    return out


# -------------------------
# Episodes on irregular timestamps
# -------------------------

def episodes_from_boolean_time(
    t: np.ndarray,
    x: np.ndarray,
    min_episode_s: float = 0.0,
    merge_gap_s: float = 0.0
) -> Tuple[int, float, float]:
    if len(t) == 0:
        return 0, 0.0, np.nan

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=int)

    on = (x == 1)
    if not np.any(on):
        return 0, 0.0, np.nan

    idx = np.where(on)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]

    intervals = [(t[s], t[e]) for s, e in zip(starts, ends)]

    if merge_gap_s > 0 and len(intervals) > 1:
        merged = []
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if (s - cur_e) <= merge_gap_s:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        intervals = merged

    durs = np.array([max(0.0, e - s) for s, e in intervals], dtype=float)

    if min_episode_s > 0:
        durs = durs[durs >= min_episode_s]

    n = int(durs.size)
    tot = float(durs.sum()) if n else 0.0
    mean = float(tot / n) if n else np.nan
    return n, tot, mean


# -------------------------
# Rasterization for overlap metrics
# -------------------------

def infer_dt_seconds(df: pd.DataFrame) -> float:
    t = df.index.to_numpy(dtype=float)
    if t.size < 3:
        return np.nan
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.nan


def rasterize_binary(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    if df.empty:
        return np.zeros(len(grid), dtype=int)
    src_vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).to_numpy()
    src = pd.DataFrame({"t": df.index.to_numpy(dtype=float), col: src_vals}).sort_values("t")
    tmp = pd.DataFrame({"t": grid})
    out = pd.merge_asof(tmp, src, on="t", direction="backward")
    return out[col].fillna(0).astype(int).to_numpy()


def overlap_jaccard(x: np.ndarray, y: np.ndarray) -> float:
    inter = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))
    return float(inter / union) if union > 0 else np.nan


def count_sync_episodes_from_overlap(z: np.ndarray, dt: float, min_overlap_s: float) -> int:
    if z.size == 0:
        return 0
    dx = np.diff(z.astype(int), prepend=0, append=0)
    starts = np.where(dx == 1)[0]
    ends = np.where(dx == -1)[0]
    durs = (ends - starts) * dt
    return int(np.sum(durs >= min_overlap_s))


# -------------------------
# Windowed features & Pearson synchrony
# -------------------------

def emotions_features(
    df: pd.DataFrame,
    t0: float,
    t1: float,
    win: float,
    step: float,
    min_success: float = 0.5
) -> pd.DataFrame:
    """
    Série fenêtrée synchronisable entre rôles.
    - joy_mean, sad_mean
    - joy_active_pct, sad_active_pct
    - AU activity pct (AU6, AU12, AU1, AU4, AU15, AU17)
    - AU co-activity pct (AU6+12, AU4+15, AU15+17)
    """
    df = df[(df.index >= t0) & (df.index <= t1)].copy()

    results = []
    for s, e in sliding_windows(t0, t1, win, step):
        chunk = df[(df.index >= s) & (df.index < e)]
        if chunk.empty:
            results.append({
                "t_start": s, "t_end": e,
                "joy_mean": np.nan, "sad_mean": np.nan,
                "joy_active_pct": np.nan, "sad_active_pct": np.nan,
                "au6_active_pct": np.nan,
                "au12_active_pct": np.nan,
                "au1_active_pct": np.nan,
                "au4_active_pct": np.nan,
                "au15_active_pct": np.nan,
                "au17_active_pct": np.nan,
                "au6_au12_coactive_pct": np.nan,
                "au4_au15_coactive_pct": np.nan,
                "au15_au17_coactive_pct": np.nan,
                "success_ok": 0.0
            })
            continue

        success_rate = chunk["success"].mean() if "success" in chunk else 1.0
        if success_rate < min_success:
            results.append({
                "t_start": s, "t_end": e,
                "joy_mean": np.nan, "sad_mean": np.nan,
                "joy_active_pct": np.nan, "sad_active_pct": np.nan,
                "au6_active_pct": np.nan,
                "au12_active_pct": np.nan,
                "au1_active_pct": np.nan,
                "au4_active_pct": np.nan,
                "au15_active_pct": np.nan,
                "au17_active_pct": np.nan,
                "au6_au12_coactive_pct": np.nan,
                "au4_au15_coactive_pct": np.nan,
                "au15_au17_coactive_pct": np.nan,
                "success_ok": float(success_rate)
            })
            continue

        results.append({
            "t_start": s, "t_end": e,
            "joy_mean": float(chunk["joy_intensity"].mean()),
            "sad_mean": float(chunk["sad_intensity"].mean()),
            "joy_active_pct": float(chunk["joy_active"].mean()),
            "sad_active_pct": float(chunk["sad_active"].mean()),
            "au6_active_pct": float(pd.to_numeric(chunk["au6_active"], errors="coerce").mean()),
            "au12_active_pct": float(pd.to_numeric(chunk["au12_active"], errors="coerce").mean()),
            "au1_active_pct": float(pd.to_numeric(chunk["au1_active"], errors="coerce").mean()),
            "au4_active_pct": float(pd.to_numeric(chunk["au4_active"], errors="coerce").mean()),
            "au15_active_pct": float(pd.to_numeric(chunk["au15_active"], errors="coerce").mean()),
            "au17_active_pct": float(pd.to_numeric(chunk["au17_active"], errors="coerce").mean()),
            "au6_au12_coactive_pct": float(pd.to_numeric(chunk["au6_au12_coactive"], errors="coerce").mean()),
            "au4_au15_coactive_pct": float(pd.to_numeric(chunk["au4_au15_coactive"], errors="coerce").mean()),
            "au15_au17_coactive_pct": float(pd.to_numeric(chunk["au15_au17_coactive"], errors="coerce").mean()),
            "success_ok": float(success_rate)
        })

    out = pd.DataFrame(results)
    expected = [
        "t_start", "t_end",
        "joy_mean", "sad_mean",
        "joy_active_pct", "sad_active_pct",
        "au6_active_pct", "au12_active_pct",
        "au1_active_pct", "au4_active_pct",
        "au15_active_pct", "au17_active_pct",
        "au6_au12_coactive_pct", "au4_au15_coactive_pct", "au15_au17_coactive_pct",
        "success_ok",
    ]
    for c in expected:
        if c not in out.columns:
            out[c] = pd.Series(dtype=float)
    return out[expected]


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(pearsonr(x, y)[0])


def compute_synchrony_pearson(features_by_role: Dict[str, pd.DataFrame],
                             metrics=("joy_mean", "sad_mean", "joy_active_pct", "sad_active_pct")) -> pd.DataFrame:
    pairs = [("calculateur", "modelisateur"),
             ("calculateur", "lecteur"),
             ("modelisateur", "lecteur")]
    rows = []
    for a, b in pairs:
        A = features_by_role[a]
        B = features_by_role[b]
        merged = pd.merge(
            A[["t_start", "t_end"] + list(metrics)],
            B[["t_start", "t_end"] + list(metrics)],
            on=["t_start", "t_end"],
            how="inner",
            suffixes=(f"_{a}", f"_{b}")
        )
        for m in metrics:
            x = merged[f"{m}_{a}"]
            y = merged[f"{m}_{b}"]
            mask = (~x.isna()) & (~y.isna())
            n = int(mask.sum())
            r = safe_pearson(x[mask].to_numpy(), y[mask].to_numpy()) if n >= 2 else np.nan
            rows.append({
                "pair": f"{a}–{b}",
                "metric": m,
                "r_pearson": r,
                "n_windows": n
            })
    return pd.DataFrame(rows)


def descriptives_windowed(df: pd.DataFrame, metrics=("joy_mean", "sad_mean", "joy_active_pct", "sad_active_pct")) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in metrics:
        if m not in df.columns:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_median"] = np.nan
            out[f"{m}_valid_ratio"] = np.nan
            continue
        s = pd.to_numeric(df[m], errors="coerce")
        valid = s.dropna()
        out[f"{m}_mean"] = float(valid.mean()) if valid.size else np.nan
        out[f"{m}_median"] = float(valid.median()) if valid.size else np.nan
        out[f"{m}_valid_ratio"] = float(valid.size / len(s)) if len(s) > 0 else np.nan
    return out


# -------------------------
# Group processing
# -------------------------

def process_one_group(
    group_dir: Path,
    thr_mode: str,
    joy_thr_abs: float,
    sad_thr_abs: float,
    q: float,
    z_k: float,
    min_episode_s: float,
    merge_gap_s: float,
    dt_mode: str,
    dt_user: float,
    min_overlap_s: float,
    win: float,
    step: float,
    min_success: float
) -> Dict[str, object]:
    condition = extract_condition(group_dir)
    scenario = extract_scenario(group_dir)
    timepoint = extract_timepoint(group_dir)

    # select files
    if condition == "VR":
        files = {}
        for role in ROLES:
            folder = group_dir / role
            cands = list(folder.glob("*_FaceTrackingData.csv"))
            if len(cands) != 1:
                raise RuntimeError(f"{folder}: attendu 1 *_FaceTrackingData.csv, trouvé {len(cands)}")
            files[role] = cands[0]
        source = "VR"
    elif condition == "PC":
        files = find_pc_facs_files(group_dir)
        source = "PC"
    else:
        raise RuntimeError(f"Condition inconnue pour {group_dir}")

    # load / preprocess / build intensities
    dfs: Dict[str, pd.DataFrame] = {}
    dt_by_role = {}
    for role, fp in files.items():
        df_raw = normalize_columns(read_csv_eu(fp))
        df = preprocess_time_align(df_raw, time_col="Timestamp")
        df = build_emotion_intensities(df)
        dfs[role] = df
        dt_by_role[role] = infer_dt_seconds(df)

    # common interval
    t0 = max(d.index.min() for d in dfs.values())
    t1 = min(d.index.max() for d in dfs.values())
    if not (np.isfinite(t0) and np.isfinite(t1)) or (t1 - t0) <= 0:
        raise RuntimeError(f"{group_dir.name}: intervalle commun invalide (t0={t0}, t1={t1})")

    interaction_dur_s = float(t1 - t0)

    # dt for overlap raster
    dt_mode = dt_mode.lower()
    if dt_mode == "user":
        dt = float(dt_user)
    else:
        vals = [v for v in dt_by_role.values() if np.isfinite(v) and v > 0]
        dt = float(max(vals)) if vals else 0.02
    if dt <= 0:
        dt = 0.02

    # cut + threshold + binary
    cut: Dict[str, pd.DataFrame] = {}
    for role, df in dfs.items():
        dfc = df.loc[t0:t1].copy()
        dfc = add_binary_states(
            dfc,
            thr_mode=thr_mode,
            joy_thr_abs=joy_thr_abs,
            sad_thr_abs=sad_thr_abs,
            q=q,
            z_k=z_k
        )
        cut[role] = dfc

    # common grid for overlap
    grid = np.arange(float(t0), float(t1) + 1e-9, dt)

    pairs = [("calculateur", "modelisateur"),
             ("calculateur", "lecteur"),
             ("modelisateur", "lecteur")]

    row: Dict[str, object] = {
        "group_id": group_dir.name,
        "condition": condition or "UNK",
        "scenario": scenario or "UNK",
        "timepoint": timepoint,
        "source_face": source,
        "interaction_dur_s": interaction_dur_s,

        "thr_mode": thr_mode,
        "q": float(q),
        "z_k": float(z_k),
        "joy_thr_abs": float(joy_thr_abs),
        "sad_thr_abs": float(sad_thr_abs),

        "min_episode_s": float(min_episode_s),
        "merge_gap_s": float(merge_gap_s),

        "dt_s": float(dt),
        "min_overlap_s": float(min_overlap_s),

        "win_s": float(win),
        "step_s": float(step),
        "min_success": float(min_success),

        "dt_calc_s": float(dt_by_role.get("calculateur", np.nan)),
        "dt_model_s": float(dt_by_role.get("modelisateur", np.nan)),
        "dt_lect_s": float(dt_by_role.get("lecteur", np.nan)),
    }
    # Aliases legacy conservés pour compatibilité descendante.
    row["group"] = row["group_id"]
    row["session"] = row["scenario"]

    # trace thresholds per role
    for role in ROLES:
        row[f"{role}_joy_thr_used"] = float(cut[role]["joy_thr_used"].iloc[0]) if len(cut[role]) else np.nan
        row[f"{role}_sad_thr_used"] = float(cut[role]["sad_thr_used"].iloc[0]) if len(cut[role]) else np.nan

    # (A) Pearson fenêtré (co-fluctuation)
    feats_by_role: Dict[str, pd.DataFrame] = {}
    for role in ROLES:
        feats_by_role[role] = emotions_features(
            cut[role],
            t0=float(t0),
            t1=float(t1),
            win=win,
            step=step,
            min_success=min_success
        )

    sync_df = compute_synchrony_pearson(
        feats_by_role,
        metrics=(
            "joy_mean", "sad_mean",
            "joy_active_pct", "sad_active_pct",
            "au6_active_pct", "au12_active_pct",
            "au1_active_pct", "au4_active_pct",
            "au15_active_pct", "au17_active_pct",
            "au6_au12_coactive_pct", "au4_au15_coactive_pct", "au15_au17_coactive_pct",
        ),
    )

    merged = pd.concat(feats_by_role.values(), ignore_index=True)
    window_metrics = [
        "joy_mean", "sad_mean",
        "joy_active_pct", "sad_active_pct",
        "au6_active_pct", "au12_active_pct",
        "au1_active_pct", "au4_active_pct",
        "au15_active_pct", "au17_active_pct",
        "au6_au12_coactive_pct", "au4_au15_coactive_pct", "au15_au17_coactive_pct",
    ]
    group_mean = (
        merged.groupby(["t_start", "t_end"])
        .agg({**{metric: "mean" for metric in window_metrics}, "success_ok": "mean"})
        .reset_index()
    )
    row.update(descriptives_windowed(group_mean, metrics=tuple(window_metrics)))

    for _, r in sync_df.iterrows():
        key = f"sync_pearson_{r['pair'].replace('–','_')}_{r['metric']}"
        row[key] = r["r_pearson"]
        row[key + "_nwin"] = r["n_windows"]

    au_pearson_metrics = {
        "au6_active_pct",
        "au12_active_pct",
        "au1_active_pct",
        "au4_active_pct",
        "au15_active_pct",
        "au17_active_pct",
        "au6_au12_coactive_pct",
        "au4_au15_coactive_pct",
        "au15_au17_coactive_pct",
    }
    au_pearson_scores = pd.to_numeric(
        sync_df.loc[sync_df["metric"].isin(au_pearson_metrics), "r_pearson"],
        errors="coerce",
    ).dropna()
    row["au_sync_pearson_mean"] = float(au_pearson_scores.mean()) if not au_pearson_scores.empty else np.nan

    # (B) Episodes + normalisation (timestamps-based)
    # (C) Overlap synchrony (co-activation)
    for emo, col in [("joy", "joy_active"), ("sad", "sad_active")]:
        for role in ROLES:
            dfc = cut[role]
            n, tot, mean = episodes_from_boolean_time(
                t=dfc.index.to_numpy(dtype=float),
                x=dfc[col].to_numpy(dtype=int),
                min_episode_s=min_episode_s,
                merge_gap_s=merge_gap_s
            )
            row[f"{emo}_{role}_n_ep"] = n
            row[f"{emo}_{role}_tot_dur_s"] = tot
            row[f"{emo}_{role}_mean_dur_s"] = mean
            row[f"{emo}_{role}_occupancy"] = (tot / interaction_dur_s) if interaction_dur_s > 0 else np.nan
            row[f"{emo}_{role}_rate_per_min"] = (n / (interaction_dur_s / 60.0)) if interaction_dur_s > 0 else np.nan

        xC = rasterize_binary(cut["calculateur"], col, grid)
        xM = rasterize_binary(cut["modelisateur"], col, grid)
        xL = rasterize_binary(cut["lecteur"], col, grid)
        tri = ((xC == 1) & (xM == 1) & (xL == 1)).astype(int)

        n_tri = count_sync_episodes_from_overlap(tri, dt=dt, min_overlap_s=min_episode_s)
        tot_tri = float(np.sum(tri) * dt)
        mean_tri = float(tot_tri / n_tri) if n_tri > 0 else np.nan

        row[f"{emo}_tri_n_ep"] = n_tri
        row[f"{emo}_tri_tot_dur_s"] = tot_tri
        row[f"{emo}_tri_mean_dur_s"] = mean_tri
        row[f"{emo}_tri_occupancy"] = (tot_tri / interaction_dur_s) if interaction_dur_s > 0 else np.nan
        row[f"{emo}_tri_rate_per_min"] = (n_tri / (interaction_dur_s / 60.0)) if interaction_dur_s > 0 else np.nan

        for a, b in pairs:
            xa = rasterize_binary(cut[a], col, grid)
            xb = rasterize_binary(cut[b], col, grid)
            row[f"{emo}_sync_{a}_{b}_jaccard"] = overlap_jaccard(xa, xb)
            z = ((xa == 1) & (xb == 1)).astype(int)
            row[f"{emo}_sync_{a}_{b}_n_sync_ep"] = count_sync_episodes_from_overlap(
                z, dt=dt, min_overlap_s=min_overlap_s
            )

    au_jaccard_scores: list[float] = []
    for au_col in [
        "au6_active",
        "au12_active",
        "au1_active",
        "au4_active",
        "au15_active",
        "au17_active",
        "au6_au12_coactive",
        "au4_au15_coactive",
        "au15_au17_coactive",
    ]:
        if not all(au_col in cut[role].columns and cut[role][au_col].notna().any() for role in ROLES):
            continue
        for a, b in pairs:
            xa = rasterize_binary(cut[a], au_col, grid)
            xb = rasterize_binary(cut[b], au_col, grid)
            score = overlap_jaccard(xa, xb)
            if np.isfinite(score):
                au_jaccard_scores.append(float(score))

    row["au_sync_jaccard_mean"] = float(np.mean(au_jaccard_scores)) if au_jaccard_scores else np.nan
    au_sync_components = [v for v in [row["au_sync_jaccard_mean"], row["au_sync_pearson_mean"]] if pd.notna(v)]
    row["au_sync_mean"] = float(np.mean(au_sync_components)) if au_sync_components else np.nan

    return row


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser(description="Analyse Joy/Sadness en épisodes + synchronie overlap + Pearson fenêtré (OpenFace & Quest)")
    p.add_argument("root_dir", help="Dossier racine (ex: data_e2)")
    p.add_argument("--out", default="face_emotion_metrics_all.csv", help="CSV global (tous groupes)")

    # thresholding
    p.add_argument("--thr_mode", choices=["quantile", "zscore", "absolute"], default="quantile",
                   help="Méthode de seuillage (robuste VR/PC: quantile)")
    p.add_argument("--q", type=float, default=0.90, help="Quantile pour thr_mode=quantile")
    p.add_argument("--z_k", type=float, default=1.0, help="k pour thr_mode=zscore (thr = mean + k*std)")
    p.add_argument("--joy_thr_abs", type=float, default=0.7, help="Seuil absolu Joy si thr_mode=absolute")
    p.add_argument("--sad_thr_abs", type=float, default=0.7, help="Seuil absolu Sad si thr_mode=absolute")

    # episode definition
    p.add_argument("--min_episode_s", type=float, default=0.30, help="Durée min d'un épisode (filtre anti-bruit)")
    p.add_argument("--merge_gap_s", type=float, default=0.20, help="Fusionne deux épisodes séparés par un OFF <= merge_gap_s")

    # overlap synchrony
    p.add_argument("--dt_mode", choices=["auto", "user"], default="auto",
                   help="Pas dt pour rasterisation overlap (auto=max dt médian rôles)")
    p.add_argument("--dt", type=float, default=0.05, help="dt (s) si dt_mode=user")
    p.add_argument("--min_overlap_s", type=float, default=0.30, help="Seuil overlap (s) pour compter un épisode synchrone dyadique")

    # windowed Pearson
    p.add_argument("--win", type=float, default=10.0, help="Taille fenêtre (s) pour Pearson fenêtré")
    p.add_argument("--step", type=float, default=2.0, help="Pas fenêtre (s) pour Pearson fenêtré")
    p.add_argument("--min_success", type=float, default=0.5, help="Taux minimal de frames valides (ici success=1.0)")

    args = p.parse_args()

    root = Path(args.root_dir)
    groups = find_groups(root)
    if not groups:
        raise SystemExit("Aucun groupe détecté (VR ou PC)")

    rows = []
    print(f"{len(groups)} groupes détectés.")
    for g in groups:
        print(f"\nTraitement : {g}")
        try:
            row = process_one_group(
                g,
                thr_mode=args.thr_mode,
                joy_thr_abs=args.joy_thr_abs,
                sad_thr_abs=args.sad_thr_abs,
                q=args.q,
                z_k=args.z_k,
                min_episode_s=args.min_episode_s,
                merge_gap_s=args.merge_gap_s,
                dt_mode=args.dt_mode,
                dt_user=args.dt,
                min_overlap_s=args.min_overlap_s,
                win=args.win,
                step=args.step,
                min_success=args.min_success
            )
            rows.append(row)
        except Exception as e:
            print(f"Erreur sur {g} : {e}")

    out = Path(args.out)
    # Sortie standardisée pour le reste du pipeline. La lecture aval reste
    # tolérante aux anciens exports EU pour préserver la rétrocompatibilité.
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nCSV global écrit : {out} ({len(rows)} groupes)")


if __name__ == "__main__":
    main()
