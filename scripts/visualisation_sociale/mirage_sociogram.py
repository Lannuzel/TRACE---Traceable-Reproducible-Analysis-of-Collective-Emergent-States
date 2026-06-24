#!/usr/bin/env python3
"""
Equivalent MIRAGE adapte au projet Longitudinale.

Cette visualisation reconstruit un sociogramme multimodal offline, inspire de
MIRAGE / SAAC, a partir des donnees brutes deja utilisees dans le pipeline du
projet. Le rendu n'est pas un plugin temps reel PsiStudio : il applique la meme
logique de fenetre glissante (20 s par defaut, pas 1 s) sur des sessions
enregistrees.

Adapatations au projet :
- taille des noeuds : temps de parole dans la fenetre
- halo du noeud : focus tache (ratio de fixations sur objets, VR uniquement)
- arete orange : egalite de parole dyadique
- arete verte + fleches : regard vers un pair (inferre via hit point + positions)
- courbe bleue : attention conjointe sur objet (JVA)
- capsule rose : synchronie faciale (proxy de synchronie interpersonnelle)
- geometrie : proximite moyenne entre participants (VR), sinon triangle fixe

Sorties :
- window_summary.csv
- node_metrics.csv
- edge_metrics.csv
- mirage_snapshot.png
- frames/*.png (optionnel)
- mirage_animation.gif (optionnel, si Pillow est disponible)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import wave
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_mpl_cache = PROJECT_ROOT / ".mplconfig"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sps
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyse_inv.face.analyze_aus_group import (  # noqa: E402
    add_binary_states,
    build_emotion_intensities,
    infer_dt_seconds,
    preprocess_time_align,
    safe_pearson,
)
from analyse_inv.gaze.analyze_gaze import (  # noqa: E402
    detect_fixations_per_participant,
    shared_object_metrics,
)
from analyse_inv.speech.analyze_audio import (  # noqa: E402
    DEFAULT_MERGE_GAP,
    DEFAULT_THR_DB,
    SKIP_AFTER_BEEP,
    detect_beep,
    effective_analysis_duration_s,
    speech_segments,
)
from common.io_utils import (  # noqa: E402
    find_groups,
    find_pc_facs_files,
    find_pc_role_wavs,
    find_wav,
    is_pc_group,
    is_vr_group,
    normalize_columns,
    read_csv_eu,
)
from common.metadata import (  # noqa: E402
    extract_condition,
    extract_scenario,
    extract_timepoint,
)
from common.temporal import sliding_windows  # noqa: E402


ROLE_ORDER = ["calculateur", "modelisateur", "lecteur"]
ROLE_LABELS = {
    "calculateur": "CAL",
    "modelisateur": "MOD",
    "lecteur": "LEC",
}
ROLE_LABELS_LONG = {
    "calculateur": "Calculateur",
    "modelisateur": "Modelisateur",
    "lecteur": "Lecteur",
}
ROLE_COLORS = {
    "calculateur": "#F4A261",
    "modelisateur": "#2A9D8F",
    "lecteur": "#457B9D",
}

BG_COLOR = "#0B132B"
PANEL_BG = "#16213E"
PANEL_EDGE = "#E0E1DD"
TEXT_COLOR = "#F8F9FA"
SPEECH_COLOR = "#F39C12"
INVALID_SPEECH_COLOR = "#D62828"
GAZE_COLOR = "#2ECC71"
JVA_COLOR = "#4FC3F7"
SYNC_COLOR = "#F4A6C3"

PEER_GAZE_RADIUS_M = 0.45
PEER_GAZE_MIN_EPISODE_S = 0.15
PEER_GAZE_MERGE_GAP_S = 0.10
TASK_OBJECT_IGNORE = {"", "none", "floor"}
DEFAULT_AUDIO_SR = 16000
DEFAULT_BEEP_SCAN_S = 60.0
AUDIO_HOP_LENGTH = 256
AUDIO_FRAME_LENGTH = 1024

NODE_RADIUS_MIN = 0.12
NODE_RADIUS_MAX = 0.26
HALO_SCALE_MIN = 1.20
HALO_SCALE_MAX = 2.10

DISPLAY_MIN_DISTANCE_M = 0.80
DISTANCE_GAIN = 0.85
DEFAULT_FIXED_DISTANCE_M = 1.45

PAIR_OFFSET = 0.12
JVA_CURVE_BULGE = 0.22
AX_LIM_X = (-1.9, 1.9)
AX_LIM_Y = (-1.7, 1.7)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[DEBUG] {message}", flush=True)


def safe_mean(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return float(arr.mean()) if not arr.empty else np.nan


def pair_key(role_a: str, role_b: str) -> str:
    a, b = sorted((role_a, role_b), key=lambda r: ROLE_ORDER.index(r))
    return f"{a}+{b}"


def compress_distance_meters(distance_m: float) -> float:
    if not np.isfinite(distance_m):
        return DEFAULT_FIXED_DISTANCE_M
    return DISPLAY_MIN_DISTANCE_M + DISTANCE_GAIN * max(0.0, float(distance_m) - DISPLAY_MIN_DISTANCE_M)


def overlap_duration(intervals: Iterable[Tuple[float, float]], start: float, end: float) -> float:
    total = 0.0
    for a, b in intervals:
        total += max(0.0, min(float(b), end) - max(float(a), start))
    return float(total)


def is_floor_like(object_id: object) -> bool:
    txt = str(object_id).strip().lower()
    if txt in TASK_OBJECT_IGNORE:
        return True
    return txt.startswith("floor")


def choose_vr_file(folder: Path, merged_name: str, pattern: str) -> Path:
    merged = folder / merged_name
    if merged.exists():
        return merged
    candidates = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier {pattern} trouve dans {folder}")
    return candidates[0]


def read_csv_eu_fast(path: Path, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=";", decimal=",", usecols=usecols, low_memory=False)
    except Exception:
        return read_csv_eu(path, usecols=usecols)


def wav_dtype_from_width(sample_width: int) -> np.dtype:
    if sample_width == 1:
        return np.dtype(np.uint8)
    if sample_width == 2:
        return np.dtype(np.int16)
    if sample_width == 4:
        return np.dtype(np.int32)
    raise ValueError(f"Largeur PCM non supportee: {sample_width} octets")


def pcm_to_float32(samples: np.ndarray, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return ((samples.astype(np.float32) - 128.0) / 128.0).astype(np.float32, copy=False)
    if sample_width == 2:
        return (samples.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    if sample_width == 4:
        return (samples.astype(np.float32) / 2147483648.0).astype(np.float32, copy=False)
    raise ValueError(f"Largeur PCM non supportee: {sample_width} octets")


def read_wav_window(path: Path, start_s: float = 0.0, duration_s: Optional[float] = None) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sr = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        total_frames = wav_file.getnframes()

        frame_start = int(max(0.0, start_s) * sr)
        frame_start = min(frame_start, total_frames)
        wav_file.setpos(frame_start)

        if duration_s is None:
            frames_to_read = total_frames - frame_start
        else:
            frames_to_read = min(int(max(0.0, duration_s) * sr), total_frames - frame_start)

        raw = wav_file.readframes(frames_to_read)

    samples = np.frombuffer(raw, dtype=wav_dtype_from_width(sample_width))
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return pcm_to_float32(samples, sample_width), sr


def detect_beep_fast(
    path: Path,
    *,
    peak_ratio: float = 0.9,
    pre_silence: float = 0.2,
    scan_s: float = DEFAULT_BEEP_SCAN_S,
) -> tuple[float, int]:
    try:
        y, sr = read_wav_window(path, start_s=0.0, duration_s=scan_s)
    except Exception:
        return detect_beep(path, debug=False)

    if y.size == 0:
        return detect_beep(path, debug=False)

    abs_y = np.abs(y)
    peak_amp = float(abs_y.max())
    if peak_amp <= 0:
        return detect_beep(path, debug=False)

    hard_thr = peak_ratio * peak_amp
    cand_idx = np.where(abs_y >= hard_thr)[0]
    if cand_idx.size == 0:
        return detect_beep(path, debug=False)

    min_gap = int(pre_silence * sr)
    for idx in cand_idx:
        if idx < min_gap:
            continue
        if float(abs_y[idx - min_gap : idx].max()) <= 0.1 * peak_amp:
            return float(idx / sr), sr
    return float(cand_idx[0] / sr), sr


def load_wav_mono_resampled(path: Path, start_s: float, target_sr: int) -> np.ndarray:
    y, sr_in = read_wav_window(path, start_s=start_s)
    if y.size == 0:
        return np.array([], dtype=np.float32)
    if sr_in == target_sr:
        return y.astype(np.float32, copy=False)
    gcd = math.gcd(int(sr_in), int(target_sr))
    up = int(target_sr // gcd)
    down = int(sr_in // gcd)
    return sps.resample_poly(y, up, down).astype(np.float32, copy=False)


def clip_intervals(intervals: Iterable[Tuple[float, float]], max_time: float) -> list[tuple[float, float]]:
    clipped = []
    for start, end in intervals:
        if start >= max_time:
            continue
        clipped.append((max(0.0, float(start)), min(float(end), max_time)))
    return clipped


def rms_db_fast(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.array([], dtype=np.float32)

    pad = AUDIO_FRAME_LENGTH // 2
    sq = np.pad(np.square(y, dtype=np.float32), (pad, pad), mode="constant")
    csum = np.empty(sq.size + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(sq, out=csum[1:])
    starts = np.arange(0, sq.size - AUDIO_FRAME_LENGTH + 1, AUDIO_HOP_LENGTH, dtype=np.int64)
    frame_energy = csum[starts + AUDIO_FRAME_LENGTH] - csum[starts]
    rms = np.sqrt(np.maximum(frame_energy / AUDIO_FRAME_LENGTH, 1e-12))
    ref = float(np.max(rms))
    if ref <= 0:
        return np.full(rms.shape, -120.0, dtype=np.float32)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12) / ref)
    return db.astype(np.float32, copy=False)


def speech_segments_fast(y: np.ndarray, sr: int, thr_db: float, merge_gap: float) -> list[tuple[float, float]]:
    e_db = rms_db_fast(y)
    if e_db.size == 0:
        return []

    times = np.arange(e_db.size, dtype=np.float32) * (AUDIO_HOP_LENGTH / float(sr))
    mask = e_db > (float(np.median(e_db)) + float(thr_db))

    segs: list[list[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = float(times[i])
            while i < len(mask) and mask[i]:
                i += 1
            end = float(times[min(i, len(times) - 1)])
            segs.append([start, end])
        i += 1

    merged: list[list[float]] = []
    for start, end in segs:
        if not merged or start - merged[-1][1] > merge_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return [(float(start), float(end)) for start, end in merged]


def resolve_group_dir(
    data_dir: Path,
    group_id: str,
    modality: Optional[str],
    scenario: Optional[str],
    timepoint: Optional[str],
) -> Path:
    candidates = []
    for group_dir in find_groups(data_dir):
        if group_dir.name != group_id:
            continue
        cond = extract_condition(group_dir)
        scen = extract_scenario(group_dir)
        tp = extract_timepoint(group_dir)
        if modality and cond != modality:
            continue
        if scenario and scen != scenario:
            continue
        if timepoint and tp != timepoint:
            continue
        candidates.append(group_dir)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun groupe correspondant pour {group_id} "
            f"(modalite={modality or '*'}, scenario={scenario or '*'}, timepoint={timepoint or '*'})"
        )
    if len(candidates) > 1:
        listing = "\n".join(str(p) for p in candidates)
        raise RuntimeError(
            "Plusieurs groupes correspondent. Precise --modality / --scenario / --timepoint.\n"
            f"{listing}"
        )
    return candidates[0]


def load_audio_segments(
    group_dir: Path,
    thr_db: float,
    merge_gap: float,
    *,
    audio_sr: int = DEFAULT_AUDIO_SR,
    beep_scan_s: float = DEFAULT_BEEP_SCAN_S,
    debug: bool = False,
) -> tuple[Dict[str, list[Tuple[float, float]]], float]:
    is_vr = is_vr_group(group_dir)
    is_pc = is_pc_group(group_dir)
    if not (is_vr or is_pc):
        raise RuntimeError(f"{group_dir} n'est ni un groupe VR ni un groupe PC valide.")

    if is_vr:
        files = {role: find_wav(group_dir / role) for role in ROLE_ORDER}
        beep_times: dict[str, float] = {}
        for role, file_path in files.items():
            beep_time, sr_i = detect_beep_fast(file_path, scan_s=beep_scan_s)
            beep_times[role] = beep_time
            debug_log(debug, f"audio {role}: bip detecte a {beep_time:.3f}s (sr={sr_i})")
        tracks: dict[str, np.ndarray] = {}
        for role, file_path in files.items():
            start_s = beep_times[role] + SKIP_AFTER_BEEP
            t0 = time.perf_counter()
            tracks[role] = load_wav_mono_resampled(file_path, start_s=start_s, target_sr=audio_sr)
            debug_log(debug, f"audio {role}: charge en {time.perf_counter() - t0:.2f}s depuis {start_s:.3f}s")
        sr = audio_sr
    else:
        sr = audio_sr
        files = find_pc_role_wavs(group_dir / "processed_openface")

        tracks = {}
        for role, file_path in files.items():
            t0 = time.perf_counter()
            tracks[role] = load_wav_mono_resampled(file_path, start_s=0.0, target_sr=sr)
            debug_log(debug, f"audio {role}: charge PC en {time.perf_counter() - t0:.2f}s")

    total_duration = effective_analysis_duration_s(tracks, sr)

    segs: dict[str, list[tuple[float, float]]] = {}
    for role in ROLE_ORDER:
        signal = tracks.get(role, np.array([], dtype=np.float32))
        segs_role = speech_segments_fast(signal, sr, thr_db, merge_gap) if len(signal) else []
        segs[role] = clip_intervals(segs_role, total_duration)
    return segs, total_duration


def preprocess_time_series_csv(
    path: Path,
    time_col: str,
    rename_map: Optional[dict[str, str]] = None,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = normalize_columns(read_csv_eu_fast(path, usecols=usecols))
    if time_col not in df.columns:
        raise ValueError(f"{path} : colonne {time_col} introuvable.")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    if df.empty:
        raise ValueError(f"{path} : aucune ligne temporelle valide.")
    df[time_col] = df[time_col] - float(df[time_col].iloc[0])

    if rename_map:
        df = df.rename(columns=rename_map)
    return df.reset_index(drop=True)


def load_vr_eye_samples(group_dir: Path) -> Dict[str, pd.DataFrame]:
    by_role: dict[str, pd.DataFrame] = {}
    for role in ROLE_ORDER:
        eye_path = choose_vr_file(group_dir / role, "merged_EyeTrackingData.csv", "*_EyeTrackingData.csv")
        df = preprocess_time_series_csv(
            eye_path,
            time_col="Time",
            usecols=["Time", "HitPointX", "HitPointY", "HitPointZ", "ObjectHit"],
            rename_map={
                "HitPointX": "hit_x",
                "HitPointY": "hit_y",
                "HitPointZ": "hit_z",
                "ObjectHit": "object_id",
            },
        )
        keep_cols = ["Time", "hit_x", "hit_y", "hit_z", "object_id"]
        missing = [col for col in keep_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{eye_path} : colonnes manquantes {missing}")
        df = df[keep_cols].copy()
        df = df.rename(columns={"Time": "time"})
        for col in ["time", "hit_x", "hit_y", "hit_z"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["object_id"] = df["object_id"].astype(str).fillna("")
        df = df.dropna(subset=["time", "hit_x", "hit_y", "hit_z"]).reset_index(drop=True)
        by_role[role] = df
    return by_role


def load_vr_positions(group_dir: Path) -> Dict[str, pd.DataFrame]:
    by_role: dict[str, pd.DataFrame] = {}
    for role in ROLE_ORDER:
        pos_path = choose_vr_file(group_dir / role, "merged_UsersPositions.csv", "*_UsersPositions.csv")
        df = preprocess_time_series_csv(
            pos_path,
            time_col="Timestamp",
            usecols=["Timestamp", "Player0_PosX", "Player0_PosY", "Player0_PosZ"],
            rename_map={
                "Player0_PosX": "x",
                "Player0_PosY": "y",
                "Player0_PosZ": "z",
            },
        )
        keep_cols = ["Timestamp", "x", "y", "z"]
        missing = [col for col in keep_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{pos_path} : colonnes manquantes {missing}")
        df = df[keep_cols].copy().rename(columns={"Timestamp": "time"})
        for col in ["time", "x", "y", "z"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time", "x", "y", "z"]).reset_index(drop=True)
        by_role[role] = df
    return by_role


def load_fixations_from_eye_samples(eye_samples: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = []
    for role, df in eye_samples.items():
        part = df[["time", "object_id"]].copy()
        part["participant"] = role
        all_rows.append(part[["participant", "time", "object_id"]])
    gaze_df = pd.concat(all_rows, ignore_index=True)
    return detect_fixations_per_participant(gaze_df, min_fix=0.20)


def load_face_streams(group_dir: Path) -> Dict[str, pd.DataFrame]:
    condition = extract_condition(group_dir)
    if condition == "VR":
        files = {
            role: choose_vr_file(group_dir / role, "merged_FaceTrackingData.csv", "*_FaceTrackingData.csv")
            for role in ROLE_ORDER
        }
    elif condition == "PC":
        files = find_pc_facs_files(group_dir)
    else:
        raise RuntimeError(f"Condition inconnue pour {group_dir}")

    streams: dict[str, pd.DataFrame] = {}
    for role, file_path in files.items():
        raw = normalize_columns(read_csv_eu_fast(file_path))
        df = preprocess_time_align(raw, time_col="Timestamp")
        df = build_emotion_intensities(df)
        df = add_binary_states(
            df,
            thr_mode="quantile",
            joy_thr_abs=0.7,
            sad_thr_abs=0.7,
            q=0.90,
            z_k=1.0,
        )
        streams[role] = df.reset_index().rename(columns={"Timestamp": "time"})
    return streams


def window_list(t0: float, t1: float, win: float, step: float) -> list[tuple[float, float]]:
    if (t1 - t0) <= win:
        return [(t0, t1)]
    return list(sliding_windows(t0, t1, win, step))


def sample_nearest_positions(times: np.ndarray, pos_df: pd.DataFrame, tolerance_s: float = 0.10) -> pd.DataFrame:
    base = pd.DataFrame({"time": np.asarray(times, dtype=float)})
    pos = pos_df.sort_values("time")[["time", "x", "y", "z"]].copy()
    return pd.merge_asof(base, pos, on="time", direction="nearest", tolerance=tolerance_s)


def episodes_from_sample_mask(
    times: np.ndarray,
    mask: np.ndarray,
    min_episode_s: float = 0.0,
    merge_gap_s: float = 0.0,
) -> tuple[int, float, float]:
    if times.size == 0:
        return 0, 0.0, np.nan

    times = np.asarray(times, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0, 0.0, np.nan

    if times.size == 1:
        dur = max(min_episode_s, 0.0)
        return (1, dur, dur) if mask[0] else (0, 0.0, np.nan)

    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt_default = float(np.median(diffs)) if diffs.size else 0.05

    intervals: list[tuple[float, float]] = []
    start_idx: Optional[int] = None
    for idx, active in enumerate(mask):
        if active and start_idx is None:
            start_idx = idx
        if start_idx is not None and ((not active) or idx == len(mask) - 1):
            end_time = times[idx] if not active else times[idx] + dt_default
            intervals.append((times[start_idx], end_time))
            start_idx = None

    if merge_gap_s > 0 and len(intervals) > 1:
        merged: list[tuple[float, float]] = []
        cur_start, cur_end = intervals[0]
        for nxt_start, nxt_end in intervals[1:]:
            if (nxt_start - cur_end) <= merge_gap_s:
                cur_end = nxt_end
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = nxt_start, nxt_end
        merged.append((cur_start, cur_end))
        intervals = merged

    durations = [max(0.0, end - start) for start, end in intervals if (end - start) >= min_episode_s]
    if not durations:
        return 0, 0.0, np.nan
    total = float(np.sum(durations))
    return int(len(durations)), total, float(total / len(durations))


def build_directional_peer_gaze_window(
    eye_df: pd.DataFrame,
    target_pos_df: pd.DataFrame,
    start: float,
    end: float,
    radius_m: float,
) -> dict[str, float]:
    chunk = eye_df[(eye_df["time"] >= start) & (eye_df["time"] < end)].copy()
    if chunk.empty:
        return {"ratio": 0.0, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}

    aligned = sample_nearest_positions(chunk["time"].to_numpy(dtype=float), target_pos_df)
    valid = aligned[["x", "y", "z"]].notna().all(axis=1).to_numpy()
    if not np.any(valid):
        return {"ratio": 0.0, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}

    hit = chunk[["hit_x", "hit_y", "hit_z"]].to_numpy(dtype=float)
    target = aligned[["x", "y", "z"]].to_numpy(dtype=float)
    dist = np.linalg.norm(hit - target, axis=1)
    mask = valid & np.isfinite(dist) & (dist <= radius_m)
    n_episodes, dur_total, dur_mean = episodes_from_sample_mask(
        chunk["time"].to_numpy(dtype=float),
        mask,
        min_episode_s=PEER_GAZE_MIN_EPISODE_S,
        merge_gap_s=PEER_GAZE_MERGE_GAP_S,
    )
    ratio = dur_total / max(end - start, 1e-6)
    return {
        "ratio": float(ratio),
        "n_episodes": int(n_episodes),
        "dur_total_s": float(dur_total),
        "dur_mean_s": float(dur_mean) if np.isfinite(dur_mean) else np.nan,
    }


def task_focus_ratio(fix_df: pd.DataFrame, role: str, start: float, end: float) -> tuple[float, int]:
    chunk = fix_df[
        (fix_df["participant"] == role)
        & (fix_df["onset"] < end)
        & (fix_df["offset"] > start)
    ].copy()
    if chunk.empty:
        return 0.0, 0
    chunk = chunk[~chunk["object_id"].apply(is_floor_like)].copy()
    if chunk.empty:
        return 0.0, 0
    onset = pd.to_numeric(chunk["onset"], errors="coerce").to_numpy(dtype=float)
    offset = pd.to_numeric(chunk["offset"], errors="coerce").to_numpy(dtype=float)
    overlap = np.maximum(0.0, np.minimum(offset, end) - np.maximum(onset, start))
    duration = float(np.nansum(overlap))
    n_fix = int(np.count_nonzero(overlap > 0))
    return float(duration / max(end - start, 1e-6)), int(n_fix)


def face_sync_score(face_a: pd.DataFrame, face_b: pd.DataFrame, start: float, end: float) -> float:
    cols = ["joy_intensity", "sad_intensity"]
    a = face_a[(face_a["time"] >= start) & (face_a["time"] < end)][["time"] + cols].copy()
    b = face_b[(face_b["time"] >= start) & (face_b["time"] < end)][["time"] + cols].copy()
    if len(a) < 3 or len(b) < 3:
        return np.nan

    dt = max(
        infer_dt_seconds(a.set_index("time")),
        infer_dt_seconds(b.set_index("time")),
        1.0 / 30.0,
    )
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / 30.0
    grid = np.arange(start, end, dt)
    if grid.size < 3:
        return np.nan

    base = pd.DataFrame({"time": grid})
    a_aligned = pd.merge_asof(base, a.sort_values("time"), on="time", direction="nearest", tolerance=dt)
    b_aligned = pd.merge_asof(base, b.sort_values("time"), on="time", direction="nearest", tolerance=dt)

    scores = []
    for col in cols:
        x = pd.to_numeric(a_aligned[col], errors="coerce")
        y = pd.to_numeric(b_aligned[col], errors="coerce")
        mask = (~x.isna()) & (~y.isna())
        if int(mask.sum()) >= 3:
            rho = safe_pearson(x[mask].to_numpy(), y[mask].to_numpy())
            if np.isfinite(rho):
                scores.append((rho + 1.0) / 2.0)
    return float(np.mean(scores)) if scores else np.nan


def mean_pair_distance(pos_a: pd.DataFrame, pos_b: pd.DataFrame, start: float, end: float) -> float:
    a = pos_a[(pos_a["time"] >= start) & (pos_a["time"] < end)][["time", "x", "y", "z"]].copy()
    b = pos_b[(pos_b["time"] >= start) & (pos_b["time"] < end)][["time", "x", "y", "z"]].copy()
    if len(a) < 2 or len(b) < 2:
        return np.nan

    dt_a = pd.to_numeric(a["time"].diff(), errors="coerce").dropna()
    dt_b = pd.to_numeric(b["time"].diff(), errors="coerce").dropna()
    dt = max(
        float(dt_a[dt_a > 0].median()) if not dt_a.empty else np.nan,
        float(dt_b[dt_b > 0].median()) if not dt_b.empty else np.nan,
        0.05,
    )
    grid = np.arange(start, end, dt)
    if grid.size < 2:
        return np.nan

    base = pd.DataFrame({"time": grid})
    aa = pd.merge_asof(base, a.sort_values("time"), on="time", direction="nearest", tolerance=dt)
    bb = pd.merge_asof(base, b.sort_values("time"), on="time", direction="nearest", tolerance=dt)
    valid = aa[["x", "y", "z"]].notna().all(axis=1) & bb[["x", "y", "z"]].notna().all(axis=1)
    if not valid.any():
        return np.nan
    va = aa.loc[valid, ["x", "y", "z"]].to_numpy(dtype=float)
    vb = bb.loc[valid, ["x", "y", "z"]].to_numpy(dtype=float)
    dist = np.linalg.norm(va - vb, axis=1)
    return float(np.mean(dist)) if dist.size else np.nan


def compute_triangle_layout(display_distances: dict[str, float]) -> dict[str, tuple[float, float]]:
    d12 = compress_distance_meters(display_distances.get(pair_key("calculateur", "modelisateur"), DEFAULT_FIXED_DISTANCE_M))
    d13 = compress_distance_meters(display_distances.get(pair_key("calculateur", "lecteur"), DEFAULT_FIXED_DISTANCE_M))
    d23 = compress_distance_meters(display_distances.get(pair_key("modelisateur", "lecteur"), DEFAULT_FIXED_DISTANCE_M))

    eps = 1e-3
    for _ in range(3):
        if d12 >= d13 + d23:
            d12 = max(d13 + d23 - eps, eps)
        if d13 >= d12 + d23:
            d13 = max(d12 + d23 - eps, eps)
        if d23 >= d12 + d13:
            d23 = max(d12 + d13 - eps, eps)

    half = d23 * 0.5
    p2 = np.array([-half, 0.0], dtype=float)
    p3 = np.array([half, 0.0], dtype=float)

    x1 = (d12 * d12 - d13 * d13) / (2.0 * max(d23, eps))
    y1_sq = max(d12 * d12 - x1 * x1, 0.0)
    y1 = -max(math.sqrt(y1_sq), 0.18 * d23)
    p1 = np.array([x1, y1], dtype=float)

    centroid = (p1 + p2 + p3) / 3.0
    p1 = p1 - centroid
    p2 = p2 - centroid
    p3 = p3 - centroid

    return {
        "calculateur": (float(p1[0]), float(p1[1])),
        "modelisateur": (float(p2[0]), float(p2[1])),
        "lecteur": (float(p3[0]), float(p3[1])),
    }


def edge_offset_points(a: np.ndarray, b: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    vec = b - a
    length = np.linalg.norm(vec)
    if length < 1e-9:
        return a.copy(), b.copy()
    direction = vec / length
    normal = np.array([-direction[1], direction[0]], dtype=float)
    delta = normal * offset
    return a + delta, b + delta


def quadratic_curve_patch(a: np.ndarray, b: np.ndarray, bulge: float, color: str, linewidth: float, alpha: float) -> PathPatch:
    mid = (a + b) / 2.0
    vec = b - a
    length = np.linalg.norm(vec)
    if length < 1e-9:
        length = 1.0
    direction = vec / length
    normal = np.array([-direction[1], direction[0]], dtype=float)
    ctrl = mid + normal * bulge
    verts = [tuple(a), tuple(ctrl), tuple(b)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    return PathPatch(path, facecolor="none", edgecolor=color, linewidth=linewidth, alpha=alpha, capstyle="round")


def make_capsule_polygon(a: np.ndarray, b: np.ndarray, half_width: float, n_arc: int = 12) -> Optional[np.ndarray]:
    """Capsule remplie englobant deux points (rectangle aux extrémités semi-circulaires).

    Retourne un tableau (N, 2) de sommets formant le contour, ou None si les points
    sont confondus. La capsule est exprimée en coordonnées de données (pas en points).
    """
    vec = b - a
    length = np.linalg.norm(vec)
    if length < 1e-9:
        return None
    d = vec / length                                  # vecteur unitaire A→B
    n = np.array([-d[1], d[0]], dtype=float)          # normale perpendiculaire

    pts: list[np.ndarray] = []
    # Demi-cercle autour de A (côté opposé à B : angles π/2 → 3π/2)
    for angle in np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc + 1):
        pts.append(a + half_width * (np.cos(angle) * d + np.sin(angle) * n))
    # Demi-cercle autour de B (côté B : angles -π/2 → π/2)
    for angle in np.linspace(-np.pi / 2, np.pi / 2, n_arc + 1):
        pts.append(b + half_width * (np.cos(angle) * d + np.sin(angle) * n))
    return np.array(pts, dtype=float)


def draw_legend(ax, layers: dict[str, bool]) -> None:
    legend_elements = [
        Line2D([0], [0], marker="o", markersize=8, color="white", markerfacecolor="#9AA0A6", linestyle="None", label="Noeud = role"),
        Line2D([0], [0], color=SPEECH_COLOR, linewidth=2.5, label="Egalite de parole"),
    ]
    if layers.get("gaze"):
        legend_elements.append(Line2D([0], [0], color=GAZE_COLOR, linewidth=2.5, label="Regard vers pair"))
    if layers.get("jva"):
        legend_elements.append(Line2D([0], [0], color=JVA_COLOR, linewidth=2.5, label="Attention conjointe"))
    if layers.get("face_sync"):
        legend_elements.append(Line2D([0], [0], color=SYNC_COLOR, linewidth=6.0, alpha=0.25, label="Synchronie faciale"))

    leg = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fontsize=9,
        facecolor=PANEL_BG,
        edgecolor=PANEL_EDGE,
        labelcolor=TEXT_COLOR,
        title="Legende",
        title_fontsize=10,
    )
    leg.get_title().set_color(TEXT_COLOR)


def format_score(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/D"
    return f"{float(value):.2f}"


def draw_group_panel(ax, window_row: pd.Series) -> None:
    """Panneau groupe bas-gauche : leaders (avec carré coloré) + scores dimension."""
    # --- Section leaders ---
    leader_lines: list[tuple[str, str, str]] = [
        (window_row.get("talking_leader", ""), "Parle le plus", "talking"),
        (window_row.get("watched_leader", ""), "Recu regard max", "watched"),
        (window_row.get("task_focus_leader", ""), "Focus tache max", "task"),
    ]

    # Bloc texte gauche (leaders) — le carré coloré est affiché via ax.text Unicode ■
    leader_text_lines = []
    leader_colors: list[str] = []
    for role, label, _ in leader_lines:
        color = ROLE_COLORS.get(role, "#9AA0A6") if role else "#9AA0A6"
        short = ROLE_LABELS.get(role, "N/D") if role else "N/D"
        leader_text_lines.append(f"■ {label}: {short}")
        leader_colors.append(color)

    # Bloc texte droite (scores)
    score_lines = [
        f"Equilibre parole    {format_score(window_row.get('speech_balance_score'))}",
        f"Attention conjointe {format_score(window_row.get('joint_attention_score'))}",
        f"Synchronie faciale  {format_score(window_row.get('face_synchrony_score'))}",
        f"Engagement          {format_score(window_row.get('engagement_score'))}",
        f"Proximite           {format_score(window_row.get('proximity_score'))}",
        f"Proxy collab        {format_score(window_row.get('collaboration_proxy_score'))}",
    ]

    # Affichage : chaque ligne leader avec sa couleur, puis les scores en blanc
    panel_x, panel_y = 0.02, 0.02
    line_gap = 0.038
    fontsize = 9

    # Cadre arrière-plan
    n_lines = len(leader_text_lines) + 1 + len(score_lines)
    box_h = n_lines * line_gap + 0.04
    from matplotlib.patches import FancyBboxPatch as _FBP
    bg = _FBP(
        (panel_x - 0.01, panel_y - 0.01),
        0.36, box_h,
        boxstyle="round,pad=0.015",
        facecolor=PANEL_BG,
        edgecolor=PANEL_EDGE,
        alpha=0.92,
        transform=ax.transAxes,
        zorder=19,
        clip_on=False,
    )
    ax.add_patch(bg)

    y_cursor = panel_y + box_h - 0.025
    for text, color in zip(leader_text_lines, leader_colors):
        ax.text(
            panel_x + 0.01, y_cursor, text,
            transform=ax.transAxes,
            va="top", ha="left",
            color=color,
            fontsize=fontsize,
            weight="bold",
            zorder=20,
        )
        y_cursor -= line_gap

    y_cursor -= line_gap * 0.3  # petit séparateur
    for text in score_lines:
        ax.text(
            panel_x + 0.01, y_cursor, text,
            transform=ax.transAxes,
            va="top", ha="left",
            color=TEXT_COLOR,
            fontsize=fontsize,
            fontfamily="monospace",
            zorder=20,
        )
        y_cursor -= line_gap


def prepare_frame_axes(ax) -> None:
    ax.clear()
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(*AX_LIM_X)
    ax.set_ylim(*AX_LIM_Y)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_frame(
    ax,
    frame_idx: int,
    window_row: pd.Series,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    available_layers: dict[str, bool],
) -> None:
    prepare_frame_axes(ax)

    frame_nodes = nodes_df[nodes_df["frame_idx"] == frame_idx].copy()
    frame_edges = edges_df[edges_df["frame_idx"] == frame_idx].copy()

    distance_map = {
        row["pair"]: row["distance_m_display"]
        for _, row in frame_edges.iterrows()
    }
    coords = compute_triangle_layout(distance_map)
    coords_arr = {role: np.array(coords[role], dtype=float) for role in ROLE_ORDER}

    for _, edge in frame_edges.iterrows():
        role_a = edge["role_a"]
        role_b = edge["role_b"]
        a = coords_arr[role_a]
        b = coords_arr[role_b]

        # --- Capsule synchronie (engloble les deux noeuds, style MIRAGE) ---
        sync_score = pd.to_numeric(pd.Series([edge.get("face_sync_score")]), errors="coerce").iloc[0]
        if available_layers.get("face_sync") and pd.notna(sync_score):
            node_a_row = frame_nodes[frame_nodes["role"] == role_a]
            node_b_row = frame_nodes[frame_nodes["role"] == role_b]
            r_a = NODE_RADIUS_MIN + (NODE_RADIUS_MAX - NODE_RADIUS_MIN) * (
                float(node_a_row["speaking_ratio_window"].iloc[0]) if not node_a_row.empty else 0.5
            )
            r_b = NODE_RADIUS_MIN + (NODE_RADIUS_MAX - NODE_RADIUS_MIN) * (
                float(node_b_row["speaking_ratio_window"].iloc[0]) if not node_b_row.empty else 0.5
            )
            # La demi-largeur de la capsule varie entre 115 % et 220 % du plus grand rayon
            max_r = max(r_a, r_b)
            half_w = max_r * (1.15 + 1.05 * float(sync_score))
            cap_pts = make_capsule_polygon(a, b, half_w)
            if cap_pts is not None:
                from matplotlib.patches import Polygon as _Polygon
                cap = _Polygon(
                    cap_pts,
                    closed=True,
                    facecolor=SYNC_COLOR,
                    edgecolor=SYNC_COLOR,
                    linewidth=0.8 + 1.5 * float(sync_score),
                    alpha=0.06 + 0.14 * float(sync_score),
                    zorder=1,
                )
                ax.add_patch(cap)

        jva_ratio = pd.to_numeric(pd.Series([edge.get("jva_ratio")]), errors="coerce").iloc[0]
        if available_layers.get("jva") and pd.notna(jva_ratio):
            patch = quadratic_curve_patch(
                a,
                b,
                bulge=JVA_CURVE_BULGE,
                color=JVA_COLOR,
                linewidth=1.5 + 5.5 * float(jva_ratio),
                alpha=0.35 + 0.50 * float(jva_ratio),
            )
            patch.set_zorder(2)
            ax.add_patch(patch)

        speech_valid = bool(edge.get("speech_valid", False))
        speech_eq = edge.get("speech_equality")
        p1, p2 = edge_offset_points(a, b, +PAIR_OFFSET)
        if speech_valid and pd.notna(speech_eq):
            speech_eq = float(speech_eq)
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=SPEECH_COLOR,
                linewidth=1.2 + 4.0 * speech_eq,
                linestyle="solid" if speech_eq >= 0.35 else (0, (2, 3)),
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=INVALID_SPEECH_COLOR,
                linewidth=2.2,
                linestyle=(0, (2, 2)),
                alpha=0.95,
                zorder=3,
            )

        if available_layers.get("gaze"):
            # Arête gaze en bas (offset négatif)
            g1, g2 = edge_offset_points(a, b, -PAIR_OFFSET)
            # g1 = côté role_a, g2 = côté role_b
            gaze_ab = float(edge.get("gaze_a_to_b_ratio", 0.0) or 0.0)
            gaze_ba = float(edge.get("gaze_b_to_a_ratio", 0.0) or 0.0)
            gaze_sum = gaze_ab + gaze_ba
            ax.plot(
                [g1[0], g2[0]],
                [g1[1], g2[1]],
                color=GAZE_COLOR,
                linewidth=1.2 + 4.5 * clamp(gaze_sum, 0.0, 1.0),
                linestyle="solid" if gaze_sum > 0 else (0, (3, 4)),
                alpha=0.90,
                zorder=4,
            )

            # Flèches directionnelles sur l'arête offset (g1=rôle_a, g2=rôle_b)
            for src_pt, dst_pt, value in [
                (g1, g2, gaze_ab),   # role_a → role_b
                (g2, g1, gaze_ba),   # role_b → role_a
            ]:
                if value <= 0:
                    continue
                arrow = FancyArrowPatch(
                    posA=tuple(src_pt),
                    posB=tuple(dst_pt),
                    arrowstyle="-|>",
                    mutation_scale=10 + 20 * clamp(value, 0.0, 1.0),
                    linewidth=0.0,
                    color=GAZE_COLOR,
                    alpha=0.95,
                    shrinkA=20,
                    shrinkB=16,
                    zorder=5,
                )
                ax.add_patch(arrow)

    for _, node in frame_nodes.iterrows():
        role = node["role"]
        x, y = coords_arr[role]
        color = ROLE_COLORS[role]

        focus_ratio = pd.to_numeric(pd.Series([node.get("task_focus_ratio")]), errors="coerce").iloc[0]
        if available_layers.get("task_focus") and pd.notna(focus_ratio):
            halo_radius = (NODE_RADIUS_MIN + (NODE_RADIUS_MAX - NODE_RADIUS_MIN) * float(node["speaking_ratio_window"])) * (
                HALO_SCALE_MIN + (HALO_SCALE_MAX - HALO_SCALE_MIN) * float(focus_ratio)
            )
            halo = Circle((x, y), halo_radius, facecolor=color, edgecolor="none", alpha=0.16, zorder=6)
            ax.add_patch(halo)

        radius = NODE_RADIUS_MIN + (NODE_RADIUS_MAX - NODE_RADIUS_MIN) * float(node["speaking_ratio_window"])
        circle = Circle((x, y), radius, facecolor=color, edgecolor="white", linewidth=2.0, alpha=0.98, zorder=8)
        ax.add_patch(circle)
        # % parole à l'intérieur du noeud (visible si le rayon est assez grand)
        speak_pct = float(node["speaking_ratio_window"])
        ax.text(
            x,
            y,
            f"{speak_pct:.0%}",
            ha="center",
            va="center",
            color="white",
            fontsize=max(7, int(9 * radius / NODE_RADIUS_MAX)),
            weight="bold",
            zorder=9,
            path_effects=[pe.withStroke(linewidth=1.5, foreground=color)],
        )
        ax.text(
            x,
            y + radius + 0.10,
            ROLE_LABELS[role],
            ha="center",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=12,
            weight="bold",
            zorder=9,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=BG_COLOR)],
        )

    title = (
        f"MIRAGE adapte - {window_row['group_id']} | {window_row['condition']} | "
        f"{window_row['scenario']} | {window_row['timepoint']}\n"
        f"Fenetre {window_row['t_start']:.1f}s -> {window_row['t_end']:.1f}s"
    )
    ax.text(
        0.5,
        0.98,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=14,
        weight="bold",
    )

    draw_legend(ax, available_layers)
    draw_group_panel(ax, window_row)


def render_frame(
    out_path: Path,
    frame_idx: int,
    window_row: pd.Series,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    available_layers: dict[str, bool],
    dpi: int = 180,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    draw_frame(
        ax=ax,
        frame_idx=frame_idx,
        window_row=window_row,
        nodes_df=nodes_df,
        edges_df=edges_df,
        available_layers=available_layers,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def build_visualization_tables(
    group_dir: Path,
    window_s: float,
    step_s: float,
    thr_db: float,
    merge_gap: float,
    min_speech_total_s: float,
    min_speech_role_s: float,
    *,
    start_at: Optional[float] = None,
    end_at: Optional[float] = None,
    max_windows: Optional[int] = None,
    audio_sr: int = DEFAULT_AUDIO_SR,
    beep_scan_s: float = DEFAULT_BEEP_SCAN_S,
    debug_timings: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    condition = extract_condition(group_dir) or "UNK"
    scenario = extract_scenario(group_dir) or "UNK"
    timepoint = extract_timepoint(group_dir)

    t_load = time.perf_counter()
    segs_by_role, audio_end = load_audio_segments(
        group_dir,
        thr_db=thr_db,
        merge_gap=merge_gap,
        audio_sr=audio_sr,
        beep_scan_s=beep_scan_s,
        debug=debug_timings,
    )
    debug_log(debug_timings, f"audio charge en {time.perf_counter() - t_load:.2f}s")

    t_load = time.perf_counter()
    face_streams = load_face_streams(group_dir)
    debug_log(debug_timings, f"face charge en {time.perf_counter() - t_load:.2f}s")

    eye_samples: dict[str, pd.DataFrame] = {}
    positions: dict[str, pd.DataFrame] = {}
    fix_df = pd.DataFrame()
    if condition == "VR":
        try:
            t_load = time.perf_counter()
            eye_samples = load_vr_eye_samples(group_dir)
            debug_log(debug_timings, f"eye samples charges en {time.perf_counter() - t_load:.2f}s")
            t_load = time.perf_counter()
            positions = load_vr_positions(group_dir)
            debug_log(debug_timings, f"positions chargees en {time.perf_counter() - t_load:.2f}s")
            t_load = time.perf_counter()
            fix_df = load_fixations_from_eye_samples(eye_samples)
            debug_log(debug_timings, f"fixations calculees en {time.perf_counter() - t_load:.2f}s")
        except Exception as exc:
            print(f"[WARN] Gaze / positions indisponibles pour {group_dir.name}: {exc}")
            eye_samples = {}
            positions = {}
            fix_df = pd.DataFrame(columns=["participant", "object_id", "onset", "offset"])

    available_layers = {
        "speech": True,
        "gaze": bool(eye_samples and positions),
        "jva": bool(not fix_df.empty),
        "face_sync": bool(face_streams),
        "task_focus": bool(not fix_df.empty),
        "proximity": bool(positions),
    }

    end_candidates = [audio_end]
    if face_streams:
        end_candidates.append(min(float(df["time"].max()) for df in face_streams.values()))
    if eye_samples:
        end_candidates.append(min(float(df["time"].max()) for df in eye_samples.values()))
    if positions:
        end_candidates.append(min(float(df["time"].max()) for df in positions.values()))
    global_end = min(v for v in end_candidates if np.isfinite(v) and v > 0)
    global_start = 0.0
    if start_at is not None:
        global_start = max(global_start, float(start_at))
    if end_at is not None:
        global_end = min(global_end, float(end_at))
    if global_end <= global_start:
        raise RuntimeError(f"Intervalle temporel invalide apres filtrage: {global_start:.1f}s -> {global_end:.1f}s")

    windows = window_list(global_start, global_end, window_s, step_s)
    if max_windows is not None:
        windows = windows[: max(0, int(max_windows))]
    if not windows:
        raise RuntimeError("Aucune fenetre a calculer avec les bornes temporelles demandees.")
    debug_log(debug_timings, f"{len(windows)} fenetres a calculer entre {global_start:.1f}s et {global_end:.1f}s")
    fix_task = fix_df[~fix_df["object_id"].apply(is_floor_like)].copy() if not fix_df.empty else fix_df.copy()

    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    progress_every = 25 if len(windows) <= 250 else 100

    for frame_idx, (start, end) in enumerate(windows):
        if debug_timings and (frame_idx == 0 or (frame_idx + 1) % progress_every == 0 or frame_idx == len(windows) - 1):
            debug_log(True, f"fenetre {frame_idx + 1}/{len(windows)} [{start:.1f}s, {end:.1f}s]")
        duration = max(end - start, 1e-6)
        speaking_s = {role: overlap_duration(segs_by_role[role], start, end) for role in ROLE_ORDER}
        speaking_ratio_window = {role: clamp(speaking_s[role] / duration, 0.0, 1.0) for role in ROLE_ORDER}

        task_focus_by_role: dict[str, float] = {role: np.nan for role in ROLE_ORDER}
        watched_incoming: dict[str, float] = {role: 0.0 for role in ROLE_ORDER}

        for role in ROLE_ORDER:
            if available_layers["task_focus"]:
                focus_ratio, n_task_fix = task_focus_ratio(fix_task, role, start, end)
            else:
                focus_ratio, n_task_fix = np.nan, 0
            task_focus_by_role[role] = focus_ratio
            node_rows.append(
                {
                    "frame_idx": frame_idx,
                    "group_id": group_dir.name,
                    "condition": condition,
                    "scenario": scenario,
                    "timepoint": timepoint,
                    "t_start": start,
                    "t_end": end,
                    "role": role,
                    "speaking_time_s": speaking_s[role],
                    "speaking_ratio_window": speaking_ratio_window[role],
                    "task_focus_ratio": focus_ratio,
                    "task_fixation_count": n_task_fix,
                }
            )

        speech_scores = []
        jva_scores = []
        face_scores = []
        pair_distance_values = []

        for role_a, role_b in combinations(ROLE_ORDER, 2):
            da = speaking_s[role_a]
            db = speaking_s[role_b]
            speech_valid = (da + db) >= min_speech_total_s and min(da, db) >= min_speech_role_s
            speech_eq = 1.0 - abs(da - db) / max(da + db, 1e-6) if speech_valid else np.nan
            if pd.notna(speech_eq):
                speech_scores.append(float(speech_eq))

            if available_layers["jva"]:
                pair_fix = fix_task[
                    (fix_task["participant"].isin([role_a, role_b]))
                    & (fix_task["onset"] < end)
                    & (fix_task["offset"] > start)
                ]
                jva_metrics = shared_object_metrics(
                    pair_fix,
                    t0=float(start),
                    t1=float(end),
                    fs_grid=20.0,
                    min_participants=2,
                    overlap_min_s=0.10,
                )
                jva_ratio = float(jva_metrics["shared_ratio"]) if np.isfinite(jva_metrics["shared_ratio"]) else np.nan
            else:
                jva_ratio = np.nan
                jva_metrics = {"n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}
            if pd.notna(jva_ratio):
                jva_scores.append(jva_ratio)

            if available_layers["gaze"]:
                gaze_ab = build_directional_peer_gaze_window(
                    eye_samples[role_a],
                    positions[role_b],
                    start,
                    end,
                    radius_m=PEER_GAZE_RADIUS_M,
                )
                gaze_ba = build_directional_peer_gaze_window(
                    eye_samples[role_b],
                    positions[role_a],
                    start,
                    end,
                    radius_m=PEER_GAZE_RADIUS_M,
                )
            else:
                gaze_ab = {"ratio": np.nan, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}
                gaze_ba = {"ratio": np.nan, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}

            if pd.notna(gaze_ab["ratio"]):
                watched_incoming[role_b] += float(gaze_ab["ratio"])
            if pd.notna(gaze_ba["ratio"]):
                watched_incoming[role_a] += float(gaze_ba["ratio"])

            face_score = face_sync_score(face_streams[role_a], face_streams[role_b], start, end) if available_layers["face_sync"] else np.nan
            if pd.notna(face_score):
                face_scores.append(float(face_score))

            distance_m = mean_pair_distance(positions[role_a], positions[role_b], start, end) if available_layers["proximity"] else np.nan
            if pd.notna(distance_m):
                pair_distance_values.append(float(distance_m))

            edge_rows.append(
                {
                    "frame_idx": frame_idx,
                    "group_id": group_dir.name,
                    "condition": condition,
                    "scenario": scenario,
                    "timepoint": timepoint,
                    "t_start": start,
                    "t_end": end,
                    "pair": pair_key(role_a, role_b),
                    "role_a": role_a,
                    "role_b": role_b,
                    "speech_equality": speech_eq,
                    "speech_valid": speech_valid,
                    "gaze_a_to_b_ratio": gaze_ab["ratio"],
                    "gaze_b_to_a_ratio": gaze_ba["ratio"],
                    "gaze_a_to_b_episodes": gaze_ab["n_episodes"],
                    "gaze_b_to_a_episodes": gaze_ba["n_episodes"],
                    "jva_ratio": jva_ratio,
                    "jva_n_episodes": jva_metrics["n_episodes"],
                    "jva_dur_total_s": jva_metrics["dur_total_s"],
                    "face_sync_score": face_score,
                    "distance_m_raw": distance_m,
                    "distance_m_display": compress_distance_meters(distance_m),
                }
            )

        talking_leader = max(ROLE_ORDER, key=lambda role: speaking_s[role]) if speaking_s else None
        watched_leader = max(ROLE_ORDER, key=lambda role: watched_incoming[role]) if any(v > 0 for v in watched_incoming.values()) else None
        task_focus_leader = (
            max(ROLE_ORDER, key=lambda role: (-1 if pd.isna(task_focus_by_role[role]) else float(task_focus_by_role[role])))
            if any(pd.notna(v) for v in task_focus_by_role.values())
            else None
        )

        # Score engagement = moyenne(focus tâche, regard reçu normalisé par rôle)
        engagement_components: list[float] = []
        max_watched = max(watched_incoming.values()) if any(v > 0 for v in watched_incoming.values()) else 0.0
        for role in ROLE_ORDER:
            tf = task_focus_by_role[role]
            wr = watched_incoming[role] / max(max_watched, 1e-6) if max_watched > 0 else 0.0
            if pd.notna(tf):
                engagement_components.append((float(tf) + float(wr)) / 2.0)
        engagement_score = safe_mean(engagement_components)

        window_rows.append(
            {
                "frame_idx": frame_idx,
                "group_id": group_dir.name,
                "condition": condition,
                "scenario": scenario,
                "timepoint": timepoint,
                "t_start": start,
                "t_end": end,
                "speech_balance_score": safe_mean(speech_scores),
                "joint_attention_score": safe_mean(jva_scores),
                "face_synchrony_score": safe_mean(face_scores),
                "task_focus_score": safe_mean([task_focus_by_role[role] for role in ROLE_ORDER]),
                "engagement_score": engagement_score,
                "mean_pair_distance_m": safe_mean(pair_distance_values),
                "talking_leader": talking_leader,
                "watched_leader": watched_leader,
                "task_focus_leader": task_focus_leader,
                "talking_leader_label": ROLE_LABELS_LONG.get(talking_leader, "N/D") if talking_leader else "N/D",
                "watched_leader_label": ROLE_LABELS_LONG.get(watched_leader, "N/D") if watched_leader else "N/D",
                "task_focus_leader_label": ROLE_LABELS_LONG.get(task_focus_leader, "N/D") if task_focus_leader else "N/D",
            }
        )

    windows_df = pd.DataFrame(window_rows)
    if not windows_df.empty and windows_df["mean_pair_distance_m"].notna().sum() >= 2:
        dmin = float(windows_df["mean_pair_distance_m"].min())
        dmax = float(windows_df["mean_pair_distance_m"].max())
        if dmax > dmin:
            windows_df["proximity_score"] = 1.0 - (windows_df["mean_pair_distance_m"] - dmin) / (dmax - dmin)
        else:
            windows_df["proximity_score"] = 1.0
    else:
        windows_df["proximity_score"] = np.nan

    component_cols = [
        "speech_balance_score",
        "joint_attention_score",
        "face_synchrony_score",
        "task_focus_score",
        "proximity_score",
    ]
    windows_df["collaboration_proxy_score"] = windows_df[component_cols].mean(axis=1, skipna=True)

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    return windows_df, nodes_df, edges_df, available_layers


def select_snapshot_frame(windows_df: pd.DataFrame, snapshot_at: Optional[float]) -> int:
    if windows_df.empty:
        raise RuntimeError("Aucune fenetre calculee pour le sociogramme.")
    if snapshot_at is None:
        return int(windows_df["frame_idx"].iloc[len(windows_df) // 2])
    idx = (windows_df["t_end"] - snapshot_at).abs().idxmin()
    return int(windows_df.loc[idx, "frame_idx"])


def resolve_frame_ids(windows_df: pd.DataFrame, max_frames: Optional[int], frame_stride: int = 1) -> list[int]:
    """Retourne la liste des frame_idx à exporter, avec sous-échantillonnage optionnel.

    Parameters
    ----------
    frame_stride : int
        Ne retenir qu'une frame sur ``frame_stride`` (1 = toutes, 10 = une sur dix).
        Utile pour réduire la taille du GIF sans recalculer les fenêtres.
    """
    frame_ids = windows_df["frame_idx"].astype(int).tolist()
    stride = max(1, int(frame_stride))
    frame_ids = frame_ids[::stride]
    if max_frames is not None:
        frame_ids = frame_ids[: max(0, int(max_frames))]
    return frame_ids


def export_frame_sequence(
    frames_dir: Path,
    frame_ids: Sequence[int],
    windows_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    available_layers: dict[str, bool],
    frame_dpi: int,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for frame_id in frame_ids:
        frame_png = frames_dir / f"frame_{frame_id:04d}.png"
        if not frame_png.exists():
            render_frame(
                out_path=frame_png,
                frame_idx=frame_id,
                window_row=windows_df[windows_df["frame_idx"] == frame_id].iloc[0],
                nodes_df=nodes_df,
                edges_df=edges_df,
                available_layers=available_layers,
                dpi=frame_dpi,
            )
        frame_paths.append(frame_png)
    return frame_paths


def export_gif_from_frames(frame_paths: Sequence[Path], out_path: Path, duration_ms: int) -> bool:
    if not frame_paths:
        raise RuntimeError("Aucune frame disponible pour exporter le GIF.")
    if Image is None:
        return False

    images = [Image.open(path) for path in frame_paths]
    try:
        first, rest = images[0], images[1:]
        first.save(
            out_path,
            save_all=True,
            append_images=rest,
            duration=max(20, int(duration_ms)),
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()
    return True


def show_live_animation(
    windows_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    available_layers: dict[str, bool],
    frame_ids: Sequence[int],
    interval_ms: int,
) -> None:
    if not frame_ids:
        raise RuntimeError("Aucune frame disponible pour l'animation live.")

    fig, ax = plt.subplots(figsize=(11, 8), dpi=120)
    fig.patch.set_facecolor(BG_COLOR)

    state = {"paused": False, "index": 0}
    def _draw_by_index(index: int) -> None:
        state["index"] = int(index)
        frame_id = int(frame_ids[index])
        window_row = windows_df[windows_df["frame_idx"] == frame_id].iloc[0]
        draw_frame(
            ax=ax,
            frame_idx=frame_id,
            window_row=window_row,
            nodes_df=nodes_df,
            edges_df=edges_df,
            available_layers=available_layers,
        )
        ax.text(
            0.98,
            0.02,
            "Espace: pause/reprise | Gauche/Droite: frame precedente/suivante",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=TEXT_COLOR,
            fontsize=9,
            alpha=0.8,
        )

    _draw_by_index(0)
    plt.tight_layout()

    def _update(index: int):
        if state["paused"]:
            return []
        _draw_by_index(index)
        return []

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(frame_ids),
        interval=max(20, int(interval_ms)),
        blit=False,
        repeat=False,
    )

    def _toggle_pause() -> None:
        state["paused"] = not state["paused"]
        if state["paused"]:
            anim.event_source.stop()
        else:
            anim.event_source.start()

    def _jump(delta: int) -> None:
        new_index = int(clamp(state["index"] + delta, 0, len(frame_ids) - 1))
        state["paused"] = True
        anim.event_source.stop()
        _draw_by_index(new_index)
        fig.canvas.draw_idle()

    def _on_key(event) -> None:
        key = (event.key or "").lower()
        if key in {" ", "space"}:
            _toggle_pause()
        elif key == "right":
            _jump(+1)
        elif key == "left":
            _jump(-1)
        elif key == "home":
            _jump(-len(frame_ids))
        elif key == "end":
            _jump(len(frame_ids))

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genere un sociogramme MIRAGE adapte au projet Longitudinale.")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/data_e2"), help="Racine des donnees brutes.")
    parser.add_argument("--group-dir", type=Path, help="Chemin direct vers un groupe (prioritaire).")
    parser.add_argument("--group-id", type=str, help="ID du groupe, ex. bim073.")
    parser.add_argument("--modality", type=str, choices=["VR", "PC"], help="Filtre modalite si --group-id est utilise.")
    parser.add_argument("--scenario", type=str, choices=["S1", "S2"], help="Filtre scenario si --group-id est utilise.")
    parser.add_argument("--timepoint", type=str, choices=["T1", "T2"], help="Filtre timepoint si --group-id est utilise.")
    parser.add_argument("--out-dir", type=Path, help="Dossier de sortie. Defaut: results/visualisation_sociale/<group>.")
    parser.add_argument("--window-s", type=float, default=20.0, help="Taille de fenetre glissante en secondes.")
    parser.add_argument("--step-s", type=float, default=1.0, help="Pas de la fenetre glissante en secondes.")
    parser.add_argument("--start-at", type=float, help="Debut optionnel de l'intervalle analyse (s).")
    parser.add_argument("--end-at", type=float, help="Fin optionnelle de l'intervalle analyse (s).")
    parser.add_argument("--snapshot-at", type=float, help="Instant (s) auquel prendre le snapshot.")
    parser.add_argument("--thr-db", type=float, default=DEFAULT_THR_DB, help="Seuil VAD audio.")
    parser.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP, help="Fusion micro-pauses pour l'audio.")
    parser.add_argument("--min-speech-total-s", type=float, default=2.0, help="Parole totale minimale pour valider l'egalite dyadique.")
    parser.add_argument("--min-speech-role-s", type=float, default=0.5, help="Parole minimale par role pour valider l'egalite dyadique.")
    parser.add_argument("--audio-sr", type=int, default=DEFAULT_AUDIO_SR, help="Frequence cible pour le traitement audio.")
    parser.add_argument("--beep-scan-s", type=float, default=DEFAULT_BEEP_SCAN_S, help="Duree maximale scannee pour detecter le bip VR.")
    parser.add_argument("--debug-timings", action="store_true", help="Affiche des timings de chargement et de progression.")
    parser.add_argument("--export-frames", action="store_true", help="Export toutes les frames PNG.")
    parser.add_argument("--export-gif", action="store_true", help="Export une animation GIF si Pillow est disponible.")
    parser.add_argument("--show-live", action="store_true", help="Ouvre une fenetre Matplotlib pour lire l'animation du sociogramme.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="Vitesse de lecture live relative au pas temporel.")
    parser.add_argument("--frame-dpi", type=int, default=110, help="Resolution des frames d'animation.")
    parser.add_argument("--max-frames", type=int, help="Limite optionnelle sur le nombre de frames exportees.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Ne garder qu'une frame sur N pour l'export (1=toutes, 10=une sur dix). Reduit la taille du GIF.")
    parser.add_argument("--max-windows", type=int, help="Limite optionnelle sur le nombre de fenetres calculees.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.group_dir is None and not args.group_id:
        parser.error("Fournir soit --group-dir, soit --group-id.")

    if args.group_dir is not None:
        group_dir = args.group_dir
    else:
        group_dir = resolve_group_dir(
            data_dir=args.data_dir,
            group_id=args.group_id,
            modality=args.modality,
            scenario=args.scenario,
            timepoint=args.timepoint,
        )

    if not group_dir.exists():
        raise FileNotFoundError(f"Groupe introuvable : {group_dir}")

    out_dir = args.out_dir or (
        Path(__file__).resolve().parents[2]
        / "results"
        / "visualisation_sociale"
        / group_dir.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_df, nodes_df, edges_df, available_layers = build_visualization_tables(
        group_dir=group_dir,
        window_s=args.window_s,
        step_s=args.step_s,
        thr_db=args.thr_db,
        merge_gap=args.merge_gap,
        min_speech_total_s=args.min_speech_total_s,
        min_speech_role_s=args.min_speech_role_s,
        start_at=args.start_at,
        end_at=args.end_at,
        max_windows=args.max_windows,
        audio_sr=args.audio_sr,
        beep_scan_s=args.beep_scan_s,
        debug_timings=args.debug_timings,
    )

    windows_df.to_csv(out_dir / "window_summary.csv", index=False, encoding="utf-8-sig")
    nodes_df.to_csv(out_dir / "node_metrics.csv", index=False, encoding="utf-8-sig")
    edges_df.to_csv(out_dir / "edge_metrics.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "group_dir": str(group_dir),
        "group_id": group_dir.name,
        "condition": extract_condition(group_dir),
        "scenario": extract_scenario(group_dir),
        "timepoint": extract_timepoint(group_dir),
        "window_s": args.window_s,
        "step_s": args.step_s,
        "start_at": args.start_at,
        "end_at": args.end_at,
        "audio_sr": args.audio_sr,
        "beep_scan_s": args.beep_scan_s,
        "show_live": bool(args.show_live),
        "playback_speed": args.playback_speed,
        "available_layers": available_layers,
        "n_frames": int(len(windows_df)),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    snapshot_frame = select_snapshot_frame(windows_df, args.snapshot_at)
    snapshot_png = out_dir / "mirage_snapshot.png"
    render_frame(
        out_path=snapshot_png,
        frame_idx=snapshot_frame,
        window_row=windows_df[windows_df["frame_idx"] == snapshot_frame].iloc[0],
        nodes_df=nodes_df,
        edges_df=edges_df,
        available_layers=available_layers,
    )

    frame_paths: list[Path] = []
    if args.export_frames or args.export_gif:
        frame_ids = resolve_frame_ids(windows_df, args.max_frames, frame_stride=args.frame_stride)
        frame_paths = export_frame_sequence(
            frames_dir=out_dir / "frames",
            frame_ids=frame_ids,
            windows_df=windows_df,
            nodes_df=nodes_df,
            edges_df=edges_df,
            available_layers=available_layers,
            frame_dpi=args.frame_dpi,
        )

    gif_path = out_dir / "mirage_animation.gif"
    if args.export_gif:
        try:
            duration_ms = int(round(args.step_s * 1000.0))
            exported = export_gif_from_frames(frame_paths, gif_path, duration_ms=duration_ms)
            if not exported:
                print("[WARN] Export GIF indisponible: Pillow n'est pas installe.")
        except Exception as exc:
            print(f"[WARN] Echec export GIF: {exc}")

    if args.show_live:
        frame_ids = resolve_frame_ids(windows_df, args.max_frames, frame_stride=args.frame_stride)
        interval_ms = int(round((args.step_s * 1000.0) / max(args.playback_speed, 1e-6)))
        show_live_animation(
            windows_df=windows_df,
            nodes_df=nodes_df,
            edges_df=edges_df,
            available_layers=available_layers,
            frame_ids=frame_ids,
            interval_ms=interval_ms,
        )

    print(f"[OK] Snapshot : {snapshot_png}")
    if args.export_gif and gif_path.exists():
        print(f"[OK] GIF : {gif_path}")
    print(f"[OK] Tableaux : {out_dir}")


if __name__ == "__main__":
    main()
