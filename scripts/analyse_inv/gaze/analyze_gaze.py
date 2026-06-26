#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_gaze.py — Analyse du regard par direction (sans maillage BIM).

Remplace l'ancienne analyse par objet (BIM mesh) pour les sessions VR dont les CSV corrigés sont disponibles
(produits par reconstruct_eyetracking.py + refine_yaw_eyetracking.py).

Principe : toutes les métriques sont calculées depuis DirCorrX/Y/Z (yaw corrigé si
disponible) ou DirX/Y/Z (brut) avec un flag `dir_source` indiquant lequel est utilisé.
ObjectHit et HitPoint sont ignorés — seule la DIRECTION compte.

Métriques produites :
  Groupe (overall + fenêtres glissantes) :
    - gaze_convergence_ratio    : ratio de temps où ≥2 participants regardent dans la même
                                  direction (angle inter-direction < seuil)
    - gaze_convergence_mean_angle_deg : angle médian dans les épisodes de convergence
    - mutual_gaze_ratio         : ratio de temps où A regarde vers B ET B vers A
                                  (angle(DirA, B_head - A_head) < seuil)
    - gaze_entropy_dir          : entropie directionnelle (dispersion des azimuts en bins)
    - gaze_focus_proxy          : 1 - gaze_entropy_dir (normalisé)

  Paires :
    - pair_convergence_ratio, pair_mutual_gaze_ratio

  Participants :
    - gaze_entropy_dir, gaze_focus_proxy, pct_dir_valid, dir_source

Usage :
    python analyze_gaze.py \
        --corrected-dir D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected \
        --out-dir       D:/Analyse_donnee/Longitudinale/results/INV/gaze_directional

    # Groupes spécifiques
    python analyze_gaze.py \
        --corrected-dir D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected \
        --out-dir D:/Analyse_donnee/Longitudinale/results/INV/gaze_directional \
        --groups bim015 bim066

Paramètres clés :
    --convergence-angle-deg  20.0   Seuil pour "regardent dans la même direction"
    --mutual-gaze-angle-deg  30.0   Seuil pour "regarde vers la tête de l'autre"
    --stable-dir-min-s       0.15   Durée minimale d'un épisode de direction stable
    --win                    30.0   Fenêtre glissante (s)
    --step                   30.0   Pas de la fenêtre (s)
    --fs-grid                20.0   Fréquence de la grille de temps (Hz)
"""

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.metadata import extract_condition, extract_scenario, extract_timepoint, upsert_meta_cols
from common.temporal import sliding_windows, episodes_from_bool
from common.stats import shannon_entropy_bits


# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
CONVERGENCE_ANGLE_DEG_DEFAULT = 20.0   # seuil convergence directionnelle
MUTUAL_GAZE_ANGLE_DEG_DEFAULT = 30.0   # seuil regard vers tête de l'autre
STABLE_DIR_MIN_S_DEFAULT      = 0.15   # durée min épisode de direction stable
N_AZIMUTH_BINS                = 16     # bins pour l'entropie directionnelle (360°/16 = 22.5°)
INTERP_MAX_GAP_S              = 1.0    # gap max pour interpoler TrueOrigin (sinon NaN)


# ---------------------------------------------------------------------------
# I/O : découverte et lecture des CSV corrigés
# ---------------------------------------------------------------------------

def _find_corrected_files(corrected_dir: Path, groups: Optional[list[str]] = None) -> dict[str, dict[str, Path]]:
    """
    Retourne {session_key: {role: path}} groupés par (timepoint, scenario, group).
    session_key = "timepoint__scenario__group"
    """
    sessions: dict[str, dict[str, Path]] = {}
    for f in sorted(corrected_dir.glob("*_EyeTrackingData_corrected.csv")):
        stem = f.name.replace("_EyeTrackingData_corrected.csv", "")
        parts = stem.split("__")
        if len(parts) != 4:
            continue
        tp, sc, grp, role = parts
        if groups and grp not in groups:
            continue
        key = f"{tp}__{sc}__{grp}"
        sessions.setdefault(key, {})[role] = f
    return sessions


def _find_source_eye_file(data_dir: Path, tp: str, sc: str, grp: str, role: str) -> Optional[Path]:
    """Retrouve le fichier EyeTrackingData source (hors _old/) pour un rôle donné."""
    role_dir = data_dir / tp / "VR" / sc / grp / role
    if not role_dir.exists():
        return None
    candidates = [p for p in role_dir.glob("*EyeTrackingData.csv") if "_old" not in str(p)]
    if not candidates:
        return None
    # Préfère merged s'il existe
    merged = [p for p in candidates if "merged" in p.name]
    return merged[0] if merged else candidates[0]


def _extract_marker_time(eye_source_path: Path) -> Optional[float]:
    """
    Lit le fichier EyeTrackingData source et retourne le temps Unity du MARKER.
    Retourne None si aucun MARKER trouvé.
    """
    try:
        raw = eye_source_path.read_bytes().decode("utf-8-sig", errors="replace")
        for line in raw.splitlines():
            if "MARKER" in line:
                t_str = line.split(";")[0].strip().replace(",", ".")
                try:
                    return float(t_str)
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def _load_corrected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=".", engine="python")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object and col not in ("ObjectHit", "direction_assumption"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_dir_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """
    Retourne (DirX_col, DirY_col, DirZ_col, source) en préférant DirCorr si disponible.
    """
    has_corr = (
        "DirCorrX" in df.columns and "DirCorrY" in df.columns and "DirCorrZ" in df.columns
        and df["DirCorrX"].notna().any()
    )
    if has_corr:
        return "DirCorrX", "DirCorrY", "DirCorrZ", "corr"
    return "DirX", "DirY", "DirZ", "raw"


def _resample_to_grid(df: pd.DataFrame, t0: float, t1: float, fs: float,
                      dx_col: str, dy_col: str, dz_col: str) -> np.ndarray:
    """
    Rééchantillonne DirX/Y/Z sur une grille régulière [t0, t1) à fs Hz.
    Retourne tableau (N_grid, 3) avec NaN aux instants sans données valides.
    Interpolation linéaire par segment continu (gaps > INTERP_MAX_GAP_S → NaN).
    """
    grid = np.arange(t0, t1, 1.0 / fs)
    N = len(grid)
    result = np.full((N, 3), np.nan)

    t = df["Time"].values
    dx = df[dx_col].values
    dy = df[dy_col].values
    dz = df[dz_col].values

    valid = ~(np.isnan(dx) | np.isnan(dy) | np.isnan(dz))
    if not valid.any():
        return result

    t_v = t[valid]
    dx_v = dx[valid]
    dy_v = dy[valid]
    dz_v = dz[valid]

    # Interpolation composante par composante
    for i, comp in enumerate([dx_v, dy_v, dz_v]):
        interp = np.interp(grid, t_v, comp, left=np.nan, right=np.nan)
        # Annule les zones de trop grand gap
        if INTERP_MAX_GAP_S < (t1 - t0):
            # Pour chaque point de la grille, cherche le gap dans les données source
            idx_right = np.searchsorted(t_v, grid, side="left").clip(0, len(t_v) - 1)
            idx_left = (idx_right - 1).clip(0, len(t_v) - 1)
            gap_left  = np.where(idx_right > 0, grid - t_v[idx_left], np.inf)
            gap_right = np.where(idx_right < len(t_v), t_v[idx_right] - grid, np.inf)
            gap = np.minimum(gap_left, gap_right)
            interp[gap > INTERP_MAX_GAP_S] = np.nan
        result[:, i] = interp

    # Re-normalise (l'interpolation peut déformer légèrement la norme)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    safe = norms[:, 0] > 1e-9
    result[safe] = result[safe] / norms[safe]
    result[~safe] = np.nan

    return result


# ---------------------------------------------------------------------------
# MÉTRIQUES DIRECTIONNELLES
# ---------------------------------------------------------------------------

def _angle_between_dirs(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Angle (degrés) entre deux tableaux de vecteurs (N, 3).
    Retourne NaN où l'un des vecteurs est NaN.
    """
    dot = np.einsum("ij,ij->i", d1, d2).clip(-1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    nan_mask = np.any(np.isnan(d1) | np.isnan(d2), axis=1)
    angle[nan_mask] = np.nan
    return angle


def _angle_to_head(dir_a: np.ndarray, pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """
    Angle (degrés) entre DirA et le vecteur (pos_B - pos_A).
    Tous tableaux (N, 3). Retourne NaN où données manquantes.
    """
    vec = pos_b - pos_a
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    safe = norms[:, 0] > 1e-6
    vec_n = np.where(np.column_stack([safe]*3), vec / np.where(norms > 0, norms, 1), np.nan)
    angle = _angle_between_dirs(dir_a, vec_n)
    nan_mask = ~safe | np.any(np.isnan(pos_a) | np.isnan(pos_b), axis=1)
    angle[nan_mask] = np.nan
    return angle


def _resample_position(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Rééchantillonne TrueOriginX/Y/Z sur la grille."""
    result = np.full((len(grid), 3), np.nan)
    for i, col in enumerate(["TrueOriginX", "TrueOriginY", "TrueOriginZ"]):
        if col not in df.columns:
            continue
        t = df["Time"].values
        v = df[col].values
        valid = ~np.isnan(v)
        if not valid.any():
            continue
        result[:, i] = np.interp(grid, t[valid], v[valid], left=np.nan, right=np.nan)
    return result


def convergence_metrics(
    dirs: dict[str, np.ndarray],  # {role: (N, 3)}
    fs: float,
    threshold_deg: float,
    overlap_min_s: float = 0.10,
) -> dict:
    """
    Ratio de temps où ≥2 participants regardent dans des directions convergentes.
    """
    roles = list(dirs.keys())
    if len(roles) < 2:
        return {"gaze_convergence_ratio": np.nan, "gaze_convergence_mean_angle_deg": np.nan,
                "gaze_convergence_n_episodes": 0, "gaze_convergence_dur_total_s": 0.0}

    N = next(iter(dirs.values())).shape[0]
    converge = np.zeros(N, dtype=bool)
    angles_in_conv = []

    for r1, r2 in itertools.combinations(roles, 2):
        angle = _angle_between_dirs(dirs[r1], dirs[r2])
        pair_conv = angle <= threshold_deg
        pair_conv[np.isnan(angle)] = False
        converge |= pair_conv
        if pair_conv.any():
            angles_in_conv.extend(angle[pair_conv & ~np.isnan(angle)].tolist())

    # Filtre micro-épisodes
    n, tot, mean = episodes_from_bool(converge, fs)
    if n > 0 and overlap_min_s > 0:
        edges = np.diff(np.concatenate(([False], converge, [False])).astype(np.int8))
        on_ix = np.where(edges == 1)[0]
        off_ix = np.where(edges == -1)[0]
        durs = (off_ix - on_ix) / fs
        keep = durs >= overlap_min_s
        if not np.all(keep):
            conv2 = np.zeros_like(converge)
            for a, b, k in zip(on_ix, off_ix, keep):
                if k:
                    conv2[a:b] = True
            converge = conv2
            n, tot, mean = episodes_from_bool(converge, fs)

    valid_frames = np.sum(~np.all(np.isnan(next(iter(dirs.values()))), axis=1))
    ratio = float(np.sum(converge) / max(valid_frames, 1))

    return {
        "gaze_convergence_ratio": ratio,
        "gaze_convergence_mean_angle_deg": float(np.mean(angles_in_conv)) if angles_in_conv else np.nan,
        "gaze_convergence_n_episodes": n,
        "gaze_convergence_dur_total_s": tot,
    }


def mutual_gaze_directional(
    dirs: dict[str, np.ndarray],    # {role: (N, 3)}
    positions: dict[str, np.ndarray],  # {role: (N, 3)}
    fs: float,
    threshold_deg: float,
    overlap_min_s: float = 0.10,
) -> dict:
    """
    Regard mutuel directionnel : A regarde vers la tête de B ET B vers la tête de A.
    Utilise TrueOrigin comme position de tête.
    """
    roles = list(dirs.keys())
    if len(roles) < 2:
        return {"mutual_gaze_ratio": np.nan, "mutual_gaze_n_episodes": 0,
                "mutual_gaze_dur_total_s": 0.0, "mutual_gaze_dur_mean_s": np.nan}

    N = next(iter(dirs.values())).shape[0]
    any_mutual = np.zeros(N, dtype=bool)
    pair_ratios = []

    for r1, r2 in itertools.combinations(roles, 2):
        if r1 not in positions or r2 not in positions:
            continue
        angle_12 = _angle_to_head(dirs[r1], positions[r1], positions[r2])
        angle_21 = _angle_to_head(dirs[r2], positions[r2], positions[r1])
        m12 = angle_12 <= threshold_deg
        m21 = angle_21 <= threshold_deg
        m12[np.isnan(angle_12)] = False
        m21[np.isnan(angle_21)] = False
        mutual = m12 & m21

        # Filtre micro-épisodes
        n, tot, _ = episodes_from_bool(mutual, fs)
        if n > 0 and overlap_min_s > 0:
            edges = np.diff(np.concatenate(([False], mutual, [False])).astype(np.int8))
            on_ix = np.where(edges == 1)[0]
            off_ix = np.where(edges == -1)[0]
            durs = (off_ix - on_ix) / fs
            keep = durs >= overlap_min_s
            if not np.all(keep):
                mut2 = np.zeros_like(mutual)
                for a, b, k in zip(on_ix, off_ix, keep):
                    if k:
                        mut2[a:b] = True
                mutual = mut2

        any_mutual |= mutual
        valid = ~(np.isnan(angle_12) & np.isnan(angle_21))
        pair_ratios.append(float(np.sum(mutual) / max(np.sum(valid), 1)))

    n, tot, mean_dur = episodes_from_bool(any_mutual, fs)
    ratio = float(np.mean(pair_ratios)) if pair_ratios else np.nan

    return {
        "mutual_gaze_ratio": ratio,
        "mutual_gaze_n_episodes": n,
        "mutual_gaze_dur_total_s": tot,
        "mutual_gaze_dur_mean_s": mean_dur,
    }


def directional_entropy(dir_grid: np.ndarray, n_bins: int = N_AZIMUTH_BINS) -> float:
    """
    Entropie de la distribution des azimuts (projection XZ) sur n_bins.
    Mesure la dispersion directionnelle d'un participant.
    Retourne NaN si moins de 10 frames valides.
    """
    valid = ~np.any(np.isnan(dir_grid), axis=1)
    if np.sum(valid) < 10:
        return np.nan
    dx = dir_grid[valid, 0]
    dz = dir_grid[valid, 2]
    azimuth = np.degrees(np.arctan2(dx, dz)) % 360.0
    bin_edges = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(azimuth, bins=bin_edges)
    return float(shannon_entropy_bits(counts.astype(float), normalize=True))


def stable_dir_episodes(dir_grid: np.ndarray, fs: float, min_dur_s: float,
                        angle_threshold_deg: float = 10.0) -> pd.DataFrame:
    """
    Détecte les épisodes où la direction reste stable (variation angulaire < threshold sur
    une fenêtre glissante de min_dur_s). Retourne DataFrame avec onset/offset.
    """
    N = dir_grid.shape[0]
    k = max(1, int(min_dur_s * fs))
    stable = np.zeros(N, dtype=bool)

    for i in range(N - k + 1):
        window = dir_grid[i:i + k]
        valid = ~np.any(np.isnan(window), axis=1)
        if valid.sum() < max(2, k // 2):
            continue
        w_v = window[valid]
        # angle max entre la direction moyenne et chaque frame
        mean_dir = w_v.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-9:
            continue
        mean_dir /= norm
        dots = np.einsum("ij,j->i", w_v, mean_dir).clip(-1, 1)
        max_angle = np.degrees(np.arccos(dots)).max()
        if max_angle <= angle_threshold_deg:
            stable[i:i + k] = True

    n, tot, mean = episodes_from_bool(stable, fs)
    if n == 0:
        return pd.DataFrame(columns=["onset", "offset", "duration_s"])

    edges = np.diff(np.concatenate(([False], stable, [False])).astype(np.int8))
    on_ix  = np.where(edges ==  1)[0]
    off_ix = np.where(edges == -1)[0]
    dt = 1.0 / fs
    rows = [{"onset": on_ix[i] * dt, "offset": off_ix[i] * dt,
             "duration_s": (off_ix[i] - on_ix[i]) * dt}
            for i in range(len(on_ix))]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ANALYSE PAR GROUPE
# ---------------------------------------------------------------------------

def run_one_group(
    session_key: str,
    role_paths: dict[str, Path],
    args,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Traite un groupe (3 rôles). session_key = "timepoint__scenario__group".
    """
    parts = session_key.split("__")
    tp, sc, grp = parts[0], parts[1], parts[2]
    data_dir: Optional[Path] = getattr(args, "data_dir", None)

    # Lecture des CSV corrigés
    role_dfs: dict[str, pd.DataFrame] = {}
    for role, path in role_paths.items():
        try:
            df = _load_corrected(path)
            if df.empty or "Time" not in df.columns:
                print(f"[WARN] {session_key}/{role}: CSV vide ou sans colonne Time")
                continue
            role_dfs[role] = df
        except Exception as e:
            print(f"[WARN] {session_key}/{role}: lecture échouée — {e}")

    if len(role_dfs) < 2:
        print(f"[WARN] {session_key}: moins de 2 rôles valides ({list(role_dfs.keys())})")
        return None, None, None

    # --- Calage sur le MARKER (même logique qu'analyze_gaze.py) ---
    # t0 = max des temps MARKER parmi les rôles (début de tâche commun)
    # Si data_dir non fourni ou MARKER introuvable, repli sur intersection des plages Time.
    t_maxs_data = [df["Time"].max() for df in role_dfs.values()]

    if args.t0 is not None:
        t0 = float(args.t0)
    elif data_dir is not None:
        marker_times = []
        for role in role_dfs:
            src = _find_source_eye_file(data_dir, tp, sc, grp, role)
            if src is not None:
                tm = _extract_marker_time(src)
                if tm is not None:
                    marker_times.append(tm)
        if marker_times:
            # t0 = MARKER le plus tardif (tous les rôles sont prêts)
            t0 = float(max(marker_times))
        else:
            # Repli : intersection des plages Time
            t_mins = [df["Time"].min() for df in role_dfs.values()]
            t0 = float(max(t_mins))
            print(f"[WARN] {session_key}: aucun MARKER trouvé, repli sur intersection")
    else:
        t_mins = [df["Time"].min() for df in role_dfs.values()]
        t0 = float(max(t_mins))

    t1 = args.t1 if args.t1 is not None else float(min(t_maxs_data))
    # Cas sessions multi-fichiers (bim002) : plages non recouvrantes → union
    if t1 <= t0:
        t0 = float(min(df["Time"].min() for df in role_dfs.values()))
        t1 = float(max(t_maxs_data))
    if t1 <= t0:
        print(f"[WARN] {session_key}: plage temporelle invalide t0={t0:.1f} t1={t1:.1f}")
        return None, None, None

    fs = float(args.fs_grid)
    grid = np.arange(t0, t1, 1.0 / fs)
    N = len(grid)
    interaction_dur = t1 - t0

    conv_thr  = float(args.convergence_angle_deg)
    mut_thr   = float(args.mutual_gaze_angle_deg)
    overlap_s = float(args.overlap_ms) / 1000.0

    # Rééchantillonnage directions et positions sur la grille
    dirs_grid:      dict[str, np.ndarray] = {}
    positions_grid: dict[str, np.ndarray] = {}
    dir_sources:    dict[str, str]        = {}

    for role, df in role_dfs.items():
        # Filtre sur la plage temporelle
        df_w = df[(df["Time"] >= t0) & (df["Time"] <= t1)].copy()
        cx, cy, cz, src = _get_dir_columns(df_w)
        dir_sources[role] = src
        dirs_grid[role]      = _resample_to_grid(df_w, t0, t1, fs, cx, cy, cz)
        positions_grid[role] = _resample_position(df_w, grid)

    roles = list(role_dfs.keys())

    # ---- Métriques globales ----
    m_conv = convergence_metrics(dirs_grid, fs, conv_thr, overlap_s)
    m_mut  = mutual_gaze_directional(dirs_grid, positions_grid, fs, mut_thr, overlap_s)

    ent_vals = {r: directional_entropy(dirs_grid[r]) for r in roles}
    ent_mean = float(np.nanmean(list(ent_vals.values())))

    def norm(x):
        return float(x / interaction_dur) if (interaction_dur > 0 and np.isfinite(x)) else np.nan

    overall_row = {
        "interaction_duration_s": interaction_dur,
        "gaze_convergence_ratio": m_conv["gaze_convergence_ratio"],
        "gaze_convergence_mean_angle_deg": m_conv["gaze_convergence_mean_angle_deg"],
        "gaze_convergence_n_episodes": m_conv["gaze_convergence_n_episodes"],
        "gaze_convergence_dur_total_s": m_conv["gaze_convergence_dur_total_s"],
        "gaze_convergence_n_episodes_per_s": norm(m_conv["gaze_convergence_n_episodes"]),
        "mutual_gaze_ratio": m_mut["mutual_gaze_ratio"],
        "mutual_gaze_n_episodes": m_mut["mutual_gaze_n_episodes"],
        "mutual_gaze_dur_total_s": m_mut["mutual_gaze_dur_total_s"],
        "mutual_gaze_dur_mean_s": m_mut["mutual_gaze_dur_mean_s"],
        "mutual_gaze_n_episodes_per_s": norm(m_mut["mutual_gaze_n_episodes"]),
        "gaze_entropy_dir_mean": ent_mean,
        "gaze_focus_proxy": float(1.0 - ent_mean) if np.isfinite(ent_mean) else np.nan,
        "dir_source": "+".join(sorted(set(dir_sources.values()))),
        "n_roles": len(roles),
    }

    overall = pd.DataFrame([overall_row])

    # ---- Métriques paires ----
    pair_rows = []
    for r1, r2 in itertools.combinations(roles, 2):
        d_sub  = {r1: dirs_grid[r1],      r2: dirs_grid[r2]}
        p_sub  = {r1: positions_grid[r1], r2: positions_grid[r2]}
        mc = convergence_metrics(d_sub, fs, conv_thr, overlap_s)
        mm = mutual_gaze_directional(d_sub, p_sub, fs, mut_thr, overlap_s)
        pair_rows.append({
            "pair": f"{r1}+{r2}",
            "pair_convergence_ratio": mc["gaze_convergence_ratio"],
            "pair_convergence_mean_angle_deg": mc["gaze_convergence_mean_angle_deg"],
            "pair_convergence_n_episodes": mc["gaze_convergence_n_episodes"],
            "pair_convergence_dur_total_s": mc["gaze_convergence_dur_total_s"],
            "pair_mutual_gaze_ratio": mm["mutual_gaze_ratio"],
            "pair_mutual_gaze_n_episodes": mm["mutual_gaze_n_episodes"],
            "pair_mutual_gaze_dur_total_s": mm["mutual_gaze_dur_total_s"],
            "pair_mutual_gaze_dur_mean_s": mm["mutual_gaze_dur_mean_s"],
        })
    pairs_df = pd.DataFrame(pair_rows)

    # ---- Métriques participants ----
    part_rows = []
    for role in roles:
        dg = dirs_grid[role]
        valid_frames = int(~np.any(np.isnan(dg), axis=1).sum() if dg.ndim == 2 else 0)
        pct_valid = float(valid_frames / max(N, 1))
        ent = ent_vals[role]
        part_rows.append({
            "participant": role,
            "gaze_entropy_dir": ent,
            "gaze_focus_proxy": float(1.0 - ent) if np.isfinite(ent) else np.nan,
            "pct_dir_valid": pct_valid,
            "dir_source": dir_sources[role],
            "interaction_duration_s": interaction_dur,
        })
    participants_df = pd.DataFrame(part_rows)

    # ---- Fenêtres glissantes ----
    win_rows = []
    for s, e in sliding_windows(t0, t1, args.win, args.step):
        gi0 = int((s - t0) * fs)
        gi1 = int((e - t0) * fs)
        gi0 = max(gi0, 0); gi1 = min(gi1, N)
        dirs_w = {r: dirs_grid[r][gi0:gi1] for r in roles}
        pos_w  = {r: positions_grid[r][gi0:gi1] for r in roles}
        mc_w = convergence_metrics(dirs_w, fs, conv_thr, overlap_s)
        mm_w = mutual_gaze_directional(dirs_w, pos_w, fs, mut_thr, overlap_s)
        win_rows.append({
            "t_start": s, "t_end": e,
            "gaze_convergence_ratio": mc_w["gaze_convergence_ratio"],
            "gaze_convergence_n_episodes": mc_w["gaze_convergence_n_episodes"],
            "gaze_convergence_dur_total_s": mc_w["gaze_convergence_dur_total_s"],
            "mutual_gaze_ratio": mm_w["mutual_gaze_ratio"],
            "mutual_gaze_n_episodes": mm_w["mutual_gaze_n_episodes"],
        })
    windows_df = pd.DataFrame(win_rows)

    # ---- Écriture ----
    out_dir = Path(args.out_dir) / grp
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = "VR"  # ces fichiers sont tous VR
    scenario  = sc
    # tp contient la valeur brute du dossier (ex: "T1_BSI_A1") — on normalise en T1/T2
    tp_upper = tp.upper()
    if tp_upper.startswith("T1"):
        timepoint = "T1"
    elif tp_upper.startswith("T2"):
        timepoint = "T2"
    else:
        timepoint = tp

    overall       = upsert_meta_cols(overall,       grp, condition, scenario, timepoint)
    pairs_df      = upsert_meta_cols(pairs_df,       grp, condition, scenario, timepoint)
    participants_df = upsert_meta_cols(participants_df, grp, condition, scenario, timepoint)
    windows_df    = upsert_meta_cols(windows_df,    grp, condition, scenario, timepoint)

    overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    pairs_df.to_csv(out_dir / "metrics_pairs.csv", index=False)
    participants_df.to_csv(out_dir / "metrics_participants.csv", index=False)
    windows_df.to_csv(out_dir / "metrics_windows.csv", index=False)

    print(f"[OK] {session_key} ({len(roles)} roles, {interaction_dur:.0f}s) -> {out_dir}")
    return overall, pairs_df, participants_df


# ---------------------------------------------------------------------------
# CLI / MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse gaze directionnelle VR (sans maillage BIM)."
    )
    parser.add_argument(
        "--corrected-dir",
        default="D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected",
        type=Path,
    )
    parser.add_argument(
        "--data-dir", default="D:/data_e2", type=Path,
        help="Dossier donnees brutes pour lire les MARKERs (defaut: D:/data_e2)"
    )
    parser.add_argument(
        "--out-dir",
        default="D:/Analyse_donnee/Longitudinale/results/INV/gaze_directional",
        type=Path,
    )
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Filtrer sur des groupes spécifiques (ex: bim015 bim066)")
    parser.add_argument("--win",  type=float, default=30.0, help="Fenetre glissante (s)")
    parser.add_argument("--step", type=float, default=30.0, help="Pas fenetre (s)")
    parser.add_argument("--fs-grid", type=float, default=20.0, help="Frequence grille (Hz)")
    parser.add_argument("--t0", type=float, default=None, help="Debut (s)")
    parser.add_argument("--t1", type=float, default=None, help="Fin (s)")
    parser.add_argument("--convergence-angle-deg", type=float, default=CONVERGENCE_ANGLE_DEG_DEFAULT)
    parser.add_argument("--mutual-gaze-angle-deg", type=float, default=MUTUAL_GAZE_ANGLE_DEG_DEFAULT)
    parser.add_argument("--stable-dir-min-s", type=float, default=STABLE_DIR_MIN_S_DEFAULT)
    parser.add_argument("--overlap-ms", type=float, default=100.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = _find_corrected_files(args.corrected_dir, args.groups)
    print(f"[Gaze directionnel] {len(sessions)} sessions trouvees.")

    all_overall, all_pairs, all_parts = [], [], []

    for key, role_paths in sorted(sessions.items()):
        try:
            o, p, u = run_one_group(key, role_paths, args)
            if o is not None:  all_overall.append(o)
            if p is not None and not p.empty: all_pairs.append(p)
            if u is not None and not u.empty: all_parts.append(u)
        except Exception as e:
            import traceback
            print(f"[WARN] Echec {key}: {e}")
            traceback.print_exc()

    if all_overall:
        pd.concat(all_overall, ignore_index=True).to_csv(
            args.out_dir / "ALL_metrics_overall.csv", index=False)
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            args.out_dir / "ALL_metrics_pairs.csv", index=False)
    if all_parts:
        pd.concat(all_parts, ignore_index=True).to_csv(
            args.out_dir / "ALL_metrics_participants.csv", index=False)

    print(f"[OK] Master files -> {args.out_dir}")
    n_ok = len(all_overall)
    n_yaw = sum(1 for df in all_overall
                if "dir_source" in df.columns and "corr" in str(df["dir_source"].iloc[0]))
    print(f"     {n_ok} groupes traites, {n_yaw} avec DirCorr (yaw corrige)")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Fonctions partagées — conservées pour compatibilité avec mirage_sociogram.py
# (analyse par objet/BIM, non utilisées par le pipeline directionnelle)
# ---------------------------------------------------------------------------

def detect_fixations_per_participant(gaze_df: pd.DataFrame, min_fix: float = 0.2) -> pd.DataFrame:
    """Détecte les fixations par participant à partir d'un DataFrame objet (ObjectHit)."""
    gaze_df = gaze_df.sort_values(["participant", "time"])
    out = []
    for pid, g in gaze_df.groupby("participant"):
        arr = g[["time", "object_id"]].to_numpy()
        if arr.size == 0:
            continue
        i = 0
        while i < len(arr):
            start = i
            oid = arr[i, 1]
            while i + 1 < len(arr) and arr[i + 1, 1] == oid:
                i += 1
            t_start = float(arr[start, 0])
            t_end = float(arr[i, 0])
            if (t_end - t_start) >= min_fix:
                out.append((pid, oid, t_start, t_end))
            i += 1
    if not out:
        return pd.DataFrame(columns=["participant", "object_id", "onset", "offset"])
    return pd.DataFrame(out, columns=["participant", "object_id", "onset", "offset"])


def shared_object_metrics(
    fix_df: pd.DataFrame,
    t0: float,
    t1: float,
    fs_grid: float,
    min_participants: int = 2,
    overlap_min_s: float = 0.10,
) -> Dict[str, float]:
    """Épisodes d'attention partagée (≥min_participants fixent le même objet simultanément)."""
    _empty = {
        "shared_ratio": np.nan, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan,
        "dur_median_s": np.nan, "dur_q25_s": np.nan, "dur_q75_s": np.nan, "dur_iqr_s": np.nan,
    }
    if fix_df.empty:
        return _empty

    grid = np.arange(t0, t1, 1.0 / fs_grid)
    if grid.size == 0:
        return _empty

    per_obj_counts: Dict[str, np.ndarray] = {}
    for _, r in fix_df.iterrows():
        oid = r["object_id"]
        if oid not in per_obj_counts:
            per_obj_counts[oid] = np.zeros_like(grid, dtype=np.int32)
        m = (grid >= max(float(r["onset"]), t0)) & (grid < min(float(r["offset"]), t1))
        per_obj_counts[oid][m] += 1

    sameobj = np.zeros_like(grid, dtype=bool)
    for arr in per_obj_counts.values():
        sameobj |= (arr >= min_participants)

    n, tot, mean = episodes_from_bool(sameobj, fs_grid)

    if n > 0 and overlap_min_s > 0:
        edges = np.diff(np.concatenate(([False], sameobj, [False])).astype(np.int8))
        on_ix = np.where(edges == 1)[0]
        off_ix = np.where(edges == -1)[0]
        durs = (off_ix - on_ix) / fs_grid
        keep = durs >= overlap_min_s
        if not np.all(keep):
            sameobj2 = np.zeros_like(sameobj)
            for a, b, k in zip(on_ix, off_ix, keep):
                if k:
                    sameobj2[a:b] = True
            sameobj = sameobj2
            n, tot, mean = episodes_from_bool(sameobj, fs_grid)

    shared_ratio = float(np.mean(sameobj)) if grid.size else np.nan

    edges2 = np.diff(np.concatenate(([False], sameobj, [False])).astype(np.int8))
    on_ix2 = np.where(edges2 == 1)[0]
    off_ix2 = np.where(edges2 == -1)[0]
    episodes_dur = list((off_ix2 - on_ix2) / fs_grid) if len(on_ix2) > 0 else []

    if episodes_dur:
        shared_dur_median = float(np.median(episodes_dur))
        shared_dur_q25    = float(np.percentile(episodes_dur, 25))
        shared_dur_q75    = float(np.percentile(episodes_dur, 75))
        shared_dur_iqr    = shared_dur_q75 - shared_dur_q25
    else:
        shared_dur_median = shared_dur_q25 = shared_dur_q75 = shared_dur_iqr = np.nan

    return {
        "shared_ratio":    shared_ratio,
        "n_episodes":      n,
        "dur_total_s":     tot,
        "dur_mean_s":      mean,
        "dur_median_s":    shared_dur_median,
        "dur_q25_s":       shared_dur_q25,
        "dur_q75_s":       shared_dur_q75,
        "dur_iqr_s":       shared_dur_iqr,
    }
