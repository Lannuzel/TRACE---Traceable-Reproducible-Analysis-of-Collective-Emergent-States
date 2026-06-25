#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse gaze (v2) : attention partagée + mutual gaze + entropy + (optionnel) speaker metrics.

Entrées attendues (gaze) :
  - Dossier groupe contenant sous-dossiers participants, chacun avec "*_EyeTrackingData.csv"
  - Après MARKER : Time;...;ObjectHit
  - ObjectHit (string) : objet regardé / collider / etc.

Optionnel (parole / speaking turns) :
  - CSV par groupe ou global avec colonnes :
      speaker,onset,offset
    où onset/offset sont en secondes depuis le début interaction (même origine que gaze après MARKER).

Sorties par groupe (dans <out_dir>/<group_id>/) :
  - shared_object_windows.csv
  - shared_object_pairs_windows.csv
  - metrics_overall.csv
  - metrics_pairs.csv
  - metrics_participants.csv

+ master files en mode batch.
"""

import os
import re
import sys
import glob
import argparse
import itertools
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd

# ---- common package imports ------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.metadata import extract_condition, extract_scenario, extract_timepoint, upsert_meta_cols
from common.io_utils import find_gaze_groups
from common.temporal import sliding_windows, episodes_from_bool
from common.stats import shannon_entropy_bits


# ----------------------------- I/O & Parsing gaze ----------------------------- #

def _find_marker_time(lines: List[str]) -> Tuple[float, int]:
    for i, line in enumerate(lines):
        if "MARKER" in line:
            tokens = re.split(r"[;\s]+", line.strip())
            for tok in tokens:
                t = tok.replace(",", ".")
                try:
                    return float(t), i
                except ValueError:
                    continue
            raise RuntimeError(f"Ligne MARKER trouvée mais temps introuvable : {line.strip()}")
    raise RuntimeError("MARKER introuvable.")


def _parse_eye_file_to_gaze_df(file_path: str, participant_name: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    t_marker, idx_marker = _find_marker_time(lines)

    rows = []
    for line in lines[idx_marker + 1:]:
        parts = line.strip().split(";", 7)
        if len(parts) < 8:
            continue

        t_str = parts[0].replace(",", ".").strip()
        obj = parts[7].strip()

        if not t_str:
            continue
        try:
            t = float(t_str) - t_marker
        except ValueError:
            continue
        if t < 0:
            continue

        if obj == "" or obj.upper() == "NONE":
            continue

        rows.append((participant_name, t, obj))

    if not rows:
        return pd.DataFrame(columns=["participant", "time", "object_id"])

    df = pd.DataFrame(rows, columns=["participant", "time", "object_id"])
    df = df.sort_values(["participant", "time"]).reset_index(drop=True)
    return df


def load_group_gaze(group_dir: str) -> pd.DataFrame:
    all_parts = []
    for p in os.listdir(group_dir):
        pdir = os.path.join(group_dir, p)
        if not os.path.isdir(pdir):
            continue

        candidates = sorted(glob.glob(os.path.join(pdir, "*_EyeTrackingData.csv")))
        if not candidates:
            continue

        eye_file = max(candidates, key=os.path.getmtime)

        try:
            dfp = _parse_eye_file_to_gaze_df(eye_file, participant_name=p)
            if not dfp.empty:
                all_parts.append(dfp)
            else:
                import logging; logging.warning(f"Données gaze vides après parsing: {group_dir}/{p}")
        except Exception as e:
            import logging; logging.warning(f"Échec parsing {eye_file} : {e}")

    if not all_parts:
        raise RuntimeError(f"Aucune donnée gaze valide trouvée dans {group_dir}")

    return pd.concat(all_parts, ignore_index=True)


# ----------------------------- Fixations ----------------------------- #

def detect_fixations_per_participant(gaze_df: pd.DataFrame, min_fix: float = 0.2) -> pd.DataFrame:
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


# ----------------------------- Entropy ----------------------------- #

def gaze_entropy_participant(fix_df: pd.DataFrame, normalize: bool = True) -> pd.DataFrame:
    """
    Entropy over object_id distribution (by total fixation duration) per participant.
    """
    rows = []
    for pid, g in fix_df.groupby("participant"):
        # duration per object
        dur = (g["offset"] - g["onset"]).clip(lower=0)
        by_obj = dur.groupby(g["object_id"]).sum()
        H = shannon_entropy_bits(by_obj.to_numpy(), normalize=normalize)
        rows.append({"participant": pid, "gaze_entropy": H, "n_objects_fixated": int(by_obj.size)})
    return pd.DataFrame(rows)


# ----------------------------- Shared visual attention (same object) ----------------------------- #

def shared_object_metrics(
    fix_df: pd.DataFrame,
    t0: float,
    t1: float,
    fs_grid: float,
    min_participants: int = 2,
    overlap_min_s: float = 0.10,
) -> Dict[str, float]:
    """
    Compute shared attention episodes where >=min_participants fixate the same object simultaneously.
    Uses regular grid to define overlap and episodes.
    """
    _empty_result = {
        "shared_ratio": np.nan, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan,
        "dur_median_s": np.nan, "dur_q25_s": np.nan, "dur_q75_s": np.nan, "dur_iqr_s": np.nan,
    }
    if fix_df.empty:
        return _empty_result

    grid = np.arange(t0, t1, 1.0 / fs_grid)
    if grid.size == 0:
        return _empty_result

    per_obj_counts = {}
    for _, r in fix_df.iterrows():
        oid = r["object_id"]
        if oid not in per_obj_counts:
            per_obj_counts[oid] = np.zeros_like(grid, dtype=np.int32)
        m = (grid >= max(float(r["onset"]), t0)) & (grid < min(float(r["offset"]), t1))
        per_obj_counts[oid][m] += 1

    sameobj = np.zeros_like(grid, dtype=bool)
    for arr in per_obj_counts.values():
        sameobj |= (arr >= min_participants)

    # enforce minimal overlap duration per episode via post-filtering on the boolean mask
    n, tot, mean = episodes_from_bool(sameobj, fs_grid)

    # remove micro-episodes < overlap_min_s
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

    # MOD-10 : statistiques robustes (distribution skewed, SD/mean > 0.85 empirique)
    edges2 = np.diff(np.concatenate(([False], sameobj, [False])).astype(np.int8))
    on_ix2 = np.where(edges2 == 1)[0]
    off_ix2 = np.where(edges2 == -1)[0]
    episodes_dur = list((off_ix2 - on_ix2) / fs_grid) if len(on_ix2) > 0 else []

    if episodes_dur:
        shared_dur_median = float(np.median(episodes_dur))
        shared_dur_q25 = float(np.percentile(episodes_dur, 25))
        shared_dur_q75 = float(np.percentile(episodes_dur, 75))
        shared_dur_iqr = shared_dur_q75 - shared_dur_q25
    else:
        shared_dur_median = np.nan
        shared_dur_q25 = np.nan
        shared_dur_q75 = np.nan
        shared_dur_iqr = np.nan

    return {
        "shared_ratio": shared_ratio,
        "n_episodes": n,
        "dur_total_s": tot,
        "dur_mean_s": mean,
        "dur_median_s": shared_dur_median,
        "dur_q25_s": shared_dur_q25,
        "dur_q75_s": shared_dur_q75,
        "dur_iqr_s": shared_dur_iqr,
    }


def shared_object_windows(
    fix_df: pd.DataFrame,
    t0: float,
    t1: float,
    win: float,
    step: float,
    fs_grid: float,
    overlap_min_s: float,
) -> pd.DataFrame:
    rows = []
    for s, e in sliding_windows(t0, t1, win, step):
        chunk = fix_df[(fix_df["onset"] < e) & (fix_df["offset"] > s)]
        m = shared_object_metrics(chunk, s, e, fs_grid, min_participants=2, overlap_min_s=overlap_min_s)
        rows.append({
            "t_start": s, "t_end": e,
            "shared_obj_ratio": m["shared_ratio"],
            "shared_obj_n_episodes": m["n_episodes"],
            "shared_obj_dur_total_s": m["dur_total_s"],
            "shared_obj_dur_mean_s": m["dur_mean_s"],
            "shared_obj_dur_median_s": m["dur_median_s"],
            "shared_obj_dur_q25_s": m["dur_q25_s"],
            "shared_obj_dur_q75_s": m["dur_q75_s"],
            "shared_obj_dur_iqr_s": m["dur_iqr_s"],
        })
    return pd.DataFrame(rows)


def shared_object_pairs_windows(
    fix_df: pd.DataFrame,
    participants: List[str],
    t0: float,
    t1: float,
    win: float,
    step: float,
    fs_grid: float,
    overlap_min_s: float,
) -> pd.DataFrame:
    rows = []
    for p1, p2 in itertools.combinations(participants, 2):
        f1 = fix_df[fix_df["participant"] == p1]
        f2 = fix_df[fix_df["participant"] == p2]
        for s, e in sliding_windows(t0, t1, win, step):
            c1 = f1[(f1["onset"] < e) & (f1["offset"] > s)]
            c2 = f2[(f2["onset"] < e) & (f2["offset"] > s)]
            if c1.empty or c2.empty:
                rows.append({
                    "pair": f"{p1}+{p2}", "t_start": s, "t_end": e,
                    "pair_shared_obj_ratio": np.nan,
                    "pair_shared_obj_n_episodes": 0,
                    "pair_shared_obj_dur_total_s": 0.0,
                    "pair_shared_obj_dur_mean_s": np.nan,
                })
                continue

            chunk = pd.concat([c1, c2], ignore_index=True)
            m = shared_object_metrics(chunk, s, e, fs_grid, min_participants=2, overlap_min_s=overlap_min_s)
            rows.append({
                "pair": f"{p1}+{p2}", "t_start": s, "t_end": e,
                "pair_shared_obj_ratio": m["shared_ratio"],
                "pair_shared_obj_n_episodes": m["n_episodes"],
                "pair_shared_obj_dur_total_s": m["dur_total_s"],
                "pair_shared_obj_dur_mean_s": m["dur_mean_s"],
                "pair_shared_obj_dur_median_s": m["dur_median_s"],
                "pair_shared_obj_dur_q25_s": m["dur_q25_s"],
                "pair_shared_obj_dur_q75_s": m["dur_q75_s"],
                "pair_shared_obj_dur_iqr_s": m["dur_iqr_s"],
            })
    return pd.DataFrame(rows)


# ----------------------------- Mutual gaze ----------------------------- #

def infer_target_participant(object_id: str, participants: List[str]) -> Optional[str]:
    """
    Heuristique : si object_id contient le nom d'un participant (case-insensitive),
    on considère que l'utilisateur regarde cette personne.
    """
    u = object_id.upper()
    hits = [p for p in participants if p.upper() in u]
    if len(hits) == 1:
        return hits[0]
    return None


def mutual_gaze_pair_metrics(
    fix_df: pd.DataFrame,
    p1: str,
    p2: str,
    t0: float,
    t1: float,
    fs_grid: float,
    sync_lag_s: float = 0.0,
    overlap_min_s: float = 0.10,
) -> Dict[str, float]:
    """
    Mutual gaze episode = p1 looks at p2 AND p2 looks at p1 (simultaneous on grid).
    sync_lag_s allows dilating masks (tolerance).
    """
    grid = np.arange(t0, t1, 1.0 / fs_grid)
    if grid.size == 0:
        return {"ratio": np.nan, "n_episodes": 0, "dur_total_s": 0.0, "dur_mean_s": np.nan}

    def mask_looks_at(src: str, tgt: str) -> np.ndarray:
        g = fix_df[fix_df["participant"] == src]
        msk = np.zeros_like(grid, dtype=bool)
        for _, r in g.iterrows():
            oid = str(r["object_id"])
            inferred = infer_target_participant(oid, [tgt])
            if inferred != tgt:
                continue
            a = max(float(r["onset"]), t0)
            b = min(float(r["offset"]), t1)
            mm = (grid >= a) & (grid < b)
            msk |= mm
        if sync_lag_s > 0:
            k = int(np.ceil(sync_lag_s * fs_grid))
            if k > 0:
                dil = msk.copy()
                for sh in range(1, k + 1):
                    dil |= np.roll(msk, sh)
                    dil |= np.roll(msk, -sh)
                dil[:k] |= msk[:k]
                dil[-k:] |= msk[-k:]
                msk = dil
        return msk

    p1_to_p2 = mask_looks_at(p1, p2)
    p2_to_p1 = mask_looks_at(p2, p1)

    mutual = p1_to_p2 & p2_to_p1

    n, tot, mean = episodes_from_bool(mutual, fs_grid)
    if n > 0 and overlap_min_s > 0:
        edges = np.diff(np.concatenate(([False], mutual, [False])).astype(np.int8))
        on_ix = np.where(edges == 1)[0]
        off_ix = np.where(edges == -1)[0]
        durs = (off_ix - on_ix) / fs_grid
        keep = durs >= overlap_min_s
        if not np.all(keep):
            mutual2 = np.zeros_like(mutual)
            for a, b, k in zip(on_ix, off_ix, keep):
                if k:
                    mutual2[a:b] = True
            mutual = mutual2
            n, tot, mean = episodes_from_bool(mutual, fs_grid)

    ratio = float(np.mean(mutual)) if grid.size else np.nan
    return {"ratio": ratio, "n_episodes": n, "dur_total_s": tot, "dur_mean_s": mean}


def mutual_gaze_group_metrics(
    fix_df: pd.DataFrame,
    participants: List[str],
    t0: float,
    t1: float,
    fs_grid: float,
    sync_lag_s: float,
    overlap_min_s: float,
) -> Dict[str, float]:
    """Group-level mutual gaze ratio: average of pair ratios."""
    vals = []
    n_eps = []
    tot_dur = []
    mean_dur = []
    for p1, p2 in itertools.combinations(participants, 2):
        m = mutual_gaze_pair_metrics(
            fix_df, p1, p2, t0, t1, fs_grid,
            sync_lag_s=sync_lag_s, overlap_min_s=overlap_min_s
        )
        if np.isfinite(m["ratio"]):
            vals.append(m["ratio"])
        n_eps.append(m["n_episodes"])
        tot_dur.append(m["dur_total_s"])
        mean_dur.append(m["dur_mean_s"])

    return {
        "mutual_gaze_ratio_mean_pairs": float(np.mean(vals)) if vals else np.nan,
        "mutual_gaze_n_episodes_sum_pairs": int(np.sum(n_eps)) if n_eps else 0,
        "mutual_gaze_dur_total_s_sum_pairs": float(np.sum(tot_dur)) if tot_dur else 0.0,
        "mutual_gaze_dur_mean_s_mean_pairs": float(np.nanmean(mean_dur)) if mean_dur else np.nan,
    }


# ----------------------------- Speaking turns (optional) ----------------------------- #

def load_speaking_turns_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    for c in ["speaker", "participant"]:
        if c in df.columns:
            df["speaker"] = df[c].astype(str)
            break
    if "speaker" not in df.columns:
        raise ValueError("Speaking turns CSV must contain a 'speaker' (or 'participant') column.")

    for c in ["onset", "start", "start_s", "t_start"]:
        if c in df.columns:
            df["onset"] = pd.to_numeric(df[c], errors="coerce")
            break
    for c in ["offset", "end", "end_s", "t_end"]:
        if c in df.columns:
            df["offset"] = pd.to_numeric(df[c], errors="coerce")
            break

    if "onset" not in df.columns or "offset" not in df.columns:
        raise ValueError("Speaking turns CSV must contain onset/offset columns (e.g., onset,offset).")

    df = df.dropna(subset=["speaker", "onset", "offset"]).copy()
    df = df[df["offset"] > df["onset"]].copy()
    return df[["speaker", "onset", "offset"]].sort_values(["onset", "offset"]).reset_index(drop=True)


def gaze_to_speaker_ratio(
    fix_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    participants: List[str],
    t0: float,
    t1: float,
    fs_grid: float,
    sync_lag_s: float = 0.0,
) -> float:
    """
    Ratio of time where each non-speaker looks at current speaker, averaged across participants/time.
    """
    if turns_df is None or turns_df.empty or fix_df.empty:
        return np.nan

    grid = np.arange(t0, t1, 1.0 / fs_grid)
    if grid.size == 0:
        return np.nan

    speaker_at = np.array([""] * grid.size, dtype=object)
    for _, r in turns_df.iterrows():
        a, b = float(r["onset"]), float(r["offset"])
        sp = str(r["speaker"])
        m = (grid >= max(a, t0)) & (grid < min(b, t1))
        speaker_at[m] = sp

    valid = speaker_at != ""
    if not np.any(valid):
        return np.nan

    look_ok = []
    for listener in participants:
        g = fix_df[fix_df["participant"] == listener]

        tgt = np.array([""] * grid.size, dtype=object)
        for _, fx in g.iterrows():
            a = max(float(fx["onset"]), t0)
            b = min(float(fx["offset"]), t1)
            mm = (grid >= a) & (grid < b)
            inferred = infer_target_participant(str(fx["object_id"]), participants)
            if inferred is None:
                continue
            tgt[mm] = inferred

        if sync_lag_s > 0:
            # dilate target mask
            k = int(np.ceil(sync_lag_s * fs_grid))
            if k > 0:
                tgt_filled = tgt.copy()
                for sh in range(1, k + 1):
                    for idx in range(grid.size):
                        if tgt_filled[idx] == "":
                            src = idx - sh
                            if 0 <= src < grid.size and tgt[src] != "":
                                tgt_filled[idx] = tgt[src]
                            else:
                                src = idx + sh
                                if 0 <= src < grid.size and tgt[src] != "":
                                    tgt_filled[idx] = tgt[src]
                tgt = tgt_filled

        ok = valid & (speaker_at != listener) & (tgt == speaker_at)
        look_ok.append(ok)

    if not look_ok:
        return np.nan

    mat = np.vstack([x.astype(float) for x in look_ok])
    return float(np.nanmean(mat))


def transition_prob_gaze_to_speech(
    fix_df: pd.DataFrame,
    turns_df: pd.DataFrame,
    participants: List[str],
    tau_s: float = 1.0,
) -> float:
    """
    P(gaze->speech): for each speaking onset of participant p,
    check if in [onset - tau, onset) participant had a fixation whose inferred target is another participant.
    """
    if turns_df is None or turns_df.empty or fix_df.empty:
        return np.nan

    onsets = []
    for _, r in turns_df.iterrows():
        sp = str(r["speaker"])
        if sp not in participants:
            continue
        onsets.append((sp, float(r["onset"])))

    if not onsets:
        return np.nan

    hit = 0
    tot = 0
    for sp, onset in onsets:
        g = fix_df[fix_df["participant"] == sp]
        w0 = onset - tau_s
        w1 = onset
        chunk = g[(g["onset"] < w1) & (g["offset"] > w0)]
        tot += 1
        ok = False
        for _, fx in chunk.iterrows():
            inferred = infer_target_participant(str(fx["object_id"]), participants)
            if inferred is not None and inferred != sp:
                ok = True
                break
        if ok:
            hit += 1

    return float(hit / tot) if tot > 0 else np.nan


# ----------------------------- Main per group ----------------------------- #

def run_one_group(group_dir: str, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gaze = load_group_gaze(group_dir)
    if gaze.empty:
        import logging; logging.warning(f"{group_dir}: aucune donnée gaze exploitable")
        return None, None, None

    fix = detect_fixations_per_participant(gaze, min_fix=args.min_fix)

    t_min = float(gaze["time"].min())
    t_max = float(gaze["time"].max())
    t0 = args.t0 if args.t0 is not None else t_min
    t1 = args.t1 if args.t1 is not None else t_max
    if t1 <= t0:
        import logging; logging.warning(f"{group_dir}: t0/t1 invalides (t0={t0}, t1={t1})")
        return None, None, None

    interaction_dur = float(t1 - t0)

    participants = sorted(gaze["participant"].unique().tolist())
    fs = float(args.fs_grid)
    overlap_min_s = float(args.overlap_ms) / 1000.0
    sync_lag_s = float(args.sync_lag_ms) / 1000.0
    tau_s = float(args.tau_ms) / 1000.0

    turns_df = None
    if args.turns_file:
        try:
            turns_df = load_speaking_turns_csv(args.turns_file)
        except Exception as e:
            import logging; logging.warning(f"Speaking turns invalid ({args.turns_file}) : {e}")
            turns_df = None

    group_id = os.path.basename(group_dir)
    out_dir = os.path.join(args.out_dir if args.out_dir else os.getcwd(), group_id)
    os.makedirs(out_dir, exist_ok=True)

    # windowed metrics
    shared_win = shared_object_windows(
        fix_df=fix, t0=t0, t1=t1, win=args.win, step=args.step,
        fs_grid=fs, overlap_min_s=overlap_min_s
    )
    shared_pairs_win = shared_object_pairs_windows(
        fix_df=fix, participants=participants, t0=t0, t1=t1,
        win=args.win, step=args.step, fs_grid=fs, overlap_min_s=overlap_min_s
    )

    shared_win.to_csv(os.path.join(out_dir, "shared_object_windows.csv"), index=False)
    shared_pairs_win.to_csv(os.path.join(out_dir, "shared_object_pairs_windows.csv"), index=False)

    # overall shared object (full interaction)
    m_shared = shared_object_metrics(
        fix_df=fix[(fix["onset"] < t1) & (fix["offset"] > t0)],
        t0=t0, t1=t1, fs_grid=fs, min_participants=2, overlap_min_s=overlap_min_s
    )

    # mutual gaze (group = avg of pairs)
    m_mutual = mutual_gaze_group_metrics(
        fix_df=fix[(fix["onset"] < t1) & (fix["offset"] > t0)],
        participants=participants, t0=t0, t1=t1, fs_grid=fs,
        sync_lag_s=sync_lag_s, overlap_min_s=overlap_min_s
    )

    # entropy per participant
    ent_df = gaze_entropy_participant(fix, normalize=not args.entropy_raw)
    ent_group_mean = float(ent_df["gaze_entropy"].mean()) if not ent_df.empty else np.nan

    # speaker metrics
    gts_ratio = gaze_to_speaker_ratio(
        fix_df=fix, turns_df=turns_df, participants=participants,
        t0=t0, t1=t1, fs_grid=fs, sync_lag_s=sync_lag_s
    ) if turns_df is not None else np.nan

    trans_prob = transition_prob_gaze_to_speech(
        fix_df=fix, turns_df=turns_df, participants=participants, tau_s=tau_s
    ) if turns_df is not None else np.nan

    def norm(x):
        return float(x / interaction_dur) if (interaction_dur > 0 and np.isfinite(x)) else np.nan

    overall = pd.DataFrame([{
        "interaction_duration_s": interaction_dur,
        "shared_obj_ratio": m_shared["shared_ratio"],
        "shared_obj_n_episodes": m_shared["n_episodes"],
        "shared_obj_dur_total_s": m_shared["dur_total_s"],
        "shared_obj_dur_mean_s": m_shared["dur_mean_s"],
        "shared_obj_dur_median_s": m_shared["dur_median_s"],
        "shared_obj_dur_q25_s": m_shared["dur_q25_s"],
        "shared_obj_dur_q75_s": m_shared["dur_q75_s"],
        "shared_obj_dur_iqr_s": m_shared["dur_iqr_s"],
        "shared_obj_n_episodes_per_s": norm(m_shared["n_episodes"]),
        "shared_obj_dur_total_ratio": norm(m_shared["dur_total_s"]),
        "mutual_gaze_ratio_mean_pairs": m_mutual["mutual_gaze_ratio_mean_pairs"],
        "mutual_gaze_n_episodes_sum_pairs": m_mutual["mutual_gaze_n_episodes_sum_pairs"],
        "mutual_gaze_dur_total_s_sum_pairs": m_mutual["mutual_gaze_dur_total_s_sum_pairs"],
        "mutual_gaze_dur_mean_s_mean_pairs": m_mutual["mutual_gaze_dur_mean_s_mean_pairs"],
        "mutual_gaze_n_episodes_per_s": norm(m_mutual["mutual_gaze_n_episodes_sum_pairs"]),
        "mutual_gaze_dur_total_ratio": norm(m_mutual["mutual_gaze_dur_total_s_sum_pairs"]),
        "gaze_entropy_mean_participants": ent_group_mean,
        "gaze_focus_proxy": float(1.0 - ent_group_mean) if np.isfinite(ent_group_mean) else np.nan,
        "gaze_to_speaker_ratio": gts_ratio,
        "transition_prob_gaze_to_speech": trans_prob,
    }])

    # pairs summary
    pair_rows = []
    for p1, p2 in itertools.combinations(participants, 2):
        chunk_pair = fix[(fix["participant"].isin([p1, p2])) & (fix["onset"] < t1) & (fix["offset"] > t0)]
        m_pair_shared = shared_object_metrics(
            fix_df=chunk_pair, t0=t0, t1=t1, fs_grid=fs, min_participants=2, overlap_min_s=overlap_min_s
        )
        m_pair_mutual = mutual_gaze_pair_metrics(
            fix_df=fix[(fix["onset"] < t1) & (fix["offset"] > t0)],
            p1=p1, p2=p2, t0=t0, t1=t1, fs_grid=fs,
            sync_lag_s=sync_lag_s, overlap_min_s=overlap_min_s
        )
        pair_rows.append({
            "pair": f"{p1}+{p2}",
            "pair_shared_obj_ratio": m_pair_shared["shared_ratio"],
            "pair_shared_obj_n_episodes": m_pair_shared["n_episodes"],
            "pair_shared_obj_dur_total_s": m_pair_shared["dur_total_s"],
            "pair_shared_obj_dur_mean_s": m_pair_shared["dur_mean_s"],
            "pair_shared_obj_dur_median_s": m_pair_shared["dur_median_s"],
            "pair_shared_obj_dur_q25_s": m_pair_shared["dur_q25_s"],
            "pair_shared_obj_dur_q75_s": m_pair_shared["dur_q75_s"],
            "pair_shared_obj_dur_iqr_s": m_pair_shared["dur_iqr_s"],
            "pair_shared_obj_n_episodes_per_s": norm(m_pair_shared["n_episodes"]),
            "pair_shared_obj_dur_total_ratio": norm(m_pair_shared["dur_total_s"]),
            "pair_mutual_gaze_ratio": m_pair_mutual["ratio"],
            "pair_mutual_gaze_n_episodes": m_pair_mutual["n_episodes"],
            "pair_mutual_gaze_dur_total_s": m_pair_mutual["dur_total_s"],
            "pair_mutual_gaze_dur_mean_s": m_pair_mutual["dur_mean_s"],
            "pair_mutual_gaze_n_episodes_per_s": norm(m_pair_mutual["n_episodes"]),
            "pair_mutual_gaze_dur_total_ratio": norm(m_pair_mutual["dur_total_s"]),
        })
    pairs_df = pd.DataFrame(pair_rows)

    participants_df = ent_df.copy()
    participants_df["interaction_duration_s"] = interaction_dur

    # meta
    condition = extract_condition(group_dir)
    scenario = extract_scenario(group_dir)
    timepoint = extract_timepoint(Path(group_dir))

    overall = upsert_meta_cols(overall, group_id, condition, scenario, timepoint)
    pairs_df = upsert_meta_cols(pairs_df, group_id, condition, scenario, timepoint)
    participants_df = upsert_meta_cols(participants_df, group_id, condition, scenario, timepoint)

    overall.to_csv(os.path.join(out_dir, "metrics_overall.csv"), index=False)
    pairs_df.to_csv(os.path.join(out_dir, "metrics_pairs.csv"), index=False)
    participants_df.to_csv(os.path.join(out_dir, "metrics_participants.csv"), index=False)

    print(f"[OK] Groupe traité : {group_id} -> {out_dir}")

    return overall, pairs_df, participants_df


# ----------------------------- CLI / Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Analyse gaze v2 (shared attention + mutual gaze + entropy + speaker).")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--group-dir", help="Chemin vers le dossier du groupe (ex. .../bim066)")
    mode.add_argument("--data-dir", help="Chemin racine à scanner récursivement pour trouver les groupes")

    parser.add_argument("--win", type=float, default=30.0)
    parser.add_argument("--step", type=float, default=30.0)
    parser.add_argument("--min-fix", type=float, default=0.20)
    parser.add_argument("--fs-grid", type=float, default=20.0)
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--t1", type=float, default=None)

    parser.add_argument("--overlap-ms", type=float, default=100.0,
                        help="Durée minimale d'un épisode (ms) pour shared attention / mutual gaze.")
    parser.add_argument("--sync-lag-ms", type=float, default=0.0,
                        help="Tolérance lead/lag (ms) en dilatant les masques (mutual gaze + gaze-to-speaker).")

    parser.add_argument("--entropy-raw", action="store_true",
                        help="Si présent, n'applique pas la normalisation par log2(n_objects).")

    parser.add_argument("--turns-file", type=str, default=None,
                        help="CSV speaking turns avec colonnes speaker,onset,offset (secondes).")
    parser.add_argument("--tau-ms", type=float, default=1000.0,
                        help="Fenêtre (ms) pour transition gaze->speech avant onset.")

    parser.add_argument("--out-dir", type=str, default=None,
                        help="Dossier de sortie (un sous-dossier par groupe est créé).")

    args = parser.parse_args()

    if args.group_dir:
        run_one_group(args.group_dir, args)
        return

    group_dirs = [str(p) for p in find_gaze_groups(Path(args.data_dir))]
    if not group_dirs:
        raise SystemExit(f"Aucun groupe détecté sous : {args.data_dir}")

    master_dir = args.out_dir if args.out_dir else os.getcwd()
    os.makedirs(master_dir, exist_ok=True)

    all_overall = []
    all_pairs = []
    all_participants = []

    for gd in group_dirs:
        try:
            o, p, u = run_one_group(gd, args)
            if o is not None:
                all_overall.append(o)
            if p is not None and not p.empty:
                all_pairs.append(p)
            if u is not None and not u.empty:
                all_participants.append(u)
        except Exception as e:
            print(f"[WARN] Échec sur {gd} : {e}")

    if all_overall:
        pd.concat(all_overall, ignore_index=True).to_csv(
            os.path.join(master_dir, "ALL_metrics_overall.csv"), index=False
        )
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            os.path.join(master_dir, "ALL_metrics_pairs.csv"), index=False
        )
    if all_participants:
        pd.concat(all_participants, ignore_index=True).to_csv(
            os.path.join(master_dir, "ALL_metrics_participants.csv"), index=False
        )

    print("[OK] Master files écrits dans :", master_dir)


if __name__ == "__main__":
    main()
