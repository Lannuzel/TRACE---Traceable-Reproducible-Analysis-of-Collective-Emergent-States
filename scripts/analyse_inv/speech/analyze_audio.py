#!/usr/bin/env python3
"""
Audio Overlap Metrics -- sans ML  (v2.3-MOD1-2-3)
===============================================================

- Un seul dossier groupe en entree (doit contenir calculateur/, modelisateur/, lecteur/).
- Calcule : overlap, silences, distribution du temps de parole, prises de tour
  rapides et interruptions overlap-based.
- Ce script produit directement audio_features.csv
  (features haut niveau via build_features de compute_audio_features).

Decisions methodologiques importantes :
- `duration_s` correspond a la duree commune effectivement analysee pour le
  groupe, apres alignement / decoupe / troncature des pistes audio.
- `overlap_s` et `overlap_ratio` decrivent la simultaneite reelle de groupe :
  union des intervalles ou au moins deux participants parlent en meme temps.
- les anciennes "interruptions" de type prise de tour juste apres la fin de
  l'autre locuteur sont conservees sous le nom `rapid_floor_takeover_*`.
- le mot `interruption` est reserve aux evenements avec chevauchement effectif
  (`interrupt_attempt_*`, `interrupt_success_*`,
  `n_attempted_interruptions`, `n_successful_interruptions`).

MOD-1 : agregation IPU->tours CA (Sacks, Schegloff & Jefferson 1974).
MOD-2 : decomposition occupancy multi-locuteurs (Cetin & Shriberg 2006).
MOD-3 : detection stricte backchannels (Truong & Heylen 2010).
         Filtre F0 omis : pas de pipeline prosodique disponible.

Exemple :
    pip install numpy scipy librosa soundfile pandas tqdm matplotlib
    python analyze_audio.py chemin/vers/data_root --out audio_features.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sps
import librosa
import matplotlib.pyplot as plt

# Add project root to path for common imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.constants import ROLES
from common.metadata import extract_condition, extract_scenario, extract_timepoint
from common.io_utils import find_pc_role_wavs, find_wav, find_groups, is_vr_group, is_pc_group
from common.temporal import (
    segment_duration,
    segment_overlap_len,
    pairwise_overlap,
    silence_intervals,
    compute_floor_exchanges,
)
HOP_LENGTH   = 256                  # ~16 ms @16 kHz
FRAME_LENGTH = 1024                 # ~64 ms
DEFAULT_THR_DB   = 6.0              # seuil VAD (dB au-dessus de la mediane)
DEFAULT_MERGE_GAP = 0.25            # s - fusion des micro-pauses
DELTA_INTERRUPT = 0.2               # s
PLOT_DPI = 150

# seuils "conversation" (ajustables en CLI)
TURN_MIN_SEC      = 1.0             # >= 1 s = tour de parole
TURN_MAX_PAUSE_SEC = 3.0            # silence intra-tour max : au-dela, le tour est coupe
BACKCH_MAX_SEC    = 0.7             # <= 0.7 s = backchannel
BACKCH_MIN_OVL    = 0.10            # >= 100 ms de chevauchement avec autrui
SKIP_AFTER_BEEP   = 0.05            # coupe 50 ms apres le bip
FLOOR_EXCHANGE_MAX_GAP = 2.0   # s : latence max pour considerer une transition de tour
MIN_POST_TAKEOVER_SEC = 0.5    # s : duree min apres la fin de A pour considerer que B garde le tour
MIN_INTERRUPT_OVERLAP = 0.1    # s : chevauchement min pour une interruption

# ----- utilitaires fichier & bip ---------------------------------------------

def detect_beep(path: Path,
                *,
                debug: bool = True,
                **kwargs) -> Tuple[float, int]:
    """
    Detection du bip par pic d'amplitude global (np.argmax).
    Le bip est concu pour etre le son le plus fort de la session.
    Retourne (instant_bip_en_secondes, sample_rate).
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y.astype(np.float32)
    abs_y = np.abs(y)
    idx = int(np.argmax(abs_y))
    t_bip = idx / sr
    if debug:
        print(f"[DEBUG] bip > {t_bip:.3f}s  (amp {abs_y[idx]:.2f})  {path.name}")
    return t_bip, sr

def align_audio(path: Path, offset: float, ref: float, sr_out: int) -> np.ndarray:
    """Decale la piste pour superposer le bip sur ref."""
    diff = offset - ref
    y, sr = librosa.load(path, sr=sr_out, mono=True)
    if diff > 0:
        y = y[int(diff * sr):]
    elif diff < 0:
        y = np.concatenate([np.zeros(int(-diff * sr), dtype=y.dtype), y])
    return y

# ----- VAD --------------------------------------------------------------------
def rms_db(y: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                              hop_length=HOP_LENGTH, center=True)[0]
    return librosa.amplitude_to_db(rms, ref=np.max)

def speech_segments(y: np.ndarray, sr: int,
                    thr_db: float, merge_gap: float,
                    abs_floor_dbfs: float = -float("inf")) -> List[Tuple[float, float]]:
    """
    VAD par seuil relatif (median + thr_db) avec seuil absolu optionnel.

    abs_floor_dbfs : seuil absolu en dBFS (0 dBFS = amplitude max numerique).
                     Ex: abs_floor_dbfs=-35 signifie que la frame doit depasser
                     -35 dBFS pour etre candidate parole.
                     Independant de la piste — meme reference pour toutes les pistes.
                     Defaut = -inf (desactive).
    """
    e_db = rms_db(y)
    times = librosa.frames_to_time(np.arange(len(e_db)), sr=sr, hop_length=HOP_LENGTH)
    mask = e_db > (np.median(e_db) + thr_db)
    if abs_floor_dbfs > -float("inf"):
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                                   hop_length=HOP_LENGTH, center=True)[0]
        e_abs = librosa.amplitude_to_db(rms, ref=1.0)
        mask = mask & (e_abs > abs_floor_dbfs)
    segs: List[List[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = times[i]
            while i < len(mask) and mask[i]:
                i += 1
            end = times[min(i, len(times)-1)]
            segs.append([start, end])
        i += 1
    # fusion micro-pauses
    merged: List[List[float]] = []
    for s, e in segs:
        if not merged or s - merged[-1][1] > merge_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(s, e) for s, e in merged]

# ----- metriques utilitaires --------------------------------------------------

def vad_vector(y: np.ndarray, sr: int, vad_thr_db: float) -> np.ndarray:
    rms  = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    db   = librosa.amplitude_to_db(rms, ref=np.max)
    return (db > (np.median(db) + vad_thr_db)).astype(float)

def synchrony_matrix(vecs: Dict[str, np.ndarray]) -> pd.DataFrame:
    centered = {r: v - v.mean() for r, v in vecs.items()}
    mat = pd.DataFrame(index=ROLES, columns=ROLES, dtype=float)
    for a in ROLES:
        for b in ROLES:
            num = np.dot(centered[a], centered[b])
            den = np.linalg.norm(centered[a]) * np.linalg.norm(centered[b])
            mat.loc[a, b] = num / den if den else np.nan
    return mat


def effective_analysis_duration_s(tracks: Dict[str, np.ndarray], sr: int) -> float:
    """
    Duree de reference commune reellement analysee pour le groupe.
    """
    if not tracks or sr <= 0:
        return 0.0
    min_len_samples = min(len(y) for y in tracks.values())
    return float(min_len_samples / sr) if min_len_samples > 0 else 0.0


def group_overlap_intervals(
    segs_by_role: Dict[str, List[Tuple[float, float]]]
) -> List[Tuple[float, float]]:
    """
    Intervalles de parole simultanee au niveau du groupe.

    On calcule ici l'union des intervalles ou au moins 2 personnes parlent,
    ce qui evite le double comptage des episodes de parole a 3.
    """
    events: List[Tuple[float, int]] = []
    for segs in segs_by_role.values():
        for start, end in segs:
            events.append((start, 1))
            events.append((end, -1))

    if not events:
        return []

    events.sort(key=lambda x: (x[0], -x[1]))
    overlaps: List[Tuple[float, float]] = []
    count = 0
    start_overlap: float | None = None

    for t, delta in events:
        prev_count = count
        count += delta
        if prev_count < 2 <= count:
            start_overlap = t
        elif prev_count >= 2 > count and start_overlap is not None and t > start_overlap:
            overlaps.append((start_overlap, t))
            start_overlap = None

    return overlaps


def total_interval_duration(intervals: List[Tuple[float, float]]) -> float:
    """Duree totale d'une liste d'intervalles non chevauchants."""
    return float(sum(end - start for start, end in intervals))


def count_rapid_floor_takeovers(
    source_segs: List[Tuple[float, float]],
    target_segs: List[Tuple[float, float]],
    max_gap: float,
    min_turn_sec: float = TURN_MIN_SEC,
) -> int:
    """
    Compte les prises de tour rapides de `target` apres la fin de `source`.

    Ce construit n'est pas une interruption au sens strict : il n'y a pas de
    chevauchement requis, seulement une reprise rapide du plancher.
    """
    filtered_source = [(s, e) for s, e in source_segs if (e - s) >= min_turn_sec]
    filtered_target = [(s, e) for s, e in target_segs if (e - s) >= min_turn_sec]
    n_takeovers = 0
    for _, a_end in filtered_source:
        for b_start, _ in filtered_target:
            if 0 < b_start - a_end <= max_gap:
                n_takeovers += 1
    return n_takeovers


def count_directed_interruptions(
    source_segs: List[Tuple[float, float]],
    target_segs: List[Tuple[float, float]],
    min_overlap: float = MIN_INTERRUPT_OVERLAP,
    min_post_takeover: float = MIN_POST_TAKEOVER_SEC,
    min_turn_sec: float = TURN_MIN_SEC,
) -> Tuple[int, int]:
    """
    Compte les interruptions overlap-based de `target` envers `source`.
    """
    attempts = 0
    successes = 0
    filtered_source = [(s, e) for s, e in source_segs if (e - s) >= min_turn_sec]
    filtered_target = [(s, e) for s, e in target_segs if (e - s) >= min_turn_sec]

    for a_start, a_end in filtered_source:
        for b_start, b_end in filtered_target:
            if a_start < b_start < a_end:
                overlap = min(a_end, b_end) - b_start
                if overlap >= min_overlap:
                    attempts += 1
                    if (b_end - a_end) >= min_post_takeover:
                        successes += 1

    return attempts, successes


# ----- MOD-1 : agregation IPU -> tours CA (Sacks, Schegloff & Jefferson 1974) -----

def aggregate_ipus_to_ca_turns(
    segs_by_role: Dict[str, List[Tuple[float, float]]],
    turn_min_s: float = 1.0,
    turn_max_pause_s: float = 3.0,
) -> List[Dict]:
    """
    Agregation IPU->tours CA (Sacks, Schegloff & Jefferson 1974 ; Levitan & Hirschberg 2011).

    Segments consecutifs du meme role sont fusionnes en un seul tour,
    sous deux conditions :
      - Aucun segment d'un autre role ne commence entre eux (critere CA classique).
      - Le silence intra-tour ne depasse pas turn_max_pause_s (defaut 3.0 s).
        Au-dela, meme si personne d'autre ne parle, le tour est coupe : un long
        silence collectif ne prolonge pas mecaniquement la prise de parole.

    Algorithme :
    1. Fusion conditionnelle des IPU du meme role (pause <= turn_max_pause_s
       ET pas de segment etranger intercale).
    2. Filtre : suppression des tours < turn_min_s.
    3. Re-fusion : apres filtrage, tours adjacents du meme role redevenus
       voisins sont consolides (meme contrainte de pause).

    Returns : liste de dicts {"role": str, "start": float, "end": float}
    """
    all_segs = []
    for role, segs in segs_by_role.items():
        for s, e in segs:
            all_segs.append((s, e, role))
    all_segs.sort(key=lambda x: x[0])

    # Passe 1 : fusion conditionnelle
    turns: List[Dict] = []
    for s, e, role in all_segs:
        if (
            turns
            and turns[-1]["role"] == role
            and (s - turns[-1]["end"]) <= turn_max_pause_s  # silence intra-tour acceptable
        ):
            turns[-1]["end"] = max(turns[-1]["end"], e)
        else:
            turns.append({"role": role, "start": s, "end": e})

    # Passe 2 : filtre duree minimale
    turns = [t for t in turns if (t["end"] - t["start"]) >= turn_min_s]

    # Passe 3 : re-fusion des adjacents meme role crees par l'elimination de micro-tours
    merged: List[Dict] = []
    for t in turns:
        if (
            merged
            and merged[-1]["role"] == t["role"]
            and (t["start"] - merged[-1]["end"]) <= turn_max_pause_s
        ):
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(t)
    return merged


# ----- MOD-2 : occupancy multi-locuteurs (Cetin & Shriberg 2006) -----

def compute_speaker_occupancy(
    segs_by_role: Dict[str, List[Tuple[float, float]]],
    total_s: float,
    fs_grid: float = 100.0,
) -> Dict[str, float]:
    """
    Decompose le temps en etats 0/1/2/3 locuteurs actifs simultanement.
    Cetin & Shriberg (2006) ICSI multi-party.
    En triade, l'overlap baseline est mecaniquement superieur au dyadique ;
    la decomposition en occupancy est plus interpretable que le ratio brut.
    """
    if total_s <= 0:
        return {
            "audio_pct_time_0_speakers": np.nan,
            "audio_pct_time_1_speaker":  np.nan,
            "audio_pct_time_2_speakers": np.nan,
            "audio_pct_time_3_speakers": np.nan,
            "audio_overlap_speaking_ratio_from_occupancy": np.nan,
        }
    n_frames = int(np.ceil(total_s * fs_grid))
    grid = np.zeros(n_frames, dtype=np.int8)
    for role_segs in segs_by_role.values():
        for s, e in role_segs:
            i0 = int(np.floor(s * fs_grid))
            i1 = int(np.ceil(e * fs_grid))
            grid[i0:min(i1, n_frames)] += 1
    counts = np.bincount(grid.clip(0, 3), minlength=4)[:4]
    pcts = counts / n_frames
    result = {
        "audio_pct_time_0_speakers": round(float(pcts[0]), 6),
        "audio_pct_time_1_speaker":  round(float(pcts[1]), 6),
        "audio_pct_time_2_speakers": round(float(pcts[2]), 6),
        "audio_pct_time_3_speakers": round(float(pcts[3]), 6),
        "audio_overlap_speaking_ratio_from_occupancy": round(float(pcts[2] + pcts[3]), 6),
    }
    total_pct = sum(result[k] for k in [
        "audio_pct_time_0_speakers", "audio_pct_time_1_speaker",
        "audio_pct_time_2_speakers", "audio_pct_time_3_speakers"
    ])
    assert abs(total_pct - 1.0) < 1e-3, (
        f"REGRESSION MOD-2: sum(pct_time_*)={total_pct:.6f} != 1.0"
    )
    return result


# ----- MOD-3 : detection stricte backchannels (Truong & Heylen 2010) -----

def is_backchannel_strict(
    s: float, e: float,
    role_segs: List[Tuple[float, float]],
    other_segs: List[Tuple[float, float]],
    turns_ca: List[Dict],
    role: str,
    bc_min_dur: float = 0.10,
    bc_max_dur: float = 0.70,
    bc_min_ovl: float = 0.10,
    bc_continuation_window: float = 0.20,
    bc_proximity_window: float = 0.50,
) -> bool:
    """
    Detection stricte backchannels, 4 filtres en cascade.
    Truong & Heylen (2010). Filtre F0 omis (pas de pipeline prosodique disponible).

    Filtres :
    1. Duree du segment dans [bc_min_dur, bc_max_dur]
    2. Doit se produire pendant le tour d'un autre locuteur : soit chevauchement
       >= bc_min_ovl, soit dans une micro-pause (fin segment autre < bc_proximity_window
       avant le debut de ce segment, ou debut segment autre < bc_proximity_window
       apres la fin de ce segment).
    3. Pas une continuation : exclusion si suit dans bc_continuation_window
       la fin d'un autre segment du meme role
    4. Pas un tour CA propre : exclusion si turns_ca contient un tour de ce role
       dont le debut est a moins de 500ms de ce segment
    """
    dur = e - s
    if not (bc_min_dur <= dur <= bc_max_dur):
        return False
    # F2 : chevauchement OU proximite immediate (micro-pause)
    def _near(os: float, oe: float) -> bool:
        if segment_overlap_len((s, e), (os, oe)) >= bc_min_ovl:
            return True
        # backchannel juste apres la fin d'un segment autre (micro-pause)
        if 0 <= s - oe <= bc_proximity_window:
            return True
        # backchannel juste avant le debut d'un segment autre
        if 0 <= os - e <= bc_proximity_window:
            return True
        return False
    if not any(_near(os, oe) for os, oe in other_segs):
        return False
    for rs, re in role_segs:
        if rs == s and re == e:
            continue
        if 0 < s - re <= bc_continuation_window:
            return False
    for t in turns_ca:
        if t["role"] == role and abs(t["start"] - s) < 0.5:
            return False
    return True


# ----- timeline plot ----------------------------------------------------------
def draw_timeline(group: str, segs: Dict[str, List[Tuple[float,float]]],
                  total: float, out_png: Path):
    ovl = group_overlap_intervals(segs)
    sil = silence_intervals(segs, total)

    fig, ax = plt.subplots(figsize=(12, 3 + 0.6*len(ROLES)), dpi=PLOT_DPI)
    ymap = {r:i for i,r in enumerate(ROLES[::-1],1)}
    for s,e in sil: ax.axvspan(s,e,color="0.9",alpha=0.6)
    for role,lst in segs.items():
        for s,e in lst: ax.barh(ymap[role], e-s, left=s, height=0.35, color="tab:blue")
    for s,e in ovl: ax.barh(0, e-s, left=s, height=0.35, color="tab:red")
    ax.set_yticks([0]+list(ymap.values())); ax.set_yticklabels(["overlap"]+ROLES[::-1])
    ax.set_xlabel("Temps (s)"); ax.set_title(f"Timeline -- {group}")
    ax.set_xlim(0,total); plt.tight_layout(); fig.savefig(out_png); plt.close(fig)

# ----- pipeline principal -----------------------------------------------------

def extract_condition_session_timepoint(path: Path) -> Tuple[str, str, str]:
    condition = extract_condition(path) or "UNK"
    scenario = extract_scenario(path) or "UNK"
    timepoint = extract_timepoint(path)
    return condition, scenario, timepoint


def process_group(gdir: Path, thr_db: float, merge_gap: float,
                  plot: bool, turn_min: float, backch_max: float, backch_ovl: float,
                  abs_floor_dbfs: float = -30.0,
                  turn_max_pause: float = TURN_MAX_PAUSE_SEC):

    # --- Detection du mode (VR vs PC) ---
    is_vr = is_vr_group(gdir)
    is_pc = is_pc_group(gdir)

    if not (is_vr or is_pc):
        raise SystemExit(
            f"Groupe invalide: {gdir}\n"
            f"Attendu soit sous-dossiers {ROLES} (VR), soit processed_openface/*__audio.wav (PC)."
        )

    condition, session, timepoint = extract_condition_session_timepoint(gdir)

    # --- Selection des fichiers wav par role ---
    if is_vr:
        files = {r: find_wav(gdir / r) for r in ROLES}
    else:
        files = find_pc_role_wavs(gdir / "processed_openface")

    # 1) Alignement + decoupe (VR) OU lecture directe (PC)
    tracks: Dict[str, np.ndarray] = {}
    vad_vecs: Dict[str, np.ndarray] = {}
    segs: Dict[str, List[Tuple[float,float]]] = {}
    total = 0.0

    if is_vr:
        beep, sr = {}, None
        for r, f in files.items():
            t, sr_i = detect_beep(f, debug=True)
            beep[r] = t
            sr = sr or sr_i

        if not beep:
            raise RuntimeError(f"Aucun bip detecte (beep vide) dans {gdir}")
        ref = min(beep.values())

        for r, f in files.items():
            y = align_audio(f, beep[r], ref, sr)
            start_idx = int((ref + SKIP_AFTER_BEEP) * sr)
            y = y[start_idx:] if start_idx < len(y) else np.array([], dtype=np.float32)
            tracks[r] = y

    else:
        sr = 16000
        for r, f in files.items():
            y, _ = librosa.load(f, sr=sr, mono=True)
            tracks[r] = y

    # Duree commune reellement analysee.
    min_len_samples = min((len(y) for y in tracks.values()), default=0)
    tracks = {r: y[:min_len_samples] for r, y in tracks.items()}
    total = effective_analysis_duration_s(tracks, sr)

    for r in ROLES:
        y = tracks.get(r, np.array([], dtype=np.float32))
        vad_vecs[r] = vad_vector(y, sr, thr_db) if len(y) else np.array([], dtype=float)
        segs[r] = speech_segments(y, sr, thr_db, merge_gap, abs_floor_dbfs) if len(y) else []

    # MOD-1 : agregation CA (Sacks, Schegloff & Jefferson 1974)
    turns_ca = aggregate_ipus_to_ca_turns(segs, turn_min_s=turn_min,
                                          turn_max_pause_s=turn_max_pause)

    # 2) metriques de base (groupe)
    if set(vad_vecs.keys()) != set(ROLES):
        raise RuntimeError(f"vad_vecs incomplet pour {gdir}: {list(vad_vecs.keys())}")

    min_len_frames = min((len(v) for v in vad_vecs.values()), default=0)
    if min_len_frames == 0:
        sync_mat = pd.DataFrame(np.nan, index=ROLES, columns=ROLES)
    else:
        vad_vecs = {r: v[:min_len_frames] for r, v in vad_vecs.items()}
        sync_mat = synchrony_matrix(vad_vecs)

    pairs = [(ROLES[0], ROLES[1]), (ROLES[0], ROLES[2]), (ROLES[1], ROLES[2])]
    overlap_intervals = group_overlap_intervals(segs)
    overlap_s = total_interval_duration(overlap_intervals)
    pairwise_overlap_s = sum(pairwise_overlap(segs[a], segs[b]) for a, b in pairs)
    pause_intervals_ = silence_intervals(segs, total)
    pause_time = sum(e-s for s,e in pause_intervals_)

    # CA-based (MOD-1)
    floor_exchange_gaps = []
    for i in range(len(turns_ca) - 1):
        tc, tn = turns_ca[i], turns_ca[i + 1]
        if tc["role"] != tn["role"]:
            gap = tn["start"] - tc["end"]
            if 0 < gap <= FLOOR_EXCHANGE_MAX_GAP:
                floor_exchange_gaps.append(gap)

    # MOD-2 : occupancy (Cetin & Shriberg 2006)
    occupancy = compute_speaker_occupancy(segs, total)

    condition, scenario, timepoint = extract_condition_session_timepoint(gdir)
    metrics = {
        # Colonnes canoniques de sortie pour le pipeline INV.
        "group_id": gdir.name,
        "condition": condition,
        "scenario": scenario,
        "timepoint": timepoint,
        "duration_s": round(total,3),
        "overlap_s": round(overlap_s,3),
        "pairwise_overlap_s": round(pairwise_overlap_s, 3),
        "pairwise_overlap_ratio": round(pairwise_overlap_s/total if total else 0, 3),
        "pause_time_s": round(pause_time,3),
        "pause_ratio": round(pause_time/total if total else 0,3),
        "mean_pause_s": round(np.mean([e-s for s,e in pause_intervals_]) if pause_intervals_ else 0.0, 3),
        "floor_exchange_pause_mean_s": round(float(np.mean(floor_exchange_gaps)) if floor_exchange_gaps else 0.0, 3),
        "n_floor_exchanges": int(len(floor_exchange_gaps)),
    }

    # MOD-2 : overlap_ratio depuis occupancy
    metrics.update(occupancy)
    metrics["overlap_ratio"] = occupancy["audio_overlap_speaking_ratio_from_occupancy"]

    # 3) metriques par role (parole/silence/turns/backchannels)
    others = {
        r: [seg for rr, lst in segs.items() if rr!=r for seg in lst]
        for r in ROLES
    }

    all_turn_durations = []
    for r in ROLES:
        seg = segs[r]
        speech_t = segment_duration(seg)
        silence_t = max(0.0, total - speech_t)

        # CA-based (MOD-1)
        turn_segs_ca = [(t["start"], t["end"]) for t in turns_ca if t["role"] == r]
        n_turns_ca = len(turn_segs_ca)
        turn_durs_ca = [e - s for s, e in turn_segs_ca]
        mean_turn_ca = float(np.mean(turn_durs_ca)) if turn_durs_ca else 0.0
        speak_turn_time_ca = sum(turn_durs_ca)
        all_turn_durations.extend(turn_durs_ca)  # agregat groupe = CA

        # Backchannels stricts (MOD-3, Truong & Heylen 2010)
        n_backch = sum(
            1 for (bs, be) in segs[r]
            if is_backchannel_strict(bs, be, segs[r], others[r], turns_ca, r,
                                      bc_min_dur=0.10, bc_max_dur=backch_max, bc_min_ovl=backch_ovl)
        )

        ratio_turns_backch = (n_turns_ca / n_backch) if n_backch > 0 else np.inf

        # NB: la somme des speech_{role}_ratio peut depasser 1.
        metrics.update({
            f"speech_{r}_s": round(speech_t,3),
            f"speech_{r}_ratio": round(speech_t/total if total else 0,3),
            f"silence_{r}_s": round(silence_t,3),
            f"speak_sil_ratio_{r}": round((speech_t/silence_t) if silence_t>0 else np.inf,3),
            f"turns_{r}_n": int(n_turns_ca),
            f"backchannels_{r}_n": int(n_backch),
            f"turns_to_backchannels_{r}": round(ratio_turns_backch,3) if np.isfinite(ratio_turns_backch) else "inf",
            f"mean_turn_{r}_s": round(mean_turn_ca,3),
            f"speaking_turn_time_{r}_s": round(speak_turn_time_ca,3),
        })

    # agregats groupe lies aux tours
    mean_turn_all_ca = float(np.mean(all_turn_durations)) if all_turn_durations else 0.0

    metrics["mean_turn_s"] = round(mean_turn_all_ca, 3)
    metrics["turn_to_pause_ratio"] = round(
        (mean_turn_all_ca / metrics["mean_pause_s"]) if metrics.get("mean_pause_s", 0) > 0 else 0.0, 3
    )

    # 4) synchronie et dynamiques de prise de tour
    rapid_takeovers_total = 0
    attempts_total = 0
    successes_total = 0
    for a,b in pairs:
        rapid_takeovers_total += count_rapid_floor_takeovers(segs[a], segs[b], DELTA_INTERRUPT, min_turn_sec=turn_min)
        rapid_takeovers_total += count_rapid_floor_takeovers(segs[b], segs[a], DELTA_INTERRUPT, min_turn_sec=turn_min)

        att_b_a, suc_b_a = count_directed_interruptions(
            segs[a], segs[b],
            min_overlap=MIN_INTERRUPT_OVERLAP,
            min_post_takeover=MIN_POST_TAKEOVER_SEC,
            min_turn_sec=turn_min,
        )
        att_a_b, suc_a_b = count_directed_interruptions(
            segs[b], segs[a],
            min_overlap=MIN_INTERRUPT_OVERLAP,
            min_post_takeover=MIN_POST_TAKEOVER_SEC,
            min_turn_sec=turn_min,
        )
        attempts_total += att_b_a + att_a_b
        successes_total += suc_b_a + suc_a_b

        metrics[f"sync_{a}_{b}"] = round(float(sync_mat.loc[a, b]), 3) if pd.notna(sync_mat.loc[a,b]) else np.nan

    metrics["rapid_floor_takeovers_total"] = int(rapid_takeovers_total)
    metrics["n_attempted_interruptions"] = int(attempts_total)
    metrics["n_successful_interruptions"] = int(successes_total)
    if attempts_total > 0:
        metrics["successful_interruption_ratio"] = round(
            float(successes_total / attempts_total),
            3,
        )
    else:
        metrics["successful_interruption_ratio"] = np.nan

    # 5) chronogramme
    if plot:
        png = gdir.name + "_timeline.png"
        draw_timeline(gdir.name, segs, total, Path(png))
        metrics["plot"] = png

    return metrics




# ----- CLI --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("Audio overlap metrics (groupe unique + metriques etendues)")
    p.add_argument("root_dir", help="Dossier racine (ex: DATA_E2)")
    p.add_argument("--out", default="audio_features.csv",
                   help="Chemin de sortie (defaut : audio_features.csv)")
    p.add_argument("--thr-db", type=float, default=DEFAULT_THR_DB)
    p.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP)
    p.add_argument("--turn-min",       type=float, default=TURN_MIN_SEC,       help="Duree minimale d'un tour (s)")
    p.add_argument("--turn-max-pause", type=float, default=TURN_MAX_PAUSE_SEC, help="Silence intra-tour max avant coupure (s, defaut 3.0)")
    p.add_argument("--backch-max", type=float, default=BACKCH_MAX_SEC, help="Duree max d'un backchannel (s)")
    p.add_argument("--backch-ovl", type=float, default=BACKCH_MIN_OVL, help="Chevauchement min avec autrui (s)")
    p.add_argument("--abs-floor-dbfs", type=float, default=-30.0,
                   help="Seuil absolu VAD en dBFS (defaut: -30 dBFS).")
    p.add_argument("--plot", action="store_true", help="Afficher + enregistrer le PNG timeline")
    args = p.parse_args()

    root = Path(args.root_dir)
    groups = find_groups(root)

    if not groups:
        raise SystemExit("Aucun groupe valide trouve dans l'arborescence.")

    all_metrics = []

    print(f"[INFO] {len(groups)} groupes detectes.")

    for gdir in groups:
        print(f"\n[RUN] Traitement : {gdir}")
        m = process_group(
            gdir,
            args.thr_db,
            args.merge_gap,
            args.plot,
            args.turn_min,
            args.backch_max,
            args.backch_ovl,
            args.abs_floor_dbfs,
            args.turn_max_pause,
        )
        all_metrics.append(m)

    raw_df = pd.DataFrame(all_metrics)

    from analyse_inv.speech.compute_audio_features import build_features  # import local
    out_df = build_features(raw_df)

    out_df.to_csv(args.out, index=False)
    print(f"\n[OK] {args.out} enregistre ({len(out_df)} groupes).")

if __name__ == "__main__":
    main()
