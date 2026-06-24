#!/usr/bin/env python3
"""
viewer.py  —  Verificateur visuel interactif du pipeline audio INV
==================================================================

Objectif : verifier visuellement que les metriques calculees par la pipeline
(tours CA, backchannels stricts, occupancy) correspondent bien aux donnees audio.

Panneau principal (de haut en bas) :
  1. Formes d'onde (calculateur / modelisateur / lecteur)
  2. IPU (VAD brut, barres bleues) + tours CA (barres vertes, plus hautes)
     + marqueurs backchannels (triangles rouges) par role
  3. Occupancy timeline : aire empilee 0/1/2/3 locuteurs
  4. Panel metriques statiques (texte)

Curseur rouge en temps reel pendant la lecture audio.

Usage :
    python viewer.py /chemin/vers/groupe [--thr-db 6] [--merge-gap 0.25]
                                          [--turn-min 1.0] [--bc-max 0.7]
                                          [--no-play]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Imports pipeline (meme logique que analyze_audio.py)
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))                        # speech/ -> pour analyze_audio
sys.path.insert(0, str(_here.parent))                 # analyse_inv/ -> pour common
sys.path.insert(0, str(_here.parents[1]))             # scripts/ -> racine projet

from common.constants import ROLES

# Re-import des constantes et fonctions pipeline directement depuis analyze_audio
from analyze_audio import (
    detect_beep,
    align_audio,
    speech_segments,
    aggregate_ipus_to_ca_turns,
    compute_speaker_occupancy,
    is_backchannel_strict,
    effective_analysis_duration_s,
    count_directed_interruptions,
    count_rapid_floor_takeovers,
    SKIP_AFTER_BEEP,
    TURN_MIN_SEC,
    BACKCH_MAX_SEC,
    BACKCH_MIN_OVL,
    DEFAULT_THR_DB,
    DEFAULT_MERGE_GAP,
    FLOOR_EXCHANGE_MAX_GAP,
    MIN_INTERRUPT_OVERLAP,
    MIN_POST_TAKEOVER_SEC,
    DELTA_INTERRUPT,
    group_overlap_intervals,
)

try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

DISP_SR = 200          # Hz pour le downsampling affichage
DEFAULT_ABS_FLOOR_DB = -30.0
# Le bip dure ~500ms — on coupe 500ms apres le debut du bip dans le viewer
# (le pipeline analyse_audio.py utilise 50ms mais ne cherche pas a afficher le bip)
VIEWER_SKIP_AFTER_BEEP = 1.0  # coupe 1s apres le pic detecte pour eliminer bip complet
ROLE_COLORS = {
    "calculateur":  "tab:blue",
    "modelisateur": "tab:orange",
    "lecteur":      "tab:green",
}
OCCUPANCY_COLORS = ["#d9d9d9", "#a8d8ea", "#f9ca74", "#e06c75"]


# ---------------------------------------------------------------------------
# Utilitaires fichiers
# ---------------------------------------------------------------------------

def find_last_wav(folder: Path) -> Path:
    wavs = list(folder.glob("*.wav")) or list(folder.glob("**/*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Aucun wav dans {folder}")
    return max(wavs, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def speech_segments_with_floor(
    y: np.ndarray,
    sr: int,
    thr_db: float,
    merge_gap: float,
    abs_floor_db: float,
) -> List[Tuple[float, float]]:
    """
    Variante de speech_segments avec double seuil :
      1. energie > median_rms + thr_db  (seuil relatif, identique au pipeline)
      2. rms_abs > abs_floor_db         (seuil absolu en dBFS, ancre sur la mediane RMS
                                         de la piste — empeche le bruit de fond de
                                         passer quand le max est un pic isole)

    La reference pour le seuil absolu est la mediane RMS de la piste (pas np.max),
    ce qui evite que un bip ou souffle court ecrase toute la dynamique.
    """
    from analyze_audio import HOP_LENGTH, FRAME_LENGTH
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                               hop_length=HOP_LENGTH, center=True)[0]
    # Seuil relatif (ref=np.max, identique pipeline)
    e_db_rel = librosa.amplitude_to_db(rms, ref=np.max)
    # Seuil absolu en dBFS (ref=1.0 = 0 dBFS)
    e_db_abs = librosa.amplitude_to_db(rms, ref=1.0)

    times = librosa.frames_to_time(np.arange(len(e_db_rel)), sr=sr, hop_length=HOP_LENGTH)
    # Seuil adaptatif : median dBFS de la piste + abs_floor_db dB
    # Permet de filtrer le bruit de fond sans penaliser les pistes silencieuses
    mask = (e_db_rel > (np.median(e_db_rel) + thr_db)) & (e_db_abs > abs_floor_db)

    segs: List[List[float]] = []
    i = 0
    while i < len(mask):
        if mask[i]:
            start = times[i]
            while i < len(mask) and mask[i]:
                i += 1
            end = times[min(i, len(times) - 1)]
            segs.append([start, end])
        i += 1
    merged: List[List[float]] = []
    for s, e in segs:
        if not merged or s - merged[-1][1] > merge_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(s, e) for s, e in merged]


def detect_beep_by_sustained_peak(
    path: Path,
    sr_out: int | None = None,
    min_sustain_s: float = 0.3,
    frame_s: float = 0.05,
) -> Tuple[float, int]:
    """
    Detection du bip par tonalite SOUTENUE dans la bande 800-2000 Hz.
    Combine deux criteres :
      1. Ratio energie beep-band / noise-band eleve (tonalite pure)
      2. Ce ratio reste eleve pendant >= min_sustain_s (soutenu)
    Discrimine le bip (~0.5s, tonalite pure) des artefacts courts et de la voix.
    """
    y, sr = librosa.load(path, sr=sr_out, mono=True)
    hop = int(frame_s * sr)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop)

    beep_band  = (freqs >= 800)  & (freqs <= 2000)
    noise_band = (freqs >= 100)  & (freqs <  800)
    ratio = D[beep_band, :].mean(0) / (D[noise_band, :].mean(0) + 1e-10)

    # Seuil = 2x la mediane du ratio (le bip depasse largement)
    thr = np.median(ratio) * 2.0
    above = (ratio >= thr).astype(np.int8)

    min_frames = int(min_sustain_s / frame_s)
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_frames:
                t_bip = float(times[i])
                print(f"[DEBUG] bip (sustain+freq) > {t_bip:.3f}s  "
                      f"(duree={(j-i)*frame_s:.3f}s  ratio={ratio[i]:.1f})  {path.name}")
                return t_bip, sr
        i += 1

    # fallback amplitude classique
    print(f"[DEBUG] bip (sustain) fallback amplitude  {path.name}")
    t_bip = float(np.argmax(np.abs(y))) / sr
    return t_bip, sr


def detect_beep_by_frequency(path: Path, sr_out: int | None = None) -> Tuple[float, int]:
    """
    Detection du bip par ratio energie 800-2000 Hz / bruit 100-800 Hz.
    Plus robuste quand la piste contient un pic d'amplitude parasite
    (claquement, saturation) qui trompe la detection par amplitude.
    """
    y, sr = librosa.load(path, sr=sr_out, mono=True)
    hop = int(sr * 0.05)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    beep_band  = (freqs >= 800)  & (freqs <= 2000)
    noise_band = (freqs >= 100)  & (freqs <  800)
    beep_energy  = D[beep_band,  :].mean(axis=0)
    noise_energy = D[noise_band, :].mean(axis=0) + 1e-10
    ratio = beep_energy / noise_energy
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop)
    best = int(np.argmax(ratio))
    t_bip = float(times[best])
    print(f"[DEBUG] bip (freq) > {t_bip:.3f}s  (ratio={ratio[best]:.1f})  {path.name}")
    return t_bip, sr


def speech_segments_dominant(
    tracks: Dict[str, np.ndarray],
    sr: int,
    thr_db: float,
    merge_gap: float,
    margin_db: float = 3.0,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Segmentation VAD avec diarization par dominance energetique inter-pistes.

    Une frame est attribuee au role R seulement si :
      1. son energie depasse median_R + thr_db  (actif sur sa propre piste)
      2. son energie est superieure a celle de TOUS les autres roles
         d'au moins margin_db dB (dominant a cet instant)

    Cela elimine les faux positifs dus a la propagation acoustique
    (un locuteur capte par les micros des autres participants).
    """
    from analyze_audio import HOP_LENGTH, FRAME_LENGTH

    # Calcul RMS pour toutes les pistes sur la meme grille de frames
    rms_all: Dict[str, np.ndarray] = {}
    e_db_all: Dict[str, np.ndarray] = {}
    medians: Dict[str, float] = {}

    for role, y in tracks.items():
        if len(y) == 0:
            rms_all[role] = np.array([])
            e_db_all[role] = np.array([])
            medians[role] = 0.0
            continue
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                                   hop_length=HOP_LENGTH, center=True)[0]
        e_db = librosa.amplitude_to_db(rms, ref=np.max)
        rms_all[role] = rms
        e_db_all[role] = e_db
        medians[role] = float(np.median(e_db))

    # Aligner les longueurs de frames
    min_frames = min(len(v) for v in e_db_all.values() if len(v) > 0)
    for role in ROLES:
        if len(e_db_all[role]) > 0:
            e_db_all[role] = e_db_all[role][:min_frames]

    times = librosa.frames_to_time(np.arange(min_frames), sr=sr, hop_length=HOP_LENGTH)

    segs: Dict[str, List[Tuple[float, float]]] = {role: [] for role in ROLES}

    # RMS absolu (meme reference pour toutes les pistes = comparaison valide)
    rms_abs: Dict[str, np.ndarray] = {}
    for role in ROLES:
        if len(e_db_all[role]) == 0:
            rms_abs[role] = np.array([])
            continue
        rms = librosa.feature.rms(
            y=tracks[role], frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH, center=True
        )[0][:min_frames]
        rms_abs[role] = rms

    # margin lineaire : dominant si rms_role > margin * rms_other
    margin_linear = 10 ** (margin_db / 20.0)

    for role in ROLES:
        if len(e_db_all[role]) == 0:
            continue
        e = e_db_all[role]
        rms_r = rms_abs[role]
        # Critere 1 : actif sur sa propre piste (seuil relatif)
        active = e > (medians[role] + thr_db)
        # Critere 2 : RMS dominant sur toutes les autres pistes
        for other in ROLES:
            if other == role or len(rms_abs[other]) == 0:
                continue
            active &= rms_r > margin_linear * rms_abs[other]

        # Conversion mask -> segments + fusion micro-pauses
        merged: List[List[float]] = []
        i = 0
        while i < len(active):
            if active[i]:
                start = times[i]
                while i < len(active) and active[i]:
                    i += 1
                end = times[min(i, len(times) - 1)]
                if not merged or start - merged[-1][1] > merge_gap:
                    merged.append([start, end])
                else:
                    merged[-1][1] = end
            else:
                i += 1
        segs[role] = [(s, e_) for s, e_ in merged]

    return segs


def load_and_align(
    gdir: Path,
    thr_db: float,
    merge_gap: float,
    abs_floor_db: float = DEFAULT_ABS_FLOOR_DB,
    beep_overrides: Dict[str, float] | None = None,
    beep_method: str = "amplitude",
) -> Tuple[Dict[str, np.ndarray], int, Dict[str, List[Tuple[float, float]]]]:
    """
    Charge, aligne sur le bip, calcule les segments IPU.

    beep_overrides : dict role -> timestamp manuel (ex. {"calculateur": 24.95})
    beep_method    : "amplitude" (defaut, identique pipeline) ou "frequency"
                     (robuste aux pics parasites)
    """
    beep_times: Dict[str, float] = {}
    sr_ref: int | None = None

    import soundfile as _sf

    print("  Detection des bips ...")
    for role in ROLES:
        wav = find_last_wav(gdir / role)
        sr_i = _sf.info(str(wav)).samplerate
        sr_ref = sr_ref or sr_i
        if beep_overrides and role in beep_overrides:
            t_bip = beep_overrides[role]
            print(f"[OVERRIDE] {role} bip force a t={t_bip:.3f}s")
        elif beep_method == "frequency":
            t_bip, _ = detect_beep_by_frequency(wav, sr_out=sr_ref)
        elif beep_method == "sustain":
            t_bip, _ = detect_beep_by_sustained_peak(wav, sr_out=sr_ref)
        elif beep_method == "peak":
            y_tmp, sr_tmp = librosa.load(wav, sr=sr_ref, mono=True)
            t_bip = float(np.argmax(np.abs(y_tmp))) / sr_tmp
            print(f"[DEBUG] bip (peak global) > {t_bip:.3f}s  {wav.name}")
        else:
            t_bip, _ = detect_beep(wav, debug=True)
        beep_times[role] = t_bip

    assert sr_ref is not None
    print(f"  Bips detectes : { {r: f'{t:.2f}s' for r, t in beep_times.items()} }")
    print(f"  -> chaque piste est coupee a son propre bip, t=0 = debut synchronise")

    tracks: Dict[str, np.ndarray] = {}
    for role in ROLES:
        wav = find_last_wav(gdir / role)
        y, sr_loaded = librosa.load(wav, sr=sr_ref, mono=True)
        t_bip = beep_times[role]
        start_idx = int((t_bip + VIEWER_SKIP_AFTER_BEEP) * sr_loaded)
        y = y[start_idx:] if start_idx < len(y) else np.array([], dtype=np.float32)
        print(f"    {role:14s}: bip={t_bip:.2f}s  start_idx={start_idx}  duree_restante={len(y)/sr_loaded:.1f}s")
        tracks[role] = y

    # Tronquer a la duree commune
    min_len = min(len(y) for y in tracks.values())
    tracks = {r: y[:min_len] for r, y in tracks.items()}

    # Diagnostic niveaux RMS par piste
    print("\n  --- Diagnostic niveaux audio ---")
    from analyze_audio import HOP_LENGTH, FRAME_LENGTH
    for role, y in tracks.items():
        if len(y) == 0:
            print(f"    {role:14s} : VIDE")
            continue
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                                   hop_length=HOP_LENGTH, center=True)[0]
        e_rel = librosa.amplitude_to_db(rms, ref=np.max)
        e_abs = librosa.amplitude_to_db(rms, ref=1.0)
        seuil_rel = float(np.median(e_rel) + thr_db)
        pct_rel  = float(np.mean(e_rel > seuil_rel) * 100)
        pct_both = float(np.mean((e_rel > seuil_rel) & (e_abs > abs_floor_db)) * 100)
        print(
            f"    {role:14s} : "
            f"median={np.median(e_abs):+5.1f} dBFS  "
            f"p90={np.percentile(e_abs,90):+5.1f} dBFS  |  "
            f"actif relatif={pct_rel:4.1f}%  "
            f"actif+floor({abs_floor_db:+.0f}dBFS)={pct_both:4.1f}%"
        )
    print(f"  -> Ajuster --abs-floor-db entre median(silence) et p90(parole)")
    print()

    print("  Calcul segments IPU (seuil absolu dBFS) ...")
    segs: Dict[str, List[Tuple[float, float]]] = {}
    for role in ROLES:
        y = tracks[role]
        segs[role] = (
            speech_segments_with_floor(y, sr_ref, thr_db, merge_gap, abs_floor_db)
            if len(y) else []
        )
        print(f"    {role:14s} : {len(segs[role])} segments IPU")

    return tracks, sr_ref, segs


def _compute_interruptions_and_takeovers(
    segs: Dict[str, List[Tuple[float, float]]],
    turn_min: float = TURN_MIN_SEC,
) -> Dict[str, object]:
    """Calcule interruptions overlap-based et rapid floor takeovers pour toutes les paires."""
    pairs = [(ROLES[i], ROLES[j]) for i in range(len(ROLES)) for j in range(i + 1, len(ROLES))]
    attempts_total = 0
    successes_total = 0
    rapid_total = 0
    result: Dict[str, object] = {}
    for a, b in pairs:
        rapid_b_a = count_rapid_floor_takeovers(segs[a], segs[b], DELTA_INTERRUPT, min_turn_sec=turn_min)
        rapid_a_b = count_rapid_floor_takeovers(segs[b], segs[a], DELTA_INTERRUPT, min_turn_sec=turn_min)
        result[f"rapid_{b}->{a}"] = rapid_b_a
        result[f"rapid_{a}->{b}"] = rapid_a_b
        rapid_total += rapid_b_a + rapid_a_b
        att_b_a, suc_b_a = count_directed_interruptions(segs[a], segs[b],
            min_overlap=MIN_INTERRUPT_OVERLAP, min_post_takeover=MIN_POST_TAKEOVER_SEC, min_turn_sec=turn_min)
        att_a_b, suc_a_b = count_directed_interruptions(segs[b], segs[a],
            min_overlap=MIN_INTERRUPT_OVERLAP, min_post_takeover=MIN_POST_TAKEOVER_SEC, min_turn_sec=turn_min)
        attempts_total += att_b_a + att_a_b
        successes_total += suc_b_a + suc_a_b
    result["n_attempted_interruptions"] = attempts_total
    result["n_successful_interruptions"] = successes_total
    result["rapid_floor_takeovers_total"] = rapid_total
    result["successful_interruption_ratio"] = round(
        successes_total / attempts_total, 3) if attempts_total > 0 else float("nan")
    return result


def compute_pipeline_metrics(
    segs: Dict[str, List[Tuple[float, float]]],
    turns_ca: List[Dict],
    backchannels: Dict[str, List[Tuple[float, float]]],
    occupancy: Dict[str, float],
    total_s: float,
    turn_min: float = TURN_MIN_SEC,
) -> Dict[str, object]:
    """Calcule les metriques agregees pour l'affichage."""
    metrics: Dict[str, object] = {}

    # --- tours CA ---
    n_turns = len(turns_ca)
    metrics["n_turns_CA"] = n_turns
    if n_turns:
        durs = [t["end"] - t["start"] for t in turns_ca]
        metrics["mean_turn_CA_s"] = round(float(np.mean(durs)), 2)
        metrics["median_turn_CA_s"] = round(float(np.median(durs)), 2)
    else:
        metrics["mean_turn_CA_s"] = float("nan")
        metrics["median_turn_CA_s"] = float("nan")

    # --- tours CA par role ---
    for role in ROLES:
        role_turns = [t for t in turns_ca if t["role"] == role]
        metrics[f"n_turns_{role}"] = len(role_turns)

    # --- floor exchanges (CA-based) ---
    floor_gaps = []
    for i in range(len(turns_ca) - 1):
        tc, tn = turns_ca[i], turns_ca[i + 1]
        if tc["role"] != tn["role"]:
            gap = tn["start"] - tc["end"]
            if 0 < gap <= FLOOR_EXCHANGE_MAX_GAP:
                floor_gaps.append(gap)
    metrics["n_floor_exchanges"] = len(floor_gaps)
    metrics["floor_pause_mean_s"] = round(float(np.mean(floor_gaps)), 3) if floor_gaps else float("nan")

    # --- backchannels ---
    n_bc = sum(len(v) for v in backchannels.values())
    metrics["n_backchannels"] = n_bc
    metrics["bc_per_min"] = round(n_bc / (total_s / 60.0), 2) if total_s > 0 else float("nan")

    # --- backchannels par role ---
    for role in ROLES:
        metrics[f"bc_{role}_n"] = len(backchannels.get(role, []))

    # --- occupancy ---
    metrics["pct_0_spk"] = round(occupancy.get("audio_pct_time_0_speakers", float("nan")) * 100, 1)
    metrics["pct_1_spk"] = round(occupancy.get("audio_pct_time_1_speaker", float("nan")) * 100, 1)
    metrics["pct_2_spk"] = round(occupancy.get("audio_pct_time_2_speakers", float("nan")) * 100, 1)
    metrics["pct_3_spk"] = round(occupancy.get("audio_pct_time_3_speakers", float("nan")) * 100, 1)
    metrics["overlap_ratio"] = round(occupancy.get("audio_overlap_speaking_ratio_from_occupancy", float("nan")), 3)

    # --- speech ratio par role + entropie ---
    speech_times = []
    for role in ROLES:
        speech_t = sum(e - s for s, e in segs.get(role, []))
        metrics[f"speech_{role}_ratio"] = round(speech_t / total_s, 3) if total_s > 0 else float("nan")
        speech_times.append(speech_t)
    total_speech = sum(speech_times)
    if total_speech > 0:
        probs = np.array([t / total_speech for t in speech_times])
        probs = probs[probs > 0]
        metrics["participation_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 3)
    else:
        metrics["participation_entropy"] = float("nan")

    # --- interruptions et rapid takeovers ---
    metrics.update(_compute_interruptions_and_takeovers(segs, turn_min))

    return metrics


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def build_occupancy_timeline(
    segs_by_role: Dict[str, List[Tuple[float, float]]],
    total_s: float,
    fs_grid: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (t_axis, counts_grid) ou counts_grid[i] = n locuteurs a t_axis[i]."""
    n_frames = int(np.ceil(total_s * fs_grid))
    grid = np.zeros(n_frames, dtype=np.int8)
    for role_segs in segs_by_role.values():
        for s, e in role_segs:
            i0 = int(np.floor(s * fs_grid))
            i1 = int(np.ceil(e * fs_grid))
            grid[i0:min(i1, n_frames)] += 1
    t_axis = np.arange(n_frames) / fs_grid
    return t_axis, grid.clip(0, 3)


def compute_live_metrics(
    t_cur: float,
    turns_ca: List[Dict],
    backchannels: Dict[str, List[Tuple[float, float]]],
    segs: Dict[str, List[Tuple[float, float]]],
    turn_min: float = TURN_MIN_SEC,
) -> Dict[str, object]:
    """Metriques incrementales : ne compte que les evenements termines avant t_cur."""
    m: Dict[str, object] = {}

    # Tours CA termines
    done_turns = [t for t in turns_ca if t["end"] <= t_cur]
    m["n_turns_CA"] = len(done_turns)
    for role in ROLES:
        m[f"n_turns_{role}"] = sum(1 for t in done_turns if t["role"] == role)
    if done_turns:
        durs = [t["end"] - t["start"] for t in done_turns]
        m["mean_turn_CA_s"] = round(float(np.mean(durs)), 2)
        m["median_turn_CA_s"] = round(float(np.median(durs)), 2)
    else:
        m["mean_turn_CA_s"] = float("nan")
        m["median_turn_CA_s"] = float("nan")

    # Floor exchanges termines (gap entre deux tours CA consecutifs deja ecoules)
    floor_gaps = []
    for i in range(len(done_turns) - 1):
        tc, tn = done_turns[i], done_turns[i + 1]
        if tc["role"] != tn["role"]:
            gap = tn["start"] - tc["end"]
            if 0 < gap <= FLOOR_EXCHANGE_MAX_GAP:
                floor_gaps.append(gap)
    m["n_floor_exchanges"] = len(floor_gaps)
    m["floor_pause_mean_s"] = round(float(np.mean(floor_gaps)), 3) if floor_gaps else float("nan")

    # Backchannels termines
    elapsed_min = t_cur / 60.0 if t_cur > 0 else float("nan")
    total_bc = 0
    for role in ROLES:
        done_bc = [(s, e) for s, e in backchannels.get(role, []) if e <= t_cur]
        m[f"bc_{role}_n"] = len(done_bc)
        total_bc += len(done_bc)
    m["n_backchannels"] = total_bc
    m["bc_per_min"] = round(total_bc / elapsed_min, 2) if elapsed_min and elapsed_min > 0 else float("nan")

    # Occupancy en temps reel (recalcul sur [0, t_cur])
    if t_cur > 0.5:
        segs_clipped = {
            role: [(s, min(e, t_cur)) for s, e in role_segs if s < t_cur]
            for role, role_segs in segs.items()
        }
        occ = compute_speaker_occupancy(segs_clipped, t_cur)
        m["pct_0_spk"] = round(occ.get("audio_pct_time_0_speakers", float("nan")) * 100, 1)
        m["pct_1_spk"] = round(occ.get("audio_pct_time_1_speaker",  float("nan")) * 100, 1)
        m["pct_2_spk"] = round(occ.get("audio_pct_time_2_speakers", float("nan")) * 100, 1)
        m["pct_3_spk"] = round(occ.get("audio_pct_time_3_speakers", float("nan")) * 100, 1)
        m["overlap_ratio"] = round(occ.get("audio_overlap_speaking_ratio_from_occupancy", float("nan")), 3)
    else:
        for k in ("pct_0_spk", "pct_1_spk", "pct_2_spk", "pct_3_spk", "overlap_ratio"):
            m[k] = float("nan")

    # Speech ratio par role (sur [0, t_cur])
    speech_times = []
    for role in ROLES:
        speech_t = sum(min(e, t_cur) - s for s, e in segs.get(role, []) if s < t_cur)
        m[f"speech_{role}_ratio"] = round(speech_t / t_cur, 3) if t_cur > 0 else float("nan")
        speech_times.append(speech_t)

    # Entropie de Shannon sur la distribution des temps de parole par role
    total_speech = sum(speech_times)
    if total_speech > 0:
        probs = np.array([t / total_speech for t in speech_times])
        probs = probs[probs > 0]
        m["participation_entropy"] = round(float(-np.sum(probs * np.log2(probs))), 3)
    else:
        m["participation_entropy"] = float("nan")

    # Interruptions et rapid takeovers (sur segments clips a t_cur)
    if t_cur > 1.0:
        segs_clipped = {
            role: [(s, min(e, t_cur)) for s, e in role_segs if s < t_cur]
            for role, role_segs in segs.items()
        }
        m.update(_compute_interruptions_and_takeovers(segs_clipped, turn_min))
    else:
        m["n_attempted_interruptions"] = 0
        m["n_successful_interruptions"] = 0
        m["rapid_floor_takeovers_total"] = 0
        m["successful_interruption_ratio"] = float("nan")

    return m


def format_metric_text(m: Dict[str, object], t_cur: float, total_s: float) -> str:
    def _fmt(v: object, fmt: str = ".2f") -> str:
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return "—"
        return format(v, fmt)  # type: ignore[arg-type]

    lines = [
        f"t = {t_cur:6.1f} s / {total_s:.1f} s",
        "",
        "--- Parole par role ---",
        f"  calc  : {_fmt(m['speech_calculateur_ratio'], '.1%')}",
        f"  model : {_fmt(m['speech_modelisateur_ratio'], '.1%')}",
        f"  lect  : {_fmt(m['speech_lecteur_ratio'], '.1%')}",
        f"  entropie : {_fmt(m['participation_entropy'], '.3f')} bits",
        f"  (max equirep : {np.log2(len(ROLES)):.3f} bits)",
        "",
        "--- Tours CA ---",
        f"N tours CA       : {m['n_turns_CA']}",
        f"Duree moy (CA)   : {_fmt(m['mean_turn_CA_s'])} s",
        f"Duree med (CA)   : {_fmt(m['median_turn_CA_s'])} s",
        f"  calc  : {m['n_turns_calculateur']} tours",
        f"  model : {m['n_turns_modelisateur']} tours",
        f"  lect  : {m['n_turns_lecteur']} tours",
        "",
        "--- Floor exchanges ---",
        f"N exchanges      : {m['n_floor_exchanges']}",
        f"Pause moy        : {_fmt(m['floor_pause_mean_s'], '.3f')} s",
        "",
        "--- Backchannels ---",
        f"N total          : {m['n_backchannels']}",
        f"Taux / min       : {_fmt(m['bc_per_min'])}",
        f"  calc  : {m['bc_calculateur_n']}",
        f"  model : {m['bc_modelisateur_n']}",
        f"  lect  : {m['bc_lecteur_n']}",
        "",
        "--- Interruptions ---",
        f"Tentatives       : {m['n_attempted_interruptions']}",
        f"Succes           : {m['n_successful_interruptions']}",
        f"Ratio succes     : {_fmt(m['successful_interruption_ratio'], '.1%')}",
        f"Rapid takeovers  : {m['rapid_floor_takeovers_total']}",
        "",
        "--- Occupancy (cumulee) ---",
        f"0 spk  : {_fmt(m['pct_0_spk'], '.1f')}%",
        f"1 spk  : {_fmt(m['pct_1_spk'], '.1f')}%",
        f"2 spk  : {_fmt(m['pct_2_spk'], '.1f')}%",
        f"3 spk  : {_fmt(m['pct_3_spk'], '.1f')}%",
        f"Overlap ratio    : {_fmt(m['overlap_ratio'], '.3f')}",
    ]
    return "\n".join(lines)


def collect_interruption_events(
    segs: Dict[str, List[Tuple[float, float]]],
    turn_min: float = TURN_MIN_SEC,
) -> List[Dict]:
    """
    Retourne la liste des interruptions avec timestamps et roles.
    Chaque dict : {t_start, source, interruptor, success}
    """
    pairs = [(ROLES[i], ROLES[j]) for i in range(len(ROLES)) for j in range(i + 1, len(ROLES))]
    events = []
    for a, b in pairs:
        for direction, src, tgt in [(f"{b}->{a}", a, b), (f"{a}->{b}", b, a)]:
            filtered_src = [(s, e) for s, e in segs.get(src, []) if (e - s) >= turn_min]
            filtered_tgt = [(s, e) for s, e in segs.get(tgt, []) if (e - s) >= turn_min]
            for a_start, a_end in filtered_src:
                for b_start, b_end in filtered_tgt:
                    if a_start < b_start < a_end:
                        overlap = min(a_end, b_end) - b_start
                        if overlap >= MIN_INTERRUPT_OVERLAP:
                            success = (b_end - a_end) >= MIN_POST_TAKEOVER_SEC
                            events.append({
                                "t": b_start,
                                "source": src,
                                "interruptor": tgt,
                                "success": success,
                            })
    return events


def make_figure(
    tracks: Dict[str, np.ndarray],
    sr: int,
    segs: Dict[str, List[Tuple[float, float]]],
    turns_ca: List[Dict],
    backchannels: Dict[str, List[Tuple[float, float]]],
    total_s: float,
    metric_text: str,
    group_name: str,
    turn_min: float = TURN_MIN_SEC,
) -> Tuple[plt.Figure, Dict, object]:
    """
    Construit la figure multi-panneaux.
    Retourne (fig, cursors_dict, metric_text_artist).
    metric_text_artist : objet Text matplotlib a mettre a jour via set_text().
    """
    down = max(1, sr // DISP_SR)

    # ---- mise en page ----
    fig = plt.figure(figsize=(17, 11))
    fig.suptitle(f"Verificateur audio — {group_name}", fontsize=12, fontweight="bold")

    gs = GridSpec(
        5, 2,
        figure=fig,
        width_ratios=[4, 1],
        height_ratios=[1, 1, 1, 0.8, 0.8],
        hspace=0.08,
        wspace=0.04,
    )

    n_roles = len(ROLES)
    wave_axes = []
    seg_axes = []
    cursors: Dict[str, object] = {}

    # --- panneaux formes d'onde (lignes 0..2, col 0) ---
    for i, role in enumerate(ROLES):
        ax = fig.add_subplot(gs[i, 0])
        y_ds = tracks[role][::down]
        t_wave = np.arange(len(y_ds)) / DISP_SR
        ax.plot(t_wave, y_ds, color=ROLE_COLORS[role], lw=0.5, alpha=0.8)
        ax.set_ylabel(role[:5], fontsize=8, rotation=0, labelpad=28, va="center")
        ax.set_yticks([])
        ax.set_xlim(0, total_s)
        if i < n_roles - 1:
            ax.set_xticklabels([])
        wave_axes.append(ax)
        c = ax.axvline(0, color="red", lw=1.2, alpha=0.8)
        cursors[f"wave_{role}"] = c

    # --- panneau segments IPU + tours CA + backchannels (ligne 3, col 0) ---
    ax_seg = fig.add_subplot(gs[3, 0], sharex=wave_axes[0])
    ax_seg.set_ylabel("IPU/CA/BC", fontsize=8, rotation=0, labelpad=28, va="center")
    ax_seg.set_yticks([])
    ax_seg.set_xlim(0, total_s)
    ax_seg.set_xticklabels([])

    y_offsets = {role: i * 3 for i, role in enumerate(ROLES)}
    y_max = len(ROLES) * 3 + 0.5
    ax_seg.set_ylim(-0.5, y_max)

    for role in ROLES:
        yb = y_offsets[role]
        col = ROLE_COLORS[role]
        # IPU (barres fines, bleu transparent)
        for s, e in segs[role]:
            ax_seg.barh(yb + 0.4, e - s, left=s, height=0.6,
                        color=col, alpha=0.35, linewidth=0)
        # Tours CA (barres plus hautes, meme couleur, plus opaque)
        for t in turns_ca:
            if t["role"] == role:
                ax_seg.barh(yb + 1.1, t["end"] - t["start"], left=t["start"], height=0.6,
                            color=col, alpha=0.85, linewidth=0)
        # Backchannels (triangles rouges)
        for s, e in backchannels.get(role, []):
            mid = (s + e) / 2
            ax_seg.plot(mid, yb + 0.4, marker="v", color="red",
                        markersize=5, zorder=5, alpha=0.9)

    # Interruptions : etoile sur la piste de la source, couleur = succes/echec
    interruption_events = collect_interruption_events(segs, turn_min)
    for ev in interruption_events:
        yb_src = y_offsets[ev["source"]]
        color_int = "#e74c3c" if ev["success"] else "#e67e22"
        ax_seg.plot(ev["t"], yb_src + 1.8, marker="*", color=color_int,
                    markersize=8, zorder=6, alpha=0.95,
                    markeredgecolor="white", markeredgewidth=0.4)

    # Legende segments
    ipu_patch   = mpatches.Patch(color="gray", alpha=0.4, label="IPU (VAD)")
    ca_patch    = mpatches.Patch(color="gray", alpha=0.85, label="Tour CA")
    bc_marker   = plt.Line2D([0], [0], marker="v", color="red", linestyle="None",
                             markersize=6, label="Backchannel")
    int_suc     = plt.Line2D([0], [0], marker="*", color="#e74c3c", linestyle="None",
                             markersize=7, label="Interrupt. (succes)")
    int_fail    = plt.Line2D([0], [0], marker="*", color="#e67e22", linestyle="None",
                             markersize=7, label="Interrupt. (echec)")
    ax_seg.legend(handles=[ipu_patch, ca_patch, bc_marker, int_suc, int_fail],
                  loc="upper right", fontsize=7, ncol=5)

    # Etiquettes roles sur axe Y
    for role in ROLES:
        yb = y_offsets[role]
        ax_seg.text(-0.5, yb + 1.0, role[:3], fontsize=7, ha="right", va="center",
                    transform=ax_seg.get_yaxis_transform())

    c_seg = ax_seg.axvline(0, color="red", lw=1.2, alpha=0.8)
    cursors["seg"] = c_seg
    seg_axes.append(ax_seg)

    # --- panneau occupancy (ligne 4, col 0) ---
    ax_occ = fig.add_subplot(gs[4, 0], sharex=wave_axes[0])
    ax_occ.set_ylabel("Occupancy", fontsize=8, rotation=0, labelpad=28, va="center")
    ax_occ.set_xlabel("Temps (s)", fontsize=8)
    ax_occ.set_xlim(0, total_s)
    ax_occ.set_ylim(0, 3.1)

    t_occ, grid_occ = build_occupancy_timeline(segs, total_s, fs_grid=50.0)
    occ_labels = ["0 spk", "1 spk", "2 spk", "3 spk"]
    # Chaque etat k remplit exactement la bande [0, k] — superpose du plus grand au plus petit
    # pour que le silence (k=0, hauteur 0) soit representé par le fond de l'axe
    ax_occ.set_facecolor("#d0d0d0")  # fond gris = silence visible
    for k, (col, lbl) in enumerate(zip(OCCUPANCY_COLORS[1:], occ_labels[1:]), start=1):
        ax_occ.fill_between(t_occ, 0, np.where(grid_occ >= k, k, 0).astype(float),
                            step="post", color=col, alpha=1.0, label=lbl)

    silence_patch = mpatches.Patch(color="#d0d0d0", label="0 spk", edgecolor="#999999", linewidth=0.5)
    occ_handles = [silence_patch] + [
        mpatches.Patch(color=col, label=lbl)
        for col, lbl in zip(OCCUPANCY_COLORS[1:], occ_labels[1:])
    ]
    ax_occ.set_yticks([0, 1, 2, 3])
    ax_occ.set_yticklabels(["0", "1", "2", "3"], fontsize=7)
    ax_occ.set_ylim(0, 3.5)
    ax_occ.legend(handles=occ_handles, loc="upper right", fontsize=7, ncol=4)

    c_occ = ax_occ.axvline(0, color="red", lw=1.2, alpha=0.8)
    cursors["occ"] = c_occ

    # --- panneau metriques texte (colonne droite, toute la hauteur) ---
    ax_txt = fig.add_subplot(gs[:, 1])
    ax_txt.set_axis_off()
    txt_artist = ax_txt.text(
        0.05, 0.98, metric_text,
        transform=ax_txt.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(facecolor="whitesmoke", edgecolor="gray", boxstyle="round,pad=0.5"),
    )

    # Legende couleurs roles
    legend_patches = [
        mpatches.Patch(color=ROLE_COLORS[r], label=r) for r in ROLES
    ]
    fig.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=1,
               bbox_to_anchor=(0.99, 0.02))

    return fig, cursors, txt_artist


# ---------------------------------------------------------------------------
# Lecture audio
# ---------------------------------------------------------------------------

def play_audio(tracks: Dict[str, np.ndarray], sr: int) -> threading.Thread:
    """Lance la lecture stereo dans un thread daemon."""
    if not HAS_AUDIO:
        print("[AVERTISSEMENT] sounddevice non disponible — lecture audio desactivee")
        return threading.Thread(target=lambda: None, daemon=True)

    max_len = max(len(y) for y in tracks.values())
    mix = np.zeros((max_len, 2), dtype=np.float32)
    pans = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    for role, pan in zip(ROLES, pans):
        y = tracks[role]
        mix[:len(y), 0] += y * pan[0]
        mix[:len(y), 1] += y * pan[1]
    # Normalisation legere
    peak = np.abs(mix).max()
    if peak > 0:
        mix /= peak * 1.1

    def _play():
        sd.play(mix, sr)
        sd.wait()

    th = threading.Thread(target=_play, daemon=True)
    th.start()
    return th


# ---------------------------------------------------------------------------
# Boucle principale visualisation
# ---------------------------------------------------------------------------

def run_viewer(
    group_dir: Path,
    thr_db: float = DEFAULT_THR_DB,
    merge_gap: float = DEFAULT_MERGE_GAP,
    turn_min: float = TURN_MIN_SEC,
    bc_max: float = BACKCH_MAX_SEC,
    bc_min_ovl: float = BACKCH_MIN_OVL,
    abs_floor_db: float = DEFAULT_ABS_FLOOR_DB,
    beep_overrides: Dict[str, float] | None = None,
    beep_method: str = "amplitude",
    play: bool = True,
) -> None:
    print(f"\n=== Verificateur pipeline audio : {group_dir.name} ===")
    print(f"  Parametres : thr_db={thr_db}, merge_gap={merge_gap}s, "
          f"turn_min={turn_min}s, bc_max={bc_max}s, bc_min_ovl={bc_min_ovl}s, "
          f"abs_floor_db={abs_floor_db}dBFS  beep={beep_method}")

    # 1) Chargement + alignement + IPU
    tracks, sr, segs = load_and_align(
        group_dir, thr_db, merge_gap, abs_floor_db, beep_overrides, beep_method
    )
    total_s = effective_analysis_duration_s(tracks, sr)
    print(f"  Duree analysee : {total_s:.1f} s")

    # 2) Tours CA (MOD-1 : 3 passes)
    print("  Agregation tours CA ...")
    turns_ca = aggregate_ipus_to_ca_turns(segs, turn_min_s=turn_min)
    print(f"  -> {len(turns_ca)} tours CA")

    # 3) Backchannels stricts (MOD-3 : 4 filtres)
    print("  Detection backchannels stricts ...")
    backchannels: Dict[str, List[Tuple[float, float]]] = {}
    for role in ROLES:
        bcs = []
        other_all: List[Tuple[float, float]] = []
        for r2, s2 in segs.items():
            if r2 != role:
                other_all.extend(s2)
        for s, e in segs[role]:
            if is_backchannel_strict(
                s, e,
                role_segs=segs[role],
                other_segs=other_all,
                turns_ca=turns_ca,
                role=role,
                bc_min_dur=0.10,
                bc_max_dur=bc_max,
                bc_min_ovl=bc_min_ovl,
            ):
                bcs.append((s, e))
        backchannels[role] = bcs
    total_bc = sum(len(v) for v in backchannels.values())
    print(f"  -> {total_bc} backchannels stricts")

    # 4) Occupancy (MOD-2)
    print("  Calcul occupancy ...")
    occupancy = compute_speaker_occupancy(segs, total_s)

    # 5) Metriques initiales (t=0)
    metrics_t0 = compute_live_metrics(0.0, turns_ca, backchannels, segs, turn_min)
    metric_text_init = format_metric_text(metrics_t0, 0.0, total_s)

    print("\n--- Metriques finales (session complete) ---")
    metrics_final = compute_pipeline_metrics(segs, turns_ca, backchannels, occupancy, total_s, turn_min)
    for line in format_metric_text(metrics_final, total_s, total_s).split("\n"):
        print(" ", line)

    # 6) Lecture audio + animation curseur + metriques live
    print("\n  Construction figure ...")
    fig, cursors, txt_artist = make_figure(
        tracks, sr, segs, turns_ca, backchannels, total_s, metric_text_init, group_dir.name,
        turn_min=turn_min,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Cadence de mise a jour des metriques (toutes les N frames pour economiser le CPU)
    _METRIC_UPDATE_EVERY = 10   # frames matplotlib (~0.3 s)
    _frame_counter = 0

    if play:
        print("  Lecture audio (fermer la fenetre pour quitter) ...")
        th = play_audio(tracks, sr)
        start_t = time.time()
        while th.is_alive():
            t_cur = time.time() - start_t
            for c in cursors.values():
                c.set_xdata([t_cur])
            _frame_counter += 1
            if _frame_counter % _METRIC_UPDATE_EVERY == 0:
                live_m = compute_live_metrics(t_cur, turns_ca, backchannels, segs, turn_min)
                txt_artist.set_text(format_metric_text(live_m, t_cur, total_s))
            fig.canvas.draw_idle()
            plt.pause(0.03)
        # Affichage final (session complete)
        final_m = compute_live_metrics(total_s, turns_ca, backchannels, segs, turn_min)
        txt_artist.set_text(format_metric_text(final_m, total_s, total_s))
        fig.canvas.draw_idle()
        plt.pause(0.1)
    else:
        # Affichage statique : metriques finales d'emblee
        final_m = compute_live_metrics(total_s, turns_ca, backchannels, segs, turn_min)
        txt_artist.set_text(format_metric_text(final_m, total_s, total_s))
        print("  Affichage statique (--no-play) ...")
        plt.show()

    plt.close(fig)
    print("  Termine.")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verificateur visuel pipeline audio INV — IPU, tours CA, backchannels, occupancy"
    )
    ap.add_argument("group_dir", help="Repertoire du groupe (doit contenir calculateur/, modelisateur/, lecteur/)")
    ap.add_argument("--thr-db",    type=float, default=DEFAULT_THR_DB,    help=f"Seuil VAD dB (defaut {DEFAULT_THR_DB})")
    ap.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP, help=f"Fusion micro-pauses s (defaut {DEFAULT_MERGE_GAP})")
    ap.add_argument("--turn-min",  type=float, default=TURN_MIN_SEC,      help=f"Duree min tour CA s (defaut {TURN_MIN_SEC})")
    ap.add_argument("--bc-max",    type=float, default=BACKCH_MAX_SEC,    help=f"Duree max backchannel s (defaut {BACKCH_MAX_SEC})")
    ap.add_argument("--bc-min-ovl",type=float, default=BACKCH_MIN_OVL,   help=f"Overlap min backchannel s (defaut {BACKCH_MIN_OVL})")
    ap.add_argument("--abs-floor-db", type=float, default=DEFAULT_ABS_FLOOR_DB,
                    help=f"Seuil absolu VAD dBFS (defaut {DEFAULT_ABS_FLOOR_DB})")
    ap.add_argument("--beep-method", choices=["amplitude", "frequency", "sustain", "peak"], default="peak",
                    help="Methode detection bip : peak (defaut, pic global absolu), "
                         "amplitude (pic + silence avant), sustain (pic soutenu), frequency (ratio 800-2000Hz)")
    ap.add_argument("--beep-override", action="append", default=[], metavar="ROLE:T",
                    help="Force le timestamp du bip pour un role. Ex: --beep-override calculateur:24.95"
                         " (repetable pour plusieurs roles)")
    ap.add_argument("--no-play",   action="store_true",                   help="Affichage statique sans lecture audio")
    args = ap.parse_args()

    # Parse beep overrides "role:timestamp"
    beep_overrides: Dict[str, float] = {}
    for ov in args.beep_override:
        try:
            role, t = ov.split(":")
            beep_overrides[role.strip()] = float(t.strip())
        except ValueError:
            print(f"[ERREUR] --beep-override format invalide : '{ov}' (attendu role:timestamp)")

    run_viewer(
        group_dir=Path(args.group_dir),
        thr_db=args.thr_db,
        merge_gap=args.merge_gap,
        turn_min=args.turn_min,
        bc_max=args.bc_max,
        bc_min_ovl=args.bc_min_ovl,
        abs_floor_db=args.abs_floor_db,
        beep_overrides=beep_overrides or None,
        beep_method=args.beep_method,
        play=not args.no_play,
    )


if __name__ == "__main__":
    main()
