#!/usr/bin/env python3
"""
viewer_annotated.py
===================
Viewer audio interactif avec annotations comportementales superposées.

Affiche, pour chaque rôle, la forme d'onde alignée sur le bip avec :
  - Tours de parole (barres bleues)
  - Backchannels (marqueurs orange triangulaires)
  - Interruptions tentées / réussies (zones rouges / vertes)
  - Chevauchements de groupe (rangée dédiée, rouge)
  - Transitions de plancher (pauses inter-locuteurs, tirets gris)

Une rangée synthèse en bas montre le ratio de parole par rôle en temps réel.

Usage :
    python viewer_annotated.py chemin/vers/dossier_groupe [OPTIONS]

Options :
    --no-audio          Ne pas jouer le son (visualisation seule)
    --thr-db FLOAT      Seuil VAD en dB (défaut : 6.0)
    --merge-gap FLOAT   Fusion micro-pauses en s (défaut : 0.25)
    --turn-min FLOAT    Durée min tour en s (défaut : 1.0)
    --backch-max FLOAT  Durée max backchannel en s (défaut : 0.7)
    --backch-ovl FLOAT  Chevauchement min backchannel en s (défaut : 0.10)
    --save PNG          Enregistre le graphique annoté en PNG sans lancer la lecture

Dépendances : numpy, scipy, librosa, soundfile, sounddevice, matplotlib
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Ajoute la racine du projet au path (même convention que analyze_audio.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.constants import ROLES
from common.io_utils import find_wav, is_vr_group, is_pc_group, find_pc_role_wavs
from common.temporal import silence_intervals, compute_floor_exchanges

# ─────────────────────────── constantes ────────────────────────────────────────
HOP_LENGTH       = 256      # ~16 ms @ 16 kHz
FRAME_LENGTH     = 1024     # ~64 ms
DEFAULT_THR_DB   = 6.0
DEFAULT_MERGE_GAP = 0.25    # s
TURN_MIN_SEC     = 1.0      # s
TURN_MAX_PAUSE_SEC = 3.0    # silence intra-tour max avant coupure (s)
BACKCH_MAX_SEC   = 0.7      # s
BACKCH_MIN_OVL   = 0.10     # s
SKIP_AFTER_BEEP  = 0.05     # s
FLOOR_EXCHANGE_MAX_GAP = 2.0
MIN_POST_TAKEOVER_SEC  = 0.5
MIN_INTERRUPT_OVERLAP  = 0.1
DISP_SR          = 200      # Hz pour downsampling affichage

# Couleurs par rôle
ROLE_COLORS = {
    "calculateur": "tab:blue",
    "modelisateur": "tab:green",
    "lecteur":      "tab:purple",
}

# ─────────────────────────── chargement audio ───────────────────────────────────

def detect_beep(path: Path, peak_ratio: float = 0.9, pre_silence: float = 0.2) -> Tuple[float, int]:
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y.astype(np.float32)
    abs_y = np.abs(y)
    peak_amp = abs_y.max()
    hard_thr = peak_ratio * peak_amp
    cand_idx = np.where(abs_y >= hard_thr)[0]
    if cand_idx.size == 0:
        raise RuntimeError(f"Bip non détecté dans {path.name}")
    min_gap = int(pre_silence * sr)
    for idx in cand_idx:
        if idx < min_gap:
            continue
        if abs_y[idx - min_gap: idx].max() <= 0.1 * peak_amp:
            return idx / sr, sr
    return cand_idx[0] / sr, sr


def load_tracks(gdir: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """Charge et aligne les pistes (VR : sur bip, PC : troncature commune)."""
    is_vr = is_vr_group(gdir)
    if not is_vr and not is_pc_group(gdir):
        raise SystemExit(f"Groupe invalide : {gdir}")

    if is_vr:
        files = {r: find_wav(gdir / r) for r in ROLES}
        beep, sr = {}, None
        for r, f in files.items():
            t, sr_i = detect_beep(f)
            beep[r] = t
            sr = sr or sr_i
        ref = min(beep.values())
        tracks = {}
        for r, f in files.items():
            y, _ = librosa.load(f, sr=sr, mono=True)
            diff = beep[r] - ref
            if diff > 0:
                y = y[int(diff * sr):]
            elif diff < 0:
                y = np.concatenate([np.zeros(int(-diff * sr), dtype=y.dtype), y])
            start_idx = int((ref + SKIP_AFTER_BEEP) * sr)
            tracks[r] = y[start_idx:] if start_idx < len(y) else np.array([], dtype=np.float32)
    else:
        files = find_pc_role_wavs(gdir / "processed_openface")
        sr = 16000
        tracks = {}
        for r, f in files.items():
            y, _ = librosa.load(f, sr=sr, mono=True)
            tracks[r] = y

    min_len = min(len(y) for y in tracks.values())
    tracks = {r: y[:min_len] for r, y in tracks.items()}
    return tracks, sr


# ─────────────────────────── VAD / segmentation ─────────────────────────────────

def speech_segments(
    y: np.ndarray, sr: int, thr_db: float, merge_gap: float,
    abs_floor_dbfs: float = -30.0,
) -> List[Tuple[float, float]]:
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=True)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=HOP_LENGTH)
    mask = db > (np.median(db) + thr_db)
    if abs_floor_dbfs > -float("inf"):
        e_abs = librosa.amplitude_to_db(rms, ref=1.0)
        mask = mask & (e_abs > abs_floor_dbfs)
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


# ─────────────────────────── calcul des événements ──────────────────────────────

def compute_events(
    segs: Dict[str, List[Tuple[float, float]]],
    total: float,
    turn_min: float,
    backch_max: float,
    backch_ovl_min: float,
    turn_max_pause: float = TURN_MAX_PAUSE_SEC,
) -> dict:
    """
    Calcule tous les événements à annoter.

    Retourne un dict avec :
      turns        : {role: [(s, e), ...]}
      backchannels : {role: [(s, e), ...]}
      overlaps     : [(s, e), ...]           — co-parole ≥2 locuteurs
      interrupts   : [(s, e, success:bool), ...] — niveau groupe
      floor_gaps   : [(t_end_A, t_start_B, gap_dur), ...]
    """
    # Tours : segments IPU fusionnés avec contrainte de pause intra-tour
    def _merge_ipus(seg_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        merged: List[List[float]] = []
        for s, e in sorted(seg_list):
            if merged and (s - merged[-1][1]) <= turn_max_pause:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged if (e - s) >= turn_min]

    turns = {r: _merge_ipus(seg) for r, seg in segs.items()}

    # Backchannels
    others = {r: [seg for rr, lst in segs.items() if rr != r for seg in lst] for r in segs}
    backchannels: Dict[str, List[Tuple[float, float]]] = {}
    for r, seg in segs.items():
        bc = []
        for s, e in seg:
            if (e - s) > backch_max:
                continue
            for os, oe in others[r]:
                ovl = max(0.0, min(e, oe) - max(s, os))
                if ovl >= backch_ovl_min:
                    bc.append((s, e))
                    break
        backchannels[r] = bc

    # Overlaps de groupe
    events_sweep: List[Tuple[float, int]] = []
    for seg_list in segs.values():
        for s, e in seg_list:
            events_sweep.append((s, 1))
            events_sweep.append((e, -1))
    events_sweep.sort(key=lambda x: (x[0], -x[1]))
    overlaps: List[Tuple[float, float]] = []
    count = 0
    start_ov: float | None = None
    for t, delta in events_sweep:
        prev = count
        count += delta
        if prev < 2 <= count:
            start_ov = t
        elif prev >= 2 > count and start_ov is not None and t > start_ov:
            overlaps.append((start_ov, t))
            start_ov = None

    # Interruptions (overlap-based, directionnelles, tous rôles)
    role_list = list(segs.keys())
    pairs = [(role_list[i], role_list[j]) for i in range(len(role_list)) for j in range(len(role_list)) if i != j]
    interrupts: List[Tuple[float, float, bool]] = []
    for a_role, b_role in pairs:
        filt_a = [(s, e) for s, e in segs[a_role] if (e - s) >= turn_min]
        filt_b = [(s, e) for s, e in segs[b_role] if (e - s) >= turn_min]
        for a_s, a_e in filt_a:
            for b_s, b_e in filt_b:
                if a_s < b_s < a_e:
                    ovl = min(a_e, b_e) - b_s
                    if ovl >= MIN_INTERRUPT_OVERLAP:
                        success = (b_e - a_e) >= MIN_POST_TAKEOVER_SEC
                        # Représenté par la zone de chevauchement effectif
                        interrupts.append((b_s, min(a_e, b_e), success))

    # Floor gaps (timestamps)
    all_turns_sorted = []
    for role, seg in segs.items():
        for s, e in seg:
            if (e - s) >= turn_min:
                all_turns_sorted.append((s, e, role))
    all_turns_sorted.sort(key=lambda x: x[0])
    floor_gaps: List[Tuple[float, float, float]] = []
    for i in range(len(all_turns_sorted) - 1):
        s1, e1, r1 = all_turns_sorted[i]
        s2, e2, r2 = all_turns_sorted[i + 1]
        if r1 == r2:
            continue
        gap = s2 - e1
        if 0.0 <= gap <= FLOOR_EXCHANGE_MAX_GAP:
            floor_gaps.append((e1, s2, gap))

    return {
        "turns":        turns,
        "backchannels": backchannels,
        "overlaps":     overlaps,
        "interrupts":   interrupts,
        "floor_gaps":   floor_gaps,
    }


# ─────────────────────────── métriques résumé ───────────────────────────────────

def compute_summary(
    segs: Dict[str, List[Tuple[float, float]]],
    events: dict,
    total: float,
) -> Dict[str, str]:
    """Calcule un résumé textuel des métriques clés."""
    lines = {}
    for r in segs:
        turns = events["turns"][r]
        bc    = events["backchannels"][r]
        speech_s = sum(e - s for s, e in segs[r])
        n_turns  = len(turns)
        mean_turn = float(np.mean([e - s for s, e in turns])) if turns else 0.0
        ratio = speech_s / total if total > 0 else 0.0
        bc_rate = len(bc) / (total / 60) if total > 0 else 0.0
        lines[r] = (
            f"parole {ratio:.0%}  |  "
            f"{n_turns} tours (moy {mean_turn:.1f}s)  |  "
            f"{len(bc)} backchannels ({bc_rate:.1f}/min)"
        )

    ovl_dur = sum(e - s for s, e in events["overlaps"])
    ovl_ratio = ovl_dur / total if total > 0 else 0.0
    n_int   = len(events["interrupts"])
    n_succ  = sum(1 for _, _, ok in events["interrupts"] if ok)
    int_ratio = n_succ / n_int if n_int > 0 else float("nan")
    lines["_group"] = (
        f"overlap {ovl_ratio:.1%}  |  "
        f"{n_int} interruptions (réussies {n_succ}, ratio {int_ratio:.0%})"
        if n_int > 0 else
        f"overlap {ovl_ratio:.1%}  |  aucune interruption détectée"
    )
    floor_durs = [g for _, _, g in events["floor_gaps"]]
    if floor_durs:
        lines["_group"] += f"  |  floor gap moy {np.mean(floor_durs):.2f}s"

    return lines


# ─────────────────────────── rendu matplotlib ───────────────────────────────────

def build_figure(
    tracks: Dict[str, np.ndarray],
    sr: int,
    events: dict,
    summary: Dict[str, str],
    group_name: str,
) -> Tuple[plt.Figure, dict]:
    """
    Construit la figure annotée.

    Retourne (fig, axline_refs) où axline_refs = {role: Line2D curseur}.
    """
    n_roles = len(ROLES)
    # Rangées : 1 par rôle + 1 overlaps/groupe
    n_rows = n_roles + 1
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(16, 2.5 * n_rows),
        gridspec_kw={"height_ratios": [1.2] * n_roles + [0.6]},
    )
    fig.suptitle(f"Viewer annoté — {group_name}", fontsize=13, fontweight="bold")

    down = max(1, sr // DISP_SR)
    t_max = len(next(iter(tracks.values()))) / sr
    ax_cursors: dict = {}

    for idx, role in enumerate(ROLES):
        ax = axes[idx]
        y = tracks[role]
        y_ds = y[::down]
        t_ds = np.arange(len(y_ds)) / DISP_SR
        color = ROLE_COLORS[role]

        # Forme d'onde (grisée)
        ax.fill_between(t_ds, y_ds, -y_ds, color="0.85", linewidth=0)
        ax.plot(t_ds, y_ds, color="0.65", linewidth=0.4)

        # Tours de parole (bandes colorées translucides)
        for s, e in events["turns"][role]:
            ax.axvspan(s, e, color=color, alpha=0.25, linewidth=0)

        # Backchannels (marqueurs triangles orange)
        for s, e in events["backchannels"][role]:
            mid = (s + e) / 2
            ax.annotate(
                "▲", xy=(mid, 0), fontsize=7, color="darkorange",
                ha="center", va="center",
                annotation_clip=True,
            )

        # Interruptions sur la piste du locuteur interrompu
        # On repère les interruptions où ce rôle est "a" (celui qui parle en premier)
        for i_s, i_e, success in events["interrupts"]:
            # On approxime : si la zone chevauche un tour de ce rôle
            for t_s, t_e in events["turns"][role]:
                if t_s < i_s < t_e:
                    edge_color = "limegreen" if success else "crimson"
                    ax.axvspan(i_s, i_e, color=edge_color, alpha=0.45, linewidth=0)
                    break

        # Floor gaps (tirets verticaux gris fins à la fin du tour A)
        for t_end, t_start, gap in events["floor_gaps"]:
            # On trace uniquement si ce rôle termine le tour (approximation : t_end dans ses tours)
            for ts, te in events["turns"][role]:
                if abs(te - t_end) < 0.05:
                    ax.axvline(t_end, color="0.5", linestyle=":", linewidth=0.8, alpha=0.7)
                    break

        ax.set_xlim(0, t_max)
        ax.set_ylabel(role, fontsize=9, color=color, fontweight="bold")
        ax.set_yticks([])
        ax.set_ylim(-1.05, 1.05)
        ax.text(
            0.002, 0.96, summary.get(role, ""),
            transform=ax.transAxes,
            fontsize=7, va="top", color="0.3",
        )

        # Curseur (mis à jour en temps réel)
        cursor = ax.axvline(0, color="red", linewidth=1.2, zorder=10)
        ax_cursors[role] = cursor

    # Rangée synthèse (overlaps + légende)
    ax_ov = axes[-1]
    ax_ov.set_facecolor("0.97")
    for s, e in events["overlaps"]:
        ax_ov.axvspan(s, e, color="red", alpha=0.5, linewidth=0)
    # Silences collectifs (gris)
    sil = silence_intervals({r: events["turns"][r] for r in ROLES}, t_max)
    for s, e in sil:
        ax_ov.axvspan(s, e, color="0.88", alpha=0.8, linewidth=0)

    ax_ov.set_xlim(0, t_max)
    ax_ov.set_ylim(0, 1)
    ax_ov.set_yticks([])
    ax_ov.set_ylabel("overlap\n+ silences", fontsize=8, color="red")
    ax_ov.set_xlabel("Temps (s)", fontsize=9)
    ax_ov.text(
        0.002, 0.92, summary.get("_group", ""),
        transform=ax_ov.transAxes,
        fontsize=7, va="top", color="0.3",
    )
    cursor_ov = ax_ov.axvline(0, color="red", linewidth=1.2, zorder=10)
    ax_cursors["_overlap"] = cursor_ov

    # Légende
    legend_elements = [
        mpatches.Patch(color="tab:blue",   alpha=0.3,  label="Tour de parole"),
        mpatches.Patch(color="red",        alpha=0.5,  label="Chevauchement groupe"),
        mpatches.Patch(color="limegreen",  alpha=0.5,  label="Interruption réussie"),
        mpatches.Patch(color="crimson",    alpha=0.5,  label="Interruption tentée"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="darkorange",
               markersize=7, label="Backchannel"),
        Line2D([0], [0], color="0.5", linestyle=":", linewidth=1, label="Fin tour (floor gap)"),
        Line2D([0], [0], color="red",  linewidth=1.2, label="Curseur lecture"),
    ]
    axes[0].legend(
        handles=legend_elements, loc="upper right",
        fontsize=7, framealpha=0.7, ncol=4,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, ax_cursors


# ─────────────────────────── lecture audio ──────────────────────────────────────

def make_stereo_mix(tracks: Dict[str, np.ndarray], sr: int) -> np.ndarray:
    """Mix stéréo basique : calc=gauche, mod=centre, lect=droite."""
    pans = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
    max_len = max(len(y) for y in tracks.values())
    mix = np.zeros((max_len, 2), dtype=np.float32)
    for role, pan in zip(ROLES, pans):
        y = tracks.get(role, np.array([], dtype=np.float32))
        mix[:len(y), 0] += y * pan[0]
        mix[:len(y), 1] += y * pan[1]
    return mix


# ─────────────────────────── main ───────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Viewer audio annoté (tours, backchannels, interruptions, overlaps)"
    )
    ap.add_argument("group_dir", help="Répertoire du groupe (VR ou PC)")
    ap.add_argument("--no-audio",   action="store_true",  help="Désactiver la lecture audio")
    ap.add_argument("--thr-db",     type=float, default=DEFAULT_THR_DB)
    ap.add_argument("--merge-gap",  type=float, default=DEFAULT_MERGE_GAP)
    ap.add_argument("--turn-min",   type=float, default=TURN_MIN_SEC)
    ap.add_argument("--turn-max-pause", type=float, default=TURN_MAX_PAUSE_SEC,
                    help="Silence intra-tour max avant coupure (s, defaut 3.0)")
    ap.add_argument("--backch-max", type=float, default=BACKCH_MAX_SEC)
    ap.add_argument("--backch-ovl", type=float, default=BACKCH_MIN_OVL)
    ap.add_argument("--abs-floor-db", type=float, default=-30.0,
                    help="Seuil absolu VAD en dBFS (defaut: -30 dBFS)")
    ap.add_argument("--save",       metavar="PNG", default=None,
                    help="Enregistre le graphique sans lecture (ex: out.png)")
    args = ap.parse_args()

    gdir = Path(args.group_dir)
    if not gdir.exists():
        raise SystemExit(f"❌ Dossier introuvable : {gdir}")

    print(f"• Chargement des pistes audio — {gdir.name} …")
    tracks, sr = load_tracks(gdir)
    total = len(next(iter(tracks.values()))) / sr
    print(f"  Durée commune : {total:.1f} s  |  SR : {sr} Hz")

    print("• Segmentation VAD …")
    segs: Dict[str, List[Tuple[float, float]]] = {}
    for role in ROLES:
        y = tracks.get(role, np.array([], dtype=np.float32))
        segs[role] = speech_segments(y, sr, args.thr_db, args.merge_gap, args.abs_floor_db) if len(y) else []

    print("• Calcul des événements annotés …")
    events  = compute_events(segs, total, args.turn_min, args.backch_max, args.backch_ovl,
                             args.turn_max_pause)
    summary = compute_summary(segs, events, total)

    # Affichage console du résumé
    print("\n══ Résumé métriques ══════════════════════════════════")
    for role in ROLES:
        print(f"  {role:15s} : {summary.get(role, '')}")
    print(f"  {'[groupe]':15s} : {summary.get('_group', '')}")
    print("═════════════════════════════════════════════════════\n")

    print("• Construction de la figure …")
    fig, ax_cursors = build_figure(tracks, sr, events, summary, gdir.name)

    # Mode sauvegarde seule
    if args.save:
        out_path = Path(args.save)
        fig.savefig(out_path, dpi=150)
        print(f"✅ Figure enregistrée : {out_path}")
        plt.close(fig)
        return

    # Mode lecture interactive
    if args.no_audio:
        print("• Mode sans audio — fermer la fenêtre pour quitter.")
        plt.show()
        return

    try:
        import sounddevice as sd
    except ImportError:
        print("⚠ sounddevice non disponible — affichage sans audio.")
        plt.show()
        return

    print("• Lecture synchronisée — fermer la fenêtre pour quitter …")
    mix = make_stereo_mix(tracks, sr)

    def audio_thread() -> None:
        sd.play(mix, sr)
        sd.wait()

    th = threading.Thread(target=audio_thread, daemon=True)
    th.start()

    start_t = time.time()
    while th.is_alive():
        t_cur = time.time() - start_t
        for cursor in ax_cursors.values():
            cursor.set_xdata([t_cur])
        try:
            plt.pause(0.033)  # ~30 fps
        except Exception:
            break

    plt.close(fig)


if __name__ == "__main__":
    main()
