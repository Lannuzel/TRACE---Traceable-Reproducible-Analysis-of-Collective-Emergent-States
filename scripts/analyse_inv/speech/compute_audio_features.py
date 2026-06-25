#!/usr/bin/env python3
"""
Compute higher-level audio features from metrics.csv (output of analyze_audio.py).

- Keeps analyze_audio.py intact (metrics.csv = low-level measures).
- Produces audio_features.csv = theory-driven features (TMS proxies, balance, coordination, etc.)
- Propagates interruption diagnostics (`n_attempted_interruptions`,
  `n_successful_interruptions`, `successful_interruption_ratio`) when available.
- `interruptions_total` / `interruptions_rate_per_min` are defined from the
  canonical overlap-based interruption construct, not from fast floor changes.
- `rapid_floor_takeover_*` remains a distinct construct and is kept separate.

Usage:
    python compute_audio_features.py metrics.csv --out audio_features.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.constants import ROLES
from common.stats import safe_div, coef_var, shannon_entropy

def row_sum_by_prefix(row: pd.Series, prefix: str) -> float:
    """Somme les colonnes commençant par `prefix` sur une ligne."""
    cols = [c for c in row.index if c.startswith(prefix)]
    if not cols:
        return np.nan
    vals = pd.to_numeric(row[cols], errors="coerce").fillna(0.0).values
    return float(vals.sum())


def participation_entropy_from_row(row: pd.Series, speech_cols: list[str]) -> float:
    """
    Entropie de participation à partir des temps de parole bruts.

    On utilise ici les `speech_*_s` plutôt que les ratios. La fonction
    `shannon_entropy` renormalise en interne les contributions positives, donc
    la somme > 1 des `speech_*_ratio` en présence d'overlap ne pose pas de
    problème conceptuel si l'on part des durées brutes.
    """
    vals = pd.to_numeric(row[speech_cols], errors="coerce").values.astype(float)
    return shannon_entropy(vals)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Standardisation des clés d'identification (group_id / scenario canoniques)
    if "group_id" not in out.columns and "group" in out.columns:
        out["group_id"] = out["group"]
    if "scenario" not in out.columns and "session" in out.columns:
        out["scenario"] = out["session"]

    # --- basic aggregates ---
    out["total_turns"] = 0
    out["total_speech_s"] = 0.0
    out["max_speech_ratio"] = np.nan
    out["audio_turn_balance_cv"] = np.nan
    out["speech_balance_cv"] = np.nan
    out["audio_participation_entropy"] = np.nan
    out["audio_distrib_speech"] = np.nan
    out["interruptions_total"] = np.nan
    out["interruptions_rate_per_min"] = np.nan
    out["rapid_floor_takeovers_total"] = np.nan
    out["legacy_int_total"] = np.nan
    out["n_backchannels"] = np.nan
    out["backchannel_rate_per_min"] = np.nan
    out["duration_s"] = pd.to_numeric(out.get("duration_s", np.nan), errors="coerce")
    out["n_attempted_interruptions"] = pd.to_numeric(
        out.get("n_attempted_interruptions", np.nan),
        errors="coerce",
    )
    out["n_successful_interruptions"] = pd.to_numeric(
        out.get("n_successful_interruptions", np.nan),
        errors="coerce",
    )
    out["successful_interruption_ratio"] = pd.to_numeric(
        out.get("successful_interruption_ratio", np.nan),
        errors="coerce",
    )

    # Gather per-role arrays row-wise
    turn_cols = [f"turns_{r}_n" for r in ROLES]
    speech_cols = [f"speech_{r}_s" for r in ROLES]
    speech_ratio_cols = [f"speech_{r}_ratio" for r in ROLES]

    # Make sure numeric
    for c in turn_cols + speech_cols + speech_ratio_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # total turns / speech
    out["total_turns"] = out[turn_cols].sum(axis=1, numeric_only=True)
    out["total_speech_s"] = out[speech_cols].sum(axis=1, numeric_only=True)

    # balance / entropy (3-person generalizations)
    out["audio_turn_balance_cv"] = out[turn_cols].apply(lambda r: coef_var(r.values), axis=1)
    # Le CV est invariant à un facteur d'échelle commun : utiliser les durées
    # de parole brutes est donc plus lisible que d'utiliser les ratios.
    out["speech_balance_cv"] = out[speech_cols].apply(lambda r: coef_var(r.values), axis=1)
    out["audio_participation_entropy"] = out.apply(
        lambda r: participation_entropy_from_row(r, speech_cols),
        axis=1,
    )
    out["max_speech_ratio"] = out[speech_ratio_cols].max(axis=1)

    # Variance de la distribution de parole (mesure d'inégalité, Woolley et al. 2010)
    # Var(p) = (1/N) * sum((p_i - 1/N)^2), p_i = s_i / sum(s_j), N=3
    # Min = 0 (égalité parfaite), Max = 2/9 ≈ 0.222 (un seul locuteur)
    def _distrib_speech_var(row: pd.Series) -> float:
        vals = pd.to_numeric(row[speech_cols], errors="coerce").values.astype(float)
        total = float(np.nansum(vals))
        if total <= 0 or np.isnan(total):
            return float("nan")
        p = vals / total
        p_bar = 1.0 / len(p)
        return float(np.mean((p - p_bar) ** 2))

    out["audio_distrib_speech"] = out.apply(_distrib_speech_var, axis=1)

    # Prises de tour rapides: deja agregees dans analyze_audio.py
    if "rapid_floor_takeovers_total" in out.columns:
        out["rapid_floor_takeovers_total"] = pd.to_numeric(out["rapid_floor_takeovers_total"], errors="coerce")

    # Backchannels: on les agrège explicitement ici pour que la couche HLF
    # puisse disposer d'une métrique canonique déjà calculée. Si les colonnes
    # source sont absentes, la colonne reste NaN et HLF garde son fallback.
    backchannel_cols = [f"backchannels_{r}_n" for r in ROLES if f"backchannels_{r}_n" in out.columns]
    if backchannel_cols:
        out[backchannel_cols] = out[backchannel_cols].apply(pd.to_numeric, errors="coerce")
        out["n_backchannels"] = out[backchannel_cols].sum(axis=1, min_count=1)
        out["backchannel_rate_per_min"] = out.apply(
            lambda r: safe_div(r["n_backchannels"], safe_div(r.get("duration_s", np.nan), 60.0)),
            axis=1,
        )

    # Compatibilité legacy: si un ancien CSV contient encore int_*, on les garde
    # seulement comme audit local et non comme définition canonique d'interruption.
    legacy_int_cols = [c for c in out.columns if c.startswith("int_")]
    if legacy_int_cols:
        out["legacy_int_total"] = out.apply(lambda r: row_sum_by_prefix(r, "int_"), axis=1)

    # Interruptions canoniques: overlap-based.
    directional_attempt_cols = [c for c in out.columns if c.startswith("interrupt_attempt_")]
    directional_success_cols = [c for c in out.columns if c.startswith("interrupt_success_")]
    if directional_attempt_cols:
        computed_attempts = out.apply(lambda r: row_sum_by_prefix(r, "interrupt_attempt_"), axis=1)
        out["n_attempted_interruptions"] = out["n_attempted_interruptions"].fillna(computed_attempts)
    if directional_success_cols:
        computed_successes = out.apply(lambda r: row_sum_by_prefix(r, "interrupt_success_"), axis=1)
        out["n_successful_interruptions"] = out["n_successful_interruptions"].fillna(computed_successes)

    out["interruptions_rate_per_min"] = out.apply(
        lambda r: safe_div(r["n_attempted_interruptions"], safe_div(r.get("duration_s", np.nan), 60.0)),
        axis=1
    )

    # interruption success ratio
    computed_success_ratio = out.apply(
        lambda r: safe_div(r["n_successful_interruptions"], r["n_attempted_interruptions"]),
        axis=1,
    )
    out["successful_interruption_ratio"] = out["successful_interruption_ratio"].fillna(computed_success_ratio)

    # Nettoyage final : supprimer les colonnes obsolètes / pairwise directionnelles
    cols_to_drop = []
    # Alias group/session redondants avec group_id/scenario
    for c in ["group", "session"]:
        if c in out.columns:
            cols_to_drop.append(c)
    # interruptions_total remplacé par n_attempted_interruptions
    if "interruptions_total" in out.columns:
        cols_to_drop.append("interruptions_total")
    # Colonnes _old legacy
    cols_to_drop += [c for c in out.columns if c.endswith("_old")]
    # Colonnes pairwise directionnelles (→)
    cols_to_drop += [c for c in out.columns if "→" in c]
    # Colonnes tms_*
    cols_to_drop += [c for c in out.columns if c.startswith("tms_")]
    # Colonnes floor_exchange_pause_mean_s_old / n_floor_exchanges_old / overlap_ratio_old
    cols_to_drop += [c for c in out.columns if c in (
        "floor_exchange_pause_mean_s_old", "n_floor_exchanges_old", "overlap_ratio_old"
    )]
    if cols_to_drop:
        out = out.drop(columns=list(dict.fromkeys(cols_to_drop)), errors="ignore")

    # Alias audio_* pour les variables MOD-1/2/3 (utilisés par behavioral_indices et hlf)
    if "mean_turn_s" in out.columns:
        out["audio_avg_speaking_turn_duration_s"] = out["mean_turn_s"]
    if "total_turns" in out.columns:
        out["audio_total_speaking_turns"] = out["total_turns"]
    if "floor_exchange_pause_mean_s" in out.columns:
        out["audio_floor_exchange_pause_mean_s"] = out["floor_exchange_pause_mean_s"]
    if "backchannel_rate_per_min" in out.columns:
        out["audio_backchannel_rate_per_min"] = out["backchannel_rate_per_min"]
    if "pause_ratio" in out.columns:
        out["audio_pause_ratio"] = out["pause_ratio"]

    # total_turns_old (conservé pour l'audit MOD-1 uniquement)
    turn_cols_old = [f"turns_{r}_n_old" for r in ROLES if f"turns_{r}_n_old" in out.columns]
    if turn_cols_old:
        out["total_turns_old"] = out[[c for c in turn_cols_old]].sum(axis=1, numeric_only=True)

    # Audit MOD-1 (B-MOD1-a) : log les groupes ou total_turns_CA > total_turns_IPU.
    # Ce cas est attendu quand des micro-segments (< turn_min individuellement) fusionnent
    # en tours CA valides. Ce n'est pas une regression — c'est le comportement correct
    # de l'agregation CA. On log pour transparence, on n'arrete pas le pipeline.
    if "total_turns_old" in out.columns and "total_turns" in out.columns:
        for idx, row in out.iterrows():
            nv = row.get("total_turns", np.nan)
            ov = row.get("total_turns_old", np.nan)
            if pd.notna(nv) and pd.notna(ov) and nv > ov + 0.5:
                import logging; logging.debug(
                    f"[AUDIT MOD-1] total_turns_CA ({int(nv)}) > total_turns_IPU ({int(ov)}) "
                    f"for group {row.get('group_id', idx)} "
                    f"-- micro-segments fusionnes en tours CA valides (comportement attendu)."
                )

    # --- optional: z-score within condition/scenario to remove scale effects ---
    # (useful if VR vs PC differ a lot)
    group_keys = [k for k in ["condition", "scenario", "timepoint"] if k in out.columns]
    if not group_keys:
        group_keys = [k for k in ["condition", "session", "timepoint"] if k in out.columns]
    z_cols = [
        "total_turns", "mean_turn_s", "overlap_ratio", "pause_ratio", "audio_pause_ratio",
        "interruptions_rate_per_min", "audio_turn_balance_cv", "speech_balance_cv",
        "audio_participation_entropy", "max_speech_ratio"
    ]
    for c in z_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if group_keys:
                out[f"z_{c}"] = out.groupby(group_keys)[c].transform(
                    lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) not in (0, np.nan) else np.nan)
                )
            else:
                s = out[c]
                out[f"z_{c}"] = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else np.nan)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", help="Path to metrics.csv produced by analyze_audio.py")
    ap.add_argument("--out", default="audio_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out = build_features(df)
    out.to_csv(args.out, index=False)
    print(f"✅ Wrote: {args.out} ({len(out)} rows)")

if __name__ == "__main__":
    main()
