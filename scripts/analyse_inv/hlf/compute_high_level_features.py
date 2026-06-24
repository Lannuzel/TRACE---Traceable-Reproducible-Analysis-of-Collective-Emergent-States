#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_high_level_features_v5.py

Compute final multimodal features for collaboration / cohesion / TMS modelling
from already-computed modality-level metrics.

Audio input: `audio_features.csv` (produit par analyze_audio.py).

Main additions compared with v4:
- Adds group_base_id (e.g., bim066_2 -> bim066)
- Excludes bim065_2 but keeps bim065
- Produces:
    1) compact final dataset
    2) full audit dataset
    3) missingness / availability summary dataset
- Keeps provenance columns (*_source)
- Keeps exploratory composites for backward compatibility
"""

from __future__ import annotations

import argparse
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---- common package imports ------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.metadata import (
    add_group_base_id, exclude_invalid_groups, ensure_key_cols,
)
from common.io_utils import normalize_columns, read_csv_smart, safe_numeric
from common.stats import (
    clip01, ratio, coalesce, first_valid_series, first_valid_source,
    nanmean_rows, zscore_series, zscore_df,
)


# ----------------------------
# Local helper (not in common)
# ----------------------------

def _pick_first_existing(df: pd.DataFrame, cands: List[str], out: str) -> pd.DataFrame:
    df = df.copy()
    for c in cands:
        if c in df.columns:
            df[out] = df[c]
            return df
    df[out] = np.nan
    return df


def _read_face_csv_robust(path: str) -> pd.DataFrame:
    """
    Lecture tolérante du CSV face.

    Compatibilité visée :
    - nouveau format standard du pipeline (`,` + `.`)
    - anciens exports EU (`;` + `,`)
    """
    best_df: pd.DataFrame | None = None
    best_n_cols = -1
    for opts in (
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": ","},
    ):
        try:
            df = pd.read_csv(path, engine="python", **opts)
        except Exception:
            continue
        if df.shape[1] > best_n_cols:
            best_df = df
            best_n_cols = df.shape[1]

    if best_df is None:
        best_df = read_csv_smart(path)
    return best_df


# ----------------------------
# Loaders
# ----------------------------

def load_face(path: str) -> pd.DataFrame:
    df = _read_face_csv_robust(path)
    df = normalize_columns(df)

    df = ensure_key_cols(
        df,
        group_col_candidates=["group_id", "group"],
        condition_col_candidates=["condition"],
        scenario_col_candidates=["scenario", "session"],
        timepoint_col_candidates=["timepoint", "tp", "Timepoint"],
    )

    numeric_cols = [
        "interaction_dur_s",
        "joy_tri_rate_per_min", "joy_tri_occupancy",
        "sad_tri_rate_per_min", "sad_tri_occupancy",
        "joy_sync_calculateur_modelisateur_jaccard",
        "joy_sync_calculateur_lecteur_jaccard",
        "joy_sync_modelisateur_lecteur_jaccard",
        "sad_sync_calculateur_modelisateur_jaccard",
        "sad_sync_calculateur_lecteur_jaccard",
        "sad_sync_modelisateur_lecteur_jaccard",
        "joy_mean_mean", "sad_mean_mean",
        "joy_active_pct_mean", "sad_active_pct_mean",
        "joy_mean_median", "sad_mean_median",
        "joy_active_pct_median", "sad_active_pct_median",
        "joy_mean_valid_ratio", "sad_mean_valid_ratio",
        "joy_active_pct_valid_ratio", "sad_active_pct_valid_ratio",
        "au6_active_pct_mean", "au12_active_pct_mean", "au15_active_pct_mean", "au17_active_pct_mean",
        "au6_au12_coactive_pct_mean",
        "au15_au17_coactive_pct_mean",
        "au_sync_mean", "au_sync_jaccard_mean", "au_sync_pearson_mean",
    ]
    df = safe_numeric(df, numeric_cols)

    joy_jacc_cols = [
        c for c in [
            "joy_sync_calculateur_modelisateur_jaccard",
            "joy_sync_calculateur_lecteur_jaccard",
            "joy_sync_modelisateur_lecteur_jaccard",
        ] if c in df.columns
    ]
    sad_jacc_cols = [
        c for c in [
            "sad_sync_calculateur_modelisateur_jaccard",
            "sad_sync_calculateur_lecteur_jaccard",
            "sad_sync_modelisateur_lecteur_jaccard",
        ] if c in df.columns
    ]

    df["joy_sync_jaccard_mean"] = df[joy_jacc_cols].mean(axis=1, skipna=True) if joy_jacc_cols else np.nan
    df["sad_sync_jaccard_mean"] = df[sad_jacc_cols].mean(axis=1, skipna=True) if sad_jacc_cols else np.nan

    pearson_joy = [c for c in df.columns if c.startswith("sync_pearson_") and c.endswith("_joy_mean")]
    pearson_sad = [c for c in df.columns if c.startswith("sync_pearson_") and c.endswith("_sad_mean")]
    pearson_joy_act = [c for c in df.columns if c.startswith("sync_pearson_") and c.endswith("_joy_active_pct")]
    pearson_sad_act = [c for c in df.columns if c.startswith("sync_pearson_") and c.endswith("_sad_active_pct")]

    df["joy_sync_pearson_mean"] = df[pearson_joy].mean(axis=1, skipna=True) if pearson_joy else np.nan
    df["sad_sync_pearson_mean"] = df[pearson_sad].mean(axis=1, skipna=True) if pearson_sad else np.nan
    df["joy_active_sync_pearson_mean"] = df[pearson_joy_act].mean(axis=1, skipna=True) if pearson_joy_act else np.nan
    df["sad_active_sync_pearson_mean"] = df[pearson_sad_act].mean(axis=1, skipna=True) if pearson_sad_act else np.nan

    if "joy_tri_occupancy" in df.columns:
        df["joy_tri_occupancy"] = clip01(df["joy_tri_occupancy"])
    if "sad_tri_occupancy" in df.columns:
        df["sad_tri_occupancy"] = clip01(df["sad_tri_occupancy"])

    if "joy_tri_occupancy" in df.columns and "sad_tri_occupancy" in df.columns:
        df["affect_balance_occ"] = df["joy_tri_occupancy"] - df["sad_tri_occupancy"]
        df["pos_neg_occ_ratio"] = ratio(df["joy_tri_occupancy"], df["sad_tri_occupancy"])

    if "joy_tri_rate_per_min" in df.columns and "sad_tri_rate_per_min" in df.columns:
        df["affect_balance_rate"] = df["joy_tri_rate_per_min"] - df["sad_tri_rate_per_min"]
        df["pos_neg_rate_ratio"] = ratio(df["joy_tri_rate_per_min"], df["sad_tri_rate_per_min"])

    if "joy_sync_jaccard_mean" in df.columns and "sad_sync_jaccard_mean" in df.columns:
        df["affect_sync_jaccard_contrast"] = df["joy_sync_jaccard_mean"] - df["sad_sync_jaccard_mean"]
    if "joy_sync_pearson_mean" in df.columns and "sad_sync_pearson_mean" in df.columns:
        df["affect_sync_pearson_contrast"] = df["joy_sync_pearson_mean"] - df["sad_sync_pearson_mean"]

    if "sad_mean_mean" in df.columns:
        df["face_sad_intensity_mean"] = df["sad_mean_mean"]

    if "sad_tri_rate_per_min" in df.columns:
        df["sad_tri_rate_sqrt"] = np.sqrt(
            pd.to_numeric(df["sad_tri_rate_per_min"], errors="coerce").clip(lower=0)
        )

    keep = [
        "group_id", "condition", "scenario", "timepoint",
        "interaction_dur_s",
        "joy_tri_rate_per_min", "joy_tri_occupancy",
        "sad_tri_rate_per_min", "sad_tri_occupancy",
        "joy_sync_jaccard_mean", "sad_sync_jaccard_mean",
        "joy_sync_pearson_mean", "sad_sync_pearson_mean",
        "joy_active_sync_pearson_mean", "sad_active_sync_pearson_mean",
        "joy_mean_mean", "sad_mean_mean", "face_sad_intensity_mean",
        "joy_active_pct_mean", "sad_active_pct_mean",
        "joy_mean_median", "sad_mean_median",
        "joy_active_pct_median", "sad_active_pct_median",
        "joy_mean_valid_ratio", "sad_mean_valid_ratio",
        "joy_active_pct_valid_ratio", "sad_active_pct_valid_ratio",
        "affect_balance_occ", "pos_neg_occ_ratio",
        "affect_balance_rate", "pos_neg_rate_ratio",
        "affect_sync_jaccard_contrast", "affect_sync_pearson_contrast",
        "au6_active_pct_mean", "au12_active_pct_mean", "au15_active_pct_mean", "au17_active_pct_mean",
        "au6_au12_coactive_pct_mean", "au15_au17_coactive_pct_mean",
        "au_sync_mean", "au_sync_jaccard_mean", "au_sync_pearson_mean",
        "sad_tri_rate_sqrt",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["group_id", "condition", "scenario", "timepoint"])
    df = add_group_base_id(df)
    df = exclude_invalid_groups(df)
    return df


def load_gaze_group(path: str) -> pd.DataFrame:
    df = read_csv_smart(path, sep=",")
    df = normalize_columns(df)

    df = ensure_key_cols(
        df,
        group_col_candidates=["group_id", "group", "Group", "groupId", "groupID"],
        condition_col_candidates=["condition", "Condition"],
        scenario_col_candidates=["scenario", "session", "Session"],
        timepoint_col_candidates=["timepoint", "tp", "Timepoint"],
    )

    df = _pick_first_existing(df, ["interaction_duration_s", "interaction_dur_s", "duration_s"], "interaction_dur_s")

    numeric_cols = [
        "interaction_dur_s",
        # Nouvelle analyse directionnelle (gaze_directional/)
        "gaze_convergence_ratio", "gaze_convergence_n_episodes", "gaze_convergence_dur_total_s",
        "gaze_convergence_mean_angle_deg", "gaze_convergence_n_episodes_per_s",
        "mutual_gaze_ratio", "mutual_gaze_n_episodes", "mutual_gaze_dur_total_s",
        "mutual_gaze_dur_mean_s", "mutual_gaze_n_episodes_per_s",
        "gaze_entropy_dir_mean",
        # Aliases legacy (ancienne analyse par objet) — fallback si CSV ancien fourni
        "shared_obj_ratio", "shared_obj_n_episodes", "shared_obj_dur_total_s", "shared_obj_dur_mean_s",
        "shared_obj_dur_median_s", "shared_obj_dur_q25_s", "shared_obj_dur_q75_s", "shared_obj_dur_iqr_s",
        "shared_obj_n_episodes_per_s", "shared_obj_dur_total_ratio",
        "mutual_gaze_ratio_mean_pairs",
        "mutual_gaze_n_episodes_sum_pairs", "mutual_gaze_dur_total_s_sum_pairs", "mutual_gaze_dur_mean_s_mean_pairs",
        "mutual_gaze_dur_total_ratio",
        "gaze_entropy_mean_participants",
        "gaze_to_speaker_ratio",       # absent de la nouvelle analyse
        "transition_prob_gaze_to_speech",  # absent de la nouvelle analyse
        "group_shared_obj_pct_mean",
        "group_shared_episode_median_s",
    ]
    df = safe_numeric(df, numeric_cols)

    # Aliases legacy → nouvelles colonnes (si CSV ancien fourni à la place du nouveau)
    if "gaze_convergence_ratio" not in df.columns and "shared_obj_ratio" in df.columns:
        df["gaze_convergence_ratio"] = pd.to_numeric(df["shared_obj_ratio"], errors="coerce")
    if "gaze_convergence_n_episodes" not in df.columns and "shared_obj_n_episodes" in df.columns:
        df["gaze_convergence_n_episodes"] = pd.to_numeric(df["shared_obj_n_episodes"], errors="coerce")
    if "gaze_convergence_dur_total_s" not in df.columns and "shared_obj_dur_total_s" in df.columns:
        df["gaze_convergence_dur_total_s"] = pd.to_numeric(df["shared_obj_dur_total_s"], errors="coerce")
    if "mutual_gaze_ratio" not in df.columns and "mutual_gaze_ratio_mean_pairs" in df.columns:
        df["mutual_gaze_ratio"] = pd.to_numeric(df["mutual_gaze_ratio_mean_pairs"], errors="coerce")
    if "mutual_gaze_n_episodes" not in df.columns and "mutual_gaze_n_episodes_sum_pairs" in df.columns:
        df["mutual_gaze_n_episodes"] = pd.to_numeric(df["mutual_gaze_n_episodes_sum_pairs"], errors="coerce")
    if "mutual_gaze_dur_total_s" not in df.columns and "mutual_gaze_dur_total_s_sum_pairs" in df.columns:
        df["mutual_gaze_dur_total_s"] = pd.to_numeric(df["mutual_gaze_dur_total_s_sum_pairs"], errors="coerce")
    if "gaze_entropy_dir_mean" not in df.columns and "gaze_entropy_mean_participants" in df.columns:
        df["gaze_entropy_dir_mean"] = pd.to_numeric(df["gaze_entropy_mean_participants"], errors="coerce")

    # Alias durée moyenne convergence (ancienne = shared_obj_dur_mean_s)
    if "shared_obj_ratio" not in df.columns and "group_shared_obj_pct_mean" in df.columns:
        df["gaze_convergence_ratio"] = pd.to_numeric(df["group_shared_obj_pct_mean"], errors="coerce")
    if "gaze_convergence_dur_total_s" in df.columns:
        df["gaze_convergence_episode_dur_mean_s"] = df["gaze_convergence_dur_total_s"]
        df["log_gaze_convergence_episode_dur_mean_s"] = np.log1p(
            pd.to_numeric(df["gaze_convergence_dur_total_s"], errors="coerce").clip(lower=0)
        )

    if "gaze_entropy_dir_mean" in df.columns:
        df["gaze_focus_proxy"] = 1.0 - pd.to_numeric(df["gaze_entropy_dir_mean"], errors="coerce")

    keep = [
        "group_id", "condition", "scenario", "timepoint",
        "interaction_dur_s",
        # Colonnes directionnelles (nouvelles)
        "gaze_convergence_ratio", "gaze_convergence_n_episodes", "gaze_convergence_dur_total_s",
        "gaze_convergence_mean_angle_deg", "gaze_convergence_n_episodes_per_s",
        "gaze_convergence_episode_dur_mean_s", "log_gaze_convergence_episode_dur_mean_s",
        "mutual_gaze_ratio", "mutual_gaze_n_episodes", "mutual_gaze_dur_total_s",
        "mutual_gaze_dur_mean_s", "mutual_gaze_n_episodes_per_s",
        "gaze_entropy_dir_mean",
        "gaze_focus_proxy",
        # Colonnes legacy conservées pour traçabilité (si présentes)
        "shared_obj_ratio", "shared_obj_n_episodes", "shared_obj_dur_total_s",
        "mutual_gaze_ratio_mean_pairs", "gaze_entropy_mean_participants",
        "gaze_to_speaker_ratio", "transition_prob_gaze_to_speech",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].drop_duplicates(subset=["group_id", "condition", "scenario", "timepoint"])
    df = add_group_base_id(df)
    df = exclude_invalid_groups(df)
    return df


def load_gaze_pair(path: str) -> pd.DataFrame:
    df = read_csv_smart(path, sep=",")
    df = normalize_columns(df)

    df = ensure_key_cols(
        df,
        group_col_candidates=["group_id", "group", "Group"],
        condition_col_candidates=["condition", "Condition"],
        scenario_col_candidates=["scenario", "session", "Session"],
        timepoint_col_candidates=["timepoint", "tp", "Timepoint"],
    )

    numeric_cols = [
        # Colonnes directionnelles (nouvelles)
        "pair_convergence_ratio", "pair_convergence_n_episodes", "pair_convergence_dur_total_s",
        "pair_convergence_mean_angle_deg",
        "pair_mutual_gaze_ratio", "pair_mutual_gaze_n_episodes",
        "pair_mutual_gaze_dur_total_s", "pair_mutual_gaze_dur_mean_s",
        # Legacy (ancienne analyse par objet) — fallback
        "pair_shared_obj_pct_mean", "pair_shared_episode_median_s",
        "pair_shared_obj_ratio", "pair_shared_obj_n_episodes",
        "pair_shared_obj_dur_total_s", "pair_shared_obj_dur_mean_s",
        "pair_shared_obj_n_episodes_per_s", "pair_shared_obj_dur_total_ratio",
        "pair_mutual_gaze_n_episodes_per_s", "pair_mutual_gaze_dur_total_ratio",
    ]
    df = safe_numeric(df, numeric_cols)

    # Aliases legacy → nouvelles colonnes paires
    if "pair_convergence_ratio" not in df.columns and "pair_shared_obj_ratio" in df.columns:
        df["pair_convergence_ratio"] = pd.to_numeric(df["pair_shared_obj_ratio"], errors="coerce")
    if "pair_convergence_ratio" not in df.columns and "pair_shared_obj_pct_mean" in df.columns:
        df["pair_convergence_ratio"] = pd.to_numeric(df["pair_shared_obj_pct_mean"], errors="coerce")
    if "pair_shared_obj_dur_mean_s" not in df.columns and "pair_shared_episode_median_s" in df.columns:
        df["pair_shared_obj_dur_mean_s"] = pd.to_numeric(df["pair_shared_episode_median_s"], errors="coerce")

    # Agrégation conditionnelle : seulement les colonnes présentes dans le CSV
    agg_spec: dict[str, tuple[str, str]] = {}
    for out_col, src_col in [
        ("pair_convergence_ratio_mean",       "pair_convergence_ratio"),
        ("pair_convergence_dur_total_s_mean", "pair_convergence_dur_total_s"),
        ("pair_mutual_gaze_ratio_mean",       "pair_mutual_gaze_ratio"),
        ("pair_mutual_gaze_dur_mean_s_mean",  "pair_mutual_gaze_dur_mean_s"),
        # Legacy (absentes du nouveau CSV, présentes dans l'ancien)
        ("pair_shared_obj_ratio_mean",        "pair_shared_obj_ratio"),
        ("pair_shared_obj_dur_mean_s_mean",   "pair_shared_obj_dur_mean_s"),
    ]:
        if src_col in df.columns:
            agg_spec[out_col] = (src_col, "mean")

    grp = df.groupby(
        ["group_id", "condition", "scenario", "timepoint"], dropna=False
    ).agg(**agg_spec).reset_index()

    grp = add_group_base_id(grp)
    grp = exclude_invalid_groups(grp)
    return grp


def load_speech(path: str) -> pd.DataFrame:
    df = read_csv_smart(path, sep=",")
    df = normalize_columns(df)

    df = ensure_key_cols(
        df,
        group_col_candidates=["group_id", "group"],
        condition_col_candidates=["condition"],
        scenario_col_candidates=["scenario", "session"],
        timepoint_col_candidates=["timepoint", "tp", "Timepoint"],
    )

    numeric_cols = [
        "duration_s",
        "overlap_ratio",
        "pairwise_overlap_s",
        "pairwise_overlap_ratio",
        "pause_ratio",
        "pause_mean_s",
        "mean_pause_s",           # alias audio_features.csv (≡ pause_mean_s)
        "floor_exchange_pause_mean_s",
        "n_floor_exchanges",
        "mean_turn_s",
        "total_turns",
        "rapid_floor_takeovers_total",
        "interruptions_rate_per_min",
        "successful_interruption_ratio",
        "n_successful_interruptions",
        "n_attempted_interruptions",
        "backchannel_rate_per_min",
        "n_backchannels",
        "audio_participation_entropy",
        "audio_turn_balance_cv",
        "audio_pause_ratio",
        "max_speech_ratio",
        "speech_balance_cv",
    ]
    df = safe_numeric(df, numeric_cols)

    # Calculer n_backchannels depuis les totaux par rôle si la colonne directe est absente.
    # audio_features.csv fournit backchannels_{role}_n pour chaque rôle de la triade.
    per_role_bc_cols = [c for c in df.columns if c.startswith("backchannels_") and c.endswith("_n")]
    if "n_backchannels" not in df.columns and per_role_bc_cols:
        bc_numeric = [pd.to_numeric(df[c], errors="coerce") for c in per_role_bc_cols]
        df["n_backchannels"] = sum(bc_numeric)

    keep = (
        ["group_id", "condition", "scenario", "timepoint"]
        + [c for c in numeric_cols if c in df.columns]
        + (["n_backchannels"] if "n_backchannels" in df.columns else [])
    )
    # Dédupliquer keep tout en conservant l'ordre
    seen: set = set()
    keep = [c for c in keep if c not in seen and not seen.add(c)]  # type: ignore[func-returns-value]
    df = df[keep].drop_duplicates(subset=["group_id", "condition", "scenario", "timepoint"])
    df = add_group_base_id(df)
    df = exclude_invalid_groups(df)
    return df


# ----------------------------
# Feature engineering
# ----------------------------

@dataclass
class Weights:
    cred_total_turns: float = -0.33
    cred_overlap_ratio: float = -0.20
    cred_interruptions_rate: float = -0.20

    coord_total_turns: float = -0.37
    coord_mean_turn_s: float = +0.25
    coord_overlap_ratio: float = -0.25
    coord_interruptions_rate: float = -0.30
    coord_pause_ratio: float = -0.15

    spec_mean_turn_s: float = +0.57
    spec_participation_entropy: float = -0.25
    spec_max_speech_ratio: float = +0.20
    spec_speech_balance_cv: float = +0.20

    gaze_shared_obj: float = +0.20
    gaze_mutual_gaze: float = +0.10
    gaze_to_speaker: float = +0.10
    gaze_transition_g2s: float = +0.10
    gaze_entropy_penalty: float = -0.10

    face_joy_sync: float = +0.10
    face_joy_rate: float = +0.10

    face_sad_occ_penalty: float = -0.10
    face_sad_sync_penalty: float = -0.05


def _is_vr(df: pd.DataFrame) -> pd.Series:
    if "condition" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["condition"].astype(str).str.upper().eq("VR")


def _force_pc_gaze_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        return df
    is_pc = df["condition"].astype(str).str.upper().eq("PC")
    if not np.any(is_pc):
        return df

    gaze_cols = [
        c for c in df.columns
        if (
            c.startswith("gaze_convergence_")
            or c.startswith("gaze_entropy_dir")
            or c.startswith("gaze_focus_")
            or c.startswith("gaze_mutual_")
            or c.startswith("mutual_gaze_")
            or c.startswith("gaze_")
            or c.startswith("transition_prob")
            or c.startswith("pair_convergence_")
            or c.startswith("pair_mutual_")
            # Legacy
            or c.startswith("shared_obj_")
            or c.startswith("pair_shared_")
        )
    ]
    if gaze_cols:
        df.loc[is_pc, gaze_cols] = np.nan
    return df


def _apply_reference_duration_and_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "duration_s" in df.columns:
        dur_ref = pd.to_numeric(df["duration_s"], errors="coerce")
        # Vérifier la cohérence avec interaction_dur_s (gaze) si disponible
        if "interaction_dur_s" in df.columns:
            dur_gaze = pd.to_numeric(df["interaction_dur_s"], errors="coerce")
            divergence = (dur_ref - dur_gaze).abs()
            bad = divergence[divergence > 5.0].dropna()
            if not bad.empty:
                logging.warning(
                    f"[DURÉE] Divergence audio/gaze > 5s pour {len(bad)} groupes "
                    f"(max={divergence.max():.1f}s). "
                    "Vérifier que audio duration_s et gaze interaction_dur_s "
                    "couvrent bien la même fenêtre temporelle."
                )
    else:
        dur_ref = coalesce(
            df.get("interaction_dur_s", None),
            df.get("interaction_duration_s", None),
        )

    df["interaction_dur_s_ref"] = dur_ref
    dur_min = df["interaction_dur_s_ref"] / 60.0

    # Convergence directionnelle (nouvelle analyse)
    if "gaze_convergence_n_episodes" in df.columns:
        n_ep = pd.to_numeric(df["gaze_convergence_n_episodes"], errors="coerce")
    elif "gaze_convergence_n_episodes_per_s" in df.columns:
        n_ep = pd.to_numeric(df["gaze_convergence_n_episodes_per_s"], errors="coerce") * df["interaction_dur_s_ref"]
    elif "shared_obj_n_episodes" in df.columns:  # fallback legacy
        n_ep = pd.to_numeric(df["shared_obj_n_episodes"], errors="coerce")
    elif "shared_obj_n_episodes_per_s" in df.columns:
        n_ep = pd.to_numeric(df["shared_obj_n_episodes_per_s"], errors="coerce") * df["interaction_dur_s_ref"]
    else:
        n_ep = None

    if n_ep is not None:
        df["gaze_convergence_episode_rate_per_min_ref"] = n_ep / (dur_min + 1e-6)
        # Alias legacy pour compatibilité avec inv_features_config
        df["shared_obj_episode_rate_per_min_ref"] = df["gaze_convergence_episode_rate_per_min_ref"]

    if "gaze_convergence_dur_total_s" in df.columns:
        tot_s = pd.to_numeric(df["gaze_convergence_dur_total_s"], errors="coerce")
        df["gaze_convergence_dur_total_ratio_ref"] = tot_s / (df["interaction_dur_s_ref"] + 1e-6)
        df["shared_obj_dur_total_ratio_ref"] = df["gaze_convergence_dur_total_ratio_ref"]
    elif "shared_obj_dur_total_s" in df.columns:  # fallback legacy
        tot_s = pd.to_numeric(df["shared_obj_dur_total_s"], errors="coerce")
        df["gaze_convergence_dur_total_ratio_ref"] = tot_s / (df["interaction_dur_s_ref"] + 1e-6)
        df["shared_obj_dur_total_ratio_ref"] = df["gaze_convergence_dur_total_ratio_ref"]
    elif "shared_obj_dur_total_ratio" in df.columns:
        df["shared_obj_dur_total_ratio_ref"] = pd.to_numeric(df["shared_obj_dur_total_ratio"], errors="coerce")

    # Mutual gaze directionnel
    if "mutual_gaze_n_episodes" in df.columns:
        n_ep_m = pd.to_numeric(df["mutual_gaze_n_episodes"], errors="coerce")
    elif "mutual_gaze_n_episodes_sum_pairs" in df.columns:  # fallback legacy
        n_ep_m = pd.to_numeric(df["mutual_gaze_n_episodes_sum_pairs"], errors="coerce")
    elif "mutual_gaze_n_episodes_per_s" in df.columns:
        n_ep_m = pd.to_numeric(df["mutual_gaze_n_episodes_per_s"], errors="coerce") * df["interaction_dur_s_ref"]
    else:
        n_ep_m = None

    if n_ep_m is not None:
        df["mutual_gaze_episode_rate_per_min_ref"] = n_ep_m / (dur_min + 1e-6)

    if "mutual_gaze_dur_total_s" in df.columns:
        tot_s_m = pd.to_numeric(df["mutual_gaze_dur_total_s"], errors="coerce")
        df["mutual_gaze_dur_total_ratio_ref"] = tot_s_m / (df["interaction_dur_s_ref"] + 1e-6)
    elif "mutual_gaze_dur_total_s_sum_pairs" in df.columns:  # fallback legacy
        tot_s_m = pd.to_numeric(df["mutual_gaze_dur_total_s_sum_pairs"], errors="coerce")
        df["mutual_gaze_dur_total_ratio_ref"] = tot_s_m / (df["interaction_dur_s_ref"] + 1e-6)
    elif "mutual_gaze_dur_total_ratio" in df.columns:
        df["mutual_gaze_dur_total_ratio_ref"] = pd.to_numeric(df["mutual_gaze_dur_total_ratio"], errors="coerce")

    return df


def _zscore_speech_and_face(df: pd.DataFrame, cols: List[str], z_by: Optional[List[str]]) -> pd.DataFrame:
    return zscore_df(df, cols, by=z_by, prefix="z_")


def _zscore_gaze_only_vr(df: pd.DataFrame, cols: List[str], z_by: Optional[List[str]]) -> pd.DataFrame:
    df = df.copy()
    if not cols:
        return df

    vr_mask = _is_vr(df)
    if not np.any(vr_mask):
        return df

    tmp = df.loc[vr_mask].copy()
    tmp = zscore_df(tmp, cols, by=z_by, prefix="z_")

    for c in cols:
        zc = "z_" + c
        if zc in tmp.columns:
            df.loc[vr_mask, zc] = tmp[zc].values

    return df


def add_final_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gaze_mutual_gaze_ratio"] = first_valid_series(
        df,
        # Nouvelle analyse directionnelle en priorité, puis fallbacks legacy
        ["mutual_gaze_ratio", "mutual_gaze_ratio_mean_pairs", "pair_mutual_gaze_ratio_mean", "mutual_gaze_dur_total_ratio_ref"],
    )
    df["gaze_mutual_gaze_ratio_source"] = first_valid_source(
        df,
        ["mutual_gaze_ratio", "mutual_gaze_ratio_mean_pairs", "pair_mutual_gaze_ratio_mean", "mutual_gaze_dur_total_ratio_ref"],
    )

    # gaze_to_speaker_ratio absent de la nouvelle analyse directionnelle
    # df["gaze_to_speaker_ratio_final"] = first_valid_series(df, ["gaze_to_speaker_ratio"])
    # df["gaze_to_speaker_ratio_final_source"] = first_valid_source(df, ["gaze_to_speaker_ratio"])

    df["gaze_shared_visual_attention_ratio"] = first_valid_series(
        df,
        # Nouvelle analyse directionnelle en priorité, puis fallbacks legacy
        ["gaze_convergence_ratio", "pair_convergence_ratio_mean",
         "shared_obj_ratio", "shared_obj_dur_total_ratio_ref", "pair_shared_obj_ratio_mean"],
    )
    df["gaze_shared_visual_attention_ratio_source"] = first_valid_source(
        df,
        ["gaze_convergence_ratio", "pair_convergence_ratio_mean",
         "shared_obj_ratio", "shared_obj_dur_total_ratio_ref", "pair_shared_obj_ratio_mean"],
    )

    df["gaze_entropy"] = first_valid_series(
        df,
        ["gaze_entropy_dir_mean", "gaze_entropy_mean_participants"],
    )
    df["gaze_entropy_source"] = first_valid_source(
        df,
        ["gaze_entropy_dir_mean", "gaze_entropy_mean_participants"],
    )

    df["audio_total_speaking_turns"] = first_valid_series(df, ["total_turns"])
    df["audio_total_speaking_turns_source"] = first_valid_source(df, ["total_turns"])

    df["audio_avg_speaking_turn_duration_s"] = first_valid_series(df, ["mean_turn_s"])
    df["audio_avg_speaking_turn_duration_s_source"] = first_valid_source(df, ["mean_turn_s"])

    # Définition durcie : la pause d'échange de tour ne doit plus retomber sur
    # une pause moyenne de groupe (`mean_pause_s`) qui mesure un construit plus large.
    df["audio_floor_exchange_pause_mean_s"] = first_valid_series(
        df,
        ["floor_exchange_pause_mean_s"],
    )
    df["audio_floor_exchange_pause_mean_s_source"] = first_valid_source(
        df,
        ["floor_exchange_pause_mean_s"],
    )

    df["audio_overlap_speaking_ratio"] = first_valid_series(df, ["overlap_ratio"])
    df["audio_overlap_speaking_ratio_source"] = first_valid_source(df, ["overlap_ratio"])

    if "successful_interruption_ratio" in df.columns:
        df["audio_successful_interruption_ratio"] = pd.to_numeric(df["successful_interruption_ratio"], errors="coerce")
        df["audio_successful_interruption_ratio_source"] = np.where(
            pd.to_numeric(df["successful_interruption_ratio"], errors="coerce").notna(),
            "successful_interruption_ratio (overlap-based)",
            pd.NA,
        )
    elif {"n_successful_interruptions", "n_attempted_interruptions"}.issubset(df.columns):
        df["audio_successful_interruption_ratio"] = ratio(df["n_successful_interruptions"], df["n_attempted_interruptions"])
        df["audio_successful_interruption_ratio_source"] = np.where(
            df["audio_successful_interruption_ratio"].notna(),
            "n_successful_interruptions/n_attempted_interruptions (overlap-based)",
            pd.NA,
        )
    else:
        df["audio_successful_interruption_ratio"] = np.nan
        df["audio_successful_interruption_ratio_source"] = pd.NA

    if "backchannel_rate_per_min" in df.columns:
        df["audio_backchannel_rate_per_min"] = pd.to_numeric(df["backchannel_rate_per_min"], errors="coerce")
        df["audio_backchannel_rate_per_min_source"] = np.where(
            df["audio_backchannel_rate_per_min"].notna(),
            "backchannel_rate_per_min",
            pd.NA,
        )
    elif {"n_backchannels", "duration_s"}.issubset(df.columns):
        df["audio_backchannel_rate_per_min"] = pd.to_numeric(df["n_backchannels"], errors="coerce") / (
            pd.to_numeric(df["duration_s"], errors="coerce") / 60.0 + 1e-6
        )
        df["audio_backchannel_rate_per_min_source"] = np.where(
            df["audio_backchannel_rate_per_min"].notna(),
            "n_backchannels/duration_s",
            pd.NA,
        )
    else:
        df["audio_backchannel_rate_per_min"] = np.nan
        df["audio_backchannel_rate_per_min_source"] = pd.NA

    # Sourire de Duchenne : AU6 (orbicularis oculi) + AU12 (zygomaticus major).
    # Sourire sincere (Ekman, Davidson & Friesen 1990, JPSP). Exclut le sourire social (AU12 seul).
    if "au6_au12_coactive_pct_mean" in df.columns:
        df["face_smile_ratio"] = pd.to_numeric(df["au6_au12_coactive_pct_mean"], errors="coerce")
        df["face_smile_ratio_source"] = np.where(df["face_smile_ratio"].notna(), "au6_au12_coactive_pct_mean", pd.NA)
    elif {"au6_active_pct_mean", "au12_active_pct_mean"}.issubset(df.columns):
        df["face_smile_ratio"] = nanmean_rows(
            np.vstack([
                pd.to_numeric(df["au6_active_pct_mean"], errors="coerce").values,
                pd.to_numeric(df["au12_active_pct_mean"], errors="coerce").values,
            ])
        )
        df["face_smile_ratio_source"] = np.where(pd.Series(df["face_smile_ratio"]).notna(), "mean(au6_active_pct_mean,au12_active_pct_mean)", pd.NA)
    else:
        df["face_smile_ratio"] = first_valid_series(df, ["joy_tri_occupancy", "joy_active_pct_mean", "joy_active_pct_median"])
        df["face_smile_ratio_source"] = first_valid_source(df, ["joy_tri_occupancy", "joy_active_pct_mean", "joy_active_pct_median"])

    if "au15_au17_coactive_pct_mean" in df.columns:
        df["face_negative_affect_ratio"] = pd.to_numeric(df["au15_au17_coactive_pct_mean"], errors="coerce")
        df["face_negative_affect_ratio_source"] = np.where(df["face_negative_affect_ratio"].notna(), "au15_au17_coactive_pct_mean", pd.NA)
    elif {"au15_active_pct_mean", "au17_active_pct_mean"}.issubset(df.columns):
        df["face_negative_affect_ratio"] = nanmean_rows(
            np.vstack([
                pd.to_numeric(df["au15_active_pct_mean"], errors="coerce").values,
                pd.to_numeric(df["au17_active_pct_mean"], errors="coerce").values,
            ])
        )
        df["face_negative_affect_ratio_source"] = np.where(pd.Series(df["face_negative_affect_ratio"]).notna(), "mean(au15_active_pct_mean,au17_active_pct_mean)", pd.NA)
    else:
        df["face_negative_affect_ratio"] = first_valid_series(df, ["sad_tri_occupancy", "sad_active_pct_mean", "sad_active_pct_median"])
        df["face_negative_affect_ratio_source"] = first_valid_source(df, ["sad_tri_occupancy", "sad_active_pct_mean", "sad_active_pct_median"])

    # MOD-6 : renommage (Ekman & Friesen FACS 1978)
    # Operationnalisation AU15+AU17 = marqueur de tristesse, PAS affect negatif global.
    if "face_negative_affect_ratio" in df.columns:
        df["face_negative_affect_ratio_old"] = df["face_negative_affect_ratio"]
        df["face_sadness_marker_ratio"] = df["face_negative_affect_ratio"]

    # MOD-5 : Pearson uniquement (Hess & Fischer 2013, Emotional mimicry as social regulation)
    # Ancien fallback multi-source conserve en _old
    if "face_facial_synchrony" in df.columns:
        df["face_facial_synchrony_old"] = df["face_facial_synchrony"]
    if "au_sync_pearson_mean" in df.columns:
        df["face_facial_synchrony"] = pd.to_numeric(df["au_sync_pearson_mean"], errors="coerce")
        df["face_facial_synchrony_source"] = "au_sync_pearson_mean"
        df.loc[df["face_facial_synchrony"].isna(), "face_facial_synchrony_source"] = "missing"
    else:
        df["face_facial_synchrony"] = np.nan
        df["face_facial_synchrony_source"] = "missing"

    for c in [
        "gaze_mutual_gaze_ratio",
        "gaze_to_speaker_ratio_final",
        "gaze_shared_visual_attention_ratio",
        "audio_overlap_speaking_ratio",
        "audio_successful_interruption_ratio",
        "face_smile_ratio",
        "face_negative_affect_ratio",
    ]:
        if c in df.columns:
            df[c] = clip01(df[c])

    return df


def compute_composites(df: pd.DataFrame, z_by: Optional[List[str]] = None, w: Weights = Weights()) -> pd.DataFrame:
    df = df.copy()
    df = _force_pc_gaze_to_nan(df)
    df = _apply_reference_duration_and_rates(df)

    # gaze_joint_attention_idx_raw : convergence directionnelle + regard mutuel
    conv_col = next((c for c in ["gaze_convergence_ratio", "shared_obj_ratio"] if c in df.columns), None)
    mut_col  = next((c for c in ["mutual_gaze_ratio", "mutual_gaze_ratio_mean_pairs"] if c in df.columns), None)
    if conv_col and mut_col:
        df["gaze_joint_attention_idx_raw"] = (
            pd.to_numeric(df[conv_col], errors="coerce") +
            pd.to_numeric(df[mut_col], errors="coerce")
        ) / 2.0

    # gaze_speaker_coupling_idx_raw : absent de la nouvelle analyse — conservé si colonnes legacy présentes
    if "gaze_to_speaker_ratio" in df.columns and "transition_prob_gaze_to_speech" in df.columns:
        df["gaze_speaker_coupling_idx_raw"] = (
            pd.to_numeric(df["gaze_to_speaker_ratio"], errors="coerce") +
            pd.to_numeric(df["transition_prob_gaze_to_speech"], errors="coerce")
        ) / 2.0

    # Densité épisodes de convergence
    ep_col  = next((c for c in ["gaze_convergence_n_episodes", "shared_obj_n_episodes"] if c in df.columns), None)
    dur_col = next((c for c in ["gaze_convergence_dur_total_s", "shared_obj_dur_total_s"] if c in df.columns), None)
    if ep_col and dur_col:
        df["gaze_convergence_episode_density_raw"] = ratio(
            pd.to_numeric(df[ep_col], errors="coerce"),
            pd.to_numeric(df[dur_col], errors="coerce"),
        )
        df["shared_obj_episode_density_raw"] = df["gaze_convergence_episode_density_raw"]

    speech_face_cols = [
        "total_turns",
        "overlap_ratio",
        "interruptions_rate_per_min",
        "mean_turn_s",
        "pause_ratio",
        "participation_entropy",
        "max_speech_ratio",
        "speech_balance_cv",
        "joy_tri_rate_per_min",
        "joy_tri_occupancy",
        "sad_tri_occupancy",
        "joy_sync_jaccard_mean",
        "sad_sync_jaccard_mean",
        "joy_sync_pearson_mean",
        "sad_sync_pearson_mean",
        "affect_balance_occ",
        "affect_sync_jaccard_contrast",
        "affect_sync_pearson_contrast",
    ]
    speech_face_cols = [c for c in speech_face_cols if c in df.columns]
    df = _zscore_speech_and_face(df, speech_face_cols, z_by=z_by)

    gaze_cols = [
        # Nouvelles colonnes directionnelles
        "gaze_convergence_ratio",
        "mutual_gaze_ratio",
        "gaze_entropy_dir_mean",
        "gaze_convergence_episode_rate_per_min_ref",
        "gaze_convergence_dur_total_ratio_ref",
        "mutual_gaze_episode_rate_per_min_ref",
        "mutual_gaze_dur_total_ratio_ref",
        "gaze_joint_attention_idx_raw",
        "gaze_convergence_episode_density_raw",
        # Legacy (conservées pour traçabilité, z-scorées si présentes)
        "shared_obj_ratio",
        "mutual_gaze_ratio_mean_pairs",
        "gaze_entropy_mean_participants",
        "gaze_to_speaker_ratio",
        "transition_prob_gaze_to_speech",
        "shared_obj_episode_rate_per_min_ref",
        "shared_obj_dur_total_ratio_ref",
        "gaze_speaker_coupling_idx_raw",
        "shared_obj_episode_density_raw",
    ]
    gaze_cols = [c for c in gaze_cols if c in df.columns]
    df = _zscore_gaze_only_vr(df, gaze_cols, z_by=z_by)

    cred_terms = []
    if "z_total_turns" in df.columns:
        cred_terms.append(w.cred_total_turns * df["z_total_turns"])
    if "z_overlap_ratio" in df.columns:
        cred_terms.append(w.cred_overlap_ratio * df["z_overlap_ratio"])
    if "z_interruptions_rate_per_min" in df.columns:
        cred_terms.append(w.cred_interruptions_rate * df["z_interruptions_rate_per_min"])
    if "z_joy_tri_rate_per_min" in df.columns:
        cred_terms.append(w.face_joy_rate * df["z_joy_tri_rate_per_min"])
    if "z_sad_tri_occupancy" in df.columns:
        cred_terms.append(w.face_sad_occ_penalty * df["z_sad_tri_occupancy"])
    if "z_gaze_to_speaker_ratio" in df.columns:
        cred_terms.append(w.gaze_to_speaker * df["z_gaze_to_speaker_ratio"])
    if "z_transition_prob_gaze_to_speech" in df.columns:
        cred_terms.append(w.gaze_transition_g2s * df["z_transition_prob_gaze_to_speech"])
    # Entropie directionnelle en priorité, fallback legacy
    if "z_gaze_entropy_dir_mean" in df.columns:
        cred_terms.append(w.gaze_entropy_penalty * df["z_gaze_entropy_dir_mean"])
    elif "z_gaze_entropy_mean_participants" in df.columns:
        cred_terms.append(w.gaze_entropy_penalty * df["z_gaze_entropy_mean_participants"])

    df["tms_credibility_idx"] = np.nan if len(cred_terms) == 0 else np.sum(np.vstack([t.values for t in cred_terms]), axis=0)

    coord_terms = []
    if "z_total_turns" in df.columns:
        coord_terms.append(w.coord_total_turns * df["z_total_turns"])
    if "z_mean_turn_s" in df.columns:
        coord_terms.append(w.coord_mean_turn_s * df["z_mean_turn_s"])
    if "z_overlap_ratio" in df.columns:
        coord_terms.append(w.coord_overlap_ratio * df["z_overlap_ratio"])
    if "z_interruptions_rate_per_min" in df.columns:
        coord_terms.append(w.coord_interruptions_rate * df["z_interruptions_rate_per_min"])
    if "z_pause_ratio" in df.columns:
        coord_terms.append(w.coord_pause_ratio * df["z_pause_ratio"])
    # Priorité aux colonnes directionnelles, fallback legacy
    if "z_gaze_convergence_ratio" in df.columns:
        coord_terms.append(w.gaze_shared_obj * df["z_gaze_convergence_ratio"])
    elif "z_shared_obj_ratio" in df.columns:
        coord_terms.append(w.gaze_shared_obj * df["z_shared_obj_ratio"])
    if "z_mutual_gaze_ratio" in df.columns:
        coord_terms.append(w.gaze_mutual_gaze * df["z_mutual_gaze_ratio"])
    elif "z_mutual_gaze_ratio_mean_pairs" in df.columns:
        coord_terms.append(w.gaze_mutual_gaze * df["z_mutual_gaze_ratio_mean_pairs"])
    if "z_joy_sync_jaccard_mean" in df.columns:
        coord_terms.append(w.face_joy_sync * df["z_joy_sync_jaccard_mean"])
    if "z_sad_sync_jaccard_mean" in df.columns:
        coord_terms.append(w.face_sad_sync_penalty * df["z_sad_sync_jaccard_mean"])

    df["tms_coordination_idx"] = np.nan if len(coord_terms) == 0 else np.sum(np.vstack([t.values for t in coord_terms]), axis=0)

    spec_terms = []
    if "z_mean_turn_s" in df.columns:
        spec_terms.append(w.spec_mean_turn_s * df["z_mean_turn_s"])
    if "z_participation_entropy" in df.columns:
        spec_terms.append(w.spec_participation_entropy * df["z_participation_entropy"])
    if "z_max_speech_ratio" in df.columns:
        spec_terms.append(w.spec_max_speech_ratio * df["z_max_speech_ratio"])
    if "z_speech_balance_cv" in df.columns:
        spec_terms.append(w.spec_speech_balance_cv * df["z_speech_balance_cv"])

    df["tms_specialization_idx"] = np.nan if len(spec_terms) == 0 else np.sum(np.vstack([t.values for t in spec_terms]), axis=0)

    aff_terms = []
    for c, sign in [
        ("z_joy_tri_occupancy", +1.0),
        ("z_joy_sync_jaccard_mean", +1.0),
        ("z_sad_tri_occupancy", -1.0),
        ("z_sad_sync_jaccard_mean", -1.0),
    ]:
        if c in df.columns:
            aff_terms.append(sign * df[c])

    df["affect_alignment_idx"] = np.nan if len(aff_terms) == 0 else nanmean_rows(np.vstack([t.values for t in aff_terms]))

    pear_terms = []
    for c in ["z_joy_sync_pearson_mean", "z_sad_sync_pearson_mean"]:
        if c in df.columns:
            pear_terms.append(df[c])
    df["face_sync_pearson_global_idx"] = np.nan if len(pear_terms) == 0 else nanmean_rows(np.vstack([t.values for t in pear_terms]))

    # MOD-9 : indice legacy (ancienne analyse par objet) — conservé pour traçabilité
    gaze_terms_old = []
    for c in ["z_shared_obj_ratio", "z_mutual_gaze_ratio_mean_pairs"]:
        if c in df.columns:
            gaze_terms_old.append(df[c])
    if "z_gaze_entropy_mean_participants" in df.columns:
        gaze_terms_old.append(w.gaze_entropy_penalty * df["z_gaze_entropy_mean_participants"])
    df["gaze_attention_coordination_idx_old"] = np.nan if len(gaze_terms_old) == 0 else nanmean_rows(np.vstack([t.values for t in gaze_terms_old]))

    # MOD-11 : formule directionnelle — nanmean(z_gaze_convergence_ratio, -z_gaze_entropy_dir_mean)
    # Remplace MOD-10 (shared_obj_ratio → gaze_convergence_ratio, gaze_entropy_mean_participants → gaze_entropy_dir_mean)
    gaze_terms = []
    conv_z = next((c for c in ["z_gaze_convergence_ratio", "z_shared_obj_ratio"] if c in df.columns), None)
    if conv_z:
        gaze_terms.append(df[conv_z])
    ent_z = next((c for c in ["z_gaze_entropy_dir_mean", "z_gaze_entropy_mean_participants"] if c in df.columns), None)
    if ent_z:
        gaze_terms.append(-1.0 * df[ent_z])
    df["gaze_attention_coordination_idx"] = np.nan if len(gaze_terms) == 0 else nanmean_rows(np.vstack([t.values for t in gaze_terms]))

    sp_terms = []
    for c in ["z_gaze_to_speaker_ratio", "z_transition_prob_gaze_to_speech"]:
        if c in df.columns:
            sp_terms.append(df[c])
    df["gaze_speaker_coupling_idx"] = np.nan if len(sp_terms) == 0 else nanmean_rows(np.vstack([t.values for t in sp_terms]))

    return df


# ----------------------------
# Merge / outputs
# ----------------------------

def parse_list(s: Optional[str]) -> List[str]:
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _merge_keys(frames: List[pd.DataFrame]) -> List[str]:
    if not frames:
        return ["group_id", "condition", "scenario", "timepoint"]

    has_tp = False
    for df in frames:
        if "timepoint" in df.columns:
            vals = set(df["timepoint"].astype(str).fillna("UNK").unique().tolist())
            if any(v not in ("UNK", "nan", "None") for v in vals):
                has_tp = True
                break

    base = ["group_id", "group_base_id", "condition", "scenario"]
    return base + ["timepoint"] if has_tp else base


def _drop_merge_suffixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résout toutes les colonnes dupliquées col_x / col_y issues d'un merge pandas.
    Pour chaque paire (col_x, col_y), crée col = coalesce(col_x, col_y) et supprime les deux.
    Si col existe déjà sans suffixe, la version _x/_y est simplement supprimée.
    """
    df = df.copy()

    # Trouver toutes les colonnes se terminant par _x ayant un pendant _y
    x_cols = [c for c in df.columns if c.endswith("_x")]
    for cx in x_cols:
        base = cx[:-2]  # retire "_x"
        cy = base + "_y"
        if cy not in df.columns:
            continue
        # Si la version canonique (sans suffixe) existe déjà, supprimer les doublons
        if base in df.columns:
            df.drop(columns=[cx, cy], inplace=True)
        else:
            # Créer la version canonique = coalesce(_x, _y)
            sx = pd.to_numeric(df[cx], errors="coerce")
            sy = pd.to_numeric(df[cy], errors="coerce")
            df[base] = sx.fillna(sy)
            df.drop(columns=[cx, cy], inplace=True)

    return df


def _drop_audio_raw_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes audio brutes quand la version canonique `audio_*`
    existe deja, pour eviter la duplication de variables dans le fichier audit.
    """
    alias_map = {
        "total_turns": "audio_total_speaking_turns",
        "mean_turn_s": "audio_avg_speaking_turn_duration_s",
        "floor_exchange_pause_mean_s": "audio_floor_exchange_pause_mean_s",
        "pause_mean_s": "audio_floor_exchange_pause_mean_s",
        "overlap_ratio": "audio_overlap_speaking_ratio",
        "successful_interruption_ratio": "audio_successful_interruption_ratio",
        "backchannel_rate_per_min": "audio_backchannel_rate_per_min",
    }

    drop_cols = [
        raw_col
        for raw_col, canonical_col in alias_map.items()
        if raw_col in df.columns and canonical_col in df.columns
    ]
    if drop_cols:
        logging.info(f"Dropping audio raw aliases from audit output: {drop_cols}")
        return df.drop(columns=drop_cols)
    return df


def _reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [
        "group_id", "group_base_id", "condition", "scenario", "timepoint",
        "interaction_dur_s_ref",
        "gaze_mutual_gaze_ratio",
        "gaze_to_speaker_ratio_final",
        "gaze_shared_visual_attention_ratio",
        "gaze_entropy",
        "audio_total_speaking_turns",
        "audio_avg_speaking_turn_duration_s",
        "audio_floor_exchange_pause_mean_s",
        "audio_overlap_speaking_ratio",
        "audio_successful_interruption_ratio",
        "audio_backchannel_rate_per_min",
        "face_smile_ratio",
        "face_negative_affect_ratio",
        "face_facial_synchrony",
        "tms_credibility_idx",
        "tms_coordination_idx",
        "tms_specialization_idx",
        "affect_alignment_idx",
        "face_sync_pearson_global_idx",
        "gaze_attention_coordination_idx",
        "gaze_speaker_coupling_idx",
        "n_missing_speech_core",
        "n_missing_gaze_core",
        "n_missing_face_core",
        # MOD-8 : z-scores affect alignment
        "z_joy_tri_occupancy",
        "z_joy_sync_jaccard_mean",
        "z_sad_tri_occupancy",
        "z_sad_sync_jaccard_mean",
        # MOD-9 : z-scores gaze
        "z_shared_obj_ratio",
        "z_mutual_gaze_ratio_mean_pairs",
        "z_gaze_entropy_mean_participants",
    ]
    front_existing = [c for c in front if c in df.columns]
    other = [c for c in df.columns if c not in front_existing]
    return df[front_existing + other]


def make_compact_output(df: pd.DataFrame) -> pd.DataFrame:
    compact_cols = [
        "group_id", "group_base_id", "condition", "scenario", "timepoint",
        "interaction_dur_s_ref",
        # --- Gaze directionnelle (nouvelles colonnes, MOD-11) ---
        "gaze_convergence_ratio",
        "gaze_convergence_dur_total_s",
        "gaze_convergence_n_episodes",
        "gaze_convergence_mean_angle_deg",
        "gaze_convergence_episode_dur_mean_s",
        "log_gaze_convergence_episode_dur_mean_s",
        "mutual_gaze_ratio",
        "mutual_gaze_n_episodes",
        "mutual_gaze_dur_total_s",
        "gaze_entropy_dir_mean",
        "gaze_focus_proxy",
        "pair_convergence_ratio_mean",
        "pair_mutual_gaze_ratio_mean",
        "gaze_joint_attention_idx_raw",
        "gaze_convergence_episode_rate_per_min_ref",
        "gaze_convergence_episode_density_raw",
        # --- Gaze HL composites ---
        "gaze_mutual_gaze_ratio",
        "gaze_shared_visual_attention_ratio",
        "gaze_entropy",
        "gaze_attention_coordination_idx",
        "gaze_speaker_coupling_idx",
        # --- Sources gaze ---
        "gaze_mutual_gaze_ratio_source",
        "gaze_shared_visual_attention_ratio_source",
        "gaze_entropy_source",
        # --- Audio ---
        "audio_total_speaking_turns",
        "audio_avg_speaking_turn_duration_s",
        "audio_floor_exchange_pause_mean_s",
        "audio_overlap_speaking_ratio",
        "audio_successful_interruption_ratio",
        "audio_backchannel_rate_per_min",
        "audio_participation_entropy",
        "audio_pause_ratio",
        # --- Face ---
        "face_smile_ratio",
        "face_negative_affect_ratio",
        "face_facial_synchrony",
        # --- Composites TMS/affect ---
        "tms_credibility_idx",
        "tms_coordination_idx",
        "tms_specialization_idx",
        "affect_alignment_idx",
        "face_sync_pearson_global_idx",
        # --- Sources audio/face ---
        "audio_total_speaking_turns_source",
        "audio_avg_speaking_turn_duration_s_source",
        "audio_floor_exchange_pause_mean_s_source",
        "audio_overlap_speaking_ratio_source",
        "audio_successful_interruption_ratio_source",
        "audio_backchannel_rate_per_min_source",
        "face_smile_ratio_source",
        "face_negative_affect_ratio_source",
        "face_facial_synchrony_source",
        # --- Diagnostics ---
        "n_missing_speech_core",
        "n_missing_gaze_core",
        "n_missing_face_core",
        # MOD-8 : z-scores affect
        "z_joy_tri_occupancy",
        "z_joy_sync_jaccard_mean",
        "z_sad_tri_occupancy",
        "z_sad_sync_jaccard_mean",
        # MOD-11 : z-scores gaze directionnelle
        "z_gaze_convergence_ratio",
        "z_mutual_gaze_ratio",
        "z_gaze_entropy_dir_mean",
        # Legacy z-scores (si presents)
        "z_shared_obj_ratio",
        "z_mutual_gaze_ratio_mean_pairs",
        "z_gaze_entropy_mean_participants",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    return df[compact_cols].copy()


def make_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    final_feature_cols = [
        "gaze_mutual_gaze_ratio",
        "gaze_to_speaker_ratio_final",
        "gaze_shared_visual_attention_ratio",
        "gaze_entropy",
        "audio_total_speaking_turns",
        "audio_avg_speaking_turn_duration_s",
        "audio_floor_exchange_pause_mean_s",
        "audio_overlap_speaking_ratio",
        "audio_successful_interruption_ratio",
        "audio_backchannel_rate_per_min",
        "face_smile_ratio",
        "face_negative_affect_ratio",
        "face_facial_synchrony",
    ]
    final_feature_cols = [c for c in final_feature_cols if c in out.columns]

    if not final_feature_cols:
        return pd.DataFrame()

    long_rows = []
    id_cols = [c for c in ["group_id", "group_base_id", "condition", "scenario", "timepoint"] if c in out.columns]

    for _, row in out.iterrows():
        for feat in final_feature_cols:
            long_rows.append({
                **{k: row[k] for k in id_cols},
                "feature": feat,
                "is_available": int(pd.notna(row[feat]))
            })

    long_df = pd.DataFrame(long_rows)

    summary = (
        long_df
        .groupby(["feature", "condition", "scenario", "timepoint"], dropna=False)
        .agg(
            n_rows=("is_available", "size"),
            n_available=("is_available", "sum"),
        )
        .reset_index()
    )
    summary["pct_available"] = 100 * summary["n_available"] / summary["n_rows"]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=str, default=None, help="face emotion metrics csv (sep=';')")
    ap.add_argument("--gaze-group", type=str, default=None, help="gaze group metrics csv (sep=',')")
    ap.add_argument("--gaze-pair", type=str, default=None, help="gaze pair metrics csv (sep=',')")
    ap.add_argument(
        "--speech",
        type=str,
        default=None,
        help="Chemin vers audio_features.csv (produit par analyze_audio.py).",
    )

    ap.add_argument("--out", type=str, required=True, help="output compact csv path")
    ap.add_argument("--out-audit", type=str, default=None, help="output full audit csv path")
    ap.add_argument("--out-missing", type=str, default=None, help="output missingness csv path")

    ap.add_argument(
        "--zscore-within",
        type=str,
        default="condition,scenario",
        help="comma-separated keys for within-group zscore (e.g., 'condition,scenario,timepoint' or '' for global)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    out_path = Path(args.out)
    out_audit = Path(args.out_audit) if args.out_audit else out_path.with_name(out_path.stem + "_audit.csv")
    out_missing = Path(args.out_missing) if args.out_missing else out_path.with_name(out_path.stem + "_missingness.csv")

    frames: List[pd.DataFrame] = []

    if args.speech:
        logging.info(f"Loading speech: {args.speech}")
        frames.append(load_speech(args.speech))
    if args.gaze_group:
        logging.info(f"Loading gaze group: {args.gaze_group}")
        frames.append(load_gaze_group(args.gaze_group))
    if args.gaze_pair:
        logging.info(f"Loading gaze pair: {args.gaze_pair}")
        frames.append(load_gaze_pair(args.gaze_pair))
    if args.face:
        logging.info(f"Loading face: {args.face}")
        frames.append(load_face(args.face))

    if len(frames) == 0:
        raise SystemExit(
            "No input provided. Use at least --speech (preferred: audio_features.csv) plus optionally --face/--gaze-*."
        )

    merge_keys = _merge_keys(frames)
    logging.info(f"Merging on keys: {merge_keys}")

    merged = None
    for df in frames:
        if "timepoint" in merge_keys and "timepoint" not in df.columns:
            df = df.copy()
            df["timepoint"] = "UNK"
        if "group_base_id" in merge_keys and "group_base_id" not in df.columns:
            df = add_group_base_id(df)
        merged = df if merged is None else merged.merge(df, on=merge_keys, how="outer")

    assert merged is not None
    merged = _drop_merge_suffixes(merged)
    merged = add_group_base_id(merged)
    merged = exclude_invalid_groups(merged)

    z_by = parse_list(args.zscore_within)
    if len(z_by) == 0:
        z_by = None
        logging.info("Z-scoring globally (no within-group keys).")
    else:
        z_by = [c for c in z_by if c in merged.columns]
        logging.info(f"Z-scoring within: {z_by if z_by else 'GLOBAL (filtered empty)'}")
        if not z_by:
            z_by = None

    merged = compute_composites(merged, z_by=z_by, w=Weights())
    merged = add_final_feature_columns(merged)

    core_speech = [c for c in ["total_turns", "mean_turn_s", "overlap_ratio", "pause_ratio", "interruptions_rate_per_min"] if c in merged.columns]
    merged["n_missing_speech_core"] = merged[core_speech].isna().sum(axis=1) if core_speech else 0

    core_gaze = [c for c in ["gaze_convergence_ratio", "mutual_gaze_ratio", "gaze_entropy_dir_mean",
                              "shared_obj_ratio", "mutual_gaze_ratio_mean_pairs", "gaze_entropy_mean_participants"] if c in merged.columns]
    merged["n_missing_gaze_core"] = merged[core_gaze].isna().sum(axis=1) if core_gaze else 0

    core_face = [c for c in ["joy_tri_occupancy", "sad_tri_occupancy", "joy_sync_jaccard_mean", "sad_sync_jaccard_mean"] if c in merged.columns]
    merged["n_missing_face_core"] = merged[core_face].isna().sum(axis=1) if core_face else 0

    merged = _drop_audio_raw_aliases(merged)

    merged = _reorder_output_columns(merged)
    compact = make_compact_output(merged)
    missing = make_missingness_summary(compact)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_missing.parent.mkdir(parents=True, exist_ok=True)

    compact.to_csv(out_path, index=False)
    merged.to_csv(out_audit, index=False)
    missing.to_csv(out_missing, index=False)

    logging.info(f"Wrote compact: {out_path} (rows={len(compact)}, cols={compact.shape[1]})")
    logging.info(f"Wrote audit:   {out_audit} (rows={len(merged)}, cols={merged.shape[1]})")
    logging.info(f"Wrote missing: {out_missing} (rows={len(missing)}, cols={missing.shape[1]})")


if __name__ == "__main__":
    main()
