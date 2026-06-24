# -*- coding: utf-8 -*-
"""
inv_features_config.py
======================
Configuration centralisée des features INV (Indices Non Verbaux).

Ce fichier est la source unique de vérité pour :
- la liste des features INV par famille ;
- les priorités pour le pruning automatique ;
- les relations de redondance documentées ;
- les features core affichées dans les rapports ;
- les alias canoniques (notamment audio).

Utilisé par :
- analyze_inv_structure.py
- report_ci_group_pdf.py

Lecture rapide :
1. `INV_FEATURES` décrit chaque variable (famille, priorité, redondance, rôle rapport).
2. `CORE_*` est dérivé automatiquement depuis `INV_FEATURES`.
3. `FEATURE_PRIORITY` sert au hard pruning dans l'analyse PCA.

Important :
- La section 3 du rapport ne choisit pas ses variables X à partir de la PCA.
- Elle part des listes `CORE_*`, puis conserve uniquement les colonnes réellement
  présentes dans le dataset chargé pour le bloc (`Speech`, `Face`, `Gaze`,
  `High-level`).
- La PCA, elle, travaille sur l'espace de features préparé par
  `analyze_inv_structure.py`, avec ou sans pruning.

Règles de priorité pour le pruning :
1. moins de NaN
2. priorité métier (plus petit rang = conservé)
3. nom plus court = plus canonique
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================================
# PCA — RÈGLES D'INCLUSION / EXCLUSION
# ----------------------------------------------------------------------------
# Ces constantes contrôlent quelles colonnes sont éligibles à la PCA et aux
# analyses INV (filtres techniques, exclusions explicites, règles AU).
# ============================================================================

# Colonnes identifiant / contexte à exclure de l'analyse numérique brute
ID_COLS = {
    "group_id",
    "group_base_id",
    "condition",
    "scenario",
    "timepoint",
    "interaction_dur_s_ref",
    "interaction_dur_s",
    "interaction_duration_s",
    "n_missing_speech_core",
    "n_missing_gaze_core",
    "n_missing_face_core",
    "duration_s",
}

# Suffixes/préfixes à exclure d'une sélection analytique standard
EXCLUDE_SUFFIXES = ("_source","_old", "_raw", "_zscore", "_zscored", "_z", "_log", "_sqrt", "_cube_root", "_median")
EXCLUDE_PREFIXES = ("z_", "log_", "sqrt_", "cube_root_", "tms_")

# Exclusions techniques supplémentaires (PCA + analyses prédictives)
INV_EXCLUDED_PREFIXES = ("tms_",)
INV_EXCLUDED_SUFFIXES = ("_median",)
INV_EXCLUDED_SUBSTRINGS = ("_median",)

# AU individuelles (ex. au6_active_pct_mean) — exclues des analyses prédictives
# et de la PCA. Les variables de synchronie AU (au_sync_*) sont conservées.
_AU_INDIVIDUAL_SUBSTRINGS = (
    "au6_active_pct",
    "au12_active_pct",
    "au1_active_pct",
    "au4_active_pct",
    "au15_active_pct",
    "au17_active_pct",
    "au6_au12_coactive_pct",
    "au4_au15_coactive_pct",
    "au15_au17_coactive_pct",
)

# Variables exclues par nom exact — redondantes ou non-analytiques
# Exclues explicitement de la PCA et des analyses prédictives, même si elles ne sont pas filtrées par les règles de préfixes/suffixes.
_EXCLUDED_EXACT_NAMES = frozenset({
    "pos_neg_rate_ratio",
    "pos_neg_occ_ratio",
    "sad_active_pct_mean",
    "sad_mean_mean",          # alias de face_sad_intensity_mean
    "sad_tri_rate_sqrt",
    "sad_tri_occupancy",
    "sad_tri_rate_per_min",
    "sad_sync_pearson_mean",
    "sad_sync_jaccard_mean",
    "sad_active_sync_pearson_mean",
    "joy_mean_mean",
    "joy_active_pct_mean",
    "joy_tri_rate_per_min",
    "joy_tri_occupancy",
    "affect_balance_occ",
    "affect_alignment_idx",
    "face_sad_intensity_mean",
    "face_sync_pearson_global_idx",
    # Audio : préférer audio_avg_speaking_turn_duration_s
    # "audio_pause_ratio",
    # "audio_floor_exchange_pause_mean_s",
    "audio_total_speaking_turns",
    "mean_pause_s",
    "n_floor_exchanges",
    # Audio : préférer audio_participation_entropy
    "audio_turn_balance_cv",
    "max_speech_ratio",
    # Gaze directionnelle : préférer gaze_convergence_episode_rate_per_min_ref
    "gaze_convergence_n_episodes",
    "gaze_convergence_n_episodes_per_s",
    "gaze_convergence_dur_total_ratio_ref",
    "gaze_convergence_episode_rate_per_min_ref",
    "gaze_attention_coordination_idx",
    "gaze_attention_coordination_idx_old",
    # Legacy (ancienne analyse par objet)
    "shared_obj_n_episodes",
    "shared_obj_n_episodes_per_s",
    "shared_obj_dur_q25_s",
    "shared_obj_dur_total_ratio_ref",
    "shared_obj_episode_rate_per_min_ref",
})

REGRESSION_FORCE_INCLUDE: list[str] = [
    #"gaze_shared_visual_attention_ratio",
    # "gaze_entropy_mean_participants",
    # "log_gaze_shared_obj_episode_dur_mean_s",
]

# ============================================================================
# CONSTANTES GÉNÉRALES
# ============================================================================

# Paires protégées du pruning — les deux features sont conservées même si |r| > seuil.
# Utiliser quand deux variables mesurent des aspects distincts malgré une corrélation élevée.
PRUNING_PROTECTED_PAIRS: frozenset[frozenset[str]] = frozenset({
    #frozenset({"audio_total_speaking_turns", "audio_overlap_speaking_ratio"}),
    # frozenset({"audio_participation_entropy", "audio_avg_speaking_turn_duration_s"}),
    frozenset({"gaze_entropy_dir_mean", "gaze_shared_visual_attention_ratio"}),
    frozenset({"gaze_entropy_mean_participants", "gaze_shared_visual_attention_ratio"}),  # legacy
    # frozenset({"gaze_entropy_mean_participants", "log_gaze_shared_obj_episode_dur_mean_s"}),
    # frozenset({"gaze_shared_obj_episode_dur_mean_s", "log_gaze_shared_obj_episode_dur_mean_s"}),
    # frozenset({"gaze_shared_visual_attention_ratio", "log_gaze_shared_obj_episode_dur_mean_s"}),
})

REDUNDANCY_CORR_THRESHOLD = 0.85

# Optionnel : permet une évolution future par famille.
FAMILY_REDUNDANCY_THRESHOLDS: dict[str, float] = {
    "audio": REDUNDANCY_CORR_THRESHOLD,
    "face": REDUNDANCY_CORR_THRESHOLD,
    "gaze": REDUNDANCY_CORR_THRESHOLD,
    "tms": REDUNDANCY_CORR_THRESHOLD,
}

ALLOWED_FAMILIES = {"audio", "face", "gaze", "tms"}

# ============================================================================
# VARIABLES RIEDL / HACKMAN / CI (variables cibles potentielles)
# ============================================================================

# Colonnes Y utilisées dans les matrices INV ↔ Riedl du rapport.
# Sélectionnée pour conserver les indicateurs les plus interprétables sur
# l'effort, la stratégie et la compétence de groupe.
CORE_RIEDL_COLS = [
    "effort_task_sum",
    "effort_task_norm",
    "strategy_ratio_mean",
    "strategy_norm",
    "skill_congruence_mean",
    "skill_mean",
    "skill_max",
    "contribution_mean",
]

# Suffixes/préfixes à exclure d'une sélection analytique standard
EXCLUDE_SUFFIXES = ("_source",)
EXCLUDE_PREFIXES = ("z_",)

# Association préfixe -> famille INV
MODALITY_PREFIX: dict[str, list[str]] = {
    "audio": [
        "audio_", "speech_", "silence_", "speak_sil_", "turns_", "turn_",
        "backchannels_", "overlap_", "pause_", "interrupt",
        "interrupt_attempt_", "interrupt_success_",
        "n_attempted_interruptions", "n_successful_interruptions",
        "rapid_floor_takeover_", "rapid_floor_takeovers",
        "pairwise_overlap", "floor_exchange_", "n_floor_exchanges",
        "speaking_", "total_speech", "max_speech", "participation_",
        "mean_turn", "mean_pause", "specialization_", "tms_",
        # Legacy pré-refactor audio : anciennes pseudo-interruptions basées
        # sur une reprise rapide du plancher plutôt que sur un chevauchement.
        "int_",
    ],
    "face": [
        "face_", "smile_", "negative_affect_", "joy_", "sad_",
        "au", "affect_", "pos_neg_", "sync_",
    ],
    "gaze": [
        "gaze_", "gaze_convergence_", "gaze_entropy_dir",
        "mutual_gaze", "pair_convergence_", "pair_mutual_gaze",
        # Legacy (ancienne analyse par objet)
        #"shared_obj", "pair_shared_obj", "transition_prob_gaze",
    ],
}


# Alias canonique -> alias brut (nom pré-refactor dans les CSVs) — toutes familles
FEATURE_ALIAS_PAIRS: tuple[tuple[str, str], ...] = (
    ("audio_total_speaking_turns", "total_turns"),
    ("audio_avg_speaking_turn_duration_s", "mean_turn_s"),
    ("audio_floor_exchange_pause_mean_s", "floor_exchange_pause_mean_s"),
    ("audio_floor_exchange_pause_mean_s", "pause_mean_s"),
    ("audio_overlap_speaking_ratio", "overlap_ratio"),
    ("audio_successful_interruption_ratio", "successful_interruption_ratio"),
    ("audio_backchannel_rate_per_min", "backchannel_rate_per_min"),
    ("audio_participation_entropy", "participation_entropy"),
    ("audio_turn_balance_cv", "turn_balance_cv"),
    ("audio_pause_ratio", "pause_ratio"),
    ("gaze_shared_obj_episode_dur_mean_s", "shared_obj_dur_mean_s"),
    ("face_sad_intensity_mean", "sad_mean_mean"),
)
# Alias de compatibilité — les scripts existants importent AUDIO_ALIAS_PAIRS
AUDIO_ALIAS_PAIRS = FEATURE_ALIAS_PAIRS


# ============================================================================
# RÈGLES DE SÉLECTION DANS LE RAPPORT
# ----------------------------------------------------------------------------
# Section 3 (corrélations INV ↔ Riedl / TCI / Performance / Questionnaire)
# - X = liste `CORE_*` du bloc courant, elle-même dérivée de `INV_FEATURES`,
#   puis intersectée avec les colonnes réellement disponibles dans le dataset
#   chargé par `report_ci_group_pdf.py`.
# - Y = variables cibles externes (`CORE_RIEDL_COLS`, TCI, performance,
#   questionnaire).
# - En mode pruning, un filtrage supplémentaire est appliqué seulement sur les
#   blocs `Face` et `High-level` si `inv_pruned_features.csv` est disponible.
#
# Section 5 (PCA)
# - Le script `analyze_inv_structure.py` travaille sur le jeu high-level
#   préparé pour la PCA et applique éventuellement un hard pruning par
#   corrélation |r| > seuil.
# - Les listes `CORE_*` servent surtout au rapport, pas à construire la PCA.
# ============================================================================

# ============================================================================
# DÉFINITION CENTRALISÉE DES FEATURES INV
# ----------------------------------------------------------------------------
# Champs recommandés :
# - family              : famille conceptuelle
# - priority            : priorité métier (plus bas = mieux)
# - core                : affichée dans la section modale du rapport
# - core_hl             : affichée dans la section synthèse / high-level
# - report_block        : bloc du rapport ciblé si nécessaire (ex: "speech")
# - redundant_with      : relation de redondance documentée
# - drop_if_redundant   : feature à exclure de préférence si redondance confirmée
# - description         : description courte
# - reason              : justification scientifique ou technique
# - calc_method         : méthode de calcul ou de dérivation (brute, ratio,
#                         moyenne, taux, composite, proxy, alias, audit, etc.)
# - theoretical_core    : variable centrale pour l'interprétation
# - report_preferred    : à privilégier dans les blocs rapport
# - pca_preferred       : à privilégier lors du pruning PCA
# - regression_preferred: candidate prioritaire pour la régression v2
# ============================================================================

INV_FEATURES: dict[str, dict[str, Any]] = {
    # =======================================================================
    # AUDIO — versions canoniques
    # =======================================================================
    # Le champ report_block="audio" sur ces features est purement documentaire.
    # Dans _derive_constants(), seule la valeur "speech" est testée explicitement ;
    # les features audio canoniques sont routées vers CORE_AUDIO via family="audio".
    # La valeur "audio" n'a donc aucun effet sur le routing.
    "audio_avg_speaking_turn_duration_s": {
        "family": "audio",
        "priority": 1,
        "core": True,
        "core_hl": True,
        "report_block": "audio",  # documentaire uniquement (routing via family)
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Levinson & Torreira (2015) — Frontiers in Psychology",
        "description": "Durée moyenne des tours de parole (CA)",
        "calc_method": (
            "Tours CA agrégés en 3 passes depuis les IPU VAD : fusion des IPU consécutifs "
            "du même locuteur → filtre ≥ 1.0 s → re-fusion des tours adjacents de même rôle ; "
            "durée moyenne sur l'ensemble des rôles — analyze_audio.py + compute_audio_features.py"
        ),
    },
    "audio_overlap_speaking_ratio": {
        "family": "audio",
        "priority": 2,
        "core": True,
        "core_hl": True,
        "report_block": "audio",  # documentaire uniquement (routing via family)
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Hung & Gatica-Perez (2010) — IEEE Trans. Multimedia",
        "description": "Ratio de chevauchements de parole",
        "calc_method": (
            "overlap_s / total où overlap_s = union des intervalles avec ≥ 2 locuteurs "
            "simultanés, total = durée totale d'interaction — analyze_audio.py"
        ),
    },
    "audio_floor_exchange_pause_mean_s": {
        "family": "audio",
        "priority": 3,

        "core": True,
        "core_hl": True,
        "report_block": "audio",  # documentaire uniquement (routing via family)
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Heldner & Edlund (2010) — J. Phonetics (Floor Transfer Offset)",
        "description": "Pause moyenne lors des échanges de tour",
        "calc_method": (
            "mean(floor_exchange_gaps) où floor_exchange_gaps = silences entre "
            "fin du tour CA du locuteur A et début du tour CA du locuteur B "
            "lors d'un changement de locuteur (calculé sur les tours CA agrégés) "
            "— analyze_audio.py + compute_audio_features.py"
        ),
    },
    "audio_total_speaking_turns": {
        "family": "audio",
        "priority": 4,
        "core": True,
        "core_hl": False,
        "report_block": "audio",
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre total de tours de parole",
        "calc_method": (
            "Somme des turns_{role}_n sur tous les rôles, au niveau groupe "
            "— compute_audio_features.py"
        ),
        "reason": "Biaisé par la durée d'interaction ; audio_avg_speaking_turn_duration_s capte mieux la dynamique conversationnelle",
    },
    "audio_backchannel_rate_per_min": {
        "family": "audio",
        "priority": 5,
        "core": True,
        "core_hl": True,
        "report_block": "audio",  # documentaire uniquement (routing via family)
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": False,
        "reference": "Gravano & Hirschberg (2011) — Speech Communication",
        "description": "Taux de backchannels (signaux d'écoute)",
        "calc_method": (
            "n_backchannels / (duration_s / 60) ; backchannel détecté par 4 filtres : "
            "durée IPU ∈ [0.10–0.70 s], chevauchement ≥ 100 ms avec un autre locuteur, "
            "non-continuation (aucun IPU même rôle ±200 ms), non-tour-CA (aucun tour CA ±500 ms) "
            "— analyze_audio.py + compute_audio_features.py"
        ),
    },
    "audio_successful_interruption_ratio": {
        "family": "audio",
        "priority": 6,
        "core": True,
        "core_hl": True,
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": False,
        "reference": "Hung & Gatica-Perez (2010) — IEEE Trans. Multimedia",
        "description": "Ratio d'interruptions réussies (overlap-based)",
        "calc_method": (
            "successes_total / attempts_total si attempts > 0 sinon NaN ; "
            "interruptions overlap-based (min_overlap=0.1 s, min_post_takeover=0.5 s) "
            "— analyze_audio.py + compute_audio_features.py"
        ),
        "reason": (
            "Construit canonique audio défini sur les interruptions avec "
            "chevauchement effectif ; peut être NaN s'il n'existe aucune "
            "tentative d'interruption dans le groupe"
        ),
    },
    "tms_coordination_idx": {
        "family": "audio",
        "priority": 999,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Proxy heuristique audio-only de coordination",
        "calc_method": (
            "Somme pondérée z-scorée (weights Dataclass) : "
            "z_total_turns(−0.37) + z_mean_turn_s(+0.25) + z_overlap_ratio(−0.25) "
            "+ z_interruptions_rate(−0.30) + z_pause_ratio(−0.15) "
            "+ z_shared_obj_ratio(+0.20) + z_mutual_gaze(+0.10) "
            "+ z_joy_sync_jaccard(+0.10) + z_sad_sync_jaccard(−0.05) "
            "— compute_high_level_features.py"
        ),
        "reason": "Indice theory-driven ; non estimé empiriquement",
    },
    "tms_specialization_idx": {
        "family": "audio",
        "priority": 999,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Proxy heuristique audio-only de spécialisation",
        "calc_method": (
            "Somme pondérée z-scorée (weights Dataclass) : "
            "z_mean_turn_s(+0.57) + z_participation_entropy(−0.25) "
            "+ z_max_speech_ratio(+0.20) + z_speech_balance_cv(+0.20) "
            "— compute_high_level_features.py"
        ),
        "reason": "Indice theory-driven ; non estimé empiriquement",
    },
    "tms_credibility_idx": {
        "family": "audio",
        "priority": 999,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Proxy heuristique audio-only de crédibilité",
        "calc_method": (
            "Somme pondérée z-scorée (weights Dataclass) : "
            "z_total_turns(−0.33) + z_overlap_ratio(−0.20) + z_interruptions_rate(−0.20) "
            "+ z_joy_tri_rate(+0.10) + z_sad_tri_occupancy(−0.10) "
            "+ z_gaze_to_speaker(+0.10) + z_transition_prob(+0.10) "
            "+ z_gaze_entropy(−0.10) "
            "— compute_high_level_features.py"
        ),
        "reason": "Indice theory-driven ; non estimé empiriquement",
    },

    # =======================================================================
    # AUDIO — alias / versions brutes CSV
    # =======================================================================
    "mean_turn_s": {
        "family": "audio",
        "priority": 50,
        "core": False,
        "core_hl": False,
        "report_block": "speech",
        "redundant_with": ["audio_avg_speaking_turn_duration_s"],
        "drop_if_redundant": True,
        "description": "Alias brut de la durée moyenne des tours",
        "calc_method": "Alias direct de audio_avg_speaking_turn_duration_s",
        "reason": "Alias de audio_avg_speaking_turn_duration_s — préférer le canonique audio_*",
    },
    "overlap_ratio": {
        "family": "audio",
        "priority": 51,
        "core": False,
        "core_hl": False,
        "report_block": "speech",
        "redundant_with": ["audio_overlap_speaking_ratio"],
        "drop_if_redundant": True,
        "description": "Alias brut du ratio de chevauchements de groupe",
        "calc_method": "Alias direct de audio_overlap_speaking_ratio",
        "reason": "Alias de audio_overlap_speaking_ratio — préférer le canonique audio_*",
    },
    "audio_pause_ratio": {
        "family": "audio",
        "priority": 52,
        "core": True,
        "core_hl": True,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Ratio de silence collectif du groupe",
        "calc_method": (
            "Ratio entre la durée totale de silence collectif (0 locuteur actif) "
            "et la durée totale d'interaction — compute_audio_features.py"
        ),
        "reason": (
            "Distincte d'audio_floor_exchange_pause_mean_s : ratio global de silence "
            "vs durée moyenne des inter-tours. Capte les temps morts collectifs."
        ),
    },
    "pause_ratio": {
        "family": "audio",
        "priority": 53,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_pause_ratio"],
        "drop_if_redundant": True,
        "description": "Alias brut du ratio de silence collectif",
        "calc_method": "Alias direct de audio_pause_ratio",
        "reason": "Alias de audio_pause_ratio — préférer le canonique audio_*",
    },
    "total_turns": {
        "family": "audio",
        "priority": 53,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_total_speaking_turns"],
        "drop_if_redundant": True,
        "description": "Compte absolu brut des tours de parole",
        "calc_method": "Alias direct de audio_total_speaking_turns",
        "reason": "Alias de audio_total_speaking_turns",
    },
    "floor_exchange_pause_mean_s": {
        "family": "audio",
        "priority": 54,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_floor_exchange_pause_mean_s"],
        "drop_if_redundant": True,
        "description": "Alias brut de la pause moyenne de transition inter-locuteurs",
        "calc_method": "Alias direct de audio_floor_exchange_pause_mean_s",
        "reason": "Alias de audio_floor_exchange_pause_mean_s",
    },
    "pause_mean_s": {
        "family": "audio",
        "priority": 55,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_floor_exchange_pause_mean_s"],
        "drop_if_redundant": True,
        "description": "Alias brut alternatif de la pause moyenne",
        "calc_method": "Alias direct de audio_floor_exchange_pause_mean_s",
        "reason": "Alias de audio_floor_exchange_pause_mean_s",
    },
    "successful_interruption_ratio": {
        "family": "audio",
        "priority": 56,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_successful_interruption_ratio"],
        "drop_if_redundant": True,
        "description": "Alias brut du ratio d'interruptions réussies overlap-based",
        "calc_method": "Alias direct de audio_successful_interruption_ratio",
        "reason": "Alias de audio_successful_interruption_ratio",
    },
    "n_attempted_interruptions": {
        "family": "audio",
        "priority": 93,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre de tentatives d'interruptions overlap-based",
        "calc_method": (
            "Compte brut des tentatives d'interruption overlap-based "
            "(min_overlap=0.1 s, min_post_takeover=0.5 s) "
            "— analyze_audio.py"
        ),
        "reason": "Compte brut utile pour audit ; la version normalisée est interruptions_rate_per_min",
    },
    "n_successful_interruptions": {
        "family": "audio",
        "priority": 94,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre d'interruptions réussies overlap-based",
        "calc_method": (
            "Compte brut des interruptions réussies : tentative overlap-based "
            "ayant abouti à une prise de tour confirmée "
            "— analyze_audio.py"
        ),
        "reason": "Compte brut utile pour audit ; complément du ratio d'interruptions réussies",
    },
    "rapid_floor_takeovers_total": {
        "family": "audio",
        "priority": 95,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre total de prises de tour rapides sans chevauchement requis",
        "calc_method": (
            "Compte des transitions 0 < b_start − a_end ≤ DELTA_INTERRUPT (0.2 s) "
            "sans chevauchement requis — analyze_audio.py"
        ),
        "reason": "Construit distinct des interruptions ; conservé pour audit conversationnel",
    },
    "pairwise_overlap_s": {
        "family": "audio",
        "priority": 96,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Somme des chevauchements pairwise (audit)",
        "calc_method": (
            "Somme des pairwise_overlap(segs[a], segs[b]) pour toutes les paires "
            "(sans déduplication inter-paires) — analyze_audio.py"
        ),
        "reason": "Variable d'audit ; distincte du chevauchement de groupe sans double comptage",
    },
    "pairwise_overlap_ratio": {
        "family": "audio",
        "priority": 97,
        "core": False,
        "core_hl": False,
        "redundant_with": ["pairwise_overlap_s"],
        "drop_if_redundant": False,  # intentionnel : les deux conservés pour audit
        "description": "Ratio pairwise de chevauchement (audit)",
        "calc_method": (
            "pairwise_overlap_s / duration_s "
            "— analyze_audio.py"
        ),
        "reason": (
            "Variable d'audit dérivée de pairwise_overlap_s / duration_s. "
            "Les deux versions (brute et normalisée) sont conservées intentionnellement "
            "pour audit ; aucune n'est supprimée (drop_if_redundant=False sur les deux)."
        ),
    },
    "n_floor_exchanges": {
        "family": "audio",
        "priority": 98,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre de transitions inter-locuteurs valides",
        "calc_method": (
            "Compte brut du nombre de transitions de locuteur valides détectées "
            "— analyze_audio.py"
        ),
        "reason": "Compte d'audit complémentaire à audio_floor_exchange_pause_mean_s",
    },
    "interruptions_rate_per_min": {
        "family": "audio",
        "priority": 99,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Taux d'interruptions overlap-based par minute",
        "calc_method": (
            "interruptions_total / (duration_s / 60) "
            "— compute_audio_features.py"
        ),
        "reason": (
            "Construit canonique dérivé de n_attempted_interruptions / duration_s ; "
            "distinct des rapid_floor_takeover_*"
        ),
    },
    "audio_participation_entropy": {
        "family": "audio",
        "priority": 10,
        "core": True,
        "core_hl": True,
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Stening et al. (2023) — Proceedings CogSci",
        "description": "Entropie de participation à la parole",
        "calc_method": (
            "Entropie de Shannon (bits) sur la distribution des temps de parole par participant ; "
            "valeur basse = dominance, valeur haute = équilibre "
            "— compute_audio_features.py"
        ),
        "reason": "Sélectionné en remplacement de audio_turn_balance_cv dans la sélection v2",
    },
    "participation_entropy": {
        "family": "audio",
        "priority": 50,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_participation_entropy"],
        "drop_if_redundant": True,
        "description": "Alias de audio_participation_entropy (nom pré-refactor)",
        "calc_method": "Alias direct de audio_participation_entropy",
        "reason": "Alias conservé pour compatibilité pipeline ; préférer audio_participation_entropy",
    },
    "max_speech_ratio": {
        "family": "audio",
        "priority": 11,
        "core": False,
        "core_hl": False,
        "redundant_with": ["participation_entropy"],
        "drop_if_redundant": True,
        "description": "Ratio de parole du participant le plus actif",
        "calc_method": (
            "max des speech_ratio par rôle (speech_s_role / total_speech_s) "
            "— compute_audio_features.py"
        ),
        "reason": "Représenté dans le core set par turn_balance_cv (pruning empirique |r|>REDUNDANCY_CORR_THRESHOLD)",
    },
    "audio_turn_balance_cv": {
        "family": "audio",
        "priority": 12,
        "core": True,
        "core_hl": True,
        "redundant_with": ["audio_participation_entropy"],
        "drop_if_redundant": False,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "CV du nombre de tours de parole entre participants",
        "calc_method": (
            "Coefficient de variation (std/mean) du nombre de tours par rôle ; "
            "calculé sur les durées brutes de tours (invariant à l'échelle) "
            "— compute_audio_features.py"
        ),
        "reason": "Mesure d'inégalité de tours complémentaire à audio_participation_entropy",
    },
    "turn_balance_cv": {
        "family": "audio",
        "priority": 60,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_turn_balance_cv"],
        "drop_if_redundant": True,
        "description": "Alias de audio_turn_balance_cv (nom pré-refactor)",
        "calc_method": "Alias direct de audio_turn_balance_cv",
        "reason": "Alias conservé pour compatibilité pipeline ; préférer audio_turn_balance_cv",
    },
    "speech_balance_cv": {
        "family": "audio",
        "priority": 13,
        "core": False,
        "core_hl": False,
        "redundant_with": ["turn_balance_cv"],
        "drop_if_redundant": False,
        "description": "CV des durées de parole entre participants",
        "calc_method": (
            "Coefficient de variation (std/mean) des durées de parole par rôle ; "
            "calculé sur les durées brutes de parole (invariant à l'échelle) "
            "— compute_audio_features.py"
        ),
        "reason": "Représenté dans le core set par turn_balance_cv (pruning empirique |r|>REDUNDANCY_CORR_THRESHOLD)",
    },
    "audio_distrib_speech": {
        "family": "audio",
        "priority": 14,
        "core": True,
        "core_hl": False,
        "redundant_with": ["speech_balance_cv", "audio_participation_entropy"],
        "drop_if_redundant": False,
        "description": "Variance de la distribution de parole (inégalité de participation, Woolley)",
        "calc_method": (
            "Var(p) = (1/N) · Σ(p_i − 1/N)² où p_i = s_i / Σs_j, N=3 ; "
            "p_i = proportion de parole du rôle i (durées brutes) ; "
            "p̄ = 1/3 (distribution uniforme de référence) ; "
            "min = 0 (égalité parfaite), max = 2/9 ≈ 0.222 (un seul locuteur) "
            "— compute_audio_features.py"
        ),
        "reason": (
            "Mesure scalaire d'inégalité de participation au sens de Woolley et al. (2010) ; "
            "complémentaire à l'entropie (qui sature à l'équirépartition) et au CV "
            "(qui n'est pas borné et sensible aux valeurs nulles)"
        ),
    },
    "backchannel_rate_per_min": {
        "family": "audio",
        "priority": 57,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_backchannel_rate_per_min"],
        "drop_if_redundant": True,
        "description": "Alias brut du taux de backchannels",
        "calc_method": "Alias direct de audio_backchannel_rate_per_min",
        "reason": "Alias de audio_backchannel_rate_per_min",
    },

    # =======================================================================
    # FACE / AFFECT
    # =======================================================================
    "face_facial_synchrony": {
        "family": "face",
        "priority": 20,
        "core": True,
        "core_hl": True,
        "redundant_with": [
            "au_sync_jaccard_mean",
            "au_sync_pearson_mean",
            "au_sync_mean",
            "face_sync_pearson_global_idx",
        ],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Nummenmaa et al. (2023) — Communications Biology",
        "description": "Synchronie faciale interpersonnelle (Pearson par AU)",
        "calc_method": (
            "Source : au_sync_pearson_mean (correlations de Pearson par AU entre paires). "
            "Pour chaque paire (a,b) et chaque métrique AU "
            "{au6_active_pct, au12_active_pct, au1_active_pct, au4_active_pct, "
            "au15_active_pct, au17_active_pct, "
            "au6_au12_coactive_pct, au4_au15_coactive_pct, au15_au17_coactive_pct} : "
            "r_(ab,k) = Pearson(AU_k_a(fenêtre), AU_k_b(fenêtre)) ; "
            "au_sync_pearson_mean = mean(r_(ab,k)) sur 9 métriques × 3 paires. "
            "Source tracée dans face_facial_synchrony_source "
            "— face/analyze_aus_group.py + compute_high_level_features.py"
        ),
        "reason": (
            "Le HLF privilégie les synchronies AU-level (au_sync_*) quand elles "
            "sont produites par analyze_aus_group.py, puis documente tout "
            "fallback éventuel via face_facial_synchrony_source. "
            "Les variants AU-level (au_sync_*) et face_sync_pearson_global_idx "
            "sont documentés comme redondants avec ce composite."
        ),
    },
    "face_negative_affect_ratio": {
        "family": "face",
        "priority": 21,
        "core": True,
        "core_hl": True,
        "redundant_with": ["au15_au17_coactive_pct_mean"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Ekman & Friesen (1978) — Facial Action Coding System",
        "description": "Marqueur de tristesse (AU15+AU17 co-actifs)",
        "calc_method": (
            "Source : au15_au17_coactive_pct_mean — proportion de frames où AU15 "
            "(depressor anguli oris) et AU17 (chin raiser) sont co-actifs ; "
            "marqueur FACS de tristesse (Ekman & Friesen 1978). "
            "Complément : au4_au15_coactive_pct_mean (BrowLowerer + LipCornerDepressor). "
            "Source tracée dans face_negative_affect_ratio_source "
            "— compute_high_level_features.py"
        ),
        "reason": (
            "Le HLF privilégie le proxy AU-level au15_au17_coactive_pct_mean ; "
            "au4_au15_coactive_pct_mean est disponible comme mesure complémentaire intégrant AU4. "
            "Si indisponible, la provenance effective est tracée dans "
            "face_negative_affect_ratio_source. "
            "au15_au17_coactive_pct_mean est documenté comme redondant avec cette feature."
        ),
    },
    "face_smile_ratio": {
        "family": "face",
        "priority": 22,
        "core": True,
        "core_hl": True,  # aligné avec face_negative_affect_ratio (même architecture HLF first_valid_series)
        "redundant_with": ["au6_au12_coactive_pct_mean"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Ekman, Davidson & Friesen (1990) — Psychological Science",
        "description": "Sourire de Duchenne (AU6+AU12 co-actifs)",
        "calc_method": (
            "Source : au6_au12_coactive_pct_mean — proportion de frames où AU6 "
            "(orbicularis oculi) et AU12 (zygomaticus major) sont co-actifs ; "
            "sourire de Duchenne / sourire sincère (Ekman, Davidson & Friesen 1990). "
            "Source tracée dans face_smile_ratio_source "
            "— compute_high_level_features.py"
        ),
        "reason": (
            "Le HLF privilégie le proxy AU-level au6_au12_coactive_pct_mean ; "
            "si indisponible, la provenance effective est tracée dans "
            "face_smile_ratio_source. "
            "au6_au12_coactive_pct_mean est documenté comme redondant avec cette feature."
        ),
    },
    "affect_alignment_idx": {
        "family": "face",
        "priority": 23,
        "core": True,
        "core_hl": True,
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": False,
        "report_preferred": True,
        "reference": "Contribution originale (pas de référence directe)",
        "description": "Alignement affectif entre participants (composite)",
        "calc_method": (
            "nanmean([+z_joy_tri_occupancy, +z_joy_sync_jaccard_mean, "
            "−z_sad_tri_occupancy, −z_sad_sync_jaccard_mean]) "
            "— compute_high_level_features.py"
        ),
    },
    "face_sync_pearson_global_idx": {
        "family": "face",
        "priority": 24,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Synchronie faciale globale (Pearson)",
        "calc_method": (
            "nanmean([z_joy_sync_pearson_mean, z_sad_sync_pearson_mean]) "
            "— compute_high_level_features.py"
        ),
        "reason": "Redondance empirique attendue avec face_facial_synchrony ; supprimé du core set publication",
    },
    "affect_balance_occ": {
        "family": "face",
        "priority": 25,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": True,
        "description": "Balance affect positif/négatif (occurrences)",
        "calc_method": (
            "joy_tri_occupancy − sad_tri_occupancy "
            "— compute_high_level_features.py"
        ),
        "reason": "|r|≈0.99 avec face_smile_ratio",
    },
    "joy_mean_mean": {
        "family": "face",
        "priority": 26,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": True,
        "description": "Intensité moyenne de joy",
        "calc_method": (
            "Moyenne inter-participants de l'intensité moyenne du signal joy "
            "sur l'ensemble de la session (moyenne des moyennes individuelles)"
        ),
        "reason": "|r|≈1.0 avec face_smile_ratio ; remplacé par face_smile_ratio (AU6+AU12) dans le core set",
    },
    "au6_active_pct_mean": {
        "family": "face",
        "priority": 101,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU6 actif",
        "calc_method": (
            "Moyenne inter-participants de au6_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Produit par analyze_aus_group.py si le signal AU6 est disponible",
    },
    "au12_active_pct_mean": {
        "family": "face",
        "priority": 102,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU12 actif",
        "calc_method": (
            "Moyenne inter-participants de au12_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Variable AU-level brute : conservée pour audit/interprétation PCA uniquement",
    },
    "au15_active_pct_mean": {
        "family": "face",
        "priority": 103,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU15 actif",
        "calc_method": (
            "Moyenne inter-participants de au15_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Variable AU-level brute : conservée pour audit/interprétation PCA uniquement",
    },
    "au17_active_pct_mean": {
        "family": "face",
        "priority": 104,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU17 actif",
        "calc_method": (
            "Moyenne inter-participants de au17_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Produit par analyze_aus_group.py si le signal AU17 (ChinRaiser) est disponible",
    },
    "au6_au12_coactive_pct_mean": {
        "family": "face",
        "priority": 105,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": False,
        "description": "% moyen de co-activation AU6 + AU12",
        "calc_method": (
            "Proxy AU-level du sourire : moyenne inter-participants du pourcentage de "
            "frames où AU6==1 ET AU12==1 simultanément (co-activation binaire) "
            "— analyze_aus_group.py"
        ),
        "reason": "Proxy AU-level privilégié par le HLF pour face_smile_ratio",
    },
    "au1_active_pct_mean": {
        "family": "face",
        "priority": 110,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU1 actif (InnerBrowRaiser)",
        "calc_method": (
            "Moyenne inter-participants de au1_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Variable AU-level brute : conservée pour audit/interprétation PCA uniquement",
    },
    "au4_active_pct_mean": {
        "family": "face",
        "priority": 111,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "% moyen de frames avec AU4 actif (BrowLowerer / fronceur)",
        "calc_method": (
            "Moyenne inter-participants de au4_active_pct calculé par fenêtre "
            "puis moyenné sur les rôles via descriptives_windowed "
            "— analyze_aus_group.py"
        ),
        "reason": "Variable AU-level brute : conservée pour audit/interprétation PCA uniquement",
    },
    "au4_au15_coactive_pct_mean": {
        "family": "face",
        "priority": 112,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_negative_affect_ratio", "au15_au17_coactive_pct_mean"],
        "drop_if_redundant": False,
        "description": "% moyen de co-activation AU4 + AU15 (BrowLowerer + LipCornerDepressor)",
        "calc_method": (
            "Proxy AU-level étendu de l'affect négatif : moyenne inter-participants du "
            "pourcentage de frames où AU4==1 ET AU15==1 simultanément (co-activation binaire). "
            "Combinaison FACS classique de tristesse (Ekman & Friesen 1978) incluant le "
            "fronceur des sourcils — analyze_aus_group.py"
        ),
        "reason": (
            "Complément de au15_au17_coactive_pct_mean : intègre AU4 (BrowLowerer), "
            "marqueur robuste de tristesse absent de la co-activation AU15+AU17 originale"
        ),
    },
    "au15_au17_coactive_pct_mean": {
        "family": "face",
        "priority": 106,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": False,
        "description": "% moyen de co-activation AU15 + AU17",
        "calc_method": (
            "Proxy AU-level de l'affect négatif : moyenne inter-participants du pourcentage "
            "de frames où AU15==1 ET AU17==1 simultanément (co-activation binaire) "
            "— analyze_aus_group.py"
        ),
        "reason": "Proxy AU-level privilégié par le HLF pour face_negative_affect_ratio",
    },
    "au_sync_jaccard_mean": {
        "family": "face",
        "priority": 107,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony"],
        "drop_if_redundant": False,
        "description": "Synchronie AU-level moyenne (Jaccard)",
        "calc_method": (
            "Moyenne des scores de Jaccard sur 9 colonnes AU × 3 paires : "
            "{au6, au12, au1, au4, au15, au17, au6+12, au4+15, au15+17} ; "
            "jaccard(binary_AU_a, binary_AU_b) pour chaque paire et chaque AU "
            "— analyze_aus_group.py"
        ),
        "reason": "Agrégat global de synchronie binaire entre participants, produit si les AUs sont disponibles",
    },
    "au_sync_pearson_mean": {
        "family": "face",
        "priority": 108,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony"],
        "drop_if_redundant": False,
        "description": "Synchronie AU-level moyenne (Pearson)",
        "calc_method": (
            "Moyenne des corrélations de Pearson sur 9 métriques AU × 3 paires "
            "{au6, au12, au1, au4, au15, au17, au6+12, au4+15, au15+17} "
            "(fenêtrage glissant) — analyze_aus_group.py"
        ),
        "reason": "Agrégat global de co-fluctuation AU-level entre participants",
    },
    "au_sync_mean": {
        "family": "face",
        "priority": 109,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony", "au_sync_jaccard_mean", "au_sync_pearson_mean"],
        "drop_if_redundant": False,
        "description": "Synthèse globale de synchronie AU-level",
        "calc_method": (
            "mean([au_sync_jaccard_mean, au_sync_pearson_mean]) si les deux sont disponibles "
            "— analyze_aus_group.py"
        ),
        "reason": "Moyenne de au_sync_jaccard_mean et au_sync_pearson_mean lorsque disponibles",
    },
    "face_sad_intensity_mean": {
        "family": "face",
        "priority": 27,
        "core": True,
        "core_hl": True,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "description": "Intensité moyenne de sad (signal brut)",
        "calc_method": (
            "Moyenne inter-participants de l'intensité moyenne du signal sad "
            "sur l'ensemble de la session — face_emotion_metrics_all.csv (sad_mean_mean)"
        ),
        "reason": (
            "Complément de face_negative_affect_ratio (AU15+AU17 binaire) : capte l'intensité continue "
            "du signal sad plutôt que la co-activation binaire des AUs"
        ),
    },
    "sad_mean_mean": {
        "family": "face",
        "priority": 70,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_sad_intensity_mean"],
        "drop_if_redundant": True,
        "description": "Alias de face_sad_intensity_mean (nom pré-refactor)",
        "calc_method": "Alias direct de face_sad_intensity_mean",
        "reason": "Alias conservé pour compatibilité pipeline ; préférer face_sad_intensity_mean",
    },
    "joy_active_pct_mean": {
        "family": "face",
        "priority": 28,
        "core": True,
        "core_hl": False,
        "redundant_with": ["joy_mean_mean"],
        "drop_if_redundant": True,
        "description": "% de frames où joy est actif",
        "calc_method": (
            "Moyenne inter-participants du pourcentage de frames où le signal joy "
            "dépasse le seuil d'activation (présence/absence binaire)"
        ),
        "reason": "Conceptuellement proche de face_smile_ratio ; supprimé du core set publication",
    },
    "sad_active_pct_mean": {
        "family": "face",
        "priority": 29,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_mean_mean"],
        "drop_if_redundant": True,
        "description": "% de frames où sad est actif",
        "calc_method": (
            "Moyenne inter-participants du pourcentage de frames où le signal sad "
            "dépasse le seuil d'activation (présence/absence binaire)"
        ),
        "reason": "Redondant avec sad_mean_mean",
    },
    "joy_tri_occupancy": {
        "family": "face",
        "priority": 60,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": True,
        "description": "Occupancy brute de joy",
        "calc_method": (
            "tot_tri / interaction_dur_s où tot_tri = durée totale des frames "
            "où les 3 participants sont simultanément actifs sur joy "
            "— analyze_aus_group.py"
        ),
        "reason": "|r|≈1.0 avec face_smile_ratio",
    },
    "joy_tri_rate_per_min": {
        "family": "face",
        "priority": 58,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Taux d'épisodes de joie triadique par minute",
        "calc_method": (
            "n_tri / (interaction_dur_s / 60) où n_tri = nombre d'épisodes "
            "triAdiques joy simultanés — analyze_aus_group.py"
        ),
    },
    "sad_tri_rate_per_min": {
        "family": "face",
        "priority": 59,
        "core": True,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": True,
        "description": "Taux d'épisodes de tristesse triadique par minute",
        "calc_method": (
            "n_tri / (interaction_dur_s / 60) où n_tri = nombre d'épisodes "
            "triadiques sad simultanés — analyze_aus_group.py"
        ),
        "reason": "Valeur ajoutée limitée en présence de face_negative_affect_ratio",
    },
    "sad_tri_rate_sqrt": {
        "family": "face",
        "priority": 57,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_tri_rate_per_min"],
        "drop_if_redundant": False,
        "description": "sqrt(sad_tri_rate_per_min) — usage descriptif uniquement",
        "calc_method": (
            "sqrt(sad_tri_rate_per_min) — compute_high_level_features.py. "
            "Stabilise la variance de la distribution de comptage de taux."
        ),
        "reason": "Transformation racine carrée pour usage exploratoire/descriptif ; non incluse dans la régression",
    },
    "sad_tri_occupancy": {
        "family": "face",
        "priority": 61,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": True,
        "description": "Occupancy brute de sad",
        "calc_method": (
            "tot_tri / interaction_dur_s où tot_tri = durée totale des frames "
            "où les 3 participants sont simultanément actifs sur sad "
            "— analyze_aus_group.py"
        ),
        "reason": "|r|≈1.0 avec face_negative_affect_ratio",
    },
    "joy_mean_median": {
        "family": "face",
        "priority": 62,
        "core": False,
        "core_hl": False,
        "redundant_with": ["joy_mean_mean"],
        "drop_if_redundant": True,
        "description": "Intensité médiane de joy",
        "calc_method": (
            "Médiane inter-participants de l'intensité du signal joy "
            "(version robuste aux outliers de joy_mean_mean)"
        ),
        "reason": "|r|≈0.99 avec joy_mean_mean",
    },
    "sad_mean_median": {
        "family": "face",
        "priority": 63,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_mean_mean"],
        "drop_if_redundant": True,
        "description": "Intensité médiane de sad",
        "calc_method": (
            "Médiane inter-participants de l'intensité du signal sad "
            "(version robuste aux outliers de sad_mean_mean)"
        ),
        "reason": "|r|≈1.0 avec sad_mean_mean",
    },
    "affect_balance_rate": {
        "family": "face",
        "priority": 64,
        "core": True,
        "core_hl": True,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": False,
        "description": "Balance affective (rate)",
        "calc_method": (
            "joy_tri_rate_per_min − sad_tri_rate_per_min "
            "— compute_high_level_features.py"
        ),
        "reason": "Indicateur INV_TRS dans la SEM ; core_hl=True pour propagation dans merged_master",
    },
    "joy_sync_jaccard_mean": {
        "family": "face",
        "priority": 65,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_smile_ratio"],
        "drop_if_redundant": True,
        "description": "Synchronie joy (Jaccard)",
        "calc_method": (
            "Moyenne inter-paires des indices de Jaccard calculés sur les séries "
            "binaires d'activation joy entre participants"
        ),
        "reason": "|r|≈0.96 avec face_smile_ratio",
    },
    "joy_sync_pearson_mean": {
        "family": "face",
        "priority": 66,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony"],
        "drop_if_redundant": True,
        "description": "Synchronie joy (Pearson)",
        "calc_method": (
            "Moyenne inter-paires des corrélations de Pearson calculées sur "
            "les séries temporelles d'intensité joy entre participants"
        ),
        "reason": "|r|≈0.97 avec face_facial_synchrony",
    },
    "joy_active_sync_pearson_mean": {
        "family": "face",
        "priority": 67,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_facial_synchrony"],
        "drop_if_redundant": True,
        "description": "Synchronie joy active (Pearson)",
        "calc_method": (
            "Moyenne inter-paires des corrélations de Pearson calculées uniquement "
            "sur les frames où joy est actif pour au moins un participant"
        ),
        "reason": "|r|≈0.94 avec face_facial_synchrony",
    },
    "sad_active_sync_pearson_mean": {
        "family": "face",
        "priority": 68,
        "core": True,
        "core_hl": False,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": True,
        "description": "Synchronie sad active (Pearson)",
        "calc_method": (
            "Moyenne inter-paires des corrélations de Pearson calculées uniquement "
            "sur les frames où sad est actif pour au moins un participant"
        ),
        "reason": "|r|≈0.88 avec face_negative_affect_ratio",
    },
    "sad_sync_jaccard_mean": {
        "family": "face",
        "priority": 69,
        "core": False,
        "core_hl": False,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": True,
        "description": "Synchronie sad (Jaccard)",
        "calc_method": (
            "Moyenne inter-paires des indices de Jaccard calculés sur les séries "
            "binaires d'activation sad entre participants"
        ),
        "reason": "Mesure plus brute que le ratio d'affect négatif et la synchronie composite",
    },
    "sad_sync_pearson_mean": {
        "family": "face",
        "priority": 76,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_sync_jaccard_mean", "face_negative_affect_ratio"],
        "drop_if_redundant": True,
        "description": "Synchronie sad (Pearson)",
        "calc_method": (
            "Moyenne inter-paires des corrélations de Pearson calculées sur "
            "les séries temporelles d'intensité sad entre participants"
        ),
        "reason": "Très proche conceptuellement des autres métriques de synchronie négative",
    },
    "affect_sync_jaccard_contrast": {
        "family": "face",
        "priority": 77,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Contraste de synchronie affective positive vs négative (Jaccard)",
        "calc_method": (
            "Différence entre la synchronie joy (Jaccard) et la synchronie sad (Jaccard) "
            "au niveau groupe (joy_sync_jaccard_mean - sad_sync_jaccard_mean)"
        ),
    },
    "affect_sync_pearson_contrast": {
        "family": "face",
        "priority": 78,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Contraste de synchronie affective positive vs négative (Pearson)",
        "calc_method": (
            "Différence entre la synchronie joy (Pearson) et la synchronie sad (Pearson) "
            "au niveau groupe (joy_sync_pearson_mean - sad_sync_pearson_mean)"
        ),
    },
    "pos_neg_occ_ratio": {
        "family": "face",
        "priority": 79,
        "core": False,
        "core_hl": False,
        "redundant_with": ["affect_balance_occ"],
        "drop_if_redundant": True,
        "description": "Ratio d'occurrences positives / négatives",
        "calc_method": (
            "Ratio entre le nombre d'occurrences d'affect positif et le nombre "
            "d'occurrences d'affect négatif au niveau groupe "
            "(version ratio de affect_balance_occ)"
        ),
        "reason": "Redondance potentielle avec face_smile_ratio à confirmer empiriquement ; exclu du core set par précaution",
    },
    "pos_neg_rate_ratio": {
        "family": "face",
        "priority": 80,
        "core": False,
        "core_hl": False,
        "redundant_with": ["affect_balance_rate"],
        "drop_if_redundant": True,
        "description": "Ratio de taux d'épisodes positifs / négatifs",
        "calc_method": (
            "Ratio entre le taux d'épisodes affectifs positifs et le taux "
            "d'épisodes affectifs négatifs au niveau groupe "
            "(version ratio de affect_balance_rate)"
        ),
        "reason": "Version ratio de la balance affective de taux",
    },
    "joy_mean_valid_ratio": {
        "family": "face",
        "priority": 81,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Ratio de validité du signal joy",
        "calc_method": (
            "Variable d'audit : ratio de frames où le signal joy est jugé valide "
            "(non manquant, non aberrant) sur l'ensemble de la session, "
            "agrégé au niveau groupe"
        ),
        "reason": "Indicateur de qualité du signal plus que variable comportementale centrale",
    },
    "sad_mean_valid_ratio": {
        "family": "face",
        "priority": 82,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Ratio de validité du signal sad",
        "calc_method": (
            "Variable d'audit : ratio de frames où le signal sad est jugé valide "
            "(non manquant, non aberrant) sur l'ensemble de la session, "
            "agrégé au niveau groupe"
        ),
        "reason": "Indicateur de qualité du signal plus que variable comportementale centrale",
    },
    "joy_active_pct_median": {
        "family": "face",
        "priority": 83,
        "core": False,
        "core_hl": False,
        "redundant_with": ["joy_active_pct_mean", "joy_mean_mean"],
        "drop_if_redundant": True,
        "description": "Médiane du pourcentage d'activation joy",
        "calc_method": (
            "Médiane inter-participants du pourcentage de frames où joy est actif "
            "(version robuste aux outliers de joy_active_pct_mean)"
        ),
        "reason": "Version robuste mais conceptuellement proche de joy_active_pct_mean",
    },
    "sad_active_pct_median": {
        "family": "face",
        "priority": 84,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_active_pct_mean", "sad_mean_mean"],
        "drop_if_redundant": True,
        "description": "Médiane du pourcentage d'activation sad",
        "calc_method": (
            "Médiane inter-participants du pourcentage de frames où sad est actif "
            "(version robuste aux outliers de sad_active_pct_mean)"
        ),
        "reason": "Version robuste mais conceptuellement proche de sad_active_pct_mean",
    },
    "joy_active_pct_valid_ratio": {
        "family": "face",
        "priority": 85,
        "core": False,
        "core_hl": False,
        "redundant_with": ["joy_mean_valid_ratio"],
        "drop_if_redundant": True,
        "description": "Ratio de validité du signal joy actif",
        "calc_method": (
            "Variable d'audit : ratio de frames où le signal joy actif est valide, "
            "calculé uniquement sur les frames où joy dépasse le seuil d'activation"
        ),
        "reason": "Indicateur de qualité du signal",
    },
    "sad_active_pct_valid_ratio": {
        "family": "face",
        "priority": 86,
        "core": False,
        "core_hl": False,
        "redundant_with": ["sad_mean_valid_ratio"],
        "drop_if_redundant": True,
        "description": "Ratio de validité du signal sad actif",
        "calc_method": (
            "Variable d'audit : ratio de frames où le signal sad actif est valide, "
            "calculé uniquement sur les frames où sad dépasse le seuil d'activation"
        ),
        "reason": "Indicateur de qualité du signal",
    },

    # =======================================================================
    # GAZE — ANALYSE DIRECTIONNELLE (nouvelle, sans maillage BIM)
    # =======================================================================
    "gaze_convergence_ratio": {
        "family": "gaze",
        "priority": 30,
        "core": True,
        "core_hl": True,
        "redundant_with": ["shared_obj_ratio"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "analyze_gaze_directional.py",
        "description": "Ratio de temps où ≥2 participants regardent dans la même direction (< 20°)",
        "calc_method": (
            "Fraction du temps où au moins 2 participants ont un angle inter-direction < 20° "
            "— analyze_gaze_directional.py (remplace shared_obj_ratio)"
        ),
    },
    "gaze_convergence_n_episodes": {
        "family": "gaze",
        "priority": 50,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_n_episodes"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Nombre d'épisodes de convergence directionnelle",
        "calc_method": "Comptage des épisodes où gaze_convergence_ratio > 0 — analyze_gaze_directional.py",
    },
    "gaze_convergence_dur_total_s": {
        "family": "gaze",
        "priority": 50,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_dur_total_s"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Durée totale des épisodes de convergence directionnelle (s)",
        "calc_method": "Somme des durées d'épisodes de convergence — analyze_gaze_directional.py",
    },
    "gaze_convergence_mean_angle_deg": {
        "family": "gaze",
        "priority": 55,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Angle moyen entre directions de regard lors des épisodes de convergence (°)",
        "calc_method": "Moyenne des angles inter-directions lors des épisodes < 20° — analyze_gaze_directional.py",
    },
    "mutual_gaze_ratio": {
        "family": "gaze",
        "priority": 32,
        "core": True,
        "core_hl": True,
        "redundant_with": ["mutual_gaze_ratio_mean_pairs"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "analyze_gaze_directional.py",
        "description": "Ratio de regard mutuel directionnel (A→tête B ET B→tête A simultanément)",
        "calc_method": (
            "Fraction du temps où l'angle entre DirCorr_A et (pos_B - pos_A) < 30° "
            "ET l'angle entre DirCorr_B et (pos_A - pos_B) < 30° — analyze_gaze_directional.py"
        ),
    },
    "mutual_gaze_n_episodes": {
        "family": "gaze",
        "priority": 55,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_n_episodes_sum_pairs"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Nombre d'épisodes de regard mutuel directionnel",
        "calc_method": "Comptage des épisodes de regard mutuel — analyze_gaze_directional.py",
    },
    "mutual_gaze_dur_total_s": {
        "family": "gaze",
        "priority": 55,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_dur_total_s_sum_pairs"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Durée totale des épisodes de regard mutuel directionnel (s)",
        "calc_method": "Somme des durées de regard mutuel — analyze_gaze_directional.py",
    },
    "gaze_entropy_dir_mean": {
        "family": "gaze",
        "priority": 31,
        "core": True,
        "core_hl": True,
        "redundant_with": ["gaze_entropy_mean_participants"],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "analyze_gaze_directional.py",
        "description": "Entropie directionnelle moyenne (dispersion des azimuts en 16 bins de 22.5°)",
        "calc_method": (
            "Entropie de Shannon normalisée sur la distribution des azimuts (projection XZ) "
            "en 16 bins — analyze_gaze_directional.py (remplace gaze_entropy_mean_participants)"
        ),
    },
    "log_gaze_convergence_episode_dur_mean_s": {
        "family": "gaze",
        "priority": 52,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_convergence_dur_total_s"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "reference": "analyze_gaze_directional.py",
        "description": "log1p(durée totale convergence directionnelle) — pour régression",
        "calc_method": (
            "log1p(gaze_convergence_dur_total_s) — compute_high_level_features.py. "
            "Équivalent directionnel de log_gaze_shared_obj_episode_dur_mean_s."
        ),
        "reason": "Distribution brute potentiellement asymétrique ; log-transform conservé par symétrie avec l'ancienne pipeline",
    },
    "gaze_convergence_episode_rate_per_min_ref": {
        "family": "gaze",
        "priority": 53,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_convergence_ratio"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Taux d'épisodes de convergence par minute (normalisé par durée de référence)",
        "calc_method": "gaze_convergence_n_episodes / (interaction_dur_s_ref / 60) — compute_high_level_features.py",
        "reason": "Redondant avec gaze_convergence_ratio à durée fixée ; préférer gaze_convergence_ratio",
    },
    "gaze_convergence_episode_density_raw": {
        "family": "gaze",
        "priority": 54,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_convergence_ratio"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Densité d'épisodes de convergence (n_episodes / dur_total_s)",
        "calc_method": "ratio(gaze_convergence_n_episodes, gaze_convergence_dur_total_s) — compute_high_level_features.py",
    },

    # =======================================================================
    # GAZE — ANALYSE PAR OBJET (legacy, ancienne analyse)
    # =======================================================================
    "gaze_entropy_mean_participants": {
        "family": "gaze",
        "priority": 30,
        "core": False,          # remplacée par gaze_entropy_dir_mean (analyse directionnelle)
        "core_hl": False,
        "redundant_with": ["gaze_entropy_dir_mean"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "reference": "Yarbus (1967) — Eye Movements and Vision",
        "description": "Entropie moyenne du regard (legacy — analyse par objet)",
        "calc_method": (
            "Moyenne inter-participants de l'entropie de Shannon (bits, normalisée) "
            "calculée sur la distribution des durées de fixation par AOI "
            "— analyze_gaze.py. "
            "MOD-11 (2026-06) : remplacée par gaze_entropy_dir_mean (analyse directionnelle)."
        ),
        "reason": "Remplacée par gaze_entropy_dir_mean ; conservée comme fallback legacy",
    },
    "shared_obj_ratio": {
        "family": "gaze",
        "priority": 33,
        "core": True,
        "core_hl": False,
        "redundant_with": ["gaze_shared_visual_attention_ratio"],
        "drop_if_redundant": False,
        "description": "Ratio de temps passé sur objets partagés",
        "calc_method": (
            "float(np.mean(sameobj)) sur une grille temporelle régulière "
            "où sameobj = frames avec ≥ 2 participants fixant le même objet "
            "— analyze_gaze.py"
        ),
        "reason": (
            "|r|≈1.0 avec gaze_shared_visual_attention_ratio ; source brute VR. "
            "core_hl=False : gaze_shared_visual_attention_ratio est le représentant HL."
        ),
    },
    "gaze_shared_obj_episode_dur_mean_s": {
        "family": "gaze",
        "priority": 32,
        "core": False,          # legacy — remplacée par gaze_convergence_dur_total_s
        "core_hl": False,
        "redundant_with": ["gaze_convergence_dur_total_s"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "reference": "Biancardi et al. (2023) — J. Multimodal User Interfaces",
        "description": "Durée moyenne par épisode d'objet partagé (legacy)",
        "calc_method": (
            "dur_total_s / n_episodes — analyze_gaze.py. "
            "MOD-11 (2026-06) : remplacée par gaze_convergence_dur_total_s."
        ),
        "reason": "Remplacée par gaze_convergence_dur_total_s (analyse directionnelle)",
    },
    "log_gaze_shared_obj_episode_dur_mean_s": {
        "family": "gaze",
        "priority": 31,
        "core": False,          # legacy — remplacée par log_gaze_convergence_episode_dur_mean_s
        "core_hl": False,
        "redundant_with": ["log_gaze_convergence_episode_dur_mean_s"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "reference": "Biancardi et al. (2023) — J. Multimodal User Interfaces",
        "description": "log1p(durée moyenne par épisode d'objet partagé) — legacy",
        "calc_method": (
            "log1p(gaze_shared_obj_episode_dur_mean_s) — compute_high_level_features.py. "
            "MOD-11 (2026-06) : remplacée par log_gaze_convergence_episode_dur_mean_s."
        ),
        "reason": "Remplacée par log_gaze_convergence_episode_dur_mean_s (analyse directionnelle)",
    },
    "shared_obj_dur_mean_s": {
        "family": "gaze",
        "priority": 80,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_shared_obj_episode_dur_mean_s"],
        "drop_if_redundant": True,
        "description": "Alias de gaze_shared_obj_episode_dur_mean_s (nom pré-refactor)",
        "calc_method": "Alias direct de gaze_shared_obj_episode_dur_mean_s",
        "reason": "Alias conservé pour compatibilité pipeline ; préférer gaze_shared_obj_episode_dur_mean_s",
    },
    "gaze_shared_visual_attention_ratio": {
        "family": "gaze",
        "priority": 38,  # priorité plus basse que gaze_convergence_ratio (30) → droppée si redondante
        "core": True,
        "core_hl": True,
        "redundant_with": ["gaze_convergence_ratio", "shared_obj_ratio", "gaze_joint_attention_idx_raw"],
        "drop_if_redundant": True,   # laisse gaze_convergence_ratio survivre au pruning
        "regression_preferred": False,  # gaze_convergence_ratio est le representant direct
        "report_preferred": True,
        "reference": "Richardson & Dale (2005) — Psychological Science",
        "description": "Attention visuelle partagée (chaîne de fallback : gaze_convergence_ratio en source primaire)",
        "calc_method": (
            "first_valid_series([gaze_convergence_ratio, pair_convergence_ratio_mean, "
            "shared_obj_ratio, shared_obj_dur_total_ratio_ref, pair_shared_obj_ratio_mean]) "
            "— premier résultat non-NaN de la chaîne ; "
            "MOD-11 (2026-06) : gaze_convergence_ratio devient source primaire (analyse directionnelle) "
            "— compute_high_level_features.py"
        ),
        "reason": (
            "Représentant synthétique HL de la coordination visuelle ; "
            "corrélé avec gaze_attention_coordination_idx (r=0.709) et gaze_entropy (r=-0.736) "
            "— collinéarité documentée, maintenu comme indicateur core distinct"
        ),
    },
    "gaze_entropy": {
        "family": "gaze",
        "priority": 34,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_entropy_dir_mean", "gaze_entropy_mean_participants"],
        "drop_if_redundant": True,
        "description": "Dispersion du regard (alias : directionnelle en priorité, objet en fallback)",
        "calc_method": (
            "first_valid_series([gaze_entropy_dir_mean, gaze_entropy_mean_participants]) — "
            "MOD-11 (2026-06) : gaze_entropy_dir_mean devient source primaire "
            "— compute_high_level_features.py"
        ),
        "reason": "Alias synthétique — |r|≈0.92 avec ses sources",
    },
    "gaze_attention_coordination_idx": {
        "family": "gaze",
        "priority": 35,
        "core": True,
        "core_hl": True,
        "redundant_with": [],
        "drop_if_redundant": False,
        "regression_preferred": True,
        "report_preferred": True,
        "reference": "Contribution originale (pas de référence directe)",
        "description": "Coordination de l'attention (composite : convergence directionnelle − entropie direction)",
        "calc_method": (
            "MOD-11 (2026-06) : nanmean([z_gaze_convergence_ratio, −1.0 × z_gaze_entropy_dir_mean]) "
            "à poids égaux (1/2 chacun) avec fallback legacy sur z_shared_obj_ratio / z_gaze_entropy_mean_participants "
            "— compute_high_level_features.py. "
            "Remplace MOD-10 (shared_obj_ratio → gaze_convergence_ratio, analyse directionnelle sans maillage BIM)."
        ),
    },
    "gaze_mutual_gaze_ratio": {
        "family": "gaze",
        "priority": 39,  # priorité plus basse que mutual_gaze_ratio (32) → droppée si redondante
        "core": True,
        "core_hl": False,  # drop_if_redundant=True incompatible avec core_hl=True
        "redundant_with": ["mutual_gaze_ratio"],
        "drop_if_redundant": True,   # laisse mutual_gaze_ratio survivre au pruning
        "description": "Ratio global de regard mutuel (composite HL — directionnelle en priorité)",
        "calc_method": (
            "first_valid_series([mutual_gaze_ratio, mutual_gaze_ratio_mean_pairs, "
            "pair_mutual_gaze_ratio_mean, mutual_gaze_dur_total_ratio_ref]) — premier résultat non-NaN "
            "— compute_high_level_features.py"
        ),
        "reason": "Variable VR-only valide ; exclue précédemment pour NaN PC",
    },
    "gaze_to_speaker_ratio_final": {
        "family": "gaze",
        "priority": 37,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": True,
        "description": "Ratio regard vers le locuteur",
        "calc_method": (
            "first_valid_series([gaze_to_speaker_ratio]) — alias direct via "
            "first_valid_series dans le pipeline HLF "
            "— compute_high_level_features.py"
        ),
        "reason": "Données absentes du dataset",
    },
    "mutual_gaze_ratio_mean_pairs": {
        "family": "gaze",
        "priority": 38,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Ratio moyen de regard mutuel par paire (⚠ structurellement nul en VR)",
        "calc_method": (
            "Moyenne des float(np.mean(mutual)) calculés par paire de participants "
            "— analyze_gaze.py. "
            "⚠ Audit 2026-05 : valeur = 0.0 pour les 11 groupes VR (variance=0). "
            "Retiré de gaze_attention_coordination_idx (MOD-10) — terme mort après z-score."
        ),
        "reason": (
            "Version amont redondante avec gaze_mutual_gaze_ratio ; non retenue dans le core set VR. "
            "Valeur structurellement nulle en VR (dispositif lunettes, pas de détection regard-vers-visage)."
        ),
    },
    "gaze_to_speaker_ratio": {
        "family": "gaze",
        "priority": 39,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_to_speaker_ratio_final"],
        "drop_if_redundant": True,
        "description": "Ratio brut de regard vers le locuteur actif",
        "calc_method": (
            "nanmean sur les listeners du ratio (durée regardant le locuteur actif) / "
            "(durée totale valide avec locuteur actif) — analyze_gaze.py"
        ),
        "reason": "Données absentes du dataset",
    },
    "gaze_speaker_coupling_idx": {
        "family": "gaze",
        "priority": 40,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_speaker_coupling_idx_raw"],
        "drop_if_redundant": True,
        "description": "Indice composite de couplage regard-parole",
        "calc_method": (
            "nanmean([z_gaze_to_speaker_ratio, z_transition_prob_gaze_to_speech]) "
            "— compute_high_level_features.py"
        ),
        "reason": "Composite invalide : dépend de gaze_to_speaker_ratio, données absentes",
    },
    "pair_shared_obj_ratio_mean": {
        "family": "gaze",
        "priority": 41,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Ratio moyen d'objet partagé au niveau dyadique",
        "calc_method": (
            "Moyenne des ratios de co-fixation sur un même objet calculés au niveau "
            "de chaque paire de participants (agrégat dyadique de shared_obj_ratio)"
        ),
    },
    "pair_mutual_gaze_ratio_mean": {
        "family": "gaze",
        "priority": 42,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_ratio_mean_pairs"],
        "drop_if_redundant": True,
        "description": "Ratio moyen de regard mutuel au niveau dyadique",
        "calc_method": (
            "Moyenne des ratios de regard mutuel calculés au niveau de chaque paire ; "
            "agrégation légèrement différente de mutual_gaze_ratio_mean_pairs "
            "(source ou pondération distincte dans analyze_gaze.py)"
        ),
        "reason": "Agrégation dyadique très proche de mutual_gaze_ratio_mean_pairs",
    },
    "transition_prob_gaze_to_speech": {
        "family": "gaze",
        "priority": 43,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": True,
        "description": "Probabilité de transition du regard vers la prise de parole",
        "calc_method": (
            "hit/tot où hit = nombre d'onsets de parole précédés d'un regard "
            "vers un autre participant dans la fenêtre [onset−tau, onset) "
            "— analyze_gaze.py"
        ),
        "reason": "Données absentes du dataset",
    },
    "shared_obj_episode_rate_per_min_ref": {
        "family": "gaze",
        "priority": 31,
        "core": True,
        "core_hl": True,
        "redundant_with": ["shared_obj_n_episodes_per_s", "shared_obj_n_episodes"],
        "drop_if_redundant": False,
        "description": "Taux d'épisodes d'objet partagé par minute de référence",
        "calc_method": (
            "n_ep / (duration_min + 1e-6) où n_ep = shared_obj_n_episodes "
            "et duration_min = interaction_dur_s_ref / 60 "
            "— compute_high_level_features.py"
        ),
        "reason": "Indicateur INV_TAS dans la SEM ; core_hl=True pour propagation dans merged_master",
    },
    "pair_shared_obj_dur_mean_s_mean": {
        "family": "gaze",
        "priority": 45,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_dur_mean_s"],
        "drop_if_redundant": True,
        "description": "Durée moyenne des épisodes d'objet partagé au niveau dyadique",
        "calc_method": (
            "Moyenne inter-paires des durées moyennes d'épisode de co-fixation "
            "(agrégat dyadique de shared_obj_dur_mean_s)"
        ),
        "reason": "Mesure proche de shared_obj_dur_mean_s mais agrégée par paire",
    },
    "pair_mutual_gaze_dur_mean_s_mean": {
        "family": "gaze",
        "priority": 46,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Durée moyenne des épisodes de regard mutuel au niveau dyadique",
        "calc_method": (
            "Moyenne inter-paires des durées moyennes des épisodes de regard mutuel "
            "(mutual_gaze_dur_total_s / mutual_gaze_n_episodes, agrégé par paire)"
        ),
    },
    "mutual_gaze_n_episodes_sum_pairs": {
        "family": "gaze",
        "priority": 47,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre total d'épisodes de regard mutuel (somme des paires)",
        "calc_method": (
            "Somme du nombre d'épisodes de regard mutuel sur toutes les paires "
            "de participants de la triade"
        ),
    },
    "mutual_gaze_episode_rate_per_min_ref": {
        "family": "gaze",
        "priority": 48,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_n_episodes_per_s", "mutual_gaze_n_episodes_sum_pairs"],
        "drop_if_redundant": False,
        "description": "Taux d'épisodes de regard mutuel par minute de référence",
        "calc_method": (
            "Nombre total d'épisodes de regard mutuel rapporté à une durée de référence "
            "normalisée en minutes (mutual_gaze_n_episodes_sum_pairs / duration_ref_min)"
        ),
    },
    "mutual_gaze_dur_total_ratio_ref": {
        "family": "gaze",
        "priority": 49,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_dur_total_ratio"],
        "drop_if_redundant": False,
        "description": "Ratio de durée totale de regard mutuel normalisé sur une durée de référence",
        "calc_method": (
            "mutual_gaze_dur_total_s_sum_pairs / (interaction_dur_s_ref + 1e-6) "
            "— compute_high_level_features.py"
        ),
    },
    "gaze_focus_proxy": {
        "family": "gaze",
        "priority": 70,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_entropy_mean_participants"],
        "drop_if_redundant": True,
        "description": "Proxy de focus visuel",
        "calc_method": (
            "1.0 − gaze_entropy_mean_participants (transformation monotone décroissante) "
            "— analyze_gaze.py + compute_high_level_features.py"
        ),
        "reason": "|r|≈1.0 avec gaze_entropy_mean_participants",
    },
    "shared_obj_dur_total_s": {
        "family": "gaze",
        "priority": 71,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_ratio"],
        "drop_if_redundant": True,
        "description": "Durée totale sur objets partagés",
        "calc_method": (
            "Somme des durées de tous les épisodes de co-fixation sur un même objet, "
            "en secondes (version absolue de shared_obj_ratio, dépendante de la durée)"
        ),
        "reason": "|r|≈0.98 avec shared_obj_ratio",
    },
    "shared_obj_dur_total_ratio": {
        "family": "gaze",
        "priority": 72,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_ratio"],
        "drop_if_redundant": True,
        "description": "Ratio de durée totale sur objets partagés",
        "calc_method": (
            "Ratio entre la durée totale de co-fixation sur un même objet et la durée "
            "totale d'interaction (shared_obj_dur_total_s / interaction_duration_s)"
        ),
        "reason": "|r|≈1.0 avec shared_obj_ratio",
    },
    "shared_obj_n_episodes": {
        "family": "gaze",
        "priority": 34,
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Nombre absolu d'épisodes d'objet partagé",
        "calc_method": (
            "Compte brut du nombre d'épisodes de co-fixation sur un même objet "
            "détectés sur toute la session au niveau groupe"
        ),
        "reason": "Compte absolu plutôt qu'indice normalisé",
    },
    "shared_obj_n_episodes_per_s": {
        "family": "gaze",
        "priority": 74,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_n_episodes"],
        "drop_if_redundant": True,
        "description": "Taux d'épisodes d'objet partagé",
        "calc_method": (
            "Nombre d'épisodes de co-fixation rapporté à la durée d'interaction "
            "en secondes (shared_obj_n_episodes / interaction_duration_s)"
        ),
        "reason": "|r|≈0.97 avec shared_obj_n_episodes",
    },
    "gaze_joint_attention_idx_raw": {
        "family": "gaze",
        "priority": 75,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_shared_visual_attention_ratio"],
        "drop_if_redundant": True,
        "description": "Attention conjointe brute",
        "calc_method": (
            "(shared_obj_ratio + mutual_gaze_ratio_mean_pairs) / 2 "
            "— compute_high_level_features.py"
        ),
        "reason": "|r|≈1.0 avec gaze_shared_visual_attention_ratio",
    },
    "mutual_gaze_dur_total_s_sum_pairs": {
        "family": "gaze",
        "priority": 87,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_dur_total_ratio", "mutual_gaze_dur_total_ratio_ref"],
        "drop_if_redundant": True,
        "description": "Durée totale de regard mutuel (somme des paires)",
        "calc_method": (
            "Somme des durées totales de regard mutuel sur toutes les paires de la triade, "
            "en secondes (version absolue dépendante de la durée d'interaction)"
        ),
        "reason": "Version absolue dépendante de la durée d'interaction",
    },
    "mutual_gaze_n_episodes_per_s": {
        "family": "gaze",
        "priority": 88,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_episode_rate_per_min_ref", "mutual_gaze_n_episodes_sum_pairs"],
        "drop_if_redundant": True,
        "description": "Taux brut d'épisodes de regard mutuel par seconde",
        "calc_method": (
            "Nombre total d'épisodes de regard mutuel rapporté à la durée d'interaction "
            "en secondes (mutual_gaze_n_episodes_sum_pairs / interaction_duration_s)"
        ),
        "reason": "Version brute du taux d'épisodes de regard mutuel",
    },
    "mutual_gaze_dur_total_ratio": {
        "family": "gaze",
        "priority": 89,
        "core": False,
        "core_hl": False,
        "redundant_with": ["mutual_gaze_dur_total_ratio_ref"],
        "drop_if_redundant": True,
        "description": "Ratio de durée totale de regard mutuel",
        "calc_method": (
            "Ratio entre la durée totale de regard mutuel (somme des paires) et la "
            "durée effective d'interaction (interaction_duration_s), "
            "version brute de mutual_gaze_dur_total_ratio_ref"
        ),
        "reason": "Mesure brute très proche de la version normalisée sur durée de référence",
    },
    "shared_obj_dur_total_ratio_ref": {
        "family": "gaze",
        "priority": 90,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_dur_total_ratio"],
        "drop_if_redundant": True,
        "description": "Ratio de durée totale d'objet partagé normalisé sur durée de référence",
        "calc_method": (
            "Variable dérivée de shared_obj_dur_total_s normalisée par une durée de "
            "référence (interaction_dur_s_ref) plutôt que par la durée effective, "
            "version de référence de shared_obj_dur_total_ratio"
        ),
        "reason": "Redondant avec shared_obj_episode_rate_per_min_ref ; non retenu dans le core set VR",
    },
    "shared_obj_episode_density_raw": {
        "family": "gaze",
        "priority": 91,
        "core": False,
        "core_hl": False,
        "redundant_with": ["shared_obj_episode_rate_per_min_ref", "shared_obj_n_episodes_per_s"],
        "drop_if_redundant": True,
        "description": "Densité brute d'épisodes d'objet partagé",
        "calc_method": (
            "Variable d'audit : densité brute d'épisodes de co-fixation avant "
            "normalisation sur durée de référence ; version intermédiaire de "
            "shared_obj_episode_rate_per_min_ref"
        ),
        "reason": "Mesure dérivée du taux d'épisodes",
    },
    "gaze_speaker_coupling_idx_raw": {
        "family": "gaze",
        "priority": 92,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_speaker_coupling_idx"],
        "drop_if_redundant": True,
        "description": "Version brute de l'indice de couplage regard-parole",
        "calc_method": (
            "(gaze_to_speaker_ratio + transition_prob_gaze_to_speech) / 2 "
            "— compute_high_level_features.py"
        ),
        "reason": "Idem",
    },

}


# ============================================================================
# DRAPEAUX STRUCTURELS COMPLÉMENTAIRES
# ----------------------------------------------------------------------------
# Ces drapeaux dissocient les usages sans modifier les listes historiques
# `CORE_*` ni la logique de pruning pilotée par `priority`.
# ============================================================================

_FEATURE_SELECTION_FLAGS = (
    "theoretical_core",
    "report_preferred",
    "pca_preferred",
    "regression_preferred",
)

for _cfg in INV_FEATURES.values():
    for _flag in _FEATURE_SELECTION_FLAGS:
        _cfg.setdefault(_flag, False)


# ============================================================================
# CONSTRUCTION DES ALIAS
# ============================================================================

def _build_alias_maps() -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Construit :
    - alias_to_canonical : alias -> canonique
    - canonical_to_aliases : canonique -> liste alias
    """
    alias_to_canonical: dict[str, str] = {}
    canonical_to_aliases: dict[str, list[str]] = {}

    for canonical, alias in FEATURE_ALIAS_PAIRS:
        alias_to_canonical[alias] = canonical
        canonical_to_aliases.setdefault(canonical, []).append(alias)

    return alias_to_canonical, canonical_to_aliases


ALIAS_TO_CANONICAL, CANONICAL_TO_ALIASES = _build_alias_maps()


# ============================================================================
# CONSTANTES DÉRIVÉES DEPUIS INV_FEATURES
# ----------------------------------------------------------------------------
# On garde `INV_FEATURES` comme source unique de vérité.
# Les listes `CORE_*` utilisées par le rapport sont reconstruites à partir
# des métadonnées portées par chaque feature.
# ============================================================================

def _derive_constants() -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """
    Génère les listes CORE_* et FEATURE_PRIORITY à partir de INV_FEATURES.

    Convention utilisée :
    - `core=True` + `report_block="speech"`  -> bloc Speech
    - `core=True` + `family="audio"`         -> bloc Audio canonique
    - `core=True` + `family="face"`          -> bloc Face
    - `core=True` + `family="gaze"`          -> bloc Gaze
    - `core_hl=True`                         -> bloc High-level
    """
    core_audio: list[str] = []
    core_speech: list[str] = []
    core_face: list[str] = []
    core_gaze: list[str] = []
    core_hl: list[str] = []

    for feat, cfg in INV_FEATURES.items():
        family = cfg.get("family", "")
        is_core = bool(cfg.get("core", False))
        is_core_hl = bool(cfg.get("core_hl", False))
        report_block = str(cfg.get("report_block", "")).lower().strip()

        if is_core:
            if report_block == "speech":
                core_speech.append(feat)
            elif family == "audio":
                core_audio.append(feat)
            elif family == "face":
                core_face.append(feat)
            elif family == "gaze":
                core_gaze.append(feat)

        if is_core_hl:
            core_hl.append(feat)

    sort_key = lambda k: INV_FEATURES[k].get("priority", 999)

    core_audio.sort(key=sort_key)
    core_speech.sort(key=sort_key)
    core_face.sort(key=sort_key)
    core_gaze.sort(key=sort_key)
    core_hl.sort(key=sort_key)

    priority_list = sorted(INV_FEATURES.keys(), key=sort_key)
    return core_audio, core_speech, core_face, core_gaze, core_hl, priority_list


CORE_AUDIO, CORE_SPEECH, CORE_FACE, CORE_GAZE, CORE_HL, FEATURE_PRIORITY = _derive_constants()

# Mapping bloc du rapport -> variables X candidates.
# Le rapport fera ensuite l'intersection avec les colonnes réellement présentes.
CORE_MAP = {
    "Audio": CORE_AUDIO,
    "Speech": CORE_SPEECH,
    "Face": CORE_FACE,
    "Gaze": CORE_GAZE,
    "High-level": CORE_HL,
}


# ============================================================================
# RÈGLES ANALYTIQUES PARTAGÉES (rapport v2 + usages aval)
# ----------------------------------------------------------------------------
# Ces constantes complètent la config centrale pour éviter de maintenir une
# seconde source de vérité dédiée au rapport v2.
# ============================================================================

COHESION_COMPONENTS = ["SOC", "TSK", "COM"]
TMS_DIMENSIONS = ["COR", "CRE", "SPE"]
QUESTIONNAIRE_RELIABILITY_DIMENSIONS = TMS_DIMENSIONS + COHESION_COMPONENTS
COHESION_SCORE_COL = "Cohesion_questionnaire_score"
QUESTIONNAIRE_ANALYSIS_COLS = TMS_DIMENSIONS + [COHESION_SCORE_COL]

PERFORMANCE_ANALYSIS_COLS = ["Score_perf_tsk"]
PERFORMANCE_EXCLUDED_COLS = {
    "Score_perf_M1M3",
    "Score_perf_tsk_mean",
    "M1_consignes_%",
    "M2_nombre_%",
    "M3_precision_%",
    "M4_temps_%",
}

# Analyse performance : scénario utilisé uniquement comme covariable.
SCENARIO_COVARIATE_ONLY = True


def _dedupe_keep_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _canonicalize_audio_speech_features(names: list[str] | tuple[str, ...]) -> list[str]:
    alias_to_canonical = {alias: canonical for canonical, alias in FEATURE_ALIAS_PAIRS}
    canonicalized: list[str] = []
    for name in names:
        canonicalized.append(alias_to_canonical.get(name, name))
    return _dedupe_keep_order(canonicalized)


def is_excluded_inv_feature(col: str) -> bool:
    """Filtre PCA/analyses INV (exclusions strictes)."""
    low = str(col).strip().lower()
    if low in _EXCLUDED_EXACT_NAMES:
        return True
    if any(s in low for s in _AU_INDIVIDUAL_SUBSTRINGS):
        return True
    if any(s in low for s in INV_EXCLUDED_SUBSTRINGS):
        return True
    if any(low.endswith(s) for s in INV_EXCLUDED_SUFFIXES):
        return True
    if low.startswith("tms_") and low.endswith("_idx"):
        return True
    return False


def filter_inv_feature_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Filtre une liste de features selon les règles PCA/INV."""
    return [name for name in names if not is_excluded_inv_feature(name)]


def _is_excluded_report_feature(col: str) -> bool:
    """Filtre technique pour les listes de corrélations du rapport.
    N'applique PAS _EXCLUDED_EXACT_NAMES (réservé à la PCA).
    """
    low = str(col).strip().lower()
    if any(s in low for s in _AU_INDIVIDUAL_SUBSTRINGS):
        return True
    if any(s in low for s in INV_EXCLUDED_SUBSTRINGS):
        return True
    if any(low.endswith(s) for s in INV_EXCLUDED_SUFFIXES):
        return True
    if low.startswith("tms_") and low.endswith("_idx"):
        return True
    return False


def filter_report_feature_names(names: list[str] | tuple[str, ...]) -> list[str]:
    """Variante rapport : garde les variables exclues de la PCA mais utiles en descriptif."""
    return [name for name in names if not _is_excluded_report_feature(name)]


def filter_inv_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Retourne une vue filtrée du dataframe selon is_excluded_inv_feature."""
    if df is None:
        return None
    keep_cols = [c for c in df.columns if not is_excluded_inv_feature(c)]
    return df[keep_cols].copy()


# Bloc "Speech" du rapport : variables audio/parole canoniques `audio_*`.
CORE_SPEECH_V2 = filter_inv_feature_names(
    _canonicalize_audio_speech_features(list(CORE_AUDIO) + list(CORE_SPEECH))
)
CORE_FACE_V2 = filter_inv_feature_names(list(CORE_FACE))
CORE_GAZE_V2 = filter_inv_feature_names(list(CORE_GAZE))
CORE_HL_V2 = filter_inv_feature_names(list(CORE_HL))
CORE_MAP_V2 = {
    key: filter_inv_feature_names(list(values))
    for key, values in CORE_MAP.items()
}

# Listes rapport (corrélations) : incluent les variables exclues de la PCA mais utiles
# en descriptif (ex. joy_active_pct_mean, sad_tri_rate_per_min, shared_obj_episode_rate_per_min_ref).
CORE_SPEECH_REPORT = filter_report_feature_names(
    _canonicalize_audio_speech_features(list(CORE_AUDIO) + list(CORE_SPEECH))
)
CORE_FACE_REPORT = filter_report_feature_names(list(CORE_FACE))
CORE_GAZE_REPORT = filter_report_feature_names(list(CORE_GAZE))
CORE_HL_REPORT = filter_report_feature_names(list(CORE_HL))
CORE_MAP_REPORT = {
    key: filter_report_feature_names(list(values))
    for key, values in CORE_MAP.items()
}


RETAINED_INV_FEATURES = {
    "speech": CORE_SPEECH_V2,
    "face": CORE_FACE_V2,
    "gaze": CORE_GAZE_V2,
    "high_level": CORE_HL_V2,
}


def keep_questionnaire_analysis_col(col: str) -> bool:
    return col in QUESTIONNAIRE_ANALYSIS_COLS


def keep_questionnaire_component_col(col: str) -> bool:
    return col in COHESION_COMPONENTS


def keep_performance_analysis_col(col: str) -> bool:
    return col in PERFORMANCE_ANALYSIS_COLS


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_core_features(family: str) -> list[str]:
    """
    Retourne les features core pour une famille donnée.

    Paramètres
    ----------
    family : str
        "audio", "face", "gaze", "high-level", "hl" ou "all"

    Retour
    ------
    list[str]
        Liste des features core pour cette famille.

    Note
    ----
    `speech` retourne les variables brutes du bloc Speech du rapport, alors que
    `audio` retourne les features canoniques `audio_*`.

    `all` retourne CORE_AUDIO + CORE_SPEECH + CORE_FACE + CORE_GAZE.
    CORE_HL est exclu intentionnellement — il recoupe partiellement les autres
    blocs et sert un bloc distinct dans le rapport. Utiliser get_core_features("hl")
    pour obtenir CORE_HL séparément.
    """
    family_lower = family.lower().replace("-", "").replace("_", "")

    if family_lower == "all":
        # Note : CORE_HL est exclu intentionnellement — il recoupe partiellement
        # CORE_AUDIO, CORE_FACE et CORE_GAZE et sert un bloc distinct dans le rapport.
        # Utiliser get_core_features("hl") pour obtenir CORE_HL.
        return CORE_AUDIO + CORE_SPEECH + CORE_FACE + CORE_GAZE
    if family_lower == "audio":
        return CORE_AUDIO.copy()
    if family_lower == "speech":
        return CORE_SPEECH.copy()
    if family_lower in ("face", "affect"):
        return CORE_FACE.copy()
    if family_lower == "gaze":
        return CORE_GAZE.copy()
    if family_lower in ("highlevel", "hl"):
        return CORE_HL.copy()

    return []


def get_feature_priority() -> list[str]:
    """Retourne la liste des features triées par priorité."""
    return FEATURE_PRIORITY.copy()


def get_canonical_feature_name(feature: str) -> str:
    """
    Retourne le nom canonique d'une feature.
    Si la feature n'est pas un alias connu, retourne son nom inchangé.
    """
    return ALIAS_TO_CANONICAL.get(feature, feature)


def get_aliases(feature: str) -> list[str]:
    """
    Retourne les alias connus d'une feature canonique.
    """
    canonical = get_canonical_feature_name(feature)
    return CANONICAL_TO_ALIASES.get(canonical, []).copy()


def get_features_by_family(
    family: str,
    include_drop_if_redundant: bool = True,
) -> list[str]:
    """
    Retourne les features d'une famille.

    Paramètres
    ----------
    family : str
        "audio", "face", "gaze", "tms"
    include_drop_if_redundant : bool
        Si False, exclut les features explicitement marquées comme
        'drop_if_redundant=True'.

    Retour
    ------
    list[str]
        Features triées par priorité.
    """
    family_lower = family.lower()
    result: list[str] = []

    for feat, cfg in INV_FEATURES.items():
        if cfg.get("family", "").lower() != family_lower:
            continue
        if not include_drop_if_redundant and bool(cfg.get("drop_if_redundant", False)):
            continue
        result.append(feat)

    return sorted(result, key=lambda k: INV_FEATURES[k].get("priority", 999))


def _get_flagged_features(flag_name: str, family: str) -> list[str]:
    """
    Retourne les features marquées par un drapeau structurel.

    Paramètres
    ----------
    flag_name : str
        Nom du drapeau dans `INV_FEATURES`.
    family : str
        "audio", "face", "gaze" ou "all"
    """
    family_lower = family.lower().replace("-", "").replace("_", "")
    if family_lower not in {"audio", "face", "gaze", "all"}:
        return []

    out: list[str] = []
    for feat, cfg in INV_FEATURES.items():
        if not bool(cfg.get(flag_name, False)):
            continue
        feat_family = str(cfg.get("family", "")).lower()
        if family_lower != "all" and feat_family != family_lower:
            continue
        out.append(feat)

    return sorted(out, key=lambda k: INV_FEATURES[k].get("priority", 999))


def get_theoretical_core_features(family: str) -> list[str]:
    """Retourne les features theoretical_core=True pour une famille."""
    return _get_flagged_features("theoretical_core", family)


def get_report_preferred_features(family: str) -> list[str]:
    """Retourne les features report_preferred=True pour une famille."""
    return _get_flagged_features("report_preferred", family)


def get_pca_preferred_features(family: str) -> list[str]:
    """Retourne les features pca_preferred=True pour une famille."""
    return _get_flagged_features("pca_preferred", family)


def get_regression_preferred_features(family: str) -> list[str]:
    """Retourne les features regression_preferred=True pour une famille."""
    return _get_flagged_features("regression_preferred", family)


REGRESSION_PREFERRED_AUDIO = get_regression_preferred_features("audio")
REGRESSION_PREFERRED_FACE = get_regression_preferred_features("face")
REGRESSION_PREFERRED_GAZE = get_regression_preferred_features("gaze")

REGRESSION_RETAINED_INV_FEATURES: dict[str, list[str]] = {
    "speech": REGRESSION_PREFERRED_AUDIO,
    "face": REGRESSION_PREFERRED_FACE,
    "gaze": REGRESSION_PREFERRED_GAZE,
}


def is_redundant(feat_a: str, feat_b: str) -> bool:
    """
    Vérifie si deux features sont documentées comme redondantes.
    """
    a = get_canonical_feature_name(feat_a)
    b = get_canonical_feature_name(feat_b)

    cfg_a = INV_FEATURES.get(a, {})
    cfg_b = INV_FEATURES.get(b, {})

    return (
        b in cfg_a.get("redundant_with", [])
        or a in cfg_b.get("redundant_with", [])
    )


def should_drop_if_redundant(feature: str) -> bool:
    """
    Indique si une feature doit être supprimée préférentiellement
    lorsqu'une redondance est confirmée.
    """
    canonical = get_canonical_feature_name(feature)
    cfg = INV_FEATURES.get(canonical, {})
    return bool(cfg.get("drop_if_redundant", False))


def get_redundancy_info(feature: str) -> dict[str, Any] | None:
    """
    Retourne les informations documentées de redondance pour une feature.
    """
    canonical = get_canonical_feature_name(feature)
    cfg = INV_FEATURES.get(canonical)
    if cfg is None:
        return None

    redundant_with = cfg.get("redundant_with", [])
    if not redundant_with:
        return None

    return {
        "feature": canonical,
        "redundant_with": redundant_with,
        "drop_if_redundant": bool(cfg.get("drop_if_redundant", False)),
        "reason": cfg.get("reason", f"|r| > {REDUNDANCY_CORR_THRESHOLD}"),
    }


def get_feature_metadata(feature: str) -> dict[str, Any] | None:
    """
    Retourne les métadonnées complètes d'une feature.
    """
    canonical = get_canonical_feature_name(feature)
    return INV_FEATURES.get(canonical)


def get_family_threshold(family: str) -> float:
    """
    Retourne le seuil de redondance applicable à une famille.
    """
    family_lower = family.lower()
    return FAMILY_REDUNDANCY_THRESHOLDS.get(family_lower, REDUNDANCY_CORR_THRESHOLD)


def infer_family_from_name(feature: str) -> str | None:
    """
    Infère la famille d'une feature à partir de son nom.
    Utile pour traiter des colonnes nouvelles non encore documentées.
    """
    canonical = get_canonical_feature_name(feature)

    if canonical in INV_FEATURES:
        return str(INV_FEATURES[canonical].get("family"))

    name = canonical.lower()
    for family, prefixes in MODALITY_PREFIX.items():
        if any(name.startswith(prefix.lower()) for prefix in prefixes):
            return family

    return None


# ============================================================================
# VALIDATION INTERNE
# ============================================================================

def validate_config(raise_on_duplicate_priority: bool = False) -> None:
    """
    Vérifie la cohérence interne de la configuration.

    Contrôles effectués
    -------------------
    - famille autorisée
    - priorité présente
    - report_block valide si renseigné
    - références de redondance valides
    - cohérence des alias
    - doublons éventuels de priorité
    - présence du champ calc_method sur toutes les features
    - incohérence core_hl=True + drop_if_redundant=True
    - familles autorisées non utilisées
    """
    seen_priorities: dict[int, str] = {}
    allowed_report_blocks = {"", "audio", "speech", "face", "gaze", "high-level", "highlevel", "hl"}

    for feat, cfg in INV_FEATURES.items():
        family = cfg.get("family")
        if family not in ALLOWED_FAMILIES:
            raise ValueError(f"Famille invalide pour '{feat}': {family}")

        priority = cfg.get("priority")
        if priority is None:
            raise ValueError(f"Priority manquante pour '{feat}'")
        if not isinstance(priority, int):
            raise TypeError(f"Priority non entière pour '{feat}': {priority}")

        report_block = str(cfg.get("report_block", "")).lower().strip()
        if report_block not in allowed_report_blocks:
            raise ValueError(f"report_block invalide pour '{feat}': {report_block}")

        if priority in seen_priorities:
            msg = (
                f"Priorité dupliquée {priority} pour "
                f"'{seen_priorities[priority]}' et '{feat}'"
            )
            if raise_on_duplicate_priority:
                raise ValueError(msg)
            else:
                print(f"[WARN] {msg}")
        else:
            seen_priorities[priority] = feat

        for other in cfg.get("redundant_with", []):
            if other not in INV_FEATURES:
                raise ValueError(
                    f"'{feat}' référence une feature inconnue dans redundant_with: '{other}'"
                )

        # Vérification de la présence du champ calc_method
        calc_method = cfg.get("calc_method", "")
        if not calc_method or not str(calc_method).strip():
            print(f"[WARN] calc_method manquant ou vide pour '{feat}'")

    for canonical, alias in FEATURE_ALIAS_PAIRS:
        if canonical not in INV_FEATURES:
            raise ValueError(
                f"Alias incohérent : canonique inconnue '{canonical}'"
            )
        if alias not in INV_FEATURES:
            raise ValueError(
                f"Alias incohérent : alias non documenté '{alias}'"
            )
        canon_family = INV_FEATURES[canonical]["family"]
        alias_family = INV_FEATURES[alias]["family"]
        if canon_family != alias_family:
            raise ValueError(
                f"Alias incohérent : famille différente entre '{canonical}' ({canon_family}) "
                f"et '{alias}' ({alias_family})"
            )
        # Vérification que les alias ont bien drop_if_redundant=True
        if not INV_FEATURES[alias].get("drop_if_redundant", False):
            print(
                f"[WARN] Alias '{alias}' n'a pas drop_if_redundant=True "
                f"(canonique : '{canonical}')"
            )

    # Vérification de cohérence core_hl + drop_if_redundant
    for feat, cfg in INV_FEATURES.items():
        if cfg.get("core_hl", False) and cfg.get("drop_if_redundant", False):
            msg = (
                f"Incohérence : '{feat}' est core_hl=True ET drop_if_redundant=True. "
                f"Une feature marquée pour suppression ne doit pas figurer dans CORE_HL."
            )
            if raise_on_duplicate_priority:
                raise ValueError(msg)
            else:
                print(f"[WARN] {msg}")

    # Avertir si une feature destinée à la régression est aussi marquée
    # comme candidate préférentielle à la suppression en cas de redondance.
    for feat, cfg in INV_FEATURES.items():
        if cfg.get("regression_preferred", False) and cfg.get("drop_if_redundant", False):
            print(
                f"[WARN] '{feat}' est regression_preferred=True mais "
                f"drop_if_redundant=True — incohérence potentielle"
            )

    # Avertir si deux features documentées comme redondantes sont toutes deux
    # prioritaires pour la régression v2.
    for feat, cfg in INV_FEATURES.items():
        if not cfg.get("regression_preferred", False):
            continue
        for other in cfg.get("redundant_with", []):
            if INV_FEATURES.get(other, {}).get("regression_preferred", False):
                print(
                    f"[WARN] '{feat}' et '{other}' sont toutes deux "
                    f"regression_preferred=True et documentées redondantes"
                )

    # Vérification des familles autorisées effectivement utilisées
    used_families = {cfg.get("family") for cfg in INV_FEATURES.values()}
    unused_families = ALLOWED_FAMILIES - used_families
    if unused_families:
        print(
            f"[INFO] Familles autorisées non utilisées dans INV_FEATURES : "
            f"{sorted(unused_families)}"
        )

    # Vérification des listes core
    for feat in CORE_AUDIO + CORE_SPEECH + CORE_FACE + CORE_GAZE + CORE_HL:
        if feat not in INV_FEATURES:
            raise ValueError(f"Feature core inconnue : '{feat}'")


validate_config()
