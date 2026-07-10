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

Seuil de pruning corrélationnel :
- Le seuil de pruning corrélationnel EFFECTIF est fixé à 0.85
  (`REDUNDANCY_CORR_THRESHOLD`). Toute mention d'un seuil différent dans les
  rapports en aval (en-têtes annonçant 0.80, logs appliquant 0.90, etc.) est
  un défaut de synchronisation à corriger DANS LE PIPELINE APPELANT, pas ici.
  Cette valeur est l'unique source de vérité ; `validate_config()` émet un
  warning si un override incohérent est détecté (voir `assert_pruning_threshold`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================================
# HISTORIQUE — Révision post-audit pruning PCA VR (N=12)
# ----------------------------------------------------------------------------
# Cette révision applique les décisions issues de l'audit du pruning PCA
# effectué en mode --inv-analysis-mode pruning sur le scope VR-only.
#
# Décisions principales :
# 1. Seuil de pruning harmonisé à 0.85 (déclaration = exécution).
# 2. gaze_attention_coordination_idx exclu de l'espace PCA (composite
#    algébrique de constituants retenus).
# 3. Convention rate/count inversée : rate = représentant canonique.
# 4. Paire mean_pause_s / audio_backchannel_rate_per_min protégée du
#    pruning corrélationnel (artefact suspecté à N=11).
# 5. Variables near-zero variance identifiées (audit délégué au pipeline).
#
# Points non résolus (à trancher hors config) :
# - Ratio N/p final reste ~2 obs/var : réduction supplémentaire ou
#   repositionnement de la PCA en usage descriptif à discuter avec les
#   encadrants.
# - Cluster regard mutuel : arbitrage définitif du représentant unique
#   demande une inspection empirique complémentaire.
# ============================================================================


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
EXCLUDE_SUFFIXES = ("_source","_old", "_raw", "_zscore", "_zscored", "_z", "_log", "_sqrt", "_cube_root", "_median", "_raw")
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
    # révision post-audit VR N=12 — composite algébrique exclu de la PCA :
    # gaze_attention_coordination_idx = nanmean([z_gaze_convergence_ratio,
    # −1.0 × z_gaze_entropy_dir_mean]). Ses deux constituants (gaze_convergence_ratio,
    # gaze_entropy_dir_mean) sont retenus séparément après pruning ; la PCA verrait
    # alors trois fois la même information linéaire (triple comptage). Exclu de la
    # PCA/analyses prédictives — reste utilisable en descriptif et SEM via
    # CORE_HL_REPORT (la fiche est conservée dans INV_FEATURES).
    "gaze_attention_coordination_idx",

    # Audio : préférer audio_avg_speaking_turn_duration_s
    # "audio_pause_ratio",
    # "audio_floor_exchange_pause_mean_s",
    # "audio_total_speaking_turns",
    # "mean_pause_s",
    # "n_floor_exchanges",
    # # Audio : préférer audio_participation_entropy
    # "audio_turn_balance_cv",
    # "max_speech_ratio",

    # Gaze directionnelle — convention rate/count : on ne conserve dans la PCA que
    # gaze_convergence_ratio (représentant canonique du construct convergence).
    # Les dérivées count/durée absolue sont exclues car confondues avec la durée
    # d'interaction (cf. audit VR N=12) : gaze_convergence_n_episodes (count),
    # gaze_convergence_dur_total_s (durée absolue, |r|=0.852 avec le rate
    # episode_rate_per_min_ref qu'elle absorbait — on tranche pour l'exclusion,
    # le ratio normalisé restant porté par gaze_convergence_ratio), et
    # gaze_convergence_episode_density_raw (variable d'audit _raw).
    "gaze_convergence_episode_density_raw",
    "gaze_convergence_n_episodes",
    "gaze_convergence_dur_total_s",
    "log_gaze_convergence_episode_dur_mean_s",
    # "gaze_convergence_n_episodes_per_s",
    # "gaze_convergence_dur_total_ratio_ref",
    # "gaze_convergence_episode_rate_per_min_ref",
    # "gaze_attention_coordination_idx",
    # "gaze_attention_coordination_idx_old",
    # Legacy (ancienne analyse par objet)
    "shared_obj_n_episodes",
    "shared_obj_n_episodes_per_s",
    "shared_obj_dur_q25_s",
    "shared_obj_dur_total_ratio_ref",
    "shared_obj_episode_rate_per_min_ref",
    # révision post-audit VR N=12 — oubli corrigé : variable d'audit legacy
    # (analyse par objet abandonnée) qui survivait au pruning et polluait
    # l'espace analytique PCA. Exclue au même titre que ses sœurs shared_obj_*.
    "shared_obj_episode_density_raw",
    # NB — Cluster regard mutuel : l'exclusion de toutes les facettes mutual_gaze_* /
    # pair_mutual_gaze_* SAUF mutual_gaze_ratio est gérée par une RÈGLE DE PRÉFIXE
    # dans is_excluded_inv_feature() (voir MUTUAL_GAZE_KEEP_ONLY), pas par des noms
    # exacts ici (le cluster a trop de facettes pour une liste nom par nom).
})

# ----------------------------------------------------------------------------
# révision post-audit VR N=12 — Cluster regard mutuel : représentant unique
# ----------------------------------------------------------------------------
# Le regard mutuel est quasi absent en VR (ratio moyen ≈ 0.008). Le construct est
# décliné en ~6 facettes (ratio, durée moyenne/totale, épisodes, taux, agrégats
# dyadiques pair_*), faiblement corrélées entre elles (|r| < 0.85), donc non
# départageables par le seul pruning corrélationnel. Pour N=12, on impose UN seul
# représentant : mutual_gaze_ratio (directionnel VR-natif, core, regression_preferred).
# Toute autre variable dont le nom commence par un de ces préfixes est exclue de la
# PCA/analyses prédictives (elles restent dans les CSV d'audit high-level).
MUTUAL_GAZE_EXCLUDE_PREFIXES = ("mutual_gaze_", "pair_mutual_gaze_")
MUTUAL_GAZE_KEEP_ONLY = "mutual_gaze_ratio"

REGRESSION_FORCE_INCLUDE: list[str] = [
    # Remplaçants directionnels des legacy gaze_shared_visual_attention_ratio / gaze_entropy_mean_participants :
    # - gaze_convergence_ratio  (priorité 30 — équivalent directionnel de gaze_shared_visual_attention_ratio)
    # - gaze_entropy_dir_mean   (priorité 31 — équivalent directionnel de gaze_entropy_mean_participants)
    # À décommenter si le pruning les élimine à tort :
    # "gaze_convergence_ratio",
    # "gaze_entropy_dir_mean",
    # "face_negative_affect_extended_ratio",   # affect négatif élargi (AU15+AU17, AU4+AU15, AU1)
    # Legacy (ancienne analyse objet — plus utilisées) :
    # "gaze_shared_visual_attention_ratio",
    # "gaze_entropy_mean_participants",
    # "log_gaze_shared_obj_episode_dur_mean_s",
]

# ============================================================================
# CONSTANTES GÉNÉRALES
# ============================================================================

# Paires protégées du pruning — les deux features sont conservées même si |r| > seuil.
# Utiliser quand deux variables mesurent des aspects distincts malgré une corrélation élevée.
PRUNING_PROTECTED_PAIRS: frozenset[frozenset[str]] = frozenset({
    # révision post-audit VR N=12 — Cluster audio distribution/overlap :
    # audio_total_speaking_turns, audio_overlap_speaking_ratio, audio_pause_ratio,
    # audio_participation_entropy, audio_turn_balance_cv, max_speech_ratio sont
    # mutuellement corrélées (|r|=0.67–0.89 à N=12). Les protéger deux à deux
    # (surtout avec la protection PAR FEATURE) faisait survivre TOUT le cluster
    # (5+ quasi-doublons) et dégradait le ratio N/p. Protections retirées : le
    # pruning corrélationnel tranche par priorité (représentants canoniques =
    # audio_overlap_speaking_ratio prio 2, audio_total_speaking_turns prio 4).
    # Voir audit VR N=12.
    # frozenset({"audio_total_speaking_turns", "audio_overlap_speaking_ratio"}),
    # frozenset({"audio_pause_ratio", "audio_overlap_speaking_ratio"}),
    # frozenset({"audio_participation_entropy", "audio_turn_balance_cv"}),
    # frozenset({"audio_participation_entropy", "audio_overlap_speaking_ratio"}),

    # Gaze — convergence (quantité) et entropie (dispersion) : complémentaires
    frozenset({"gaze_convergence_ratio", "gaze_entropy_dir_mean"}),
    
    # # Face : affect négatif élargi vs classique — complémentaires, ne pas pruner
    frozenset({"face_negative_affect_ratio", "face_negative_affect_extended_ratio"}),
    frozenset({"affect_balance_rate", "face_facial_synchrony"}),
    
    # # Legacy (ancienne analyse objet)
    # frozenset({"gaze_entropy_dir_mean", "gaze_shared_visual_attention_ratio"}),
    # frozenset({"gaze_entropy_mean_participants", "gaze_shared_visual_attention_ratio"}),  # legacy

    # frozenset({"gaze_entropy_mean_participants", "log_gaze_shared_obj_episode_dur_mean_s"}),
    # frozenset({"gaze_shared_obj_episode_dur_mean_s", "log_gaze_shared_obj_episode_dur_mean_s"}),
    # frozenset({"gaze_shared_visual_attention_ratio", "log_gaze_shared_obj_episode_dur_mean_s"}),

    # révision post-audit VR N=12 — Corrélation empirique élevée à N=11 (r=0.928)
    # probablement artefactuelle : les deux mesurent des dimensions conversationnelles
    # conceptuellement indépendantes (silence collectif vs signaux d'écoute).
    # Protégés du pruning corrélationnel pour éviter une suppression fragile de mean_pause_s.
    # frozenset({"mean_pause_s", "audio_backchannel_rate_per_min"}),
})

REDUNDANCY_CORR_THRESHOLD = 0.85  # révision post-audit VR N=12 — seuil EFFECTIF unique (voir docstring d'en-tête)

# Optionnel : permet une évolution future par famille.
FAMILY_REDUNDANCY_THRESHOLDS: dict[str, float] = {
    "audio": REDUNDANCY_CORR_THRESHOLD,
    "face": REDUNDANCY_CORR_THRESHOLD,
    "gaze": REDUNDANCY_CORR_THRESHOLD,
    "tms": REDUNDANCY_CORR_THRESHOLD,
}

# ----------------------------------------------------------------------------
# NEAR-ZERO VARIANCE — contrôle empirique délégué au pipeline
# ----------------------------------------------------------------------------
# révision post-audit VR N=12 :
# Certaines features restent conceptuellement valides (susceptibles de varier
# sur un autre corpus) mais présentent une variance quasi nulle sur le dataset
# VR courant (N=12). Le fichier de config NE CONNAÎT PAS les données : il expose
# uniquement un seuil et une liste de candidates. Le contrôle effectif (calcul du
# CV réel puis exclusion éventuelle) est délégué au pipeline en aval, qui dispose
# des séries observées.
NEAR_ZERO_VARIANCE_CV_THRESHOLD = 0.05  # révision post-audit VR N=12 — CV < 5 % → variance vraiment plate.
# NB : abaissé de 0.10 à 0.05 pour ne pas écarter des variables core/SEM à variabilité
# faible mais non négligeable (gaze_entropy_dir_mean CV≈0.096, audio_participation_entropy
# CV≈0.063). Ces dernières retombent, si redondantes, sous le pruning corrélationnel normal.

# Candidates identifiées lors du diagnostic VR N=12 (audit délégué au pipeline) :
# - face_negative_affect_extended_ratio : SD=0.002, moyenne=0.05 (CV ~4 %).
# - gaze_convergence_mean_angle_deg      : bornée par le seuil de convergence à
#   20°, plage empirique 12.4°–14.0° (SD=0.53°).
# Ces fiches restent inchangées dans INV_FEATURES ; seul le contrôle empirique
# en aval peut décider de les écarter sur ce corpus.
NEAR_ZERO_VARIANCE_CANDIDATES: frozenset[str] = frozenset({
    "face_negative_affect_extended_ratio",
    "gaze_convergence_mean_angle_deg",
})


def is_near_zero_variance(
    series: "pd.Series",
    threshold: float = NEAR_ZERO_VARIANCE_CV_THRESHOLD,
) -> bool:
    """
    Retourne True si la série a un coefficient de variation < seuil (variance
    quasi nulle). Utilitaire délégué au pipeline en aval : la config n'a pas
    accès aux données, l'appelant fournit la série observée.

    CV = |SD / moyenne|. Retourne False si la série est vide, entièrement NaN,
    ou de moyenne quasi nulle (CV indéfini → non discriminant).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    mean = float(s.mean())
    if abs(mean) < 1e-12:
        return False
    cv = abs(float(s.std(ddof=1)) / mean)
    return cv < threshold


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

# NB (revue, log_gaze) : EXCLUDE_SUFFIXES/EXCLUDE_PREFIXES sont définis une seule
# fois, ligne 100-101. Une redéfinition étroite (seulement "z_"/"_source") existait
# ici et écrasait silencieusement la version large (incluant "log_", "sqrt_",
# "cube_root_") importée par pca_inv.py — log_gaze_shared_obj_episode_dur_mean_s
# et consorts n'étaient donc PAS exclus de la PCA malgré la doc du catalogue INV
# qui l'affirmait. Supprimée pour lever la contradiction inter-rapports.

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
    # révision post-audit VR N=12 — Convention rate/count :
    # dans toute paire (rate_per_min, count_absolute), le rate est le représentant
    # canonique (priorité plus basse). Les counts absolus sont conservés pour audit
    # mais marqués drop_if_redundant=True. Justification : à N=12 avec des durées
    # d'interaction variables entre groupes, un count brut est confondu avec la durée ;
    # la rate normalisée est le représentant défendable.
    #
    # L'activation effective des suppressions documentées (drop_if_redundant=True)
    # dépend du seuil de pruning (0.85, cf. REDUNDANCY_CORR_THRESHOLD) et de la
    # présence de la feature canonique dans l'espace analytique. Voir audit VR N=12.
    #
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
            "+ z_gaze_entropy_dir_mean(−0.10) "
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
        "redundant_with": ["interruptions_rate_per_min"],  # révision post-audit VR N=12
        "drop_if_redundant": True,  # révision post-audit VR N=12 — count brut confondu avec la durée ; préférer le rate
        "description": "Nombre de tentatives d'interruptions overlap-based",
        "calc_method": (
            "Compte brut des tentatives d'interruption overlap-based "
            "(min_overlap=0.1 s, min_post_takeover=0.5 s) "
            "— analyze_audio.py"
        ),
        "reason": (
            "Compte brut utile pour audit ; la version normalisée canonique est "
            "interruptions_rate_per_min. révision post-audit VR N=12 : count absolu "
            "confondu avec la durée d'interaction (N=12), marqué drop_if_redundant=True."
        ),
    },
    "n_successful_interruptions": {
        "family": "audio",
        "priority": 94,
        "core": False,
        "core_hl": False,
        "redundant_with": ["audio_successful_interruption_ratio"],  # révision post-audit VR N=12
        "drop_if_redundant": True,  # révision post-audit VR N=12 — count brut confondu avec la durée ; préférer le ratio
        "description": "Nombre d'interruptions réussies overlap-based",
        "calc_method": (
            "Compte brut des interruptions réussies : tentative overlap-based "
            "ayant abouti à une prise de tour confirmée "
            "— analyze_audio.py"
        ),
        "reason": (
            "Compte brut utile pour audit ; le représentant canonique normalisé est "
            "audio_successful_interruption_ratio. révision post-audit VR N=12 : count "
            "absolu confondu avec la durée d'interaction (N=12), marqué drop_if_redundant=True."
        ),
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
        "priority": 8,  # révision post-audit VR N=12 — 99 → 8 : rate = représentant canonique (à côté de audio_successful_interruption_ratio)
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
            "distinct des rapid_floor_takeover_*. révision post-audit VR N=12 : "
            "promu représentant canonique de la paire (rate, count) — n_attempted_interruptions "
            "est désormais drop_if_redundant=True."
        ),
    },
    "audio_participation_entropy": {
        "family": "audio",
        "priority": 3,
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
    # révision post-audit VR N=12 — Cluster distribution de parole (résultats empiriques) :
    # - audio_turn_balance_cv : |r|=0.856 avec audio_participation_entropy (> seuil 0.85)
    #   → empiriquement redondante, supprimée par le pruning. Fiche correcte
    #     (redundant_with=["audio_participation_entropy"]).
    # - max_speech_ratio : |r|=0.389 avec audio_participation_entropy (<< seuil)
    #   → NON redondante, survit légitimement. Sa fiche a été corrigée
    #     (redundant_with vidé, drop_if_redundant=False) — l'ancienne mention de
    #     redondance était factuellement fausse et créait un écart config/rapport.
    # audio_participation_entropy est le représentant canonique du cluster (protégé).
    "max_speech_ratio": {
        "family": "audio",
        "priority": 11,
        "core": False,
        "core_hl": False,
        # révision post-audit VR N=12 — redondance documentée retirée :
        # empiriquement |r|(max_speech_ratio, audio_participation_entropy) = 0.389
        # sur le corpus VR (N=12), bien SOUS le seuil 0.85. La feature n'est donc
        # PAS redondante et survit légitimement au pruning. Ancienne fiche
        # (redundant_with=["participation_entropy"], drop_if_redundant=True) était
        # factuellement fausse : corrigée pour éviter l'écart config/rapport.
        "redundant_with": [],
        "drop_if_redundant": False,
        "reference": "Contribution originale (pas de référence directe)",
        "description": "Ratio de parole du participant le plus actif",
        "calc_method": (
            "max des speech_ratio par rôle (speech_s_role / total_speech_s) "
            "— compute_audio_features.py"
        ),
        "reason": (
            "Capte le déséquilibre extrême (dominance d'un locuteur), distinct de "
            "audio_participation_entropy (distribution globale) : |r|≈0.39 à N=12, "
            "non redondante empiriquement."
        ),
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
    "mean_pause_s": {
        "family": "audio",
        "priority": 58,  # révision post-audit VR N=12 — fiche ajoutée (requise par PRUNING_PROTECTED_PAIRS)
        "core": False,
        "core_hl": False,
        "redundant_with": [],
        "drop_if_redundant": False,
        "description": "Durée moyenne des pauses collectives (silence à 0 locuteur)",
        "calc_method": (
            "Moyenne des durées des intervalles de silence collectif (0 locuteur actif) "
            "sur l'ensemble de l'interaction — compute_audio_features.py"
        ),
        "reason": (
            "révision post-audit VR N=12 : fiche ajoutée car mean_pause_s figure dans "
            "PRUNING_PROTECTED_PAIRS (paire protégée avec audio_backchannel_rate_per_min, "
            "r=0.928 à N=11 jugé artefactuel). Distincte de audio_floor_exchange_pause_mean_s "
            "(pause moyenne d'inter-tour) et de audio_pause_ratio (ratio global de silence) : "
            "capte la durée typique d'un temps mort collectif."
        ),
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
        "description": "Engagement expressif bas du visage (AU15+AU17 co-actifs)",
        "calc_method": (
            "Source : au15_au17_coactive_pct_mean — proportion de frames où AU15 "
            "(depressor anguli oris) et AU17 (chin raiser) sont co-actifs. "
            "Requalification (audit) : indicateur d'ENGAGEMENT EXPRESSIF de la partie "
            "basse du visage, et NON un marqueur de tristesse à part entière — la "
            "co-activation AU15+AU17 est une condition nécessaire mais non suffisante "
            "de la tristesse FACS (Ekman & Friesen 1978), sans les AU oculaires. "
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
    "face_negative_affect_extended_ratio": {
        "family": "face",
        "priority": 23,
        "core": True,
        "core_hl": True,
        "redundant_with": ["face_negative_affect_ratio"],
        "drop_if_redundant": False,
        "regression_preferred": False,
        "report_preferred": True,
        "reference": "Ekman & Friesen (1978) — Facial Action Coding System",
        "description": "Affect négatif élargi (AU15+AU17, AU4+AU15, AU1 — nanmean)",
        "calc_method": (
            "MOD-12 (2026-06) : nanmean(au15_au17_coactive_pct_mean, au4_au15_coactive_pct_mean, "
            "au1_active_pct_mean). Intègre le marqueur de tristesse classique (AU15+AU17), "
            "la composante frontale-labiale (AU4+AU15, BrowLowerer+LipCornerDepressor) "
            "et le releveur interne du sourcil (AU1, InnerBrowRaiser). "
            "Source tracée dans face_negative_affect_extended_ratio_source "
            "— compute_high_level_features.py"
        ),
        "reason": (
            "Complément de face_negative_affect_ratio : version élargie incluant AU4 et AU1 "
            "pour capter la composante frontale de la détresse (sourcils froncés + relevés internes). "
            "Non redondant avec face_negative_affect_ratio (r attendu ~0.7–0.9 selon corpus). "
            "drop_if_redundant=False : à conserver en parallèle pour comparaison prédictive."
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
        "reference": "analyze_gaze.py",
        "description": "Ratio de temps où ≥2 participants regardent dans la même direction (< 20°)",
        "calc_method": (
            "Fraction du temps où au moins 2 participants ont un angle inter-direction < 20° "
            "— analyze_gaze.py (remplace shared_obj_ratio)"
        ),
    },
    "gaze_convergence_n_episodes": {
        "family": "gaze",
        "priority": 50,
        "core": False,
        "core_hl": False,
        # révision post-audit VR N=12 — count absolu confondu avec la durée ;
        # représentant canonique = gaze_convergence_episode_rate_per_min_ref (rate normalisé).
        "redundant_with": ["gaze_convergence_episode_rate_per_min_ref", "shared_obj_n_episodes"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Nombre d'épisodes de convergence directionnelle",
        "calc_method": "Comptage des épisodes où gaze_convergence_ratio > 0 — analyze_gaze.py",
        "reason": (
            "révision post-audit VR N=12 : count absolu confondu avec la durée d'interaction ; "
            "préférer gaze_convergence_episode_rate_per_min_ref (rate normalisé)."
        ),
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
        "calc_method": "Somme des durées d'épisodes de convergence — analyze_gaze.py",
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
        "calc_method": "Moyenne des angles inter-directions lors des épisodes < 20° — analyze_gaze.py",
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
        "reference": "analyze_gaze.py",
        "description": "Ratio de regard mutuel directionnel (A→tête B ET B→tête A simultanément)",
        "calc_method": (
            "Fraction du temps où l'angle entre DirCorr_A et (pos_B - pos_A) < 30° "
            "ET l'angle entre DirCorr_B et (pos_A - pos_B) < 30° — analyze_gaze.py"
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
        "calc_method": "Comptage des épisodes de regard mutuel — analyze_gaze.py",
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
        "calc_method": "Somme des durées de regard mutuel — analyze_gaze.py",
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
        "reference": "analyze_gaze.py",
        "description": "Entropie directionnelle moyenne (dispersion des azimuts en 16 bins de 22.5°)",
        "calc_method": (
            "Entropie de Shannon normalisée sur la distribution des azimuts (projection XZ) "
            "en 16 bins — analyze_gaze.py (remplace gaze_entropy_mean_participants)"
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
        "reference": "analyze_gaze.py",
        "description": "log1p(durée TOTALE de convergence directionnelle, s) — exclue de la PCA (durée absolue)",
        "calc_method": (
            "log1p(gaze_convergence_dur_total_s) — compute_high_level_features.py. "
            "⚠ Le nom '..._episode_dur_mean_s' est TROMPEUR : la valeur calculée est la "
            "durée TOTALE de convergence (gaze_convergence_dur_total_s), et NON une durée "
            "moyenne par épisode. Le nom est un artefact hérité de l'ancienne pipeline objet "
            "(log_gaze_shared_obj_episode_dur_mean_s) ; conservé pour ne pas casser les CSV "
            "existants, mais description et transformation alignées ici sur le calcul réel. "
            "Exclue de la PCA (durée absolue, cf. _EXCLUDED_EXACT_NAMES / convention rate/count)."
        ),
        "reason": (
            "Distribution brute potentiellement asymétrique ; log-transform conservé par symétrie "
            "avec l'ancienne pipeline. Nom hérité trompeur (durée totale, pas moyenne par épisode) — "
            "voir calc_method."
        ),
    },
    "gaze_convergence_episode_rate_per_min_ref": {
        "family": "gaze",
        # révision post-audit VR N=12 — 53 → 32 : rate = représentant canonique de la
        # paire (rate, count) face à gaze_convergence_n_episodes (juste après gaze_convergence_ratio=30).
        # NB : priorité 32 partagée avec mutual_gaze_ratio (doublon assumé — familles gaze
        # distinctes ; validate_config émet un [WARN] non bloquant).
        "priority": 32,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_convergence_ratio"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Taux d'épisodes de convergence par minute (normalisé par durée de référence)",
        "calc_method": "gaze_convergence_n_episodes / (interaction_dur_s_ref / 60) — compute_high_level_features.py",
        "reason": (
            "Redondant avec gaze_convergence_ratio à durée fixée ; préférer gaze_convergence_ratio. "
            "révision post-audit VR N=12 : promu représentant canonique face au count brut "
            "gaze_convergence_n_episodes (rate normalisé préféré au count absolu)."
        ),
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
        "priority": 91,
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
        "priority": 92,
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
        "priority": 93,
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
            "corrélé avec gaze_attention_coordination_idx (r=0.709) et gaze_entropy_dir_mean (r=-0.736) "
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
        "redundant_with": ["gaze_convergence_ratio", "gaze_entropy_dir_mean"],
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
        # révision post-audit VR N=12 — Exclu de la PCA/analyses prédictives via
        # _EXCLUDED_EXACT_NAMES : composite algébrique de gaze_convergence_ratio et
        # gaze_entropy_dir_mean, tous deux retenus séparément — évite le triple
        # comptage dans l'espace PCA. Reste utilisable en descriptif et SEM via
        # CORE_HL_REPORT (fiche conservée intentionnellement, redundant_with mis à jour).
        "reason": (
            "Composite algébrique de gaze_convergence_ratio (poids +0.5) et "
            "gaze_entropy_dir_mean (poids −0.5). Exclu de la PCA car ses deux "
            "constituants survivent au pruning ; conservé pour la SEM et le rapport descriptif."
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
        "priority": 94,
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
        "reason": "Ancien indicateur INV_TAS (SEM legacy) ; remplacé par gaze_convergence_ratio dans INV_TAS MOD-11. core_hl=True conservé pour propagation dans merged_master.",
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
        "redundant_with": ["mutual_gaze_ratio"],  # révision post-audit VR N=12
        "drop_if_redundant": True,  # révision post-audit VR N=12 — facette secondaire du cluster regard mutuel
        "description": "Durée moyenne des épisodes de regard mutuel au niveau dyadique",
        "calc_method": (
            "Moyenne inter-paires des durées moyennes des épisodes de regard mutuel "
            "(mutual_gaze_dur_total_s / mutual_gaze_n_episodes, agrégé par paire)"
        ),
        "reason": (
            "révision post-audit VR N=12 : facette secondaire d'un cluster regard mutuel "
            "quasi absent en VR (ratio moyen ≈ 0.008). Exclue de la PCA via "
            "_EXCLUDED_EXACT_NAMES ; représentant unique conservé = mutual_gaze_ratio."
        ),
    },
    # révision post-audit VR N=12 — pair_convergence_ratio_mean : agrégat dyadique
    # de gaze_convergence_ratio, déjà documenté drop_if_redundant=True (fiche correcte).
    # L'activation effective de la suppression dépend du seuil de pruning (0.85) et de
    # la présence de gaze_convergence_ratio (canonique niveau groupe) dans l'espace
    # analytique. Voir audit VR N=12.
    "pair_convergence_ratio_mean": {
        "family": "gaze",
        "priority": 44,
        "core": False,
        "core_hl": False,
        "redundant_with": ["gaze_convergence_ratio"],
        "drop_if_redundant": True,
        "regression_preferred": False,
        "report_preferred": False,
        "description": "Ratio de convergence directionnelle moyen par paire (agrégat dyadique de gaze_convergence_ratio)",
        "calc_method": (
            "Moyenne inter-paires du ratio de convergence directionnelle "
            "(proportion du temps où les deux membres de la paire regardent dans une direction "
            "< 20° d'écart) — compute_high_level_features.py"
        ),
        "reason": "Mesure dyadique complémentaire à gaze_convergence_ratio (niveau groupe)",
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
    # révision post-audit VR N=12 — Cluster regard mutuel :
    # mutual_gaze_dur_total_ratio_ref, mutual_gaze_episode_rate_per_min_ref et
    # pair_mutual_gaze_dur_mean_s_mean mesurent des facettes redondantes du même
    # regard mutuel. Représentant unique recommandé : mutual_gaze_ratio (directionnel
    # VR-natif). L'arbitrage définitif du représentant unique demande une inspection
    # empirique complémentaire (cf. « Points non résolus » de l'HISTORIQUE) ; les fiches
    # restent donc inchangées ici (drop_if_redundant non forcé). L'activation d'une
    # éventuelle suppression dépendra du seuil de pruning (0.85) et de la présence de
    # mutual_gaze_ratio dans l'espace analytique. Voir audit VR N=12.
    "mutual_gaze_episode_rate_per_min_ref": {
        "family": "gaze",
        "priority": 48,
        "core": False,
        "core_hl": False,
        # révision post-audit VR N=12 — facette secondaire du cluster regard mutuel
        "redundant_with": ["mutual_gaze_ratio", "mutual_gaze_n_episodes_per_s", "mutual_gaze_n_episodes_sum_pairs"],
        "drop_if_redundant": True,  # révision post-audit VR N=12
        "description": "Taux d'épisodes de regard mutuel par minute de référence",
        "calc_method": (
            "Nombre total d'épisodes de regard mutuel rapporté à une durée de référence "
            "normalisée en minutes (mutual_gaze_n_episodes_sum_pairs / duration_ref_min)"
        ),
        "reason": (
            "révision post-audit VR N=12 : facette secondaire d'un cluster regard mutuel "
            "quasi absent en VR (ratio moyen ≈ 0.008). Exclue de la PCA via "
            "_EXCLUDED_EXACT_NAMES ; représentant unique conservé = mutual_gaze_ratio."
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
    # révision post-audit VR N=12 — Cluster regard mutuel : représentant unique.
    # Exclure toute facette mutual_gaze_* / pair_mutual_gaze_* SAUF mutual_gaze_ratio.
    if low != MUTUAL_GAZE_KEEP_ONLY and any(
        low.startswith(p) for p in MUTUAL_GAZE_EXCLUDE_PREFIXES
    ):
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


# ============================================================================
# ESPACE DE FEATURES GELÉ (source unique pour TOUTES les strates VR)
# ----------------------------------------------------------------------------
# Point 1 (audit) : stepwise, Spearman-FDR, composites SEM, MLM et PCA doivent
# tourner sur le MÊME espace de features. Cet espace est dérivé du pruning PCA
# VR-only (inv_pruned_features.csv, kept=1) au chargement, puis filtré par les
# règles d'exclusion INV. Toute strate qui construit son propre espace introduit
# une divergence à proscrire : elle doit passer par get_frozen_inv_feature_space().
# ============================================================================

# Sous-dossier canonique du pruning VR-only (mode with_pruning).
FROZEN_INV_SUBDIR = "results_inv_structure_vr_only/with_pruning"


def get_frozen_inv_feature_space(
    results_dir: "Any",
    inv_subdir: str = FROZEN_INV_SUBDIR,
) -> list[str]:
    """Espace de features gelé = features kept=1 du pruning PCA VR-only.

    Dérivé du CSV au chargement (cohérent avec le dernier pruning), puis filtré
    par filter_inv_feature_names(). Retourne [] si le CSV est absent (l'appelant
    doit alors décider d'un fallback explicite plutôt que de fabriquer un espace ad hoc).

    C'est l'UNIQUE source de vérité de l'espace analytique VR : toutes les strates
    (stepwise, Spearman-FDR, composites, MLM, PCA régression) doivent l'utiliser.
    """
    from pathlib import Path as _Path

    csv_path = _Path(results_dir) / inv_subdir / "inv_pruned_features.csv"
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if df is None or df.empty or "kept" not in df.columns or "feature" not in df.columns:
        return []
    kept = df.loc[df["kept"] == 1, "feature"].astype(str).tolist()
    return filter_inv_feature_names(list(dict.fromkeys(kept)))


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

def assert_pruning_threshold(override: float | None) -> None:
    """
    Vérifie qu'un seuil de pruning passé par un pipeline en aval correspond au
    seuil canonique `REDUNDANCY_CORR_THRESHOLD` (0.85).

    À appeler depuis le pipeline appelant (analyze_inv_structure.py, rapport v2…)
    quand un seuil est passé en argument. Émet un warning non bloquant si l'override
    diffère du seuil canonique — un tel écart est un défaut de synchronisation à
    corriger dans l'appelant, pas dans la config.
    """
    if override is None:
        return
    if abs(float(override) - REDUNDANCY_CORR_THRESHOLD) > 1e-9:
        print(
            f"[WARN] Seuil de pruning override={override} != seuil canonique "
            f"REDUNDANCY_CORR_THRESHOLD={REDUNDANCY_CORR_THRESHOLD}. "
            f"Defaut de synchronisation a corriger dans le pipeline appelant."
        )


def _parse_nanmean_constituents(calc_method: str) -> list[str]:
    """
    Extrait les constituants d'un composite décrit comme
    nanmean([z_A, ±k × z_B, ...]) dans son champ calc_method.

    Retourne les noms de features sous-jacentes (sans préfixe z_ ni coefficient).
    Retourne [] si le calc_method ne correspond pas au motif nanmean([...]).
    """
    import re

    text = str(calc_method or "")
    if "nanmean(" not in text.replace(" ", "").lower():
        return []
    # Isoler le contenu du premier nanmean([...])
    m = re.search(r"nanmean\(\s*\[(.*?)\]", text, flags=re.DOTALL)
    if not m:
        return []
    inner = m.group(1)
    constituents: list[str] = []
    # Chaque token de la forme (optionnel ± coeff ×) z_<nom>
    for tok in re.findall(r"z_([A-Za-z0-9_]+)", inner):
        name = tok.strip()
        if name and name not in constituents:
            constituents.append(name)
    return constituents


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

    # ------------------------------------------------------------------
    # révision post-audit VR N=12 — Contrôle 1 :
    # Composites nanmean([z_A, z_B, ...]) dont TOUS les constituants figurent
    # dans le core PCA (CORE_HL ou le CORE_* de la même famille) → warning
    # « composite algébriquement redondant ». Un tel composite fait voir à la
    # PCA plusieurs fois la même information linéaire ; il devrait être exclu de
    # l'espace PCA (via _EXCLUDED_EXACT_NAMES) tout en restant utilisable en SEM.
    #
    # Généralisation possible (non implémentée ici) : appliquer le même contrôle
    # à toute forme de composite linéaire (somme pondérée z-scorée, pas seulement
    # nanmean), en parsant les coefficients depuis calc_method.
    _core_by_family = {
        "audio": set(CORE_AUDIO) | set(CORE_SPEECH),
        "face": set(CORE_FACE),
        "gaze": set(CORE_GAZE),
    }
    _core_all = set(CORE_HL) | set(CORE_AUDIO) | set(CORE_SPEECH) | set(CORE_FACE) | set(CORE_GAZE)
    for feat, cfg in INV_FEATURES.items():
        constituents = _parse_nanmean_constituents(cfg.get("calc_method", ""))
        if len(constituents) < 2:
            continue
        # Ne considérer que les constituants réellement documentés comme features.
        known = [c for c in constituents if c in INV_FEATURES]
        if len(known) < 2:
            continue
        family = cfg.get("family", "")
        core_scope = _core_by_family.get(family, set()) | set(CORE_HL)
        if all(c in core_scope or c in _core_all for c in known):
            # Ne pas alerter si le composite est déjà exclu de la PCA.
            if not is_excluded_inv_feature(feat):
                print(
                    f"[WARN] '{feat}' est un composite algébriquement redondant : "
                    f"tous ses constituants {known} figurent dans le core PCA. "
                    f"Envisager de l'exclure de la PCA (via _EXCLUDED_EXACT_NAMES)."
                )

    # ------------------------------------------------------------------
    # révision post-audit VR N=12 — Contrôle 2 :
    # Paires (name_rate_per_min, name_absolute) où le count a une priorité
    # INFÉRIEURE (donc « préférée ») au rate → warning suggérant l'inversion.
    # Convention : le rate normalisé est le représentant canonique.
    # Le suffixe rate est retiré pour obtenir un « stem » (radical) ; on n'apparie
    # qu'un count partageant ce stem complet comme préfixe (évite les faux positifs
    # legacy à radical court, ex. shared_obj_*).
    _rate_suffixes = (
        "_episode_rate_per_min_ref",
        "_rate_per_min_ref",
        "_rate_per_min",
    )
    for feat, cfg in INV_FEATURES.items():
        matched_suffix = next((s for s in _rate_suffixes if feat.endswith(s)), None)
        if matched_suffix is None:
            continue
        stem = feat[: -len(matched_suffix)]
        if not stem:
            continue
        rate_prio = cfg.get("priority", 999)
        for count_name, count_cfg in INV_FEATURES.items():
            if count_name == feat:
                continue
            is_count = (
                count_name.startswith("n_")
                or "_n_episodes" in count_name
                or count_name.endswith("_total")
            )
            if not is_count:
                continue
            # Rapprochement strict : le count doit partager le stem complet du rate
            # (même radical de mesure), sinon on ignore.
            if not count_name.startswith(stem):
                continue
            count_prio = count_cfg.get("priority", 999)
            if count_prio < rate_prio:
                print(
                    f"[WARN] Paire rate/count '{feat}' (prio {rate_prio}) vs "
                    f"'{count_name}' (prio {count_prio}) : le count absolu est prioritaire "
                    f"sur le rate — envisager l'inversion (rate = représentant canonique)."
                )

    # ------------------------------------------------------------------
    # révision post-audit VR N=12 — Contrôle 3 :
    # Toute feature figurant dans PRUNING_PROTECTED_PAIRS doit exister dans
    # INV_FEATURES (une paire protégée référençant une feature fantôme est une erreur).
    for pair in PRUNING_PROTECTED_PAIRS:
        for member in pair:
            if member not in INV_FEATURES:
                raise ValueError(
                    f"PRUNING_PROTECTED_PAIRS référence une feature absente de "
                    f"INV_FEATURES : '{member}'"
                )

    # Vérification des listes core
    for feat in CORE_AUDIO + CORE_SPEECH + CORE_FACE + CORE_GAZE + CORE_HL:
        if feat not in INV_FEATURES:
            raise ValueError(f"Feature core inconnue : '{feat}'")


validate_config()
