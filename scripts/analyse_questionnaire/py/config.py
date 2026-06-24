#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — Constantes pour l'analyse des questionnaires TMS / Cohésion.
"""

# --- Labels dimensions ---
DIMENSION_LABELS = {
    "COR": "Coordination (Lewis, 2003)",
    "CRE": "Credibilité (Lewis, 2003)",
    "SPE": "Specialisation (Lewis, 2003)",
    "SOC": "Social Cohesion (Sassier-Roublin et al., 2025)",
    "TSK": "Task Cohesion (Sassier-Roublin et al., 2025)",
    "COM": "Communication (Sassier-Roublin et al., 2025)",
}

# --- Likert 9 niveaux ---
ALL_LEVELS = [
    "Fortement en désaccord",
    "En désaccord",
    "Plutôt en désaccord",
    "Légèrement en désaccord",
    "Neutre",
    "Légèrement en accord",
    "Plutôt en accord",
    "En accord",
    "Fortement en accord",
]
CODE_MAP = {label: i + 1 for i, label in enumerate(ALL_LEVELS)}

# --- Items à inverser (préfixes courts) ---
INVERT_SHORT = {
    "G1Q00001.COR03",
    "G1Q00001.COR05",
    "G1Q00001.CRE04",
    "G1Q00001.CRE02",
}

# --- Seuils ---
RDROP_THRESHOLD = 0.20
SEUIL_TEST_MU = 5
# Seuil d'acceptabilité de l'alpha de Cronbach.
# La suppression exploratoire n'est déclenchée que pour les dimensions
# dont l'alpha initial est inférieur à ce seuil (valeur conventionnelle : 0.70).
ALPHA_ACCEPTABILITY_THRESHOLD = 0.70
