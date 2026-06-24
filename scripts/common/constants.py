"""
constants.py — Project-wide constants.
"""

# Participant roles in each triad
ROLES = ["calculateur", "modelisateur", "lecteur"]

# PC OpenFace role mapping (modelisateur_c -> modelisateur)
PC_ROLE_MAP = {
    "calculateur": "calculateur",
    "lecteur": "lecteur",
    "modelisateur_c": "modelisateur",
}

# Groups excluded from analysis (corrupted or incomplete data)
# Convention courante du projet :
# - on conserve `bim065`
# - on exclut `bim065_2`
EXCLUDED_GROUPS = {"bim002", "bim065_2", "bim032", "bim075"}

# Task maximum duration (seconds)
MAX_TASK_DURATION_S = 1500

# Performance metric weights
PERF_WEIGHTS = {
    "M1_consignes": 0.50,
    "M2_nombre": 0.00,
    "M3_precision": 0.50,
    "M4_temps": 0.00,
}
