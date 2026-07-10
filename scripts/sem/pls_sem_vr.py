# -*- coding: utf-8 -*-
"""
PLS-SEM exploratoire sur le sous-echantillon VR.

Ce module implemente une approximation transparente de PLS-SEM adaptee au
faible N du projet : composites formatifs a poids egaux (z-mean), puis modeles
de chemins estimes par OLS standardise avec IC bootstrap percentile.
Il ne fournit pas de test confirmatoire d'ajustement global.

Toutes les constantes parametrant le modele (composites, cibles, chemins
indirects) sont regroupees en tete de fichier afin de permettre un ajustement
sans modifier la logique de calcul.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MPLCONFIG_DIR = _PROJECT_ROOT / ".mplconfig"
_MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIG_DIR))
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student_t

# ===========================================================================
# PARAMETRES DU MODELE — toutes les constantes configurables sont ici.
# Configuration v3 alignée sur :
#   - mapping modalité → sous-système (TAS=Gaze, TMS=Speech, TRS=Face)
#   - PCA v2 (16 features, validation empirique des composites)
#   - Résultats stepwise p<0.10 (rapport 2026-05-19)
# ===========================================================================

# ---------------------------------------------------------------------------
# [1] SWITCH GLOBAL : inclure la branche affective autonome dans la SEM
# ---------------------------------------------------------------------------
# Justification : face_negative_affect_ratio a sa propre composante PCA (PC5)
# distincte de la composante face principale (PC2 = synchronie + smile + balance).
# Empiriquement, face_negative_affect prédit SOC (β=+0.79 dans M11 trivarié),
# CRE (rho=+0.62), Cohésion globale et COM en complément.
# La branche affective autonome est conservée car ses chemins sont robustes.
INCLUDE_AFFECT: bool = True

# ---------------------------------------------------------------------------
# [2] VARIABLES DEPENDANTES ET MEDIATEURS
# ---------------------------------------------------------------------------

# Dimensions TMS (états émergents cognitifs questionnaire).
TMS_DIMS = ["COR", "CRE", "SPE"]

# Dimensions sous-jacentes de la cohésion (questionnaire).
COHESION_DIMS = ["SOC", "TSK", "COM"]

# Score de cohésion globale (questionnaire).
COHESION_SCORE = "Cohesion_questionnaire_score"

# Variable dépendante finale.
PERFORMANCE = "Score_perf_tsk"

# ---------------------------------------------------------------------------
# [3] INPUT_composite — causes racines du potentiel collectif
# ---------------------------------------------------------------------------
# Configuration test 2 validée (VIF tous < 5, loadings 0.57-0.94) :
# c_score + rme_mean + strategy_norm + effort_task_norm.
# skill_congruence_mean retiré (loading 0.12 incohérent).
# Si tu veux passer à c-factor seul, remplacer par : ROOT_CAUSES = ["c_score"]
ROOT_CAUSES = ["c_score", "rme_mean", "strategy_norm", "effort_task_norm"]

# Nom du composite INPUT.
INPUT_COMPOSITE = "INPUT_composite"

# ---------------------------------------------------------------------------
# [4] Composites INV transactifs alignés sur les sous-systèmes (modalité-pure)
# ---------------------------------------------------------------------------
# Mapping : TAS=Gaze, TMS=Speech, TRS=Face (sans face_neg). Principes :
#   1. Modalité-pure : chaque indicateur appartient à un seul composite.
#   2. Peu d'indicateurs par composite (leçon N=12 : sur-compositisation dégrade).
#   3. Uniquement des features survivantes du pruning (16-feature set).
#   4. face_negative_affect_ratio HORS composites → branche affective (voir [5]/[6]).
REFINED_INV_COMPOSITES: dict[str, list[str]] = {
    # TAS : attention partagée visuelle (gaze pur). Assumé : aucune association
    # monoprédicteur robuste — le composite documente le résultat nul TAS
    # (cohérent H2 « division silencieuse du travail »).
    "INV_TAS": [
        "gaze_convergence_ratio",     # direct
        "gaze_entropy_dir_mean",      # INVERSÉ (dispersion = − attention partagée)
    ],
    # TMS : régulation conversationnelle (speech pur). Porte le signal empirique :
    # participation_entropy ρ≈−0.68/−0.73 (cohésion/CRE), overlap ρ≈−0.61/−0.68,
    # pause_ratio ρ≈+0.62. Orientation : haut = régulation contrôlée/spécialisée.
    "INV_TMS": [
        "audio_pause_ratio",              # direct
        "audio_participation_entropy",    # INVERSÉ
        "audio_overlap_speaking_ratio",   # INVERSÉ
    ],
    # TRS : synchronie expressive positive (face pur, sans affect négatif).
    "INV_TRS": [
        "face_facial_synchrony",
        "affect_balance_rate",
        "face_smile_ratio",
    ],
}

INVERTED_COMPOSITE_INDICATORS: dict[str, set[str]] = {
    "INV_TAS": {"gaze_entropy_dir_mean"},
    "INV_TMS": {"audio_participation_entropy", "audio_overlap_speaking_ratio"},
    # INV_TRS : aucune inversion.
}
# Poids égaux (z-mean). Justification : les loadings PCA de l'ancienne config
# sont périmés (espace de features différent) et le mécanisme poids-signés ×
# INVERTED_COMPOSITE_INDICATORS produisait des doubles inversions. À N=12,
# les poids égaux sont l'option la plus défendable (pas d'estimation de poids
# possible ; cf. Kommol et al. 2025 pour la pratique équivalente).
COMPOSITE_WEIGHTS: dict[str, dict[str, float]] = {}

AFFECT_TARGETS: list[str] = ["CRE", "SOC", "TSK", "COM", COHESION_SCORE]

# ---------------------------------------------------------------------------
# [5] Branche affective (face_negative_affect_ratio comme indicateur autonome)
# ---------------------------------------------------------------------------
# Actif uniquement si INCLUDE_AFFECT = True.
# Maintien en branche autonome justifié par :
#   - PC5 PCA (variance propre distincte de PC2 face positif)
#   - Cibles différentes des autres composites INV
#   - Résultats empiriques robustes (β +0.61 à +0.79 sur 5 cibles)

AFFECT_MARKER = "face_negative_affect_ratio"

# Cibles de la branche affective (justifications stepwise, rapport 2026-05-19 :
# SOC β=+0.79, TSK β=+0.53, COM prédicteur de cohésion, COHESION β=+0.84, CRE ρ=+0.62).

# ---------------------------------------------------------------------------
# [6] Cibles structurelles par composite
# ---------------------------------------------------------------------------
# Chaque composite est testé contre l'ensemble des dimensions questionnaire + cohésion.
TAS_TARGETS: list[str] = ["COR", "CRE", "SPE", "SOC", "TSK", "COM", COHESION_SCORE]
TMS_STRUCTURAL_TARGETS: list[str] = ["COR", "CRE", "SPE", "SOC", "TSK", "COM", COHESION_SCORE]
TRS_TARGETS: list[str] = ["COR", "CRE", "SPE", "SOC", "TSK", "COM", COHESION_SCORE]


# ---------------------------------------------------------------------------
# [7] Corrélations bivariées Spearman complémentaires
# ---------------------------------------------------------------------------
# Documente la dilution des corrélations fortes (rme/c_score → gaze_attention
# ρ ~ -0.93/-0.88) dans l'agrégation INPUT_composite.
BIVARIATE_CORR_ROOTS = ROOT_CAUSES
BIVARIATE_CORR_TARGET = "INV_TMS"

# ---------------------------------------------------------------------------
# [8] Ajusteurs pour chemins directs INPUT → performance (1 médiateur/équation)
# ---------------------------------------------------------------------------
# Décomposition effet direct vs indirect via chaque médiateur.
# Chaque équation : Performance ~ INPUT_composite + médiateur.
ADJUSTED_DIRECT_MEDIATORS: list[str] = [
    "INV_TAS", "INV_TMS", "INV_TRS",
    "COR", "CRE", "SPE",
    "SOC", "TSK", "COM",
    COHESION_SCORE,
]

# ---------------------------------------------------------------------------
# [9] Ancrage théorique de chaque composite
# ---------------------------------------------------------------------------
REFINED_COMPOSITE_THEORY: dict[str, str] = {
    INPUT_COMPOSITE: (
        "Input CI : potentiel collectif et processus cognitifs/comportementaux "
        "individuels agrégés (c-factor, RME, indicateurs Riedl)"
    ),
    "INV_TAS": (
        "TAS comportemental : attention partagée visuelle "
        "(Woolley & Gupta 2024 ; opérationnalisation gaze multimodale)"
    ),
    "INV_TMS": (
        "TMS comportemental : régulation des tours et coordination "
        "conversationnelle (Lewis 2003 ; Hung & Gatica-Perez 2010)"
    ),
    "INV_TRS": (
        "TRS comportemental : synchronie cognitive et fluidité expressive "
        "(Nummenmaa et al. 2023 ; Ekman, Davidson & Friesen 1990)"
    ),
    AFFECT_MARKER: (
        "Substrat affectif autonome : engagement expressif bas du visage (AU15+AU17, "
        "condition FACS nécessaire mais non suffisante de tristesse, Ekman & Friesen 1978) "
        "— branche parallèle à la triade transactive"
    ),
}

# ---------------------------------------------------------------------------
# [10] Bootstrap pour intervalles de confiance
# ---------------------------------------------------------------------------
BOOTSTRAP_B = 5000

# ---------------------------------------------------------------------------
# [11] Analyses multiniveau complémentaires (sections 3.1.6 - 3.1.8)
# ---------------------------------------------------------------------------

# Dimensions disponibles au niveau individuel (scores par participant).
INDIVIDUAL_DIMS = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]

# Noms de colonnes dans le fichier questionnaire individuel.
INDIVIDUAL_SCORE_COL = "score"
INDIVIDUAL_DIM_COL = "dimension"
INDIVIDUAL_GROUP_COL = "Groupe"
INDIVIDUAL_MODALITE_COL = "Modalite"
INDIVIDUAL_SCENARIO_COL = "Scenario"

# Scénario(s) cible(s) pour les analyses individuelles.
# Groupes VR S1 (bim006, bim010, bim066, bim073_2) sans questionnaire S2.
INDIVIDUAL_SCENARIO_TARGETS: list[str] = ["S1", "S2"]

# Paramètres MCMC pour le MLM bayésien (section 3.1.7).
# Priors faiblement informatifs suivant Gelman et al. (2013).
MLM_MCMC_CONFIG = {
    "chains": 4,
    "iter": 4000,
    "warmup": 2000,
    "rhat_threshold": 1.01,
    "ess_threshold": 400,
    "seed": 42,  # seed d'échantillonnage MCMC fixe (reproductibilité, documenté dans la note)
}
MLM_BAYESIAN_PRIORS = {
    "fixed_effects": "normal(0, 1)",
    "random_sd": "half_cauchy(0, 1)",
}

# ---------------------------------------------------------------------------
# [12] VIF intra-composite (à calculer pour validation méthodologique)
# ---------------------------------------------------------------------------
# Reporter VIF pour tous les composites formatifs, pas seulement INPUT.
# Justification : Diamantopoulos & Winklhofer (2001) recommandent le VIF
# pour les composites formatifs.
COMPUTE_VIF_INTRA_COMPOSITE: bool = True
VIF_THRESHOLD_WARNING: float = 5.0
VIF_THRESHOLD_CRITICAL: float = 10.0

# ---------------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    """Retourne la meilleure colonne disponible pour un nom canonique."""
    candidates = [name, f"{name}_y", f"{name}_x"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _canonicalize_columns(df: pd.DataFrame, canonical_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copie les colonnes suffixees vers leurs noms canoniques et trace la disponibilite."""
    out = df.copy()
    rows: list[dict[str, Any]] = []
    for name in canonical_names:
        source = _resolve_col(out, name)
        if source is not None:
            out[name] = pd.to_numeric(out[source], errors="coerce")
            rows.append({
                "variable": name,
                "source_column": source,
                "available": True,
                "n_non_missing": int(out[name].notna().sum()),
            })
        else:
            rows.append({
                "variable": name,
                "source_column": "",
                "available": False,
                "n_non_missing": 0,
            })
    return out, pd.DataFrame(rows)


def _standardize(s: pd.Series) -> pd.Series:
    values = pd.to_numeric(s, errors="coerce")
    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd


def _indicator_sign(construct: str, indicator: str) -> float:
    return -1.0 if indicator in INVERTED_COMPOSITE_INDICATORS.get(construct, set()) else 1.0


def _orientation_label(sign: float) -> str:
    return "inverted" if sign < 0 else "direct"


def _signed_indicator_label(construct: str, indicator: str) -> str:
    return f"-{indicator}" if _indicator_sign(construct, indicator) < 0 else indicator


def _safe_ols_standardized(df: pd.DataFrame, target: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OLS sur variables standardisees. Retourne coefficients de chemin et R2."""
    cols = [target] + predictors
    sub = df[cols].dropna().copy()
    sub = sub.loc[:, sub.nunique(dropna=True) > 1]
    predictors = [p for p in predictors if p in sub.columns]
    if target not in sub.columns or len(sub) < 4 or not predictors:
        return pd.DataFrame(), {
            "target": target,
            "n": int(len(sub)),
            "r2": np.nan,
            "n_predictors": len(predictors),
            "df_resid": np.nan,
            "model": f"{target} ~ " + " + ".join(predictors),
            "warning": "modele non estimable",
        }

    y = _standardize(sub[target]).to_numpy(dtype=float)
    X_cols = [_standardize(sub[p]).to_numpy(dtype=float) for p in predictors]
    X = np.column_stack([np.ones(len(sub), dtype=float)] + X_cols)
    n, p = X.shape
    df_resid = n - p
    if df_resid < 1:
        return pd.DataFrame(), {
            "target": target,
            "n": int(n),
            "r2": np.nan,
            "n_predictors": len(predictors),
            "df_resid": int(df_resid),
            "model": f"{target} ~ " + " + ".join(predictors),
            "warning": "ddl insuffisants",
        }

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if tss <= 0 else 1.0 - rss / tss
    mse = rss / df_resid
    se = np.sqrt(np.diag(xtx_inv) * mse)
    t_values = beta / se
    p_values = 2 * student_t.sf(np.abs(t_values), df=df_resid)

    rows = []
    for i, pred in enumerate(predictors, start=1):
        rows.append({
            "source": pred,
            "target": target,
            "path_coef_std": float(beta[i]),
            "se": float(se[i]),
            "t": float(t_values[i]),
            "p": float(p_values[i]),
            "n": int(n),
            "df_resid": int(df_resid),
            "model": f"{target} ~ " + " + ".join(predictors),
            "warning": "exploratoire_N_faible" if n < 30 else "",
        })
    return pd.DataFrame(rows), {
        "target": target,
        "n": int(n),
        "r2": float(r2),
        "n_predictors": len(predictors),
        "df_resid": int(df_resid),
        "model": f"{target} ~ " + " + ".join(predictors),
        "warning": "exploratoire_N_faible" if n < 30 else "",
    }


def _make_inv_composite(df: pd.DataFrame, inv_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    z_cols = []
    for feat in inv_features:
        z_col = f"z_{feat}"
        out[z_col] = _standardize(out[feat])
        z_cols.append(z_col)

    out["INV_composite"] = out[z_cols].mean(axis=1, skipna=True)
    rows = []
    for feat, z_col in zip(inv_features, z_cols):
        valid = out[[feat, "INV_composite"]].dropna()
        loading = valid[feat].corr(valid["INV_composite"]) if len(valid) >= 3 else np.nan
        rows.append({
            "construct": "INV_composite",
            "indicator": feat,
            "mode": "formatif_equal_weight_z_mean",
            "weight": round(1.0 / len(inv_features), 4) if inv_features else np.nan,
            "loading_corr_with_composite": loading,
            "n": int(len(valid)),
        })
    return out, pd.DataFrame(rows)


def _ensure_cohesion(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    if COHESION_SCORE in out.columns and out[COHESION_SCORE].notna().any():
        return out, "observe"
    dims = [c for c in ["SOC", "TSK", "COM"] if c in out.columns]
    if len(dims) >= 2:
        out[COHESION_SCORE] = out[dims].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        return out, f"derive_mean_{'+'.join(dims)}"
    return out, "missing"


def _path_lookup(paths: pd.DataFrame, source: str, target: str) -> float:
    if paths.empty:
        return np.nan
    hit = paths[(paths["source"] == source) & (paths["target"] == target)]
    if hit.empty:
        return np.nan
    return float(hit["path_coef_std"].iloc[0])


def _make_refined_composites(
    df: pd.DataFrame,
    composite_specs: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    rows: list[dict[str, Any]] = []
    available_specs: dict[str, list[str]] = {}
    for construct, indicators in composite_specs.items():
        available = [c for c in indicators if c in out.columns and out[c].notna().sum() >= 4]
        if not available:
            continue

        # Poids bruts depuis COMPOSITE_WEIGHTS (défaut = 1.0 par indicateur)
        raw_weights = COMPOSITE_WEIGHTS.get(construct, {})
        w_raw = np.array([raw_weights.get(ind, 1.0) for ind in available], dtype=float)
        w_norm = w_raw / w_raw.sum()  # normalisation somme = 1

        z_cols = []
        for i, indicator in enumerate(available):
            z_col = f"z_{construct}_{indicator}"
            sign = _indicator_sign(construct, indicator)
            out[z_col] = sign * _standardize(out[indicator])
            z_cols.append(z_col)

        # Moyenne pondérée des z-scores (poids normalisés)
        z_mat = out[z_cols].to_numpy(dtype=float)
        out[construct] = np.nansum(z_mat * w_norm[np.newaxis, :], axis=1)
        # Mettre NaN quand tous les indicateurs sont NaN
        all_nan = np.all(np.isnan(z_mat), axis=1)
        out.loc[all_nan, construct] = np.nan

        available_specs[construct] = available
        mode_label = "formatif_weighted_z" if raw_weights else "formatif_equal_weight_z_mean"
        for i, indicator in enumerate(available):
            valid = out[[indicator, construct]].dropna()
            loading = valid[indicator].corr(valid[construct]) if len(valid) >= 3 else np.nan
            sign = _indicator_sign(construct, indicator)
            rows.append({
                "construct": construct,
                "indicator": indicator,
                "mode": mode_label,
                "orientation": _orientation_label(sign),
                "weight": round(sign * w_norm[i], 4),
                "loading_corr_with_composite": loading,
                "n": int(len(valid)),
            })
    return out, pd.DataFrame(rows), available_specs



def _compute_vif_all_composites(
    df: pd.DataFrame,
    composite_specs: dict[str, list[str]],
) -> pd.DataFrame:
    """VIF intra-composite pour tous les composites (INPUT + INV_TAS/TMS/TRS).

    Pour chaque indicateur d'un composite, calcule le VIF par rapport aux autres
    indicateurs du meme composite. VIF < 5 attendu (pas de multicollinearite
    redhibitoire entre indicateurs formatifs).

    Retourne un DataFrame avec colonnes :
        composite | indicator | n | vif | threshold | status
    """
    rows: list[dict[str, Any]] = []
    for composite, indicators in composite_specs.items():
        available = [c for c in indicators if c in df.columns]
        if len(available) < 2:
            for ind in available:
                rows.append({
                    "composite": composite,
                    "indicator": ind,
                    "n": int(df[ind].notna().sum()) if ind in df.columns else 0,
                    "vif": np.nan,
                    "threshold": 5.0,
                    "status": "VIF non estimable (< 2 indicateurs)",
                })
            continue
        sub = df[available].dropna().copy()
        for ind in available:
            others = [c for c in available if c != ind]
            if len(sub) < len(others) + 4:
                rows.append({
                    "composite": composite,
                    "indicator": ind,
                    "n": int(len(sub)),
                    "vif": np.nan,
                    "threshold": 5.0,
                    "status": "VIF non estimable (n insuffisant)",
                })
                continue
            data = sub[[ind] + others].to_numpy(dtype=float)
            _, r2 = _standardized_beta_from_arrays(data, 1)
            vif = np.nan if not np.isfinite(r2) or r2 >= 1 else 1.0 / (1.0 - r2)
            if not np.isfinite(vif):
                status = "VIF non estimable"
            elif vif > 5.0:
                status = "VIF > 5 — multicollinearite"
            else:
                status = "OK"
            rows.append({
                "composite": composite,
                "indicator": ind,
                "n": int(len(sub)),
                "vif": round(float(vif), 3) if np.isfinite(vif) else np.nan,
                "threshold": 5.0,
                "status": status,
            })
    return pd.DataFrame(rows)


def _path_key(source: str, target: str, equation: str) -> tuple[str, str, str]:
    return (source, target, equation)


def _lookup_path_boot(
    paths: pd.DataFrame,
    boots: dict[tuple[str, str, str], np.ndarray],
    source: str,
    target: str,
    contains: str | None = None,
) -> tuple[float, np.ndarray, str]:
    hit = paths[(paths["source"] == source) & (paths["target"] == target)].copy()
    if contains:
        hit = hit[hit["equation"].astype(str).str.contains(contains, regex=False)]
    if hit.empty:
        return np.nan, np.array([], dtype=float), ""
    row = hit.iloc[0]
    equation = str(row["equation"])
    return (
        float(row["path_coef_std"]),
        boots.get(_path_key(source, target, equation), np.array([], dtype=float)),
        equation,
    )


def _add_indirect_row(
    rows: list[dict[str, Any]],
    label: str,
    parts: list[tuple[float, np.ndarray]],
) -> tuple[float, np.ndarray]:
    betas = [p[0] for p in parts]
    boot_arrays = [p[1] for p in parts]
    if not all(np.isfinite(b) for b in betas) or not boot_arrays:
        rows.append({"indirect_path": label, "effect": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "warning": "non estimable"})
        return np.nan, np.array([], dtype=float)
    min_len = min(len(a) for a in boot_arrays if len(a) > 0)
    if min_len == 0:
        rows.append({"indirect_path": label, "effect": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "warning": "bootstrap non estimable"})
        return np.nan, np.array([], dtype=float)
    product = np.ones(min_len, dtype=float)
    for arr in boot_arrays:
        product *= arr[:min_len]
    effect = float(np.prod(betas))
    ci = np.nanpercentile(product, [2.5, 97.5]) if np.isfinite(product).any() else [np.nan, np.nan]
    warning_parts = ["exploratoire_N_faible"]
    if any(abs(b) > 1 for b in betas):
        warning_parts.append("beta_intermediaire_abs_sup_1_instable")
    rows.append({
        "indirect_path": label,
        "effect": effect,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "warning": "; ".join(warning_parts),
    })
    return effect, product


def _standardized_beta_from_arrays(data: np.ndarray, source_idx: int) -> tuple[float, float]:
    """Coefficient standardise OLS pour y=data[:,0], X=data[:,1:]."""
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < data.shape[1] + 2:
        return np.nan, np.nan
    arr = data.astype(float)
    if not np.isfinite(arr).all():
        return np.nan, np.nan
    means = arr.mean(axis=0)
    sds = arr.std(axis=0, ddof=1)
    if np.any(~np.isfinite(sds)) or np.any(sds == 0):
        return np.nan, np.nan
    z = (arr - means) / sds
    y = z[:, 0]
    X = np.column_stack([np.ones(len(z)), z[:, 1:]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if tss <= 0 else 1.0 - rss / tss
    return float(beta[source_idx]), float(r2)


def _bootstrap_path(
    df: pd.DataFrame,
    target: str,
    predictors: list[str],
    source: str,
    *,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estime un chemin standardise avec IC bootstrap percentile."""
    predictors = [p for p in predictors if p in df.columns]
    if source not in predictors:
        predictors = [source] + predictors
    predictors = list(dict.fromkeys(predictors))
    cols = [target] + predictors
    sub = df[[c for c in cols if c in df.columns]].dropna().copy()
    equation = f"{target} ~ " + " + ".join(predictors)
    warning_parts = ["exploratoire_N_faible"]
    if len(predictors) > 2:
        warning_parts.append("plus_de_2_predicteurs")
    if len(sub) < max(4, len(predictors) + 3):
        return {
            "source": source,
            "target": target,
            "equation": equation,
            "path_coef_std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n": int(len(sub)),
            "n_predictors": len(predictors),
            "warning": "modele non estimable; " + "; ".join(warning_parts),
        }, np.full(n_boot, np.nan)

    data = sub[cols].to_numpy(dtype=float)
    source_idx = predictors.index(source) + 1
    beta, _ = _standardized_beta_from_arrays(data, source_idx)
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan, dtype=float)
    n = len(sub)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b], _ = _standardized_beta_from_arrays(data[idx, :], source_idx)
    ci = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else [np.nan, np.nan]
    if np.isfinite(beta) and abs(beta) > 1:
        warning_parts.append("beta_abs_sup_1_instable")
    return {
        "source": source,
        "target": target,
        "equation": equation,
        "path_coef_std": beta,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "n": int(n),
        "n_predictors": len(predictors),
        "warning": "; ".join(warning_parts),
    }, boot


def _r2_for_equation(df: pd.DataFrame, target: str, predictors: list[str]) -> dict[str, Any]:
    predictors = [p for p in dict.fromkeys(predictors) if p in df.columns]
    cols = [target] + predictors
    sub = df[[c for c in cols if c in df.columns]].dropna().copy()
    equation = f"{target} ~ " + " + ".join(predictors)
    if target not in sub.columns or not predictors or len(sub) < max(4, len(predictors) + 3):
        return {
            "target": target,
            "equation": equation,
            "n": int(len(sub)),
            "r2": np.nan,
            "n_predictors": len(predictors),
            "warning": "modele non estimable; exploratoire_N_faible",
        }
    data = sub[cols].to_numpy(dtype=float)
    _, r2 = _standardized_beta_from_arrays(data, 1)
    return {
        "target": target,
        "equation": equation,
        "n": int(len(sub)),
        "r2": r2,
        "n_predictors": len(predictors),
        "warning": "exploratoire_N_faible",
    }


def _compute_bivariate_spearman(
    df: pd.DataFrame,
    roots: list[str],
    target_col: str,
    n_boot: int = BOOTSTRAP_B,
    seed: int = 99,
) -> pd.DataFrame:
    """Correlations de Spearman avec IC bootstrap percentile entre causes racines et un indicateur cible.

    Ce tableau documente la dilution des correlations individuelles tres fortes
    dans l'agregation INPUT_composite (composite formatif a poids egaux).
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for root in roots:
        if root not in df.columns or target_col not in df.columns:
            rows.append({
                "root": root,
                "target": target_col,
                "n": 0,
                "rho_spearman": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "warning": "variable indisponible",
            })
            continue
        sub = df[[root, target_col]].dropna()
        n = len(sub)
        if n < 4:
            rows.append({
                "root": root,
                "target": target_col,
                "n": n,
                "rho_spearman": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "warning": "n insuffisant",
            })
            continue
        rho, _ = spearmanr(sub[root], sub[target_col])
        arr = sub.to_numpy(dtype=float)
        boot = np.full(n_boot, np.nan, dtype=float)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            s = arr[idx, :]
            if np.std(s[:, 0], ddof=1) == 0 or np.std(s[:, 1], ddof=1) == 0:
                continue
            r, _ = spearmanr(s[:, 0], s[:, 1])
            boot[b] = r
        ci = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else [np.nan, np.nan]
        rows.append({
            "root": root,
            "target": target_col,
            "n": n,
            "rho_spearman": round(float(rho), 4),
            "ci95_low": round(float(ci[0]), 4),
            "ci95_high": round(float(ci[1]), 4),
            "warning": "exploratoire_N_faible" if n < 30 else "",
        })
    return pd.DataFrame(rows)


def _plot_refined_path_diagram(paths: pd.DataFrame, outpath: Path) -> bool:
    """Trace les chemins structurels robustes avec etiquettes beta, N, et noeuds isoles grisés.

    Ameliorations v2 :
    - Etiquettes beta sur chaque fleche (signe + valeur arrondie)
    - N par chemin indique sous le beta
    - Noeuds sans aucun chemin robuste en gris avec annotation "(aucun chemin robuste)"
    - SOC, TSK, COM en trois noeuds distincts
    - Legende methodologique complete
    """
    if paths is None or paths.empty:
        return False
    plot_df = paths.copy()
    for col in ["path_coef_std", "ci95_low", "ci95_high"]:
        plot_df[col] = pd.to_numeric(plot_df.get(col), errors="coerce")

    # Chemins robustes : IC 95% excluant zero, type "structure" uniquement.
    robust = plot_df[
        (plot_df.get("path_type", "") == "structure")
        & plot_df["ci95_low"].notna()
        & plot_df["ci95_high"].notna()
        & ((plot_df["ci95_low"] > 0) | (plot_df["ci95_high"] < 0))
    ].copy()

    # Tendances : |beta| >= 0.40 mais IC chevauche zero, type "structure".
    TENDANCE_BETA_MIN = 0.40
    struct_df = plot_df[
        (plot_df.get("path_type", "") == "structure")
        & plot_df["ci95_low"].notna()
        & plot_df["ci95_high"].notna()
        & plot_df["path_coef_std"].notna()
    ].copy()
    tendance = struct_df[
        ~((struct_df["ci95_low"] > 0) | (struct_df["ci95_high"] < 0))
        & (struct_df["path_coef_std"].abs() >= TENDANCE_BETA_MIN)
    ].copy()

    # Noeuds impliques dans au moins un chemin robuste OU en tendance.
    active_sources = set(robust["source"].astype(str)) | set(tendance["source"].astype(str))
    active_targets = set(robust["target"].astype(str)) | set(tendance["target"].astype(str))
    active_nodes = active_sources | active_targets

    # Disposition des noeuds — adaptee selon INCLUDE_AFFECT.
    if INCLUDE_AFFECT:
        nodes: dict[str, tuple[float, float]] = {
            INPUT_COMPOSITE:  (0.07, 0.54),
            "INV_TAS":        (0.28, 0.86),
            "INV_TMS":        (0.28, 0.64),
            "INV_TRS":        (0.28, 0.42),
            AFFECT_MARKER:    (0.28, 0.18),
            "COR":            (0.57, 0.88),
            "CRE":            (0.57, 0.76),
            "SPE":            (0.57, 0.64),
            "SOC":            (0.57, 0.42),
            "TSK":            (0.57, 0.30),
            "COM":            (0.57, 0.18),
            COHESION_SCORE:   (0.78, 0.30),
            PERFORMANCE:      (0.78, 0.64),
        }
    else:
        # Sans branche affective : INV_T* recentres verticalement.
        nodes = {
            INPUT_COMPOSITE:  (0.07, 0.54),
            "INV_TAS":        (0.28, 0.84),
            "INV_TMS":        (0.28, 0.54),
            "INV_TRS":        (0.28, 0.24),
            "COR":            (0.57, 0.88),
            "CRE":            (0.57, 0.76),
            "SPE":            (0.57, 0.64),
            "SOC":            (0.57, 0.42),
            "TSK":            (0.57, 0.30),
            "COM":            (0.57, 0.18),
            COHESION_SCORE:   (0.78, 0.30),
            PERFORMANCE:      (0.78, 0.64),
        }

    labels: dict[str, str] = {
        INPUT_COMPOSITE:  "INPUT\ncomposite",
        "INV_TAS":        "INV_TAS\n(TAS)",
        "INV_TMS":        "INV_TMS\n(TMS)",
        "INV_TRS":        "INV_TRS\n(TRS)",
        AFFECT_MARKER:    "Affect\nnegatif",
        "COR":            "COR",
        "CRE":            "CRE",
        "SPE":            "SPE",
        "SOC":            "SOC",
        "TSK":            "TSK",
        "COM":            "COM",
        COHESION_SCORE:   "Cohesion\nglobale",
        PERFORMANCE:      "Performance",
    }

    fig, ax = plt.subplots(figsize=(11.0, 8.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Zone sous-systemes transactifs (hauteur adaptee selon INCLUDE_AFFECT).
    if INCLUDE_AFFECT:
        inv_zone_y, inv_zone_h = 0.34, 0.605
    else:
        inv_zone_y, inv_zone_h = 0.12, 0.795
    # Zone branche affective (seulement si INCLUDE_AFFECT).
    if INCLUDE_AFFECT:
        ax.add_patch(plt.Rectangle((0.205, 0.08), 0.15, 0.20,
                                    fill=True, fc="#f7eeee", ec="#b88a8a", lw=0.8, alpha=0.65))
        ax.text(0.285, 0.285, "Branche affective",
                ha="center", va="center", fontsize=7.5, color="#7f3f3f", style="italic")

    ax.add_patch(plt.Rectangle((0.210, inv_zone_y), 0.145, inv_zone_h,
                                fill=True, fc="#eef4f1", ec="#8aa99a", lw=0.8, alpha=0.65))
    ax.text(0.285, inv_zone_y + inv_zone_h + 0.010, "Sous-systemes transactifs",
            ha="center", va="bottom", fontsize=7.5, color="#2f5f4b", style="italic")

    ax.add_patch(plt.Rectangle((0.502, 0.585), 0.122, 0.355,
                                fill=True, fc="#f0eef7", ec="#8aaeb8", lw=0.8, alpha=0.55))
    ax.text(0.563, 0.570, "Dims TMS",
            ha="center", va="top", fontsize=7, color="#4a4a7f", style="italic")

    ax.add_patch(plt.Rectangle((0.502, 0.08), 0.122, 0.415,
                                fill=True, fc="#f0eef7", ec="#8a8ab8", lw=0.8, alpha=0.55))
    ax.text(0.563, 0.500, "Dims cohesion",
            ha="center", va="top", fontsize=7, color="#4a4a7f", style="italic")

    # -----------------------------------------------------------------------
    # Dessin des noeuds.
    # -----------------------------------------------------------------------
    isolated_color   = "#cccccc"   # gris pour noeuds sans chemin robuste
    isolated_ec      = "#999999"
    isolated_tc      = "#888888"

    for node, (x, y) in nodes.items():
        is_active = node in active_nodes
        if node in {"INV_TAS", "INV_TMS", "INV_TRS"}:
            fc = "#e3f0ea" if is_active else isolated_color
            ec = "#333333" if is_active else isolated_ec
        elif node == AFFECT_MARKER:
            fc = "#f4dddd" if is_active else isolated_color
            ec = "#333333" if is_active else isolated_ec
        elif node == PERFORMANCE:
            fc = "#fdf5e0"
            ec = "#333333"
        else:
            fc = "#f3f0e8" if is_active else isolated_color
            ec = "#333333" if is_active else isolated_ec

        tc = "#222222" if is_active else isolated_tc
        ax.text(
            x, y, labels.get(node, node),
            ha="center", va="center", fontsize=8,
            color=tc,
            bbox=dict(boxstyle="round,pad=0.32", fc=fc, ec=ec, lw=0.8 if is_active else 0.5),
            zorder=3,
        )
        # Annotation sous les noeuds sans aucun chemin (ni robuste ni tendance).
        skip_annotation = {"SOC", "TSK", "COM", INPUT_COMPOSITE, PERFORMANCE}
        if not is_active and node not in skip_annotation:
            ax.text(x, y - 0.065, "(aucun chemin)",
                    ha="center", va="top", fontsize=6, color=isolated_tc,
                    style="italic", zorder=3)

    # -----------------------------------------------------------------------
    # Dessin des fleches : robustes (trait plein) + tendances (pointille).
    #
    # Strategie anti-superposition : chaque paire (src, dst) regoit un rad
    # unique calcule depuis un compteur global, avec alternance de signe.
    # -----------------------------------------------------------------------

    # Compteur de fleches par paire de noeuds (src, dst) pour varier rad.
    _arrow_count: dict[tuple[str, str], int] = {}

    def _draw_arrow(ax, row, nodes, is_robust: bool) -> None:
        src = str(row["source"])
        dst = str(row["target"])
        if src not in nodes or dst not in nodes:
            return
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        beta = float(row["path_coef_std"])
        n_obs = int(row["n"]) if "n" in row and pd.notna(row["n"]) else None

        color = "#b23a2f" if beta >= 0 else "#2b6cb0"
        alpha = 0.88 if is_robust else 0.45
        lw = 0.8 + 2.8 * min(abs(beta), 1.0)
        linestyle = "solid" if is_robust else "dashed"

        # Rayon de courbure : 0 si une seule fleche sur cette paire,
        # sinon alternance de signe avec amplitude 0.20/longueur (normalise).
        key = (src, dst)
        count = _arrow_count.get(key, 0)
        _arrow_count[key] = count + 1
        if count == 0:
            rad = 0.0
        else:
            sign_rad = 1 if count % 2 == 1 else -1
            rad = sign_rad * 0.18 * (count // 2 + 1)

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=lw,
                color=color,
                alpha=alpha,
                linestyle=linestyle,
                connectionstyle=f"arc3,rad={rad:.3f}",
            ),
            zorder=2 if is_robust else 1,
        )

        # Point au sommet de l'arc de Bezier quadratique t=0.5 :
        # Pour arc3, le point de controle est perp a mi-segment d'une distance rad*L.
        # A t=0.5 : B(0.5) = 0.5*P0 + 0.5*P2 + 0.5*(point_ctrl - midpoint) * ...
        # Formule exacte : midpoint_bezier = midpoint_segment + 0.5 * rad * L * perp_unit
        dx, dy = x2 - x1, y2 - y1
        length = max((dx**2 + dy**2) ** 0.5, 1e-6)
        perp_x, perp_y = -dy / length, dx / length
        mx = (x1 + x2) / 2 + perp_x * (0.5 * rad * length)
        my = (y1 + y2) / 2 + perp_y * (0.5 * rad * length)

        sign = "+" if beta >= 0 else ""
        beta_label = f"{sign}{beta:.2f}"
        suffix = "" if is_robust else "~"
        n_label = f"n={n_obs}" if n_obs is not None else ""
        full_label = f"{beta_label}{suffix}\n{n_label}" if n_label else f"{beta_label}{suffix}"

        ax.text(
            mx, my, full_label,
            ha="center", va="center", fontsize=6.2 if is_robust else 5.2,
            color=color,
            fontweight="bold" if is_robust else "normal",
            alpha=0.95 if is_robust else 0.65,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.88),
            zorder=4 if is_robust else 3,
        )

    # Tendances d'abord (dessous), robustes ensuite (dessus).
    for _, row in tendance.iterrows():
        _draw_arrow(ax, row, nodes, is_robust=False)
    for _, row in robust.iterrows():
        _draw_arrow(ax, row, nodes, is_robust=True)

    # -----------------------------------------------------------------------
    # Legende compacte (coin inferieur gauche).
    # -----------------------------------------------------------------------
    legend_lines = [
        ("— IC 95% excluant 0", "#444444"),
        (f"-- tendance |b|>={TENDANCE_BETA_MIN}, IC chevauche 0", "#888888"),
        ("Rouge +  /  Bleu −  /  epaisseur ~ |beta|", "#444444"),
    ]
    lx, ly = 0.01, 0.01
    lh = 0.032
    lw_box = 0.30
    ax.add_patch(plt.Rectangle(
        (lx, ly), lw_box, len(legend_lines) * lh + 0.012,
        fill=True, fc="#f8f8f8", ec="#cccccc", lw=0.5, alpha=0.90, zorder=5,
    ))
    for i, (line, color_l) in enumerate(legend_lines):
        ax.text(lx + 0.007, ly + 0.006 + i * lh,
                line, ha="left", va="bottom", fontsize=5.8, color=color_l, zorder=6)

    fig.tight_layout(pad=0.3)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# FONCTIONS LEGACY : conservees pour compatibilite avec rapport/v2/main.py
# (run_pls_sem_vr et ses helpers prives)
# ---------------------------------------------------------------------------

def _compute_indirect_effects(paths: pd.DataFrame, exogenous: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    inv_perf = _path_lookup(paths, "INV_composite", PERFORMANCE)
    for src in exogenous:
        src_inv = _path_lookup(paths, src, "INV_composite")
        rows.append({
            "indirect_path": f"{src} -> INV_composite -> {PERFORMANCE}",
            "effect": src_inv * inv_perf if np.isfinite(src_inv) and np.isfinite(inv_perf) else np.nan,
        })
    for med in TMS_DIMS + [COHESION_SCORE]:
        inv_med = _path_lookup(paths, "INV_composite", med)
        med_perf = _path_lookup(paths, med, PERFORMANCE)
        rows.append({
            "indirect_path": f"INV_composite -> {med} -> {PERFORMANCE}",
            "effect": inv_med * med_perf if np.isfinite(inv_med) and np.isfinite(med_perf) else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_model_diagram(paths: pd.DataFrame, outpath: Path) -> bool:
    if paths is None or paths.empty:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axis("off")
    nodes = {
        "Exogènes\nc_score / Riedl / RME": (0.08, 0.58),
        "INV\ncomposite formatif": (0.36, 0.58),
        "TMS\nCOR CRE SPE": (0.62, 0.78),
        "Cohésion\nscore observé": (0.62, 0.38),
        "Performance\nScore_perf_tsk": (0.88, 0.58),
    }
    for label, (x, y) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="#f3f0e8", ec="#333333", lw=0.8))
    arrows = [
        ("Exogènes\nc_score / Riedl / RME", "INV\ncomposite formatif"),
        ("INV\ncomposite formatif", "TMS\nCOR CRE SPE"),
        ("INV\ncomposite formatif", "Cohésion\nscore observé"),
        ("TMS\nCOR CRE SPE", "Performance\nScore_perf_tsk"),
        ("Cohésion\nscore observé", "Performance\nScore_perf_tsk"),
        ("INV\ncomposite formatif", "Performance\nScore_perf_tsk"),
        ("Exogènes\nc_score / Riedl / RME", "Performance\nScore_perf_tsk"),
    ]
    for src, dst in arrows:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    ax.text(0.5, 0.08, "PLS-SEM exploratoire : coefficients standardisés par OLS, N VR faible.",
            ha="center", va="center", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True


def run_pls_sem_vr(
    merged_df: pd.DataFrame | None = None,
    merged_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    inv_features: list[str] | None = None,
) -> dict[str, Any]:
    """Lance l'analyse PLS-SEM exploratoire VR (version legacy) et sauvegarde les sorties."""
    if merged_df is None:
        if merged_path is None:
            raise ValueError("merged_df ou merged_path doit etre fourni")
        merged_df = _read_csv(Path(merged_path))
    out_dir = Path(output_dir) if output_dir is not None else Path("pls_sem_vr")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merged_df.copy()
    if "modalite" in df.columns:
        df = df[df["modalite"].astype(str).str.upper() == "VR"].copy()

    if inv_features is None:
        inv_features = [item for vals in REFINED_INV_COMPOSITES.values() for item in vals]
    inv_features = list(inv_features)

    strategy = "strategy_norm" if _resolve_col(df, "strategy_norm") else "strategy_ratio_mean"
    rme = "rme_mean" if _resolve_col(df, "rme_mean") else "rme_min"
    exogenous = [c for c in ["c_score", "skill_mean", strategy, "contribution_mean", rme] if c]
    required = sorted(set(inv_features + exogenous + TMS_DIMS + [COHESION_SCORE, PERFORMANCE]))

    df, availability = _canonicalize_columns(df, required)
    df, cohesion_source = _ensure_cohesion(df)
    if COHESION_SCORE not in availability["variable"].values:
        availability = pd.concat([
            availability,
            pd.DataFrame([{
                "variable": COHESION_SCORE,
                "source_column": cohesion_source,
                "available": cohesion_source != "missing",
                "n_non_missing": int(df[COHESION_SCORE].notna().sum()) if COHESION_SCORE in df.columns else 0,
            }]),
        ], ignore_index=True)
    else:
        availability.loc[availability["variable"] == COHESION_SCORE, "source_column"] = cohesion_source
        availability.loc[availability["variable"] == COHESION_SCORE, "available"] = cohesion_source != "missing"
        availability.loc[availability["variable"] == COHESION_SCORE, "n_non_missing"] = (
            int(df[COHESION_SCORE].notna().sum()) if COHESION_SCORE in df.columns else 0
        )

    available_inv = [c for c in inv_features if c in df.columns and df[c].notna().sum() >= 4]
    available_exog = [c for c in exogenous if c in df.columns and df[c].notna().sum() >= 4]
    available_tms = [c for c in TMS_DIMS if c in df.columns and df[c].notna().sum() >= 4]
    has_cohesion = COHESION_SCORE in df.columns and df[COHESION_SCORE].notna().sum() >= 4
    has_perf = PERFORMANCE in df.columns and df[PERFORMANCE].notna().sum() >= 4

    availability.to_csv(out_dir / "pls_sem_vr_availability.csv", index=False, encoding="utf-8-sig")
    blocking_reasons = []
    if len(available_inv) < 2:
        blocking_reasons.append("moins de 2 indicateurs INV disponibles")
    if not available_exog:
        blocking_reasons.append("aucune variable exogene disponible")
    if not has_perf:
        blocking_reasons.append("Score_perf_tsk indisponible")
    if not available_tms and not has_cohesion:
        blocking_reasons.append("aucun mediateur TMS/Cohesion disponible")

    if blocking_reasons:
        note = "PLS-SEM non estime : " + "; ".join(blocking_reasons)
        (out_dir / "pls_sem_vr_notes.txt").write_text(note, encoding="utf-8")
        return {"estimated": False, "note": note, "availability": availability, "output_dir": out_dir}

    df, measurement = _make_inv_composite(df, available_inv)
    model_cols = sorted(set(
        ["group_id", "timepoint", "modalite", "scenario", PERFORMANCE, "INV_composite"]
        + available_inv + available_exog + available_tms
        + ([COHESION_SCORE] if has_cohesion else [])
    ))
    model_df = df[[c for c in model_cols if c in df.columns]].copy()
    model_df.to_csv(out_dir / "pls_sem_vr_dataset.csv", index=False, encoding="utf-8-sig")
    measurement.to_csv(out_dir / "pls_sem_vr_measurement.csv", index=False, encoding="utf-8-sig")

    path_tables: list[pd.DataFrame] = []
    r2_rows: list[dict[str, Any]] = []
    tab, r2 = _safe_ols_standardized(model_df, "INV_composite", available_exog)
    path_tables.append(tab)
    r2_rows.append(r2)
    for target in available_tms + ([COHESION_SCORE] if has_cohesion else []):
        tab, r2 = _safe_ols_standardized(model_df, target, ["INV_composite"])
        path_tables.append(tab)
        r2_rows.append(r2)

    perf_predictors = ["INV_composite"] + available_tms + ([COHESION_SCORE] if has_cohesion else [])
    tab, r2 = _safe_ols_standardized(model_df, PERFORMANCE, perf_predictors)
    path_tables.append(tab)
    r2_rows.append(r2)
    for exog in available_exog:
        tab, r2 = _safe_ols_standardized(model_df, PERFORMANCE, [exog])
        if not tab.empty:
            tab["warning"] = (tab["warning"].astype(str) + "; direct_univarie_faible_N").str.strip("; ")
        r2["target"] = f"{PERFORMANCE}__direct_{exog}"
        if isinstance(r2.get("warning"), str):
            r2["warning"] = (r2["warning"] + "; direct_univarie_faible_N").strip("; ")
        path_tables.append(tab)
        r2_rows.append(r2)

    paths = pd.concat([t for t in path_tables if t is not None and not t.empty], ignore_index=True) if path_tables else pd.DataFrame()
    r2_df = pd.DataFrame(r2_rows)
    indirect = _compute_indirect_effects(paths, available_exog)

    for table in [paths, r2_df, indirect, measurement]:
        for col in ["path_coef_std", "se", "t", "p", "r2", "effect", "loading_corr_with_composite"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").round(4)

    paths.to_csv(out_dir / "pls_sem_vr_paths.csv", index=False, encoding="utf-8-sig")
    r2_df.to_csv(out_dir / "pls_sem_vr_r2.csv", index=False, encoding="utf-8-sig")
    indirect.to_csv(out_dir / "pls_sem_vr_indirect_effects.csv", index=False, encoding="utf-8-sig")

    diagram_path = out_dir / "pls_sem_vr_model.png"
    diagram_ok = _plot_model_diagram(paths, diagram_path)
    note = (
        "PLS-SEM exploratoire estime. Composite INV formatif equal-weight sur z-scores. "
        f"N={int(model_df[PERFORMANCE].notna().sum())}. Cohesion={cohesion_source}. "
        "Aucune inference confirmatoire robuste ne doit etre tiree de ce modele."
    )
    (out_dir / "pls_sem_vr_notes.txt").write_text(note, encoding="utf-8")
    return {
        "estimated": True,
        "note": note,
        "availability": availability,
        "dataset": model_df,
        "measurement": measurement,
        "paths": paths,
        "r2": r2_df,
        "indirect": indirect,
        "diagram_path": diagram_path if diagram_ok else None,
        "output_dir": out_dir,
        "inv_features": available_inv,
        "exogenous": available_exog,
        "tms_dims": available_tms,
        "cohesion_source": cohesion_source,
    }


# ---------------------------------------------------------------------------
# FONCTION PRINCIPALE : path analysis exploratoire VR
# ---------------------------------------------------------------------------

def run_refined_path_analysis_vr(
    merged_df: pd.DataFrame | None = None,
    merged_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    n_boot: int = BOOTSTRAP_B,
) -> dict[str, Any]:
    """Path analysis VR : INPUT -> sous-systemes transactifs -> etats emergents -> performance.

    Denomination : path analysis exploratoire sur composites formatifs alignes sur
    les sous-systemes transactifs (TAS / TMS / TRS) et une branche affective parallele.
    Bootstrap percentile B=BOOTSTRAP_B. Poids fixes egaux. Aucune inference causale.
    """
    if merged_df is None:
        if merged_path is None:
            raise ValueError("merged_df ou merged_path doit etre fourni")
        merged_df = _read_csv(Path(merged_path))
    out_dir = Path(output_dir) if output_dir is not None else Path("path_analysis_vr")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = merged_df.copy()
    if "modalite" in df.columns:
        df = df[df["modalite"].astype(str).str.upper() == "VR"].copy()

    # Toutes les variables potentiellement necessaires.
    _affect_vars = ([AFFECT_MARKER] + AFFECT_TARGETS) if INCLUDE_AFFECT else []
    retained_variables = sorted({v for v in (
        [PERFORMANCE, COHESION_SCORE, BIVARIATE_CORR_TARGET]
        + _affect_vars + ROOT_CAUSES + TMS_DIMS + COHESION_DIMS + TAS_TARGETS
        + [item for vals in REFINED_INV_COMPOSITES.values() for item in vals]
    ) if v})
    df, availability = _canonicalize_columns(df, retained_variables)
    df, cohesion_source = _ensure_cohesion(df)
    if COHESION_SCORE in availability["variable"].values:
        availability.loc[availability["variable"] == COHESION_SCORE, "source_column"] = cohesion_source
        availability.loc[availability["variable"] == COHESION_SCORE, "available"] = cohesion_source != "missing"
        availability.loc[availability["variable"] == COHESION_SCORE, "n_non_missing"] = (
            int(df[COHESION_SCORE].notna().sum()) if COHESION_SCORE in df.columns else 0
        )

    # Construction des composites : INPUT_composite (causes racines) + INV transactifs.
    composite_specs_full = {INPUT_COMPOSITE: ROOT_CAUSES, **REFINED_INV_COMPOSITES}
    df, measurement, composite_specs = _make_refined_composites(df, composite_specs_full)

    # Mise à jour de la disponibilité pour les COMPOSITES (INV_TAS/TMS/TRS,
    # INPUT_composite) : ils n'existent qu'après _make_refined_composites. Sans
    # cela, la table affiche par ex. « INV_TMS | False | 0 » alors que les chemins
    # correspondants tournent bien (évaluation faite avant création du composite).
    for _comp in [INPUT_COMPOSITE, "INV_TAS", "INV_TMS", "INV_TRS"]:
        if _comp not in df.columns:
            continue
        _n = int(df[_comp].notna().sum())
        _row_mask = availability["variable"] == _comp
        if _row_mask.any():
            availability.loc[_row_mask, "source_column"] = _comp
            availability.loc[_row_mask, "available"] = _n > 0
            availability.loc[_row_mask, "n_non_missing"] = _n
        else:
            availability = pd.concat([availability, pd.DataFrame([{
                "variable": _comp,
                "source_column": _comp,
                "available": _n > 0,
                "n_non_missing": _n,
            }])], ignore_index=True)

    availability.to_csv(out_dir / "path_analysis_vr_availability.csv", index=False, encoding="utf-8-sig")
    input_indicators = composite_specs.get(INPUT_COMPOSITE, [])
    inv_composites = [
        c for c in ["INV_TAS", "INV_TMS", "INV_TRS"]
        if c in df.columns and df[c].notna().sum() >= 4
    ]

    # Verification des pre-requis.
    blocking = []
    if len(input_indicators) < 2 or INPUT_COMPOSITE not in df.columns:
        blocking.append("INPUT_composite non estimable")
    if len(inv_composites) < 3:
        blocking.append("composites INV TAS/TMS/TRS incomplets")
    if INCLUDE_AFFECT and (AFFECT_MARKER not in df.columns or df[AFFECT_MARKER].notna().sum() < 4):
        blocking.append(f"{AFFECT_MARKER} indisponible")
    for target in [PERFORMANCE, *TMS_DIMS, COHESION_SCORE]:
        if target not in df.columns or df[target].notna().sum() < 4:
            blocking.append(f"{target} indisponible")
    if blocking:
        note = "Path analysis non estimee : " + "; ".join(blocking)
        (out_dir / "path_analysis_vr_notes.txt").write_text(note, encoding="utf-8")
        return {"estimated": False, "note": note, "availability": availability, "output_dir": out_dir}

    # VIF intra-composite sur les indicateurs definis dans la config SEM.
    # On utilise composite_specs_full (config theorique) et non composite_specs
    # (indicateurs disponibles apres filtrage), pour eviter d'inclure des variables
    # hors modele SEM qui feraient gonfler le nombre de predicteurs.
    vif_df = _compute_vif_all_composites(df, composite_specs_full)
    if not vif_df.empty:
        # Propager les warnings VIF dans la table measurement.
        vif_warning = {
            (str(row["composite"]), str(row["indicator"])): str(row.get("status", ""))
            for _, row in vif_df.iterrows()
            if str(row.get("status", "")) not in ("OK", "")
        }
        if "warning" not in measurement.columns:
            measurement["warning"] = ""
        for (construct, indicator), warn in vif_warning.items():
            mask = (
                (measurement["construct"].astype(str) == construct)
                & (measurement["indicator"].astype(str) == indicator)
            )
            measurement.loc[mask, "warning"] = warn

    _affect_model_cols = ([AFFECT_MARKER] + AFFECT_TARGETS) if INCLUDE_AFFECT else []
    model_cols = sorted(set(
        ["group_id", "timepoint", "modalite", "scenario", PERFORMANCE, COHESION_SCORE]
        + input_indicators + [INPUT_COMPOSITE] + inv_composites
        + _affect_model_cols
        + TMS_DIMS + COHESION_DIMS + TAS_TARGETS
        + [item for vals in composite_specs.values() for item in vals]
    ))
    model_df = df[[c for c in model_cols if c in df.columns]].copy()
    model_df.to_csv(out_dir / "path_analysis_vr_dataset.csv", index=False, encoding="utf-8-sig")
    measurement.to_csv(out_dir / "path_analysis_vr_measurement.csv", index=False, encoding="utf-8-sig")

    composite_justification = pd.DataFrame([
        {
            "construct": construct,
            "indicators": ", ".join(_signed_indicator_label(construct, indicator) for indicator in indicators),
            "inverted_indicators": ", ".join(
                indicator for indicator in indicators if _indicator_sign(construct, indicator) < 0
            ),
            "theoretical_anchor": REFINED_COMPOSITE_THEORY.get(construct, ""),
        }
        for construct, indicators in composite_specs.items()
    ])

    # -------------------------------------------------------------------
    # DEFINITION DES CHEMINS STRUCTURELS
    # Toutes les cibles sont lues depuis les constantes globales.
    # -------------------------------------------------------------------
    path_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    boot_map: dict[tuple[str, str, str], np.ndarray] = {}

    def add_path(target: str, predictors: list[str], source: str, seed_offset: int, path_type: str) -> None:
        row, boot = _bootstrap_path(model_df, target, predictors, source, n_boot=n_boot, seed=42 + seed_offset)
        row["path_type"] = path_type
        path_rows.append(row)
        boot_map[_path_key(source, target, row["equation"])] = boot
        r2 = _r2_for_equation(model_df, target, predictors)
        r2["path_type"] = path_type
        r2_rows.append(r2)

    seed = 0

    # Bloc 1 : INPUT -> composites INV transactifs.
    for comp in ["INV_TAS", "INV_TMS", "INV_TRS"]:
        if comp in inv_composites:
            add_path(comp, [INPUT_COMPOSITE], INPUT_COMPOSITE, seed, "structure")
            seed += 1

    # Bloc 2 : INPUT -> branche affective (gouverne par INCLUDE_AFFECT).
    if INCLUDE_AFFECT:
        add_path(AFFECT_MARKER, [INPUT_COMPOSITE], INPUT_COMPOSITE, seed, "structure")
        seed += 1

    # Blocs 3-5 : INV_* -> PERFORMANCE + cibles secondaires.
    # Structure identique pour les 3 composites :
    #   1. chemin direct -> PERFORMANCE (toujours teste)
    #   2. chemins -> cibles secondaires (gouvernes par *_TARGETS, guard notna)
    _inv_target_map = [
        ("INV_TAS", TAS_TARGETS),
        ("INV_TMS", TMS_STRUCTURAL_TARGETS),
        ("INV_TRS", TRS_TARGETS),
    ]
    for comp, extra_targets in _inv_target_map:
        if comp not in inv_composites:
            continue
        add_path(PERFORMANCE, [comp], comp, seed, "structure")
        seed += 1
        for target in extra_targets:
            if target in model_df.columns and model_df[target].notna().sum() >= 4:
                add_path(target, [comp], comp, seed, "structure")
                seed += 1

    # Bloc 6 : Affect -> toutes cibles (gouverne par INCLUDE_AFFECT et AFFECT_TARGETS).
    if INCLUDE_AFFECT:
        for target in AFFECT_TARGETS:
            if target in model_df.columns and model_df[target].notna().sum() >= 4:
                add_path(target, [AFFECT_MARKER], AFFECT_MARKER, seed, "structure")
                seed += 1

    # Bloc 7 : toutes dimensions questionnaire + cohesion -> performance.
    _all_dim_targets = list(dict.fromkeys(TMS_DIMS + COHESION_DIMS + [COHESION_SCORE]))
    for target in _all_dim_targets:
        if target in model_df.columns and model_df[target].notna().sum() >= 4:
            add_path(PERFORMANCE, [target], target, seed, "structure")
            seed += 1

    # Bloc 8 : effet direct INPUT -> performance.
    add_path(PERFORMANCE, [INPUT_COMPOSITE], INPUT_COMPOSITE, seed, "structure")
    seed += 1

    # Bloc 9 : chemins directs ajustes INPUT + mediateur -> performance (un mediateur a la fois).
    for adj in ADJUSTED_DIRECT_MEDIATORS:
        if adj in model_df.columns and model_df[adj].notna().sum() >= 4:
            add_path(PERFORMANCE, [INPUT_COMPOSITE, adj], INPUT_COMPOSITE, seed, "adjusted_direct")
            seed += 1

    paths = pd.DataFrame(path_rows)
    r2_df = pd.DataFrame(r2_rows).drop_duplicates(subset=["target", "equation"], keep="first")

    # -------------------------------------------------------------------
    # EFFETS INDIRECTS (lus depuis les constantes ; ajouter un chemin =
    # ajouter une entree dans la section appropriee ci-dessous).
    # -------------------------------------------------------------------
    indirect_rows: list[dict[str, Any]] = []
    indirect_boots: list[np.ndarray] = []
    indirect_effects: list[float] = []

    def add_indirect(label: str, parts: list[tuple[float, np.ndarray]]) -> None:
        effect, boot = _add_indirect_row(indirect_rows, label, parts)
        if np.isfinite(effect) and boot.size:
            indirect_effects.append(effect)
            indirect_boots.append(boot)

    # Effets indirects INPUT -> INV_* -> Performance (direct)
    # et INPUT -> INV_* -> mediateur -> Performance (mediation).
    # Identique pour TAS, TMS, TRS : miroir exact des blocs structurels.
    for comp, extra_targets in _inv_target_map:
        if comp not in inv_composites:
            continue
        # INPUT -> INV_* -> Performance (chemin indirect a 1 mediateur).
        b1, boot1, _ = _lookup_path_boot(paths, boot_map, INPUT_COMPOSITE, comp)
        b2, boot2, _ = _lookup_path_boot(paths, boot_map, comp, PERFORMANCE)
        add_indirect(f"{INPUT_COMPOSITE} -> {comp} -> {PERFORMANCE}", [(b1, boot1), (b2, boot2)])
        # INPUT -> INV_* -> mediateur -> Performance (chemin indirect a 2 mediateurs).
        for med in extra_targets:
            b1, boot1, _ = _lookup_path_boot(paths, boot_map, INPUT_COMPOSITE, comp)
            b2, boot2, _ = _lookup_path_boot(paths, boot_map, comp, med)
            b3, boot3, _ = _lookup_path_boot(paths, boot_map, med, PERFORMANCE)
            add_indirect(
                f"{INPUT_COMPOSITE} -> {comp} -> {med} -> {PERFORMANCE}",
                [(b1, boot1), (b2, boot2), (b3, boot3)],
            )

    # Chemins indirects via branche affective (gouverne par INCLUDE_AFFECT).
    if INCLUDE_AFFECT:
        b1, boot1, _ = _lookup_path_boot(paths, boot_map, INPUT_COMPOSITE, AFFECT_MARKER)
        b2, boot2, _ = _lookup_path_boot(paths, boot_map, AFFECT_MARKER, COHESION_SCORE)
        b3, boot3, _ = _lookup_path_boot(paths, boot_map, COHESION_SCORE, PERFORMANCE)
        add_indirect(
            f"{INPUT_COMPOSITE} -> {AFFECT_MARKER} -> {COHESION_SCORE} -> {PERFORMANCE}",
            [(b1, boot1), (b2, boot2), (b3, boot3)],
        )
        if "CRE" in AFFECT_TARGETS and "CRE" in model_df.columns and model_df["CRE"].notna().sum() >= 4:
            b1, boot1, _ = _lookup_path_boot(paths, boot_map, INPUT_COMPOSITE, AFFECT_MARKER)
            b2, boot2, _ = _lookup_path_boot(paths, boot_map, AFFECT_MARKER, "CRE")
            b3, boot3, _ = _lookup_path_boot(paths, boot_map, "CRE", PERFORMANCE)
            add_indirect(
                f"{INPUT_COMPOSITE} -> {AFFECT_MARKER} -> CRE -> {PERFORMANCE}",
                [(b1, boot1), (b2, boot2), (b3, boot3)],
            )

    indirect = pd.DataFrame(indirect_rows)

    # -------------------------------------------------------------------
    # DECOMPOSITION EFFET TOTAL INPUT -> Performance.
    # -------------------------------------------------------------------
    direct_effect, direct_boot, _ = _lookup_path_boot(paths, boot_map, INPUT_COMPOSITE, PERFORMANCE)
    if indirect_boots and direct_boot.size:
        min_len = min([len(direct_boot)] + [len(arr) for arr in indirect_boots])
        indirect_sum_boot = np.sum([arr[:min_len] for arr in indirect_boots], axis=0)
        total_boot = direct_boot[:min_len] + indirect_sum_boot
        indirect_sum = float(np.sum(indirect_effects)) if indirect_effects else np.nan
        total_effect = direct_effect + indirect_sum if np.isfinite(direct_effect) and np.isfinite(indirect_sum) else np.nan
        total_ci = np.nanpercentile(total_boot, [2.5, 97.5])
        indirect_ci = np.nanpercentile(indirect_sum_boot, [2.5, 97.5])
        direct_ci = np.nanpercentile(direct_boot[:min_len], [2.5, 97.5])
    else:
        indirect_sum = np.nan
        total_effect = np.nan
        direct_ci = indirect_ci = total_ci = [np.nan, np.nan]
    total_decomposition = pd.DataFrame([
        {"component": "direct", "effect": direct_effect, "ci95_low": direct_ci[0], "ci95_high": direct_ci[1], "warning": "exploratoire_N_faible"},
        {"component": "sum_indirect", "effect": indirect_sum, "ci95_low": indirect_ci[0], "ci95_high": indirect_ci[1], "warning": "exploratoire_N_faible"},
        {"component": "total", "effect": total_effect, "ci95_low": total_ci[0], "ci95_high": total_ci[1], "warning": "exploratoire_N_faible"},
    ])

    # -------------------------------------------------------------------
    # Tableau complementaire : correlations bivariees Spearman causes racines -> INV_TAS indicateur
    # Documente la dilution des correlations individuelles (rme/c_score -> gaze_attention : rho~-0.9)
    # dans l'agregation INPUT_composite (composite formatif a poids egaux).
    # -------------------------------------------------------------------
    bivariate_corr = _compute_bivariate_spearman(
        df, BIVARIATE_CORR_ROOTS, BIVARIATE_CORR_TARGET, n_boot=n_boot, seed=77,
    )

    # Arrondi final de toutes les tables numeriques.
    for table in [paths, r2_df, indirect, measurement, vif_df, total_decomposition, bivariate_corr]:
        for col in ["path_coef_std", "ci95_low", "ci95_high", "r2", "effect",
                    "loading_corr_with_composite", "vif", "rho_spearman"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").round(4)

    # Sauvegarde CSV.
    paths.to_csv(out_dir / "path_analysis_vr_paths.csv", index=False, encoding="utf-8-sig")
    r2_df.to_csv(out_dir / "path_analysis_vr_r2.csv", index=False, encoding="utf-8-sig")
    indirect.to_csv(out_dir / "path_analysis_vr_indirect_effects.csv", index=False, encoding="utf-8-sig")
    vif_df.to_csv(out_dir / "path_analysis_vr_vif_all_composites.csv", index=False, encoding="utf-8-sig")
    total_decomposition.to_csv(out_dir / "path_analysis_vr_total_decomposition.csv", index=False, encoding="utf-8-sig")
    composite_justification.to_csv(out_dir / "path_analysis_vr_composite_justification.csv", index=False, encoding="utf-8-sig")
    bivariate_corr.to_csv(out_dir / "path_analysis_vr_bivariate_corr_roots_inv_tas.csv", index=False, encoding="utf-8-sig")

    diagram_path = out_dir / "path_analysis_vr_model.png"
    diagram_ok = _plot_refined_path_diagram(paths, diagram_path)
    note = (
        f"Path analysis exploratoire sur composites formatifs alignes sur les sous-systemes transactifs "
        f"(TAS / TMS / TRS) et une branche affective parallele. "
        f"INPUT_composite : {', '.join(input_indicators)}. "
        f"Bootstrap percentile B={n_boot}. Poids fixes egaux; aucune inference causale."
    )
    (out_dir / "path_analysis_vr_notes.txt").write_text(note, encoding="utf-8")
    return {
        "estimated": True,
        "note": note,
        "availability": availability,
        "dataset": model_df,
        "measurement": measurement,
        "paths": paths,
        "r2": r2_df,
        "indirect": indirect,
        "vif": vif_df,
        "total_decomposition": total_decomposition,
        "composite_justification": composite_justification,
        "bivariate_corr": bivariate_corr,
        "diagram_path": diagram_path if diagram_ok else None,
        "output_dir": out_dir,
        "composite_specs": composite_specs,
        "roots": input_indicators,
        "n_boot": n_boot,
    }


# ---------------------------------------------------------------------------
# FONCTIONS UTILITAIRES MULTINIVEAU (sections 3.1.6 - 3.1.7)
# ---------------------------------------------------------------------------

def _load_individual_scores(
    questionnaire_path: Path,
    group_ids_filter: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Charge les scores individuels VR S2 et calcule mean/sd/cv par groupe et dimension.

    Retourne (df_individual, df_group_stats, note) ou :
    - df_individual : une ligne par (groupe, dimension, participant)
    - df_group_stats : une ligne par (groupe, dimension) avec mean, sd, cv, n
    - note : message sur la disponibilite des donnees
    """
    if not questionnaire_path.exists():
        return pd.DataFrame(), pd.DataFrame(), f"Fichier non trouve : {questionnaire_path}"

    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(questionnaire_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return pd.DataFrame(), pd.DataFrame(), "Erreur d'encodage"

    # Filtrer VR + scenarios cibles (INDIVIDUAL_SCENARIO_TARGETS).
    targets_upper = [s.upper() for s in INDIVIDUAL_SCENARIO_TARGETS]
    mask = (
        (df[INDIVIDUAL_MODALITE_COL].str.upper() == "VR")
        & (df[INDIVIDUAL_SCENARIO_COL].str.upper().isin(targets_upper))
    )
    df = df[mask].copy()
    if df.empty:
        scenarios_str = "/".join(INDIVIDUAL_SCENARIO_TARGETS)
        return pd.DataFrame(), pd.DataFrame(), f"Aucune donnee VR {scenarios_str} individuelle disponible"

    # Normaliser group_id
    df["group_id"] = df[INDIVIDUAL_GROUP_COL].astype(str).str.lower().str.strip()
    df[INDIVIDUAL_SCORE_COL] = pd.to_numeric(df[INDIVIDUAL_SCORE_COL], errors="coerce")

    if group_ids_filter:
        filter_norm = [g.lower().strip() for g in group_ids_filter]
        df = df[df["group_id"].isin(filter_norm)]

    available_dims = [d for d in INDIVIDUAL_DIMS if d in df[INDIVIDUAL_DIM_COL].unique()]

    # Agregation par groupe x dimension
    rows = []
    for (gid, dim), grp in df.groupby(["group_id", INDIVIDUAL_DIM_COL]):
        if dim not in available_dims:
            continue
        scores = grp[INDIVIDUAL_SCORE_COL].dropna()
        if len(scores) < 2:
            continue
        mean_v = float(scores.mean())
        sd_v = float(scores.std(ddof=1))
        cv_v = sd_v / mean_v if mean_v != 0 else np.nan
        rows.append({
            "group_id": gid,
            "dimension": dim,
            "mean_ind": round(mean_v, 4),
            "sd_ind": round(sd_v, 4),
            "cv_ind": round(cv_v, 4) if np.isfinite(cv_v) else np.nan,
            "n_ind": int(len(scores)),
        })

    df_group_stats = pd.DataFrame(rows)
    available_dims_found = sorted(df_group_stats["dimension"].unique().tolist()) if not df_group_stats.empty else []
    n_groups = df_group_stats["group_id"].nunique() if not df_group_stats.empty else 0
    scenarios_str = "/".join(INDIVIDUAL_SCENARIO_TARGETS)
    note = (
        f"Scores individuels VR {scenarios_str} disponibles pour {n_groups} groupes, "
        f"dimensions : {', '.join(available_dims_found)}. "
        f"Note : les groupes VR S1 (bim006, bim010, bim066, bim073_2) peuvent manquer "
        f"de questionnaire selon la disponibilite des donnees source."
    )
    return df, df_group_stats, note


def _run_sd_augmented_regressions(
    df_group_stats: pd.DataFrame,
    merged_df: pd.DataFrame,
    n_boot: int = BOOTSTRAP_B,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pour chaque dimension, compare modele mean seul vs mean+sd vers Score_perf_tsk.

    Retourne (df_paths, df_delta_r2) avec coefficients bootstrap et delta R2.
    """
    perf_col = PERFORMANCE

    # Joindre stats individuelles avec performance groupe
    perf = merged_df[["group_id", perf_col]].copy()
    perf["group_id"] = perf["group_id"].astype(str).str.lower().str.strip()

    path_rows = []
    delta_rows = []

    dims = df_group_stats["dimension"].unique() if not df_group_stats.empty else []

    for dim in sorted(dims):
        sub = df_group_stats[df_group_stats["dimension"] == dim].copy()
        sub = sub.merge(perf, on="group_id", how="inner")
        sub = sub.dropna(subset=["mean_ind", "sd_ind", perf_col])
        n = len(sub)

        if n < 4:
            continue

        # Modele 1 : perf ~ mean
        r1 = _r2_for_equation(sub.rename(columns={"mean_ind": f"mean_{dim}", perf_col: perf_col}),
                               perf_col, [f"mean_{dim}"])
        sub2 = sub.rename(columns={"mean_ind": f"mean_{dim}", "sd_ind": f"sd_{dim}"})
        sub2[perf_col] = sub[perf_col].values

        # Modele 2 : perf ~ mean + sd
        r2 = _r2_for_equation(sub2, perf_col, [f"mean_{dim}", f"sd_{dim}"])

        r2_m1 = r1.get("r2", np.nan)
        r2_m2 = r2.get("r2", np.nan)
        delta = (r2_m2 - r2_m1) if (np.isfinite(r2_m1) and np.isfinite(r2_m2)) else np.nan

        # Coefficient bootstrap sd_dim dans modele 2
        boot_row, _ = _bootstrap_path(
            sub2, perf_col, [f"mean_{dim}", f"sd_{dim}"], f"sd_{dim}",
            n_boot=n_boot, seed=hash(dim) % 10000,
        )
        boot_row["dimension"] = dim
        boot_row["r2_model1"] = round(float(r2_m1), 4) if np.isfinite(r2_m1) else np.nan
        boot_row["r2_model2"] = round(float(r2_m2), 4) if np.isfinite(r2_m2) else np.nan
        boot_row["delta_r2"] = round(float(delta), 4) if np.isfinite(delta) else np.nan
        path_rows.append(boot_row)

        # Correlation Spearman mean vs sd (risque floor/ceiling)
        from scipy.stats import spearmanr
        if len(sub) >= 4:
            rho, pval = spearmanr(sub["mean_ind"], sub["sd_ind"])
            # |rho| > 0.70 : confound fort (Aguinis et al. 2013 — SD non independante de mean)
            confound_flag = "CONFOUND_FORT" if abs(rho) > 0.70 else ("CONFOUND_MODERE" if abs(rho) > 0.50 else "ok")
            delta_rows.append({
                "dimension": dim,
                "n_groupes": n,
                "mean_sd_mean": round(float(sub["sd_ind"].mean()), 4),
                "min_sd": round(float(sub["sd_ind"].min()), 4),
                "max_sd": round(float(sub["sd_ind"].max()), 4),
                "rho_mean_vs_sd": round(float(rho), 4),
                "p_rho_mean_vs_sd": round(float(pval), 4),
                "confound_flag": confound_flag,
                "warning": "exploratoire_K_faible",
            })

    return pd.DataFrame(path_rows), pd.DataFrame(delta_rows)


def _icc_anova_point(sub: "pd.DataFrame") -> tuple[float, float, float]:
    """ICC point estimate via ANOVA one-way decomposition (modele nul analytique).

    Retourne (icc, sigma2_between, sigma2_within).
    Conserve comme valeur de reference pour convergence avec MCMC.
    """
    groups = sub["group_id"].unique()
    K = len(groups)
    group_data = [sub[sub["group_id"] == g][INDIVIDUAL_SCORE_COL].values for g in groups]
    ns = np.array([len(g) for g in group_data])
    N = int(ns.sum())
    grand_mean = float(sub[INDIVIDUAL_SCORE_COL].mean())

    ss_between = float(sum(n * (np.mean(g) - grand_mean)**2 for n, g in zip(ns, group_data)))
    ss_within = float(sum(np.sum((g - np.mean(g))**2) for g in group_data))
    df_b, df_w = K - 1, N - K
    if df_b < 1 or df_w < 1:
        return np.nan, np.nan, np.nan

    ms_between = ss_between / df_b
    ms_within = ss_within / df_w
    n0 = (N - sum(n**2 for n in ns) / N) / (K - 1) if K > 1 else np.nan
    sigma2_within = float(ms_within)
    sigma2_between = float(max((ms_between - ms_within) / n0, 0.0)) if (np.isfinite(n0) and n0 > 0) else np.nan
    denom = sigma2_between + sigma2_within
    icc = float(sigma2_between / denom) if (np.isfinite(sigma2_between) and denom > 0) else np.nan
    return icc, sigma2_between, sigma2_within


def _run_mlm_icc_analysis(
    df_individual: pd.DataFrame,
    merged_df: pd.DataFrame,
    n_boot: int = BOOTSTRAP_B,
    bayes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """MLM bayesien complet (PyMC 5) — Voie A refonte 2026.

    Modele nul (intercepts aleatoires, inchange) :
        score_ij ~ Normal(mu + u_j, sigma_e)
        u_j ~ Normal(0, sigma_u)
        mu ~ Normal(0, 1)
        sigma_u ~ HalfCauchy(0, 1)
        sigma_e ~ HalfCauchy(0, 1)
        ICC = sigma_u^2 / (sigma_u^2 + sigma_e^2)

    Modeles contextuels (18 sans affect / 24 avec : {3 composites + affect} x 6 dimensions) :
        score_ij ~ Normal(alpha + beta_c * X_c_j + u_j, sigma_e)
        Parametrisation non-centree (obligatoire a petit K) :
            z_u ~ Normal(0, 1, shape=K)  ;  u_j = z_u * sigma_u
        alpha   ~ Normal(0, 1)
        beta_c  ~ Normal(0, 1)   [prior par defaut, Burkner 2017]
        sigma_u ~ HalfCauchy(0, 1)
        sigma_e ~ HalfNormal(0, 1)

    Analyse de sensibilite aux priors : re-echantillonnage avec beta_c ~ N(0, 0.5).

    Retourne (df_icc, df_contextual_effects, note_implementation).
    df_contextual_effects : 18 lignes (composite | dimension | beta_std | ...)
                           + colonnes sensibilite prior.
    """
    import warnings
    import logging
    logging.getLogger("pymc").setLevel(logging.ERROR)
    logging.getLogger("pytensor").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="pytensor")
    warnings.filterwarnings("ignore", module="pymc")

    _PYMC_OK = False

    if bayes:
        try:
            import pymc as pm
            import arviz as az
            _PYMC_OK = True
        except ImportError:
            _PYMC_OK = False

    cfg = MLM_MCMC_CONFIG

    def _mcmc_draws_tune(config: dict) -> tuple[int, int]:
        """Dérive un couple (draws, tune) toujours valide pour pm.sample().

        Convention : `iter` = nombre TOTAL d'itérations (warmup inclus), `warmup`
        = phase de tuning. draws = iter - warmup. Robustesse : si iter <= warmup
        (ex. iter=5 pour un run de test), on garantit draws >= 1 et on borne tune
        à iter-1, afin de ne jamais passer un draws négatif à PyMC (qui échoue).
        """
        total = int(config.get("iter", 4000))
        warmup = int(config.get("warmup", total // 2))
        if total < 2:
            total = 2
        tune = max(1, min(warmup, total - 1))
        draws = max(1, total - tune)
        return draws, tune

    def _count_divergences(trace) -> int:
        """Nombre de transitions divergentes NUTS sur l'ensemble des chaînes.

        Diagnostic standard des modèles hiérarchiques à petit K : une géométrie en
        entonnoir de sigma_u peut produire des divergences malgré Rhat≈1 / ESS élevé.
        Retourne -1 si l'info n'est pas disponible.
        """
        try:
            if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
                return int(trace.sample_stats["diverging"].values.sum())
        except Exception:
            pass
        return -1

    # Seed d'échantillonnage MCMC (fixe, documenté pour la reproductibilité).
    _MCMC_SEED = int(cfg.get("seed", 42))

    # ------------------------------------------------------------------
    # Composites INV niveau groupe — calculés exactement comme en 3.1.3
    # ------------------------------------------------------------------
    merged_g = merged_df.copy()
    merged_g["group_id"] = merged_g["group_id"].astype(str).str.lower().str.strip()

    # Helper : z-mean formatif avec inversion de signe
    def _build_composite_col(df: pd.DataFrame, name: str) -> pd.Series:
        indicators = REFINED_INV_COMPOSITES.get(name, [])
        inverted = INVERTED_COMPOSITE_INDICATORS.get(name, set())
        parts = []
        for ind in indicators:
            col = _resolve_col(df, ind)
            if col is None:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            m, sd = s.mean(), s.std()
            if sd > 0:
                z = (s - m) / sd
            else:
                z = s - m
            if ind in inverted:
                z = -z
            parts.append(z)
        if not parts:
            return pd.Series(np.nan, index=df.index)
        return pd.concat(parts, axis=1).mean(axis=1)

    # Prédicteurs L2 du MLM = 3 composites transactifs + la branche affective
    # standalone (face_negative_affect_ratio z-scorée). L'affect n'est PAS un
    # composite (il est hors REFINED_INV_COMPOSITES par design) mais reste un
    # prédicteur L2 légitime : feature face disponible pour les 12 groupes (K=12).
    composite_names = list(REFINED_INV_COMPOSITES.keys())  # INV_TAS, INV_TMS, INV_TRS
    if INCLUDE_AFFECT:
        composite_names = composite_names + [AFFECT_MARKER]

    def _build_affect_col(df: pd.DataFrame) -> pd.Series:
        """Prédicteur L2 affectif standalone : face_negative_affect_ratio z-scorée."""
        col = _resolve_col(df, AFFECT_MARKER)
        if col is None:
            return pd.Series(np.nan, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        m, sd = s.mean(), s.std()
        return (s - m) / sd if sd > 0 else (s - m)

    # Construire un DataFrame groupe x prédicteur L2 (une ligne par groupe, dedupliqué)
    comp_df = merged_g[["group_id"]].drop_duplicates().copy()
    for cname in composite_names:
        if cname == AFFECT_MARKER:
            col_vals = _build_affect_col(merged_g)
        else:
            col_vals = _build_composite_col(merged_g, cname)
        tmp = merged_g[["group_id"]].copy()
        tmp[cname] = col_vals.values
        agg = tmp.groupby("group_id")[cname].mean()
        comp_df = comp_df.set_index("group_id").join(agg, how="left").reset_index()

    df_ind = df_individual.copy()
    df_ind["group_id"] = df_ind["group_id"].astype(str).str.lower().str.strip()
    df_ind[INDIVIDUAL_SCORE_COL] = pd.to_numeric(df_ind[INDIVIDUAL_SCORE_COL], errors="coerce")

    icc_rows: list[dict] = []
    contextual_rows: list[dict] = []

    dims = sorted(df_ind[INDIVIDUAL_DIM_COL].unique()) if not df_ind.empty else []

    # Helper : un seul modele contextuel PyMC5 (parametrisation non-centree)
    def _run_one_contextual(
        sub_ctx: pd.DataFrame,
        x_col: str,
        prior_sigma: float,
        seed: int,
    ) -> dict:
        """Retourne dict avec beta_mean, beta_sd, hdi_lo, hdi_hi, rhat, ess, n_divergent, method."""
        result = {
            "beta_mean": np.nan, "beta_sd": np.nan,
            "hdi_lo": np.nan, "hdi_hi": np.nan,
            "rhat": np.nan, "ess": np.nan,
            "n_divergent": np.nan,
            "method": "insuffisant",
        }
        if not _PYMC_OK:
            return result
        ctx_groups = sub_ctx["group_id"].unique()
        K_c = len(ctx_groups)
        ctx_group_idx = pd.Categorical(sub_ctx["group_id"], categories=ctx_groups).codes
        y_c = sub_ctx[INDIVIDUAL_SCORE_COL].values.astype(float)
        x_c = sub_ctx[x_col].values.astype(float)
        y_m, y_s = float(np.mean(y_c)), float(np.std(y_c)) if np.std(y_c) > 0 else 1.0
        x_m, x_s = float(np.mean(x_c)), float(np.std(x_c)) if np.std(x_c) > 0 else 1.0
        y_sc = (y_c - y_m) / y_s
        x_sc = (x_c - x_m) / x_s

        draws, tune = _mcmc_draws_tune(cfg)
        try:
            with pm.Model() as ctx_model:
                alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
                beta_c = pm.Normal("beta_c", mu=0.0, sigma=prior_sigma)
                sigma_u_c = pm.HalfCauchy("sigma_u", beta=1.0)
                sigma_e_c = pm.HalfNormal("sigma_e", sigma=1.0)
                # Parametrisation non-centree : obligatoire a petit K
                z_u = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=K_c)
                u_j_c = z_u * sigma_u_c
                mu_ij_c = alpha + beta_c * x_sc + u_j_c[ctx_group_idx]
                _ = pm.Normal("y_obs", mu=mu_ij_c, sigma=sigma_e_c, observed=y_sc)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trace = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=cfg["chains"],
                        cores=1,
                        target_accept=0.95,
                        random_seed=seed,
                        progressbar=False,
                        return_inferencedata=True,
                    )
            samples = trace.posterior["beta_c"].values.flatten()
            result["beta_mean"] = float(np.mean(samples))
            result["beta_sd"] = float(np.std(samples))
            hdi_b = az.hdi(trace, var_names=["beta_c"], hdi_prob=0.95)["beta_c"].values
            result["hdi_lo"] = float(hdi_b[0])
            result["hdi_hi"] = float(hdi_b[1])
            rhat_val = az.rhat(trace)["beta_c"].values
            ess_val = az.ess(trace)["beta_c"].values
            result["rhat"] = float(rhat_val.item()) if hasattr(rhat_val, "item") else float(rhat_val)
            result["ess"] = float(ess_val.item()) if hasattr(ess_val, "item") else float(ess_val)
            result["n_divergent"] = _count_divergences(trace)
            result["method"] = "PyMC5_MCMC_non_centree"

            # Si ESS bas, ré-échantillonnage avec plus d'itérations (2× le run
            # principal ; borné par _mcmc_draws_tune pour rester valide en run de test).
            if result["ess"] < cfg["ess_threshold"]:
                try:
                    draws2, tune2 = 2 * draws, 2 * tune
                    with pm.Model() as ctx_model2:
                        alpha2 = pm.Normal("alpha", mu=0.0, sigma=1.0)
                        beta_c2 = pm.Normal("beta_c", mu=0.0, sigma=prior_sigma)
                        sigma_u_c2 = pm.HalfCauchy("sigma_u", beta=1.0)
                        sigma_e_c2 = pm.HalfNormal("sigma_e", sigma=1.0)
                        z_u2 = pm.Normal("z_u", mu=0.0, sigma=1.0, shape=K_c)
                        u_j_c2 = z_u2 * sigma_u_c2
                        mu_ij_c2 = alpha2 + beta_c2 * x_sc + u_j_c2[ctx_group_idx]
                        _ = pm.Normal("y_obs", mu=mu_ij_c2, sigma=sigma_e_c2, observed=y_sc)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            trace2 = pm.sample(
                                draws=draws2,
                                tune=tune2,
                                chains=cfg["chains"],
                                cores=1,
                                target_accept=0.95,
                                random_seed=seed + 100,
                                progressbar=False,
                                return_inferencedata=True,
                            )
                    samples2 = trace2.posterior["beta_c"].values.flatten()
                    result["beta_mean"] = float(np.mean(samples2))
                    result["beta_sd"] = float(np.std(samples2))
                    hdi_b2 = az.hdi(trace2, var_names=["beta_c"], hdi_prob=0.95)["beta_c"].values
                    result["hdi_lo"] = float(hdi_b2[0])
                    result["hdi_hi"] = float(hdi_b2[1])
                    rhat2 = az.rhat(trace2)["beta_c"].values
                    ess2 = az.ess(trace2)["beta_c"].values
                    result["rhat"] = float(rhat2.item()) if hasattr(rhat2, "item") else float(rhat2)
                    result["ess"] = float(ess2.item()) if hasattr(ess2, "item") else float(ess2)
                    result["n_divergent"] = _count_divergences(trace2)
                    result["method"] = f"PyMC5_MCMC_non_centree_reechant({draws2 + tune2}iter)"
                except Exception:
                    pass
        except Exception as exc:
            result["method"] = f"PyMC5_echec({type(exc).__name__})"
        return result

    for dim in dims:
        if dim not in INDIVIDUAL_DIMS:
            continue

        sub = df_ind[df_ind[INDIVIDUAL_DIM_COL] == dim][["group_id", INDIVIDUAL_SCORE_COL]].dropna().copy()
        groups = sub["group_id"].unique()
        K = len(groups)
        N = len(sub)
        if K < 3 or N < 6:
            continue

        # --- ICC ANOVA (point de reference) ---
        icc_pt, s2b, s2w = _icc_anova_point(sub)

        # --- ICC bayesien (modele nul) ---
        icc_mean = icc_pt
        icc_sd = np.nan
        icc_hdi_lo, icc_hdi_hi = np.nan, np.nan
        rhat_null = np.nan
        ess_null = np.nan
        ndiv_null = np.nan
        icc_mean_hn = np.nan
        icc_hdi_lo_hn, icc_hdi_hi_hn = np.nan, np.nan  # variante prior HalfNormal
        method_icc = "ANOVA_point_only"

        def _fit_null_icc(sigma_u_prior: str, seed: int):
            """Ajuste le modèle nul ICC avec un prior donné sur sigma_u.

            sigma_u_prior : "halfcauchy" (référence) ou "halfnormal" (sensibilité).
            Retourne (icc_mean, icc_sd, hdi_lo, hdi_hi, rhat, ess, n_divergent) ou None.
            """
            group_idx = pd.Categorical(sub["group_id"], categories=groups).codes
            y_obs = sub[INDIVIDUAL_SCORE_COL].values.astype(float)
            grand_mean_obs = float(np.mean(y_obs))
            y_std = float(np.std(y_obs)) if np.std(y_obs) > 0 else 1.0
            with pm.Model():
                if sigma_u_prior == "halfnormal":
                    sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
                else:
                    sigma_u = pm.HalfCauchy("sigma_u", beta=1.0)
                sigma_e = pm.HalfCauchy("sigma_e", beta=1.0)
                mu_grand = pm.Normal("mu_grand", mu=0.0, sigma=1.0)
                u_j = pm.Normal("u_j", mu=0.0, sigma=sigma_u, shape=K)
                mu_ij = mu_grand + u_j[group_idx]
                y_sc = (y_obs - grand_mean_obs) / y_std
                _ = pm.Normal("y_obs", mu=mu_ij, sigma=sigma_e, observed=y_sc)
                _ = pm.Deterministic("ICC", sigma_u**2 / (sigma_u**2 + sigma_e**2))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _dn, _tn = _mcmc_draws_tune(cfg)
                    tr = pm.sample(draws=_dn, tune=_tn, chains=cfg["chains"], cores=1,
                                   target_accept=0.95, random_seed=seed,
                                   progressbar=False, return_inferencedata=True)
            s = tr.posterior["ICC"].values.flatten()
            hdi = az.hdi(tr, var_names=["ICC"], hdi_prob=0.95)["ICC"].values
            return (float(np.mean(s)), float(np.std(s)), float(hdi[0]), float(hdi[1]),
                    float(az.rhat(tr)["ICC"].values.item()), float(az.ess(tr)["ICC"].values.item()),
                    _count_divergences(tr))

        if _PYMC_OK and bayes:
            try:
                icc_mean, icc_sd, icc_hdi_lo, icc_hdi_hi, rhat_null, ess_null, ndiv_null = \
                    _fit_null_icc("halfcauchy", seed=_MCMC_SEED)
                method_icc = "PyMC5_MCMC_intercepts_aleatoires"
                # Diagnostic 2 : sensibilité du prior de variance (sigma_u HalfNormal).
                try:
                    icc_mean_hn, _sd_hn, icc_hdi_lo_hn, icc_hdi_hi_hn, *_ = \
                        _fit_null_icc("halfnormal", seed=_MCMC_SEED + 7)
                except Exception:
                    pass
            except Exception as exc:
                method_icc = f"PyMC5_echec({type(exc).__name__}); ANOVA_fallback"

        # Le verdict d'agrégation (ICC≥0.30) est-il stable entre les deux priors de sigma_u ?
        _agg_ref = np.isfinite(icc_mean) and icc_mean >= 0.30
        _agg_hn = np.isfinite(icc_mean_hn) and icc_mean_hn >= 0.30
        _prior_var_stable = (not np.isfinite(icc_mean_hn)) or (_agg_ref == _agg_hn)

        row_icc: dict = {
            "dimension": dim,
            "K_groupes": K,
            "N_individus": N,
            "sigma2_between_ANOVA": round(s2b, 4) if np.isfinite(s2b) else np.nan,
            "sigma2_within_ANOVA": round(s2w, 4) if np.isfinite(s2w) else np.nan,
            "ICC_ANOVA": round(icc_pt, 4) if np.isfinite(icc_pt) else np.nan,
            "ICC_bayes_mean": round(icc_mean, 4) if np.isfinite(icc_mean) else np.nan,
            "ICC_bayes_sd": round(icc_sd, 4) if np.isfinite(icc_sd) else np.nan,
            "ICC_hdi95_low": round(icc_hdi_lo, 4) if np.isfinite(icc_hdi_lo) else np.nan,
            "ICC_hdi95_high": round(icc_hdi_hi, 4) if np.isfinite(icc_hdi_hi) else np.nan,
            # Sensibilité au prior de variance (sigma_u ~ HalfNormal(0,1))
            "ICC_bayes_mean_HN": round(icc_mean_hn, 4) if np.isfinite(icc_mean_hn) else np.nan,
            "ICC_hdi95_low_HN": round(icc_hdi_lo_hn, 4) if np.isfinite(icc_hdi_lo_hn) else np.nan,
            "ICC_hdi95_high_HN": round(icc_hdi_hi_hn, 4) if np.isfinite(icc_hdi_hi_hn) else np.nan,
            "prior_var_stable": bool(_prior_var_stable),
            "Rhat_null": round(rhat_null, 4) if np.isfinite(rhat_null) else np.nan,
            "ESS_null": round(ess_null, 0) if np.isfinite(ess_null) else np.nan,
            "n_divergent": int(ndiv_null) if np.isfinite(ndiv_null) and ndiv_null >= 0 else np.nan,
            "method": method_icc,
        }
        icc_rows.append(row_icc)

        # --- Modeles contextuels : {3 composites + affect} x 6 dimensions ---
        # (18 modeles sans affect, 24 avec INCLUDE_AFFECT).
        seed_base = _MCMC_SEED + 100
        for c_idx, cname in enumerate(composite_names):
            # Fusionner les scores individus avec le composite L2
            sub_ctx = sub.merge(
                comp_df[["group_id", cname]].dropna(subset=[cname]),
                on="group_id", how="inner",
            )
            ctx_K = sub_ctx["group_id"].nunique()
            ctx_N = len(sub_ctx)

            if ctx_K < 3 or ctx_N < 6:
                warn_row = {
                    "composite": cname, "dimension": dim,
                    "beta_std": np.nan, "beta_sd": np.nan,
                    "hdi95_low": np.nan, "hdi95_high": np.nan,
                    "robust_hdi": False,
                    "Rhat": np.nan, "ESS": np.nan,
                    "K_groupes": ctx_K, "N_individus": ctx_N,
                    "method": f"insuffisant (K={ctx_K})",
                    "beta_N01": np.nan, "robust_N01": False,
                    "beta_N005": np.nan, "robust_N005": False,
                    "prior_stable": False,
                }
                contextual_rows.append(warn_row)
                if ctx_K < 3:
                    print(f"  [WARN MLM] {cname} x {dim} : K={ctx_K} insuffisant, modele ignore.")
                continue

            seed = seed_base + c_idx * 10 + dims.index(dim)

            # Prior par défaut N(0,1)
            r1 = _run_one_contextual(sub_ctx, cname, prior_sigma=1.0, seed=seed)
            # Sensibilité N(0,0.5)
            r05 = _run_one_contextual(sub_ctx, cname, prior_sigma=0.5, seed=seed + 50)

            # Convergence warnings
            if np.isfinite(r1["rhat"]) and r1["rhat"] > cfg["rhat_threshold"]:
                print(f"  [WARN MLM] Rhat > {cfg['rhat_threshold']} pour {cname} x {dim} (Rhat={r1['rhat']:.4f}).")
            if np.isfinite(r1["ess"]) and r1["ess"] < cfg["ess_threshold"]:
                print(f"  [WARN MLM] ESS < {cfg['ess_threshold']} pour {cname} x {dim} (ESS={r1['ess']:.0f}).")

            def _robust(lo, hi):
                return (np.isfinite(lo) and np.isfinite(hi)) and ((lo > 0) or (hi < 0))

            rob1 = _robust(r1["hdi_lo"], r1["hdi_hi"])
            rob05 = _robust(r05["hdi_lo"], r05["hdi_hi"])
            sign_stable = (
                np.isfinite(r1["beta_mean"]) and np.isfinite(r05["beta_mean"])
                and np.sign(r1["beta_mean"]) == np.sign(r05["beta_mean"])
            )
            prior_stable = sign_stable and (rob1 == rob05)

            contextual_rows.append({
                "composite": cname,
                "dimension": dim,
                "beta_std": round(r1["beta_mean"], 4) if np.isfinite(r1["beta_mean"]) else np.nan,
                "beta_sd": round(r1["beta_sd"], 4) if np.isfinite(r1["beta_sd"]) else np.nan,
                "hdi95_low": round(r1["hdi_lo"], 4) if np.isfinite(r1["hdi_lo"]) else np.nan,
                "hdi95_high": round(r1["hdi_hi"], 4) if np.isfinite(r1["hdi_hi"]) else np.nan,
                "robust_hdi": rob1,
                "Rhat": round(r1["rhat"], 4) if np.isfinite(r1["rhat"]) else np.nan,
                "ESS": round(r1["ess"], 0) if np.isfinite(r1["ess"]) else np.nan,
                "n_divergent": int(r1["n_divergent"]) if np.isfinite(r1.get("n_divergent", np.nan)) and r1["n_divergent"] >= 0 else np.nan,
                "K_groupes": ctx_K,
                "N_individus": ctx_N,
                "method": r1["method"],
                # Colonnes sensibilité aux priors
                "beta_N01": round(r1["beta_mean"], 4) if np.isfinite(r1["beta_mean"]) else np.nan,
                "robust_N01": rob1,
                "beta_N005": round(r05["beta_mean"], 4) if np.isfinite(r05["beta_mean"]) else np.nan,
                "robust_N005": rob05,
                "prior_stable": prior_stable,
            })

    df_ctx = pd.DataFrame(contextual_rows)
    if not df_ctx.empty and "composite" in df_ctx.columns and "beta_std" in df_ctx.columns:
        df_ctx = df_ctx.sort_values(
            ["composite", "beta_std"],
            key=lambda s: s.abs() if s.name == "beta_std" else s,
            ascending=[True, False],
        ).reset_index(drop=True)

    method_str = "PyMC5_MCMC" if (_PYMC_OK and bayes) else ("ANOVA_only" if not bayes else "ANOVA_fallback")
    _n_pred = len(composite_names)  # 3 composites (+ affect si INCLUDE_AFFECT)
    _n_ctx = _n_pred * len(INDIVIDUAL_DIMS)
    _affect_note = (
        f" Le 4e predicteur L2 est la branche affective standalone "
        f"({AFFECT_MARKER}, z-scoree niveau groupe, hors composites transactifs)."
        if INCLUDE_AFFECT else ""
    )
    # Total divergences sur l'ensemble des modeles (nuls + contextuels) pour attester
    # l'absence de pathologie d'echantillonnage (diagnostic 1).
    _ndiv_ctx = pd.to_numeric(df_ctx.get("n_divergent"), errors="coerce").fillna(0).sum() if not df_ctx.empty else 0
    _ndiv_null = sum(int(r.get("n_divergent", 0) or 0) for r in icc_rows if np.isfinite(r.get("n_divergent", np.nan) or np.nan))
    _ndiv_total = int(_ndiv_ctx) + int(_ndiv_null)
    note_impl = (
        f"Implementation : {method_str}. Seed d'echantillonnage fixe = {_MCMC_SEED} (reproductibilite). "
        "Modele nul (intercepts aleatoires) : score_ij ~ Normal(mu + u_j, sigma_e), "
        "mu_grand ~ Normal(0,1), sigma_u/sigma_e ~ HalfCauchy(0,1) (Gelman et al. 2013). "
        "ICC = sigma_u^2 / (sigma_u^2 + sigma_e^2) — moyenne posterieure + HDI 95%. "
        f"{_n_ctx} modeles contextuels ({_n_pred} predicteurs L2 x {len(INDIVIDUAL_DIMS)} dimensions) : "
        "score_ij ~ Normal(alpha + beta_c * X_c_j + u_j, sigma_e), "
        "parametrisation non-centree z_u ~ N(0,1), u_j = z_u * sigma_u (obligatoire a petit K), "
        "alpha ~ N(0,1), beta_c ~ N(0,1) [prior par defaut, Burkner 2017], "
        "sigma_u ~ HalfCauchy(0,1), sigma_e ~ HalfNormal(0,1)."
        f"{_affect_note} "
        "Sensibilite aux priors : (a) coefficient beta_c ~ N(0,0.5) pour chaque modele contextuel ; "
        "(b) variance sigma_u ~ HalfNormal(0,1) pour les modeles nuls ICC (verification que le "
        "verdict d'agregation ICC>=0.30 tient — colonnes *_HN / prior_var_stable). "
        f"Diagnostic de divergences NUTS : {_ndiv_total} transition(s) divergente(s) au total sur "
        f"les {_n_ctx}+{len(INDIVIDUAL_DIMS)} modeles (0 attendu grace a la parametrisation non-centree "
        "+ target_accept=0.95 ; une divergence signale une geometrie en entonnoir de sigma_u). "
        f"MCMC : {cfg['chains']} chaines, {cfg['iter']} iterations ({cfg['warmup']} warmup), "
        f"target_accept=0.95, convergence sur Rhat < {cfg['rhat_threshold']} et ESS > {cfg['ess_threshold']}. "
        "K=12 groupes VR (11 pour INV_TMS), n=3 individus/groupe — "
        "inference exploratoire uniquement (Maas & Hox 2005 : K >= 30 recommande). "
        "Note : la Cohesion globale n'est PAS modelisee au niveau multiniveau — c'est la moyenne "
        "des sous-dimensions SOC/TSK/COM deja modelisees ; l'inclure serait redondant (le resultat "
        f"FDR affect->Cohesion globale a sa contrepartie via les sous-dimensions)."
    )

    return pd.DataFrame(icc_rows), df_ctx, note_impl


# ---------------------------------------------------------------------------
# ENTREE LIGNE DE COMMANDE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path analysis exploratoire VR sur composites formatifs transactifs")
    parser.add_argument("--merged", required=True, help="Chemin du merged_dataset_complete_vr.csv")
    parser.add_argument("--out", required=True, help="Dossier de sortie results/sem/path_analysis_vr")
    parser.add_argument("--n_boot", type=int, default=BOOTSTRAP_B, help=f"Replications bootstrap (defaut {BOOTSTRAP_B})")
    args = parser.parse_args()
    result = run_refined_path_analysis_vr(merged_path=args.merged, output_dir=args.out, n_boot=args.n_boot)
    print(result.get("note", "Termine"))
