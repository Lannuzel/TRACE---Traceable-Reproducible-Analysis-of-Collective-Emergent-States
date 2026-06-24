"""
analyze_inv_structure.py
========================
Analyse exploratoire de la structure des indices non verbaux (INV).

Pipeline :
  1. Chargement des features (high_level_features_audit.csv)
  2. [Optionnel] Filtrage VR uniquement (--mode vr-only)
  3. Sélection des features exploitables (exclusion des colonnes de metadata,
     des doublons z-score, et des colonnes à trop de valeurs manquantes)
  4. Imputation médiane des valeurs manquantes restantes
  5. Standardisation (z-score)
  6. Matrice de corrélation + heatmap
  7. PCA (pourquoi PCA : réduction de redondance entre features corrélées,
     extraction de dimensions interactionnelles latentes pour les analyses stats)
  8. Extraction des composantes retenues (eigenvalue >1 ou variance cumulée ≥70%)
  9. Clustering hiérarchique des features (dendrogramme)
  10. Projection PCA des groupes
  11. PCA par modalité (PC vs VR) [seulement en mode 'all']

Contrainte : n=19 observations — les résultats sont exploratoires, à interpréter
avec prudence. Les composantes extraites serviront d'entrée pour les analyses
corrélation/régression avec performance, TCI et questionnaires.

Note méthodologique audio :
- les métriques d'interruption canoniques sont désormais overlap-based
  (`interruptions_rate_per_min`, `successful_interruption_ratio`,
  `n_attempted_interruptions`, `n_successful_interruptions`) ;
- les anciennes colonnes `int_*`, quand elles existent encore dans un ancien
  export, correspondent à des prises de tour rapides legacy et sont exclues de
  la PCA pour éviter une ambiguïté sémantique.

Usage :
    # Mode par défaut (PC + VR)
    python analyze_inv_structure.py
    python analyze_inv_structure.py --data path/to/features.csv --out path/to/output
    
    # Mode VR uniquement
    python analyze_inv_structure.py --mode vr-only
    python analyze_inv_structure.py --mode vr-only --min-row-completeness 0.50
    
Arguments :
    --data              Fichier CSV source (défaut: high_level_features_audit.csv)
    --out               Dossier de sortie (auto: results_inv_structure[_vr_only])
    --mode              'all' (PC+VR, défaut) ou 'vr-only' (VR uniquement)
    --max-missing       Seuil max NaN par feature (défaut: 0.20)
    --min-cumvar        Variance cumulée min pour rétention (défaut: 0.70)
    --prune-threshold   Seuil |r| pour suppression redondances (défaut: 0.80)
    --min-row-completeness  [vr-only] Min valeurs non manquantes par ligne (défaut: 0.60)
"""

from __future__ import annotations

import argparse
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Import configuration centralisée
import sys
_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    ID_COLS,
    EXCLUDE_SUFFIXES,
    EXCLUDE_PREFIXES,
    MODALITY_PREFIX,
    REDUNDANCY_CORR_THRESHOLD,
    PRUNING_PROTECTED_PAIRS,
    FEATURE_PRIORITY,
    AUDIO_ALIAS_PAIRS,
)
from config.inv_features_config import infer_family_from_name, is_excluded_inv_feature

# Import optionnel pour rotation varimax
try:
    from scipy.linalg import svd
    HAS_SCIPY_SVD = True
except ImportError:
    HAS_SCIPY_SVD = False

warnings.filterwarnings("ignore")

# Seuil maximal de valeurs manquantes autorisé par feature
MAX_MISSING_RATIO = 0.20

# Colonnes legacy à exclure explicitement de la PCA si elles apparaissent encore
# dans un ancien export audio. Elles ne décrivent pas les interruptions
# overlap-based canoniques mais des prises de tour rapides.
LEGACY_AUDIO_EXCLUDE_PREFIXES = ("int_",)

# Suffixes de merge pandas à exclure : colonnes dupliquées _x/_y issues d'un merge.
# La version sans suffixe (canonique) est conservée ; les doublons _x/_y sont bruit.
MERGE_DUPLICATE_SUFFIXES = ("_x", "_y")

# Variables questionnaire / externes qui ne sont pas des INV et doivent être exclues
# de la PCA (elles apparaissent dans merged_dataset mais pas dans high_level_features).
NON_INV_EXACT = frozenset({
    "COM", "COR", "CRE", "SOC", "SPE", "TSK",
    "Cohesion_questionnaire_score", "COHESION", "COHESION_score",
    "c_score", "c_score_allowed", "Score_perf_tsk", "M1", "M2", "M3", "M4",
    "rme_mean", "rme_max", "rme_min",
})

# Couleurs par modalité pour les plots
MODALITY_COLORS = {"PC": "#4e79a7", "VR": "#e15759"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_heatmap(
    ax,
    data: pd.DataFrame,
    *,
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
    center: float = 0,
    annot: bool = False,
    fmt: str = ".2f",
    annot_kws=None,
    linewidths: float = 0.3,
    square: bool = False,
    xticklabels: bool = True,
    yticklabels: bool = True,
):
    """
    Rend une heatmap avec seaborn si disponible, sinon via matplotlib pur.

    Le fallback matplotlib évite de bloquer tout le pipeline PCA sur une
    dépendance de visualisation optionnelle absente dans certains environnements.
    """
    if HAS_SEABORN:
        sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annot,
            fmt=fmt,
            annot_kws=annot_kws or {"size": 6},
            linewidths=linewidths,
            square=square,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        return

    values = data.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="equal" if square else "auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if xticklabels:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(list(data.columns))
    else:
        ax.set_xticks([])

    if yticklabels:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(list(data.index))
    else:
        ax.set_yticks([])

    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=linewidths)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annot:
        annot_size = (annot_kws or {}).get("size", 6)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = values[i, j]
                if np.isnan(val):
                    text = ""
                else:
                    text = f"{val:{fmt}}"
                ax.text(j, i, text, ha="center", va="center", fontsize=annot_size, color="black")

def load_data(path: Path) -> pd.DataFrame:
    """Charge le fichier de features INV."""
    df = pd.read_csv(path)
    for col_alias in ["group", "groupe"]:
        if col_alias in df.columns and "group_id" not in df.columns:
            df = df.rename(columns={col_alias: "group_id"})
    if "condition" not in df.columns and "modalite" in df.columns:
        df = df.rename(columns={"modalite": "condition"})
    return df


def select_features(df: pd.DataFrame, max_missing: float = MAX_MISSING_RATIO) -> list:
    """
    Sélectionne les colonnes numériques exploitables :
      - exclut les colonnes identifiant (ID_COLS)
      - exclut les colonnes _source et z_ (redondances)
      - exclut les anciennes colonnes audio `int_*` si présentes
      - exclut les colonnes avec plus de max_missing de NaN
    """
    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ID_COLS
        and c not in NON_INV_EXACT
        and not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)
        and not any(c.endswith(suf) for suf in MERGE_DUPLICATE_SUFFIXES)
        and not any(c.startswith(pre) for pre in EXCLUDE_PREFIXES)
        and not any(c.startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES)
        and not is_excluded_inv_feature(c)
    ]
    legacy_audio = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and any(c.startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES)
    ]
    excluded_inv = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ID_COLS
        and not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)
        and not any(c.startswith(pre) for pre in EXCLUDE_PREFIXES)
        and not any(c.startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES)
        and is_excluded_inv_feature(c)
    ]
    miss = df[numeric].isnull().mean()
    selected = [c for c in numeric if miss[c] <= max_missing]
    excluded = [c for c in numeric if miss[c] > max_missing]
    print(f"[FEATURES] {len(numeric)} candidates -> {len(selected)} retenues "
          f"(seuil manquant <={max_missing*100:.0f}%)")
    if excluded:
        print(f"  Exclues (>{max_missing*100:.0f}% NaN): {excluded}")
    if legacy_audio:
        print(f"  Exclues (legacy audio non canoniques): {legacy_audio}")
    if excluded_inv:
        print(f"  Exclues (AU individuelles / ratios exclus): {excluded_inv}")
    return selected


def prepare_matrix(df: pd.DataFrame, features: list) -> tuple:
    """
    Retourne la matrice standardisée et la liste des features finales.
    - Imputation médiane des NaN résiduels (stratégie prudente)
    - Suppression des features constantes
    - Standardisation z-score
    """
    X = df[features].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    stds = X_imp.std(axis=0)
    keep_mask = stds > 1e-6
    features_final = [f for f, k in zip(features, keep_mask) if k]
    X_imp = X_imp[:, keep_mask]
    dropped = [f for f, k in zip(features, keep_mask) if not k]
    if dropped:
        print(f"  Features constantes supprimées : {dropped}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, features_final


# ---------------------------------------------------------------------------
# Étape 3 — Matrice de corrélation
# ---------------------------------------------------------------------------

def compute_correlation(df_features: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Corrélation de Pearson + export CSV + heatmap."""
    corr = df_features.corr(method="pearson")
    corr.to_csv(out_dir / "inv_correlation_matrix.csv")
    print("[OK] inv_correlation_matrix.csv")

    n = len(corr)
    figsize = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    _render_heatmap(
        ax=ax,
        data=corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=(n <= 22),
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        square=True,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title("Matrice de corrélation — Features INV", fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "corr_matrix_inv.png", dpi=200)
    plt.close()
    print("[OK] corr_matrix_inv.png")
    return corr


# ---------------------------------------------------------------------------
# Rotation Varimax
# ---------------------------------------------------------------------------

def varimax_rotation(loadings: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> tuple:
    """
    Rotation varimax des loadings PCA.
    
    La rotation varimax maximise la variance des carrés des loadings dans chaque
    colonne (composante), ce qui conduit à une structure plus simple où chaque
    variable charge fortement sur un petit nombre de facteurs.
    
    Args:
        loadings: Matrice (n_features × n_components) des loadings originaux
        max_iter: Nombre maximal d'itérations
        tol: Seuil de convergence
    
    Returns:
        rotated_loadings: Matrice des loadings après rotation
        rotation_matrix: Matrice de rotation T telle que rotated = loadings @ T
    """
    n_features, n_comp = loadings.shape
    
    # Initialiser la matrice de rotation comme identité
    T = np.eye(n_comp)
    
    for iteration in range(max_iter):
        # Appliquer la rotation courante
        rotated = loadings @ T
        
        # Critère varimax : maximiser sum(var(loadings^2)) par composante
        # Normaliser par ligne (communalité)
        h = np.sqrt(np.sum(rotated ** 2, axis=1, keepdims=True))
        h[h < 1e-10] = 1  # éviter division par zéro
        normalized = rotated / h
        
        # Calculer la mise à jour SVD
        A = normalized.T @ (n_features * normalized ** 3 - normalized @ np.diag(np.sum(normalized ** 2, axis=0)))
        U, _, Vt = np.linalg.svd(A)
        T_new = U @ Vt
        
        # Vérifier convergence
        diff = np.max(np.abs(T_new - T))
        T = T_new
        
        if diff < tol:
            print(f"  [VARIMAX] Convergé en {iteration + 1} itérations")
            break
    else:
        print(f"  [VARIMAX] Max iterations ({max_iter}) atteint, diff={diff:.6f}")
    
    rotated_loadings = loadings @ T
    return rotated_loadings, T


# ---------------------------------------------------------------------------
# Étape 4 — PCA
# ---------------------------------------------------------------------------

def run_pca(X_scaled: np.ndarray, features: list, out_dir: Path, rotation: str = "none") -> tuple:
    """
    PCA complète sur les features standardisées.
    Justification : la PCA identifie les directions de variance maximale dans
    l'espace des INV, permettant de condenser l'information redondante en
    quelques dimensions interprétables (ex : 'coordination visuelle',
    'synchronie affective', 'régulation de la parole').
    Les composantes extraites sont orthogonales (non corrélées), ce qui
    satisfait les hypothèses des analyses de régression ultérieures.
    
    Args:
        X_scaled: Matrice de données standardisées (n_samples × n_features)
        features: Liste des noms de features
        out_dir: Dossier de sortie
        rotation: Type de rotation ('none' ou 'varimax')
    
    Returns:
        pca: Objet PCA sklearn
        scores: Scores des observations sur les composantes
    """
    n_comp = min(len(features), X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_scaled)

    # --- Variance expliquée ---
    ev_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(n_comp)],
        "eigenvalue": pca.explained_variance_,
        "variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    ev_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)
    print("[OK] pca_explained_variance.csv")

    # --- Scree plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.bar(range(1, n_comp + 1), pca.explained_variance_,
           color="#4e79a7", edgecolor="white", alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="eigenvalue = 1 (Kaiser)")
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree plot — eigenvalues")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, n_comp + 1))

    ax = axes[1]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, n_comp + 1), cumvar, "o-", color="#e15759", linewidth=2)
    ax.fill_between(range(1, n_comp + 1), cumvar, alpha=0.15, color="#e15759")
    ax.axhline(70, color="grey", linestyle="--", linewidth=1, label="70%")
    ax.axhline(80, color="orange", linestyle="--", linewidth=1, label="80%")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance cumulée expliquée")
    ax.set_title("Variance cumulée expliquée")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, n_comp + 1))

    plt.tight_layout()
    plt.savefig(out_dir / "pca_scree_plot.png", dpi=200)
    plt.close()
    print("[OK] pca_scree_plot.png")

    # --- Loadings (avec rotation optionnelle) ---
    raw_loadings = pca.components_.T.copy()  # (n_features × n_comp)
    
    if rotation == "varimax":
        print(f"  [ROTATION] Application de la rotation varimax...")
        rotated_loadings, rotation_matrix = varimax_rotation(raw_loadings)
        loadings_to_use = rotated_loadings
        rotation_suffix = "_varimax"
        
        # Sauvegarder la matrice de rotation
        rot_df = pd.DataFrame(
            rotation_matrix,
            index=[f"PC{i+1}" for i in range(n_comp)],
            columns=[f"RC{i+1}" for i in range(n_comp)],
        )
        rot_df.to_csv(out_dir / "pca_rotation_matrix.csv")
        print("[OK] pca_rotation_matrix.csv")
        
        # Recalculer les scores avec la rotation
        scores = scores @ rotation_matrix
    else:
        loadings_to_use = raw_loadings
        rotation_suffix = ""
    
    # --- Loadings CSV ---
    loadings_df = pd.DataFrame(
        loadings_to_use,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    loadings_df.to_csv(out_dir / f"pca_loadings{rotation_suffix}.csv")
    print(f"[OK] pca_loadings{rotation_suffix}.csv")
    
    # Sauvegarder aussi les loadings bruts (sans rotation) si rotation appliquée
    if rotation != "none":
        loadings_raw_df = pd.DataFrame(
            raw_loadings,
            index=features,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        )
        loadings_raw_df.to_csv(out_dir / "pca_loadings_raw.csv")
        print("[OK] pca_loadings_raw.csv")
    
    # --- Tableau complet des loadings pour le rapport (format style matrice de corrélation) ---
    # Variables en lignes, composantes en colonnes, valeurs arrondies
    # TRIÉES par importance (valeur absolue du loading) pour chaque PC
    loadings_full = loadings_df.copy()
    loadings_full = loadings_full.round(3)
    loadings_full.index.name = "Variable"
    
    # Créer un tableau trié par importance du loading dans PC1
    loadings_full["_max_abs_loading"] = loadings_full.abs().max(axis=1)
    loadings_full["_dominant_PC"] = loadings_full.iloc[:, :n_comp].abs().idxmax(axis=1)
    loadings_sorted = loadings_full.sort_values("_max_abs_loading", ascending=False)
    loadings_sorted = loadings_sorted.drop(columns=["_max_abs_loading", "_dominant_PC"])
    loadings_sorted.reset_index().to_csv(out_dir / "pca_loadings_full_table.csv", index=False)
    print("[OK] pca_loadings_full_table.csv (format tableau rapport, trié par importance)")
    
    # --- Tableau avec variables triées par importance pour CHAQUE PC ---
    loadings_by_pc = {}
    for pc_col in [f"PC{i+1}" for i in range(min(n_comp, 5))]:
        if pc_col not in loadings_df.columns:
            continue
        pc_df = loadings_df[[pc_col]].copy()
        pc_df["abs_loading"] = pc_df[pc_col].abs()
        pc_df = pc_df.sort_values("abs_loading", ascending=False)
        pc_df["importance"] = ["***" if v > 0.7 else "**" if v > 0.5 else "*" if v > 0.3 else "" 
                              for v in pc_df["abs_loading"]]
        pc_df = pc_df.drop(columns=["abs_loading"])
        pc_df.index.name = "Variable"
        loadings_by_pc[pc_col] = pc_df
        
        # Sauvegarder le tableau trié par PC
        pc_df.reset_index().to_csv(out_dir / f"pca_loadings_sorted_{pc_col}.csv", index=False)
    
    print(f"[OK] pca_loadings_sorted_PC*.csv (tableaux triés par composante)")

    # --- Heatmap des loadings (composantes retenues) ---
    # Variables triées par importance du loading max
    n_show = min(n_comp, 8)
    fig_h = max(6, len(features) * 0.3)
    fig, ax = plt.subplots(figsize=(n_show * 1.2 + 2, fig_h))
    title_suffix = " (varimax)" if rotation == "varimax" else ""
    
    # Trier les variables par importance max pour une meilleure lisibilité
    loadings_for_heatmap = loadings_df.iloc[:, :n_show].copy()
    loadings_for_heatmap["_max_abs"] = loadings_for_heatmap.abs().max(axis=1)
    loadings_for_heatmap = loadings_for_heatmap.sort_values("_max_abs", ascending=True)
    loadings_for_heatmap = loadings_for_heatmap.drop(columns=["_max_abs"])
    
    _render_heatmap(
        ax=ax,
        data=loadings_for_heatmap,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.3,
    )
    ax.set_title(f"Loadings PCA (PC1–PC{n_show}){title_suffix}\n(variables triées par importance)", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    
    # Ajouter des marqueurs pour les loadings élevés (>0.5)
    for i, var in enumerate(loadings_for_heatmap.index):
        for j, pc in enumerate(loadings_for_heatmap.columns):
            val = abs(loadings_for_heatmap.loc[var, pc])
            if val > 0.5:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                            edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(out_dir / f"pca_loadings_heatmap{rotation_suffix}.png", dpi=200)
    plt.close()
    print(f"[OK] pca_loadings_heatmap{rotation_suffix}.png (avec mise en évidence)")

    return pca, scores


# ---------------------------------------------------------------------------
# Étape 5 — Dimensions retenues
# ---------------------------------------------------------------------------

def extract_dimensions(
    pca: PCA,
    scores: np.ndarray,
    df_meta: pd.DataFrame,
    out_dir: Path,
    min_cumvar: float = 0.70,
) -> pd.DataFrame:
    """
    Sélection des composantes selon eigenvalue >1 (critère Kaiser)
    OU variance cumulée ≥ min_cumvar.
    Le dataset produit est prêt pour merge avec performanceanalyse_TCIquestionnaire.
    """
    eigenvalues = pca.explained_variance_
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    n_kaiser = int(np.sum(eigenvalues > 1.0))
    n_cumvar = int(np.searchsorted(cumvar, min_cumvar) + 1)
    n_retain = max(n_kaiser, n_cumvar, 2)
    n_retain = min(n_retain, scores.shape[1])

    print(f"  Critère Kaiser (eigenvalue>1)         : {n_kaiser} composantes")
    print(f"  Critère variance cumulée >={min_cumvar*100:.0f}%          : {n_cumvar} composantes")
    print(f"  -> {n_retain} composantes retenues")

    pc_cols = [f"PC{i+1}" for i in range(n_retain)]
    dim_df = pd.DataFrame(scores[:, :n_retain], columns=pc_cols)

    meta_cols = [c for c in ["group_id", "condition", "scenario", "timepoint"] if c in df_meta.columns]
    for col in reversed(meta_cols):
        dim_df.insert(0, col, df_meta[col].values)

    dim_df.to_csv(out_dir / "inv_dimensions.csv", index=False)
    print("[OK] inv_dimensions.csv")
    return dim_df, n_retain


# ---------------------------------------------------------------------------
# Étape 6 — Clustering hiérarchique des features
# ---------------------------------------------------------------------------

def cluster_features(corr: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Clustering hiérarchique basé sur la distance de corrélation (1 − |r|).
    Méthode Ward : minimise la variance intra-cluster.
    Permet d'identifier des familles d'indices qui mesurent des choses similaires
    (ex : 'features audio', 'synchronie faciale', 'attention visuelle conjointe').
    """
    dist_matrix = 1 - corr.abs()
    np.fill_diagonal(dist_matrix.values, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.clip(dist_matrix.values, 0, None, out=dist_matrix.values)

    condensed = squareform(dist_matrix.values, checks=False)
    Z = linkage(condensed, method="ward")

    n_feat = len(corr)
    fig_h = max(6, n_feat * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    dendrogram(
        Z,
        labels=list(corr.index),
        orientation="left",
        leaf_font_size=8,
        color_threshold=0.6 * Z[-1, 2],
        ax=ax,
    )
    ax.set_title(
        "Clustering hiérarchique des features INV\n(distance = 1 − |r|, méthode Ward)",
        fontsize=11,
    )
    ax.set_xlabel("Distance (Ward)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_dendrogram.png", dpi=200)
    plt.close()
    print("[OK] feature_dendrogram.png")

    threshold = 0.5 * Z[-1, 2]
    labels = fcluster(Z, t=threshold, criterion="distance")
    cluster_df = pd.DataFrame({
        "feature": list(corr.index),
        "cluster": labels,
    }).sort_values(["cluster", "feature"]).reset_index(drop=True)
    cluster_df.to_csv(out_dir / "feature_clusters.csv", index=False)
    print("[OK] feature_clusters.csv")
    return cluster_df


# ---------------------------------------------------------------------------
# Étape 7 — Projection PCA des groupes
# ---------------------------------------------------------------------------

# Couleurs distinctes pour les scénarios
SCENARIO_COLORS = {
    "S1": "#e74c3c",   # rouge
    "S2": "#3498db",   # bleu
}

def plot_pca_projection(dim_df: pd.DataFrame, out_dir: Path):
    """
    Scatter plot PC1 vs PC2.
    - Couleurs distinctes par scénario (S1=rouge, S2=bleu)
    - Marqueurs par condition/modalité (PC=carré, VR=cercle)
    """
    if "PC1" not in dim_df.columns or "PC2" not in dim_df.columns:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    
    # Marqueurs par condition/modalité
    cond_markers = {"PC": "s", "VR": "o", "": "D"}

    has_cond = "condition" in dim_df.columns
    has_scen = "scenario" in dim_df.columns
    group_by = [c for c in ["scenario", "condition"] if c in dim_df.columns]

    if group_by:
        for keys, sub in dim_df.groupby(group_by, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            
            # Extraire scénario et condition
            if has_scen and has_cond:
                scen = str(keys[0]).upper() if keys[0] else ""
                cond = str(keys[1]).upper() if len(keys) > 1 and keys[1] else ""
            elif has_scen:
                scen = str(keys[0]).upper() if keys[0] else ""
                cond = ""
            elif has_cond:
                scen = ""
                cond = str(keys[0]).upper() if keys[0] else ""
            else:
                scen, cond = "", ""
            
            # Couleur par scénario (critère principal de distinction visuelle)
            color = SCENARIO_COLORS.get(scen, "#888888")
            # Marqueur par condition
            marker = cond_markers.get(cond, "D")
            
            label = f"{scen} {cond}".strip() if scen or cond else "Autre"
            ax.scatter(sub["PC1"], sub["PC2"], c=color, marker=marker,
                       s=75, alpha=0.85, label=label,
                       edgecolors="white", linewidths=0.5)
            if "group_id" in sub.columns:
                for _, row in sub.iterrows():
                    ax.annotate(
                        str(row["group_id"]).replace("bim", ""),
                        (row["PC1"], row["PC2"]),
                        fontsize=6, ha="left", va="bottom", alpha=0.65,
                    )
    else:
        ax.scatter(dim_df["PC1"], dim_df["PC2"], s=60, alpha=0.8)

    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("Projection PCA — PC1 vs PC2\n(groupes × condition × scénario)", fontsize=10)

    # Dédupliquer les labels de légende
    handles, labels_ = ax.get_legend_handles_labels()
    seen, unique_h, unique_l = {}, [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, fontsize=8, loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_projection_groups.png", dpi=200)
    plt.close()
    print("[OK] pca_projection_groups.png")


def plot_pca_projection_by_variable(
    dim_df: pd.DataFrame,
    df_full: pd.DataFrame,
    out_dir: Path,
    colorby_cols: list[str] | None = None,
):
    """
    Génère des projections PCA (PC1 vs PC2) colorées par des variables continues.
    
    Permet de visualiser si la performance, les scores TMS/Cohésion ou le C-score
    sont associés à la position dans l'espace factoriel.
    
    Args:
        dim_df: DataFrame avec PC1, PC2, group_id, etc.
        df_full: DataFrame complet avec les variables de coloration
        out_dir: Dossier de sortie
        colorby_cols: Liste de colonnes à utiliser pour la coloration (auto-détecté si None)
    """
    if "PC1" not in dim_df.columns or "PC2" not in dim_df.columns:
        return
    
    if "group_id" not in dim_df.columns or "group_id" not in df_full.columns:
        print("  [SKIP] Projection par variable : group_id manquant")
        return
    
    # Candidats de variables de coloration
    if colorby_cols is None:
        candidates = [
            # Performance
            ("Score_perf_tsk", "Performance"),
            ("Score_perf_tsk_mean", "Performance"),
            ("performance_score", "Performance"),
            ("score_perf_tsk", "Performance"),
            # TMS agrégé (si calculé)
            ("TMS_score", "TMS"),
            ("TMS", "TMS"),
            ("tms_score", "TMS"),
            # Cohésion agrégée (si calculée)
            ("COHESION_score", "Cohésion"),
            ("COHESION", "Cohésion"),
            ("cohesion_score", "Cohésion"),
            # C-score (TCI)
            ("c_score", "C-score"),
            ("C_factor", "C-score"),
            ("c_factor", "C-score"),
        ]
        colorby_cols = []
        for col_name, label in candidates:
            if col_name in df_full.columns:
                colorby_cols.append((col_name, label))
    
    if not colorby_cols:
        print("  [SKIP] Aucune variable de coloration trouvée pour projection PCA")
        return
    
    # Merge les dimensions PCA avec les données complètes
    merge_key = "group_id"
    # Ajouter condition/scenario/timepoint au merge si présent pour éviter doublons
    merge_keys = [merge_key]
    for k in ["condition", "scenario", "timepoint"]:
        if k in dim_df.columns and k in df_full.columns:
            merge_keys.append(k)
    
    merged = dim_df.merge(
        df_full[[merge_key] + [c[0] if isinstance(c, tuple) else c for c in colorby_cols]].drop_duplicates(subset=[merge_key]),
        on=merge_key,
        how="left"
    )
    
    # Générer une figure pour chaque variable
    for item in colorby_cols:
        if isinstance(item, tuple):
            col_name, label = item
        else:
            col_name, label = item, item
        
        if col_name not in merged.columns:
            continue
        
        values = pd.to_numeric(merged[col_name], errors="coerce")
        valid_mask = values.notna()
        if valid_mask.sum() < 3:
            print(f"  [SKIP] Projection by {label}: trop peu de valeurs valides ({valid_mask.sum()})")
            continue
        
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        
        sc = ax.scatter(
            merged.loc[valid_mask, "PC1"],
            merged.loc[valid_mask, "PC2"],
            c=values[valid_mask],
            cmap="viridis",
            s=80,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(label, fontsize=10)
        
        # Annotations groupe
        if "group_id" in merged.columns:
            for _, row in merged[valid_mask].iterrows():
                ax.annotate(
                    str(row["group_id"]).replace("bim", ""),
                    (row["PC1"], row["PC2"]),
                    fontsize=6, ha="left", va="bottom", alpha=0.65,
                )
        
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("PC1", fontsize=10)
        ax.set_ylabel("PC2", fontsize=10)
        ax.set_title(f"Projection PCA — PC1 vs PC2\nColoré par {label}", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_label = label.replace(" ", "_").replace("/", "_")
        fig_path = out_dir / f"pca_projection_by_{safe_label}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[OK] pca_projection_by_{safe_label}.png")


# ---------------------------------------------------------------------------
# Étape 8 — PCA par modalité (PC vs VR)
# ---------------------------------------------------------------------------

def pca_by_modality(df: pd.DataFrame, features: list, out_dir: Path):
    """
    PCA séparée PC vs VR pour tester si la structure des INV diffère
    entre modalités. Compare variance expliquée et loadings.
    """
    if "condition" not in df.columns:
        return

    all_loadings = {}
    for mod in sorted(df["condition"].dropna().unique()):
        sub = df[df["condition"] == mod]
        if len(sub) < 4:
            print(f"  [SKIP] {mod} — trop peu de données (n={len(sub)})")
            continue
        try:
            X_sub = sub[features].copy()
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X_sub)
            stds = X_imp.std(axis=0)
            valid = stds > 1e-6
            X_imp = X_imp[:, valid]
            feat_valid = [f for f, v in zip(features, valid) if v]
            if len(feat_valid) < 2:
                continue
            X_s = StandardScaler().fit_transform(X_imp)
            n_comp = min(len(feat_valid), X_s.shape[0] - 1, 6)
            pca_m = PCA(n_components=n_comp).fit(X_s)
            loadings_m = pd.DataFrame(
                pca_m.components_.T,
                index=feat_valid,
                columns=[f"PC{i+1}" for i in range(n_comp)],
            )
            loadings_m.to_csv(out_dir / f"pca_loadings_{mod}.csv")
            print(f"[OK] pca_loadings_{mod}.csv")
            all_loadings[mod] = (pca_m, loadings_m, feat_valid)
        except Exception as e:
            print(f"  [WARN] PCA {mod} : {e}")

    if len(all_loadings) < 2:
        return

    # Comparaison variance expliquée cumulée
    fig, axes = plt.subplots(1, len(all_loadings), figsize=(5.5 * len(all_loadings), 4.5), sharey=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    for ax, (mod, (pca_m, _, _)) in zip(axes, all_loadings.items()):
        cumvar = np.cumsum(pca_m.explained_variance_ratio_) * 100
        n = len(cumvar)
        ax.plot(range(1, n + 1), cumvar, "o-",
                color=MODALITY_COLORS.get(mod, "#888"), linewidth=2, label=mod)
        ax.fill_between(range(1, n + 1), cumvar, alpha=0.15,
                        color=MODALITY_COLORS.get(mod, "#888"))
        ax.axhline(70, color="grey", linestyle="--", linewidth=0.8, label="70%")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title(f"PCA — {mod}")
        ax.set_xlabel("N composantes")
        if ax is axes[0]:
            ax.set_ylabel("Variance cumulée")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Variance cumulée PCA par modalité (PC vs VR)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_variance_by_modality.png", dpi=200)
    plt.close()
    print("[OK] pca_variance_by_modality.png")


# ---------------------------------------------------------------------------
# Étape 9 — PCA par type de modalité INV (audio / face / gaze)
# Objectif : détecter la redondance INTERNE à chaque type de feature
# ---------------------------------------------------------------------------

# NOTE : MODALITY_PREFIX, REDUNDANCY_CORR_THRESHOLD, FEATURE_PRIORITY et
# AUDIO_ALIAS_PAIRS sont importés depuis config.inv_features_config

# Paramètres bootstrap PCA (stabilité des composantes)
BOOTSTRAP_N_ITER = 300
BOOTSTRAP_RANDOM_SEED = 42

# NOTE : FEATURE_PRIORITY est maintenant importée depuis config.inv_features_config
# La liste complète y est documentée avec les justifications de chaque exclusion.


# ---------------------------------------------------------------------------
# Étape 3b — Hard pruning par corrélation (|r| > seuil)
# ---------------------------------------------------------------------------

def remove_correlated_features(
    df_features: pd.DataFrame,
    corr: pd.DataFrame,
    df_raw: pd.DataFrame | None = None,
    threshold: float = REDUNDANCY_CORR_THRESHOLD,
    out_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Supprime les features redondantes (|r| > threshold) — hard pruning.

    Algorithme greedy (ordre décroissant de |r|) :
      Pour chaque paire active, supprimer la feature la moins prioritaire selon :
        1. NaN dans df_raw (moins = mieux — robustesse)
        2. Rang dans FEATURE_PRIORITY (interprétabilité métier)
        3. Longueur du nom (plus court = plus canonique / simple)

    Exporte : inv_pruned_features.csv
    """
    # Préférence de pruning PCA : pilotée par `priority` dans FEATURE_PRIORITY
    # (inv_features_config.py). Plus petit entier = conservé en priorité.
    # Pour modifier les préférences PCA, ajuster les valeurs `priority`
    # dans INV_FEATURES, pas la logique ici.
    features = list(corr.index)

    # Compter les NaN sur les données brutes (avant imputation)
    if df_raw is not None:
        nan_counts = df_raw[[f for f in features if f in df_raw.columns]].isnull().sum()
        nan_counts = nan_counts.reindex(features, fill_value=0)
    else:
        nan_counts = pd.Series(0, index=features)

    def _priority(feat: str) -> int:
        try:
            return FEATURE_PRIORITY.index(feat)
        except ValueError:
            return len(FEATURE_PRIORITY)

    # Pairs triées par |r| décroissant
    upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "abs_r"})
        .sort_values("abs_r", ascending=False)
    )
    pairs = pairs[pairs["abs_r"] > threshold]

    dropped: set[str] = set()
    drop_reasons: dict[str, str] = {}

    for _, row in pairs.iterrows():
        feat_a, feat_b = row["feat_a"], row["feat_b"]
        abs_r = float(row["abs_r"])
        if feat_a in dropped or feat_b in dropped:
            continue
        if frozenset({feat_a, feat_b}) in PRUNING_PROTECTED_PAIRS:
            continue

        nan_a = int(nan_counts.get(feat_a, 0))
        nan_b = int(nan_counts.get(feat_b, 0))
        prio_a = _priority(feat_a)
        prio_b = _priority(feat_b)

        if nan_a < nan_b:
            to_drop, to_keep = feat_b, feat_a
        elif nan_b < nan_a:
            to_drop, to_keep = feat_a, feat_b
        elif prio_a < prio_b:
            to_drop, to_keep = feat_b, feat_a
        elif prio_b < prio_a:
            to_drop, to_keep = feat_a, feat_b
        else:
            to_drop = feat_b if len(feat_a) <= len(feat_b) else feat_a
            to_keep = feat_a if to_drop == feat_b else feat_b

        dropped.add(to_drop)
        drop_reasons[to_drop] = (
            f"|r|={abs_r:.3f} > {threshold} avec '{to_keep}' "
            f"(NaN: keep={nan_counts.get(to_keep, 0)}, drop={nan_counts.get(to_drop, 0)})"
        )

    features_kept = [f for f in features if f not in dropped]

    report_rows = [
        {"feature": f, "kept": int(f not in dropped), "reason": drop_reasons.get(f, "")}
        for f in features
    ]
    pruning_report = pd.DataFrame(report_rows)

    print(f"  [PRUNING] {len(features)} features -> {len(features_kept)} conservées "
          f"({len(dropped)} supprimées, |r| > {threshold})")
    for f in sorted(dropped):
        print(f"    DROP  {f:<50}  {drop_reasons[f]}")

    if out_dir is not None:
        pruning_report.to_csv(out_dir / "inv_pruned_features.csv", index=False)
        print("[OK] inv_pruned_features.csv")

    return features_kept, pruning_report


# NOTE : AUDIO_ALIAS_PAIRS est maintenant importée depuis config.inv_features_config


def _assign_inv_modality(feature: str) -> str | None:
    """
    Retourne la famille INV d'une feature en s'appuyant sur la config centrale.

    Important :
    - on utilise la logique de `infer_family_from_name()` pour respecter les
      alias et les préfixes documentés ;
    - on évite les matchs "substring" trop permissifs qui reclasseraient par
      erreur des features face comme `joy_sync_*` dans la famille audio à cause
      du préfixe audio `sync_`.
    """
    return infer_family_from_name(feature)


def _deduplicate_audio_aliases(mod_features: list[str]) -> list[str]:
    """Supprime les alias audio bruts quand la version `audio_*` est présente."""
    feat_set = set(mod_features)
    to_drop: set[str] = set()
    for canonical, raw_alias in AUDIO_ALIAS_PAIRS:
        if canonical in feat_set and raw_alias in feat_set:
            to_drop.add(raw_alias)

    if to_drop:
        print(f"  [INFO] AUDIO alias supprimés: {sorted(to_drop)}")
    return [f for f in mod_features if f not in to_drop]


def _bootstrap_pca_stability(
    sub_valid: pd.DataFrame,
    pca_ref: PCA,
    feat_valid: list[str],
    out_dir: Path,
    modality: str,
    imp_fitted: SimpleImputer | None = None,
    n_iter: int = BOOTSTRAP_N_ITER,
    seed: int = BOOTSTRAP_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap des composantes PCA d'une famille pour tester leur stabilité.

    On utilise l'imputer deja fitte sur les donnees completes (transform seulement)
    pour eviter que sklearn ne drop des colonnes all-NaN dans certains samples bootstrap
    (comportement keepempty_features=False, defaut sklearn >= 1.1).

    Exports:
      - pca_bootstrap_loading_stability_<modality>.csv
      - pca_bootstrap_component_stability_<modality>.csv
    """
    rng = np.random.default_rng(seed)
    n_obs = len(sub_valid)
    n_comp = pca_ref.n_components_
    ref_loadings = pca_ref.components_.copy()

    load_rows: list[dict] = []
    comp_rows: list[dict] = []

    for _ in range(n_iter):
        idx = rng.integers(0, n_obs, size=n_obs)
        X_b = sub_valid.iloc[idx].copy()

        # Utiliser l'imputer fitté sur les données complètes pour éviter le drop
        # de colonnes all-NaN dans les bootstrap samples (sklearn ≥ 1.1).
        if imp_fitted is not None:
            X_imp = imp_fitted.transform(X_b)
        else:
            X_imp = SimpleImputer(strategy="median").fit_transform(X_b)

        # Vérifier que le nombre de colonnes correspond à feat_valid
        if X_imp.shape[1] != len(feat_valid):
            continue  # sample instable, skip

        X_s = StandardScaler().fit_transform(X_imp)

        try:
            pca_b = PCA(n_components=n_comp).fit(X_s)
        except Exception:
            continue

        boot_load = pca_b.components_.copy()

        # Alignement de signe des composantes (indétermination de signe en PCA)
        for j in range(n_comp):
            if np.dot(boot_load[j], ref_loadings[j]) < 0:
                boot_load[j] *= -1

            denom = (np.linalg.norm(boot_load[j]) * np.linalg.norm(ref_loadings[j]))
            cosine_sim = float(np.dot(boot_load[j], ref_loadings[j]) / denom) if denom > 0 else np.nan
            comp_rows.append({
                "modality": modality,
                "component": f"PC{j+1}",
                "cosine_similarity": cosine_sim,
            })

            for k, feat in enumerate(feat_valid):
                load_rows.append({
                    "modality": modality,
                    "component": f"PC{j+1}",
                    "feature": feat,
                    "loading": float(boot_load[j, k]),
                })

    load_df = pd.DataFrame(load_rows)
    comp_df = pd.DataFrame(comp_rows)

    if load_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    load_summary = (
        load_df.groupby(["modality", "component", "feature"], as_index=False)["loading"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "loading_mean", "std": "loading_std", "median": "loading_median", "count": "n_boot"})
    )

    q = (
        load_df.groupby(["modality", "component", "feature"])["loading"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "loading_ci_low", 0.975: "loading_ci_high"})
    )
    load_summary = load_summary.merge(q, on=["modality", "component", "feature"], how="left")
    load_summary = load_summary.sort_values(["modality", "component", "feature"]).reset_index(drop=True)

    comp_summary = (
        comp_df.groupby(["modality", "component"], as_index=False)["cosine_similarity"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
        .rename(columns={
            "mean": "cosine_mean",
            "std": "cosine_std",
            "median": "cosine_median",
            "count": "n_boot",
        })
    )
    cq = (
        comp_df.groupby(["modality", "component"])["cosine_similarity"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "cosine_ci_low", 0.975: "cosine_ci_high"})
    )
    comp_summary = comp_summary.merge(cq, on=["modality", "component"], how="left")
    comp_summary = comp_summary.sort_values(["modality", "component"]).reset_index(drop=True)

    load_summary.to_csv(out_dir / f"pca_bootstrap_loading_stability_{modality}.csv", index=False)
    comp_summary.to_csv(out_dir / f"pca_bootstrap_component_stability_{modality}.csv", index=False)
    print(f"  [OK] pca_bootstrap_loading_stability_{modality}.csv")
    print(f"  [OK] pca_bootstrap_component_stability_{modality}.csv")

    return load_summary, comp_summary


def _export_full_modality_redundancy(
    df: pd.DataFrame,
    features_all: list[str],
    gaze_sub: pd.DataFrame | None,
    gaze_valid: list[str],
    threshold: float,
    out_dir: Path,
) -> None:
    """
    Calcule et exporte inv_modality_redundancy.csv sur TOUTES les features
    (avant pruning) par famille. Appelée après l'étape 9 pour inclure les
    redondances face/audio qui ont été retirées avant la PCA famille.
    """
    redundancy_rows = []

    # Non-gaze : corrélations pairwise par famille directement sur df
    non_gaze_by_mod: dict[str, list[str]] = {}
    for f in features_all:
        mod = _assign_inv_modality(f)
        if mod and mod != "gaze":
            non_gaze_by_mod.setdefault(mod, []).append(f)

    for modality, mod_features in sorted(non_gaze_by_mod.items()):
        if modality == "audio":
            mod_features = _deduplicate_audio_aliases(mod_features)
        sub = df[mod_features].apply(pd.to_numeric, errors="coerce")
        sub = sub[~sub.isna().all(axis=1)]
        feat_valid = [c for c in mod_features if sub[c].std() > 1e-6]
        if len(feat_valid) < 2:
            continue
        for feat_a, feat_b in combinations(feat_valid, 2):
            s_a, s_b = sub[feat_a], sub[feat_b]
            mask = s_a.notna() & s_b.notna()
            n_pairwise = int(mask.sum())
            if n_pairwise < 3:
                continue
            corr_val = s_a[mask].corr(s_b[mask], method="pearson")
            if pd.notna(corr_val) and abs(corr_val) > threshold:
                redundancy_rows.append({
                    "modality": modality,
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "corr_pearson": round(float(corr_val), 4),
                    "abs_corr": round(float(abs(corr_val)), 4),
                    "n_pairwise": n_pairwise,
                    "threshold": threshold,
                })

    # Gaze : utiliser le sous-ensemble VR calculé en step 9-pre
    if gaze_sub is not None and gaze_valid:
        sub_g = gaze_sub[gaze_valid].apply(pd.to_numeric, errors="coerce")
        for feat_a, feat_b in combinations(gaze_valid, 2):
            s_a, s_b = sub_g[feat_a], sub_g[feat_b]
            mask = s_a.notna() & s_b.notna()
            n_pairwise = int(mask.sum())
            if n_pairwise < 3:
                continue
            corr_val = s_a[mask].corr(s_b[mask], method="pearson")
            if pd.notna(corr_val) and abs(corr_val) > threshold:
                redundancy_rows.append({
                    "modality": "gaze",
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "corr_pearson": round(float(corr_val), 4),
                    "abs_corr": round(float(abs(corr_val)), 4),
                    "n_pairwise": n_pairwise,
                    "threshold": threshold,
                })

    if not redundancy_rows:
        return

    red_df = (
        pd.DataFrame(redundancy_rows)
        .sort_values(["modality", "abs_corr"], ascending=[True, False])
        .reset_index(drop=True)
    )
    red_df.to_csv(out_dir / "inv_modality_redundancy.csv", index=False)
    red_df.to_csv(out_dir / "inv_modality_redundancy_correlations.csv", index=False)
    print(f"  [OK] inv_modality_redundancy.csv mis à jour ({len(red_df)} paires, toutes familles)")


def pca_by_inv_modality(
    df: pd.DataFrame,
    features: list[str],
    out_dir: Path,
) -> dict:
    """
    PCA séparée par famille de features INV (audio, face, gaze).

    Pour chaque famille :
      1. Sélection des features dont le nom appartient à la famille
      2. Vérification du minimum requis (>= 3 variables, >= 4 observations)
      3. Imputation médiane + standardisation
      4. PCA avec n_components = min(6, n_features, n_samples-1)
      5. Export variance expliquée, loadings, scree plot
    6. Détection de redondance pairwise : |corr(feature_i, feature_j)| > 0.80
    7. Bootstrap PCA : stabilité des composantes et des loadings

    Returns : dict {modality: {"pca": PCA, "loadings": DataFrame, ...}}
    """
    # Assigner chaque feature à une famille
    feature_modality = {f: _assign_inv_modality(f) for f in features}
    family_features: dict[str, list[str]] = {}
    for f, mod in feature_modality.items():
        if mod:
            family_features.setdefault(mod, []).append(f)

    unassigned = [f for f, m in feature_modality.items() if m is None]
    if unassigned:
        print(f"  [INFO] {len(unassigned)} features non assignées à une famille (ignorées)")

    redundancy_rows = []
    modality_dimension_rows = []
    results = {}

    for modality, mod_features in sorted(family_features.items()):
        if modality == "audio":
            mod_features = _deduplicate_audio_aliases(mod_features)

        print(f"\n  -- Famille : {modality.upper()} ({len(mod_features)} features) --")

        if len(mod_features) < 2:
            print(f"  [SKIP] Moins de 2 features — analyse non réalisée")
            continue

        sub = df[mod_features].copy()

        # Pour certaines familles (ex. gaze), les données peuvent être disponibles
        # uniquement pour une sous-modalité (ex. VR uniquement).
        # On filtre les lignes où TOUTES les features sont NaN — ce sont des groupes
        # pour lesquels la modalité n'a pas été mesurée (ex. PC sans tracking VR).
        # L'imputation médiane se fait alors sur les observations réellement disponibles.
        row_nan_pct = sub.isna().mean(axis=1)
        all_nan_rows = row_nan_pct == 1.0
        if all_nan_rows.any():
            n_excluded = all_nan_rows.sum()
            print(f"  [INFO] {n_excluded} observations entièrement NaN exclues "
                  f"(modalité non mesurée pour ces groupes)")
            sub = sub[~all_nan_rows]

        n_obs = sub.shape[0]
        if n_obs < 4:
            print(f"  [SKIP] Trop peu d'observations après filtrage NaN (n={n_obs})")
            continue

        # Imputation + suppression features constantes
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(sub)
        stds = X_imp.std(axis=0)
        valid_mask = stds > 1e-6
        X_imp = X_imp[:, valid_mask]
        feat_valid = [f for f, v in zip(mod_features, valid_mask) if v]

        if len(feat_valid) < 2:
            print(f"  [SKIP] Moins de 2 features non-constantes")
            continue

        # Standardisation
        X_s = StandardScaler().fit_transform(X_imp)

        # PCA
        n_comp = min(6, len(feat_valid), n_obs - 1)
        pca_m = PCA(n_components=n_comp)
        pca_m.fit(X_s)

        # --- Variance expliquée ---
        ev_df = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(n_comp)],
            "eigenvalue": pca_m.explained_variance_,
            "variance_ratio": pca_m.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca_m.explained_variance_ratio_),
        }).round(4)
        ev_path = out_dir / f"pca_explained_variance_{modality}.csv"
        ev_df.to_csv(ev_path, index=False)
        print(f"  [OK] pca_explained_variance_{modality}.csv")

        # --- Loadings ---
        loadings_df = pd.DataFrame(
            pca_m.components_.T,
            index=feat_valid,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        ).round(4)
        ld_path = out_dir / f"pca_loadings_{modality}.csv"
        loadings_df.to_csv(ld_path)
        print(f"  [OK] pca_loadings_{modality}.csv")

        # --- Identification des dimensions retenues par modalité ---
        ev_ratio = pca_m.explained_variance_ratio_
        eigen = pca_m.explained_variance_
        cum = np.cumsum(ev_ratio)
        n_kaiser = int(np.sum(eigen > 1.0))
        n_cumvar = int(np.searchsorted(cum, 0.70) + 1)
        n_retain = max(1, n_kaiser, n_cumvar)
        n_retain = min(n_retain, n_comp)

        for j in range(n_comp):
            comp_name = f"PC{j+1}"
            top_idx = np.argsort(np.abs(pca_m.components_[j]))[::-1][:3]
            top_feats = ", ".join([feat_valid[k] for k in top_idx])
            modality_dimension_rows.append({
                "modality": modality,
                "component": comp_name,
                "eigenvalue": round(float(eigen[j]), 4),
                "variance_ratio": round(float(ev_ratio[j]), 4),
                "cumulative_variance": round(float(cum[j]), 4),
                "retained": int(j < n_retain),
                "top_features": top_feats,
            })

        # --- Scree plot ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        ax.bar(range(1, n_comp + 1), pca_m.explained_variance_,
               color="#4e79a7", edgecolor="white", alpha=0.85)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Kaiser (eigenvalue=1)")
        ax.set_xlabel("Composante")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Scree plot — {modality.capitalize()} (n={n_obs})")
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, n_comp + 1))

        ax = axes[1]
        cumvar = np.cumsum(pca_m.explained_variance_ratio_) * 100
        ax.plot(range(1, n_comp + 1), cumvar, "o-", color="#e15759", linewidth=2)
        ax.fill_between(range(1, n_comp + 1), cumvar, alpha=0.15, color="#e15759")
        ax.axhline(70, color="grey", linestyle="--", linewidth=1, label="70%")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_xlabel("N composantes")
        ax.set_ylabel("Variance cumulée")
        ax.set_title(f"Variance cumulée — {modality.capitalize()}")
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, n_comp + 1))

        plt.suptitle(f"PCA par famille INV — {modality.capitalize()}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        scree_path = out_dir / f"pca_scree_{modality}.png"
        plt.savefig(scree_path, dpi=200)
        plt.close()
        print(f"  [OK] pca_scree_{modality}.png")

        # --- Étape 1 : redondance pairwise via corrélation ---
        # Critère demandé : |r| > 0.8 -> features redondantes.
        sub_valid = sub[feat_valid].apply(pd.to_numeric, errors="coerce")
        for feat_a, feat_b in combinations(feat_valid, 2):
            s_a = sub_valid[feat_a]
            s_b = sub_valid[feat_b]
            mask = s_a.notna() & s_b.notna()
            n_pairwise = int(mask.sum())
            if n_pairwise < 3:
                continue

            corr_val = s_a[mask].corr(s_b[mask], method="pearson")
            if pd.notna(corr_val) and abs(corr_val) > REDUNDANCY_CORR_THRESHOLD:
                redundancy_rows.append({
                    "modality": modality,
                    "feature_a": feat_a,
                    "feature_b": feat_b,
                    "corr_pearson": round(float(corr_val), 4),
                    "abs_corr": round(float(abs(corr_val)), 4),
                    "n_pairwise": n_pairwise,
                    "threshold": REDUNDANCY_CORR_THRESHOLD,
                })

        # --- Étape 3 : bootstrap PCA pour la stabilité des composantes ---
        # Fitter un imputer sur sub_valid (feat_valid cols seulement) pour que
        # transform() dans le bootstrap produise toujours le bon nombre de colonnes,
        # même quand certaines features sont all-NaN dans un sample bootstrap.
        imp_valid = SimpleImputer(strategy="median")
        imp_valid.fit(sub_valid)
        _bootstrap_pca_stability(sub_valid, pca_m, feat_valid, out_dir, modality, imp_fitted=imp_valid)

        results[modality] = {
            "pca": pca_m,
            "loadings": loadings_df,
            "features": feat_valid,
            "ev_df": ev_df,
        }

    # --- Export redondances ---
    if modality_dimension_rows:
        dim_mod_df = (
            pd.DataFrame(modality_dimension_rows)
            .sort_values(["modality", "component"])
            .reset_index(drop=True)
        )
        dim_mod_path = out_dir / "inv_modality_dimensions.csv"
        dim_mod_df.to_csv(dim_mod_path, index=False)
        print(f"[OK] inv_modality_dimensions.csv ({len(dim_mod_df)} lignes)")

    if redundancy_rows:
        red_df = (
            pd.DataFrame(redundancy_rows)
            .sort_values(["modality", "abs_corr"], ascending=[True, False])
            .reset_index(drop=True)
        )
        red_path = out_dir / "inv_modality_redundancy.csv"
        red_df.to_csv(red_path, index=False)
        print(f"\n[OK] inv_modality_redundancy.csv ({len(red_df)} paires)")

        # Fichier dédié corrélations (alias explicite)
        red_corr_path = out_dir / "inv_modality_redundancy_correlations.csv"
        red_df.to_csv(red_corr_path, index=False)
        print(f"[OK] inv_modality_redundancy_correlations.csv ({len(red_df)} paires)")
    else:
        print("\n  [INFO] Aucune redondance détectée au seuil |r| > "
              f"{REDUNDANCY_CORR_THRESHOLD}")
        red_df = pd.DataFrame()
        print("[INFO] Aucune paire de features redondantes pour calculer des corrélations.")

    # --- Graphique synthétique : PC1 variance par famille ---
    if results:
        _plot_variance_by_inv_modality(results, out_dir)

    return results


def _plot_variance_by_inv_modality(results: dict, out_dir: Path):
    """
    Graphique synthétique : variance expliquée par PC1 (et PC2) pour chaque famille.
    Permet de voir d'un coup d'œil quelle famille est la plus 'compressible'
    (PC1 élevé = forte redondance interne).
    """
    palette = {"audio": "#4e79a7", "face": "#f28e2b", "gaze": "#76b7b2"}
    fig, ax = plt.subplots(figsize=(8, 4.5))

    modalities = sorted(results.keys())
    x = np.arange(len(modalities))
    width = 0.35

    pc1_vars = []
    pc2_vars = []
    for mod in modalities:
        ev = results[mod]["pca"].explained_variance_ratio_
        pc1_vars.append(ev[0] * 100 if len(ev) > 0 else 0)
        pc2_vars.append(ev[1] * 100 if len(ev) > 1 else 0)

    bars1 = ax.bar(x - width / 2, pc1_vars, width, label="PC1",
                   color=[palette.get(m, "#888") for m in modalities], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, pc2_vars, width, label="PC2",
                   color=[palette.get(m, "#888") for m in modalities], alpha=0.45, edgecolor="white")

    # Valeur sur chaque barre
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities], fontsize=10)
    ax.set_ylabel("Variance expliquée (%)")
    ax.set_title(
        "Variance expliquée par PC1 et PC2 — par famille INV\n"
        "(PC1 élevé = forte redondance interne dans la famille)",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(out_dir / "pca_variance_by_inv_modality.png", dpi=200)
    plt.close()
    print("[OK] pca_variance_by_inv_modality.png")


# ---------------------------------------------------------------------------
# Étape 6b — Sélection d'un représentant par cluster
# ---------------------------------------------------------------------------

def select_representative_per_cluster(
    df_raw: pd.DataFrame,
    cluster_df: pd.DataFrame,
    target_cols: list[str] | None = None,
    out_dir: Path | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Sélectionne 1 feature représentative par cluster (résultats du clustering
    hiérarchique Ward sur distance de corrélation).

    Critère de sélection (ordre de priorité) :
      1. Corrélation moyenne avec les variables cibles (performance, c_score)
         si disponibles dans df_raw — signal prédictif direct
      2. Rang dans FEATURE_PRIORITY (interprétabilité métier)
      3. Moins de NaN — robustesse empirique
      4. Nom le plus court — proxy de simplicité / canonicité

    Exporte : inv_cluster_representatives.csv
              inv_final_selected_features.csv (liste plate)
    """
    report_rows: list[dict] = []
    selected: list[str] = []

    for cl_id in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cl_id]["feature"].tolist()
        members = [f for f in members if f in df_raw.columns]

        if not members:
            continue

        if len(members) == 1:
            selected.append(members[0])
            report_rows.append({
                "cluster": cl_id, "feature": members[0], "selected": 1,
                "score_target_corr": np.nan,
                "n_nan": int(df_raw[members[0]].isna().sum()),
                "reason": "seule feature du cluster",
            })
            continue

        # Score = corrélation moyenne avec cibles (si disponibles)
        valid_targets = [t for t in (target_cols or []) if t in df_raw.columns]
        target_scores: dict[str, float] = {}
        for f in members:
            if valid_targets:
                corrs = [abs(df_raw[f].corr(df_raw[t])) for t in valid_targets]
                corrs = [c for c in corrs if pd.notna(c)]
                target_scores[f] = float(np.mean(corrs)) if corrs else 0.0
            else:
                target_scores[f] = 0.0

        def _sort_key(f: str) -> tuple:
            return (
                -target_scores[f],
                FEATURE_PRIORITY.index(f) if f in FEATURE_PRIORITY else len(FEATURE_PRIORITY),
                int(df_raw[f].isna().sum()),
                len(f),
            )

        best = min(members, key=_sort_key)
        selected.append(best)

        for f in members:
            is_sel = int(f == best)
            report_rows.append({
                "cluster": cl_id,
                "feature": f,
                "selected": is_sel,
                "score_target_corr": round(target_scores[f], 4),
                "n_nan": int(df_raw[f].isna().sum()),
                "reason": "représentant sélectionné" if is_sel else f"non retenu (repr={best})",
            })

    report_df = (
        pd.DataFrame(report_rows)
        .sort_values(["cluster", "selected"], ascending=[True, False])
        .reset_index(drop=True)
    )

    print(f"  [CLUSTER SELECT] {len(selected)} features retenues "
          f"(1 par cluster sur {cluster_df['cluster'].nunique()} clusters)")
    for _, row in report_df[report_df["selected"] == 1].iterrows():
        print(f"    Cluster {int(row['cluster']):2d}: {row['feature']}")

    if out_dir is not None:
        report_df.to_csv(out_dir / "inv_cluster_representatives.csv", index=False)
        print("[OK] inv_cluster_representatives.csv")
        pd.DataFrame({"feature": selected}).to_csv(
            out_dir / "inv_final_selected_features.csv", index=False
        )
        print(f"[OK] inv_final_selected_features.csv ({len(selected)} features)")

    return selected, report_df


# ---------------------------------------------------------------------------
# Résumé console
# ---------------------------------------------------------------------------

def print_summary(
    pca: PCA,
    features: list,
    cluster_df: pd.DataFrame,
    n_retain: int,
    modality_results: dict | None = None,
):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print("\n" + "=" * 65)
    print("RÉSUMÉ — PCA globale des features INV")
    print("=" * 65)
    print(f"  Features retenues   : {len(features)}")
    print(f"  Composantes retenues: {n_retain}")
    print()
    for i in range(min(n_retain, pca.n_components_)):
        ev = pca.explained_variance_ratio_[i]
        print(f"  PC{i+1} : {ev*100:.1f}%  (cumulé: {cumvar[i]*100:.1f}%)")
        top3_idx = np.argsort(np.abs(pca.components_[i]))[::-1][:3]
        for j in top3_idx:
            print(f"       -> {features[j]:<48} ({pca.components_[i][j]:+.3f})")
    print()
    print("Clusters de features :")
    for cl in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cl]["feature"].tolist()
        print(f"  Cluster {cl}: {', '.join(members)}")

    # --- Section PCA par famille INV ---
    if modality_results:
        print()
        print("=" * 65)
        print("ANALYSE PAR FAMILLE INV (redondance des métriques)")
        print("=" * 65)
        print("  Objectif : identifier les métriques mesurant la même chose")
        print(f"  Seuil redondance : |r| > {REDUNDANCY_CORR_THRESHOLD}")
        print()
        for modality, res in sorted(modality_results.items()):
            pca_m = res["pca"]
            feat_m = res["features"]
            ev_ratio = pca_m.explained_variance_ratio_
            print(f"  {modality.upper()} ({len(feat_m)} features)")
            # Afficher PC1 et PC2 avec leurs top features
            for i, pc_var in enumerate(ev_ratio[:min(3, len(ev_ratio))]):
                print(f"    PC{i+1} : {pc_var*100:.0f}%")
                top_idx = np.argsort(np.abs(pca_m.components_[i]))[::-1][:4]
                for j in top_idx:
                    if abs(pca_m.components_[i][j]) > 0.3:
                        print(f"         {feat_m[j]:<45} ({pca_m.components_[i][j]:+.3f})")
            print()

    print()
    print("Utilisation des dimensions (inv_dimensions.csv) :")
    print("  -> merger avec performance_task/recap_scores_all.csv, tci_*.csv, questionnaire_scores.csv")
    print("  -> corréler PC1, PC2 avec Score_perf_tsk, C_factor, dimensions COR/CRE/...")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Analysis Summary
# ---------------------------------------------------------------------------

def build_analysis_summary(
    df: pd.DataFrame,
    features_initial: list[str],
    features_selected: list[str],
    features_final: list[str],
    mode_name: str,
    apply_pruning: bool,
    prune_threshold: float,
    max_missing: float,
    out_dir: Path,
) -> None:
    """
    Génère un fichier résumé de l'analyse avec toutes les informations clés.

    Args:
        df: DataFrame brut complet
        features_initial: Features candidates avant filtrage technique
        features_selected: Features après filtrage technique (NaN, constantes)
        features_final: Features finales utilisées pour la PCA (après pruning si applicable)
        mode_name: "with_pruning" ou "without_pruning"
        apply_pruning: Si le pruning a été appliqué
        prune_threshold: Seuil de corrélation pour le pruning
        max_missing: Seuil max de NaN pour le filtrage technique
        out_dir: Dossier de sortie
    """
    summary = {
        "analysis_mode": mode_name,
        "pruning_applied": "yes" if apply_pruning else "no",
        "n_observations": len(df),
        "n_features_candidates": len(features_initial),
        "n_features_after_technical_filters": len(features_selected),
        "n_features_used_for_pca": len(features_final),
        "n_features_excluded_technical": len(features_initial) - len(features_selected),
        "n_features_excluded_pruning": len(features_selected) - len(features_final) if apply_pruning else 0,
        "technical_filter_max_nan_threshold": max_missing,
        "pruning_correlation_threshold": prune_threshold if apply_pruning else "N/A",
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "analysis_summary.csv", index=False)
    print(f"[OK] analysis_summary.csv")

    # Liste des features utilisées
    features_used_df = pd.DataFrame({"feature": features_final})
    features_used_df.to_csv(out_dir / "inv_features_used.csv", index=False)
    print(f"[OK] inv_features_used.csv ({len(features_final)} features)")

    # Rapport textuel
    report_lines = [
        "=" * 70,
        f"ANALYSE INV — MODE: {mode_name.upper().replace('_', ' ')}",
        "=" * 70,
        "",
        f"Observations: {len(df)}",
        f"Mode d'analyse: {mode_name.replace('_', ' ')}",
        f"Pruning appliqué: {'OUI' if apply_pruning else 'NON'}",
        "",
        "ÉTAPES DE FILTRAGE:",
        "-" * 70,
        f"1. Features candidates initiales: {len(features_initial)}",
        f"   (colonnes numériques, hors ID, hors _source, hors z_, hors legacy int_*)",
        "",
        f"2. Après filtres techniques: {len(features_selected)}",
        f"   - Exclusion si NaN > {max_missing*100:.0f}%",
        f"   - Exclusion si constante (std < 1e-6)",
        f"   - Exclusion des anciennes pseudo-interruptions audio `int_*` si présentes",
        f"   → {len(features_initial) - len(features_selected)} features exclues",
        "",
    ]

    if apply_pruning:
        n_pruned = len(features_selected) - len(features_final)
        report_lines.extend([
            f"3. Après hard pruning (|r| > {prune_threshold}): {len(features_final)}",
            f"   → {n_pruned} features redondantes supprimées",
            "",
            "FEATURES FINALES POUR LA PCA:",
            f"   {len(features_final)} features après pruning des redondances",
        ])
    else:
        report_lines.extend([
            f"3. Pas de pruning appliqué",
            f"   → Toutes les {len(features_selected)} features valides sont conservées",
            "",
            "FEATURES FINALES POUR LA PCA:",
            f"   {len(features_final)} features sans pruning analytique",
            f"   (les redondances sont conservées pour décrire la structure complète)",
        ])

    report_lines.extend([
        "",
        "=" * 70,
        "LISTE DES FEATURES UTILISÉES POUR LA PCA:",
        "=" * 70,
    ])

    for i, feat in enumerate(features_final, 1):
        report_lines.append(f"  {i:2d}. {feat}")

    report_lines.extend([
        "",
        "=" * 70,
        "FICHIERS GÉNÉRÉS:",
        "=" * 70,
        "  - analysis_summary.csv            : Résumé des paramètres d'analyse",
        "  - inv_features_used.csv           : Liste des features utilisées pour la PCA",
        "  - inv_correlation_matrix.csv      : Matrice de corrélation",
        "  - pca_explained_variance.csv      : Variance expliquée par composante",
        "  - pca_loadings.csv                : Contributions des features",
        "  - pca_scree_plot.png              : Graphique de variance",
        "  - inv_dimensions.csv              : Scores PCA par groupe",
        "  - feature_clusters.csv            : Clusters de features",
        "  - pca_projection_groups.png       : Projection PC1 vs PC2",
    ])

    if apply_pruning:
        report_lines.extend([
            "  - inv_pruned_features.csv           : Rapport de pruning",
            "  - inv_correlation_matrix_pruned.csv : Corrélation après pruning (CSV)",
            "  - corr_matrix_inv_pruned.png        : Heatmap de corrélation après pruning",
        ])

    report_lines.extend([
        "",
        "=" * 70,
    ])

    report_path = out_dir / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] analysis_report.txt")


# ---------------------------------------------------------------------------
# Pipeline d'analyse complet
# ---------------------------------------------------------------------------

def run_inv_analysis_pipeline(
    df: pd.DataFrame,
    df_meta: pd.DataFrame,
    out_dir: Path,
    mode_name: str,
    apply_pruning: bool,
    max_missing: float,
    min_cumvar: float,
    prune_threshold: float,
    rotation: str,
) -> None:
    """
    Exécute le pipeline complet d'analyse INV.

    Args:
        df: DataFrame complet avec toutes les features et métadonnées
        df_meta: DataFrame avec les colonnes de métadonnées (group_id, condition, etc.)
        out_dir: Dossier de sortie pour ce mode
        mode_name: "with_pruning" ou "without_pruning"
        apply_pruning: Si True, applique le hard pruning des redondances
        max_missing: Seuil maximal de valeurs manquantes pour filtrage technique
        min_cumvar: Variance cumulée minimale pour rétention des composantes
        prune_threshold: Seuil |r| pour le pruning (si apply_pruning=True)
        rotation: Type de rotation PCA ("none" ou "varimax")
    """
    print("\n" + "=" * 70)
    print(f"PIPELINE D'ANALYSE: {mode_name.upper().replace('_', ' ')}")
    print("=" * 70)
    print(f"Pruning: {'ACTIVÉ' if apply_pruning else 'DÉSACTIVÉ'}")
    print(f"Dossier de sortie: {out_dir}")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sélection des features (filtres techniques uniquement)
    print("\n[ÉTAPE 1] Sélection des features (filtres techniques)")
    features_initial = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in ID_COLS
        and c not in NON_INV_EXACT
        and not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)
        and not any(c.endswith(suf) for suf in MERGE_DUPLICATE_SUFFIXES)
        and not any(c.startswith(pre) for pre in EXCLUDE_PREFIXES)
        and not any(c.startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES)
    ]
    print(f"  Features candidates (numériques, hors ID/z_/_source): {len(features_initial)}")

    features_selected = select_features(df, max_missing=max_missing)
    if len(features_selected) < 3:
        print("[ERROR] Pas assez de features exploitables.")
        return

    # 2. Préparation de la matrice
    print("\n[ÉTAPE 2] Préparation de la matrice (imputation, standardisation)")
    X_scaled, features_after_prep = prepare_matrix(df, features_selected)
    df_scaled = pd.DataFrame(X_scaled, columns=features_after_prep)
    print(f"  -> {len(features_after_prep)} features, {X_scaled.shape[0]} observations")

    # 3. Corrélation (toujours calculée sur toutes les features valides)
    print("\n[ÉTAPE 3] Matrice de corrélation")
    corr = compute_correlation(df_scaled, out_dir)

    # 3b. Pruning (si activé)
    if apply_pruning:
        print(f"\n[ÉTAPE 3b] Hard pruning des features redondantes (|r| > {prune_threshold})")
        features_final, pruning_report = remove_correlated_features(
            df_scaled, corr, df_raw=df, threshold=prune_threshold, out_dir=out_dir
        )

        # Matrice de corrélation sur features prunées
        pruned_idx = [features_after_prep.index(f) for f in features_final]
        X_final = X_scaled[:, pruned_idx]
        corr_final = corr.loc[features_final, features_final]
        corr_final.to_csv(out_dir / "inv_correlation_matrix_pruned.csv")
        print(f"[OK] inv_correlation_matrix_pruned.csv ({len(features_final)} features)")

        n_pruned = len(features_final)
        figsize = max(10, n_pruned * 0.5)
        fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
        _render_heatmap(
            ax=ax,
            data=corr_final,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            annot=(n_pruned <= 22),
            fmt=".2f",
            annot_kws={"size": 6},
            linewidths=0.3,
            square=True,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(f"Matrice de corrélation après pruning — {n_pruned} features", fontsize=12, pad=12)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / "corr_matrix_inv_pruned.png", dpi=200)
        plt.close()
        print("[OK] corr_matrix_inv_pruned.png")
    else:
        print(f"\n[ÉTAPE 3b] Pruning DÉSACTIVÉ — Conservation de toutes les features valides")
        features_final = features_after_prep.copy()
        X_final = X_scaled
        corr_final = corr
        print(f"  -> {len(features_final)} features conservées (redondances incluses)")

        # Générer quand même un rapport de diagnostic de redondance (pour information)
        # Cela permet au rapport de montrer quelles features "seraient" redondantes
        print(f"\n[ÉTAPE 3b-diag] Diagnostic de redondance (non appliqué, pour info)")
        _, diagnostic_report = remove_correlated_features(
            df_scaled, corr, df_raw=df, threshold=prune_threshold, out_dir=None
        )
        # Sauvegarder le diagnostic avec un nom explicite
        diagnostic_report["applied"] = 0  # Indique que le pruning n'a pas été appliqué
        diagnostic_report.to_csv(out_dir / "inv_pruned_features.csv", index=False)
        print(f"[OK] inv_pruned_features.csv (diagnostic uniquement, pruning non appliqué)")

    # 4. PCA sur les features finales
    print(f"\n[ÉTAPE 4] PCA (sur {len(features_final)} features)")
    print(f"  [ROTATION] {rotation}")
    pca, scores = run_pca(X_final, features_final, out_dir, rotation=rotation)

    # 5. Extraction des dimensions retenues
    print("\n[ÉTAPE 5] Extraction des dimensions retenues")
    dim_df, n_retain = extract_dimensions(pca, scores, df_meta, out_dir, min_cumvar=min_cumvar)

    # 6. Clustering (sur features finales)
    print("\n[ÉTAPE 6] Clustering hiérarchique des features")
    cluster_df = cluster_features(corr_final, out_dir)

    # 6b. Sélection d'un représentant par cluster
    print("\n[ÉTAPE 6b] Sélection d'un représentant par cluster")
    target_candidates = ["score_perf_tsk", "c_score", "performance_z", "perf_score",
                         "Score_perf_tsk", "C_factor"]
    target_cols = [c for c in target_candidates if c in df.columns] or None
    select_representative_per_cluster(df, cluster_df, target_cols=target_cols, out_dir=out_dir)

    # 7. Projection
    print("\n[ÉTAPE 7] Projection PCA")
    plot_pca_projection(dim_df, out_dir)

    # 7b. Projections par variables continues
    print("\n[ÉTAPE 7b] Projections PCA par variables continues")
    plot_pca_projection_by_variable(dim_df, df, out_dir)

    # 8. PCA par modalité (PC vs VR)
    print("\n[ÉTAPE 8] PCA par modalité (PC vs VR)")
    pca_by_modality(df, features_final, out_dir)

    # 9. PCA par famille INV
    # Pour la famille gaze (VR uniquement), enrichir avec les colonnes gaze valides
    # même si elles dépassent le seuil NaN global.
    gaze_prefixes = MODALITY_PREFIX.get("gaze", [])
    extra_gaze = [
        c for c in df.columns
        if any(c.startswith(p) for p in gaze_prefixes)
        and c not in features_final
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)
        and not any(c.startswith(pre) for pre in EXCLUDE_PREFIXES)
    ]

    # Hard pruning des features gaze redondantes
    gaze_sub_for_export: pd.DataFrame | None = None
    gaze_valid_for_export: list[str] = []
    if extra_gaze:
        all_gaze = [f for f in features_final
                     if any(f.startswith(p) for p in gaze_prefixes)] + extra_gaze
        gaze_sub = df[all_gaze].copy()
        row_all_nan = gaze_sub.isna().all(axis=1)
        gaze_sub = gaze_sub[~row_all_nan]
        if len(gaze_sub) >= 3:
            gaze_valid = [c for c in all_gaze if gaze_sub[c].std() > 1e-6]
            if gaze_valid and apply_pruning:
                gaze_corr = gaze_sub[gaze_valid].corr()
                print(f"\n[ÉTAPE 9-pre] Hard pruning gaze ({len(gaze_valid)} features, "
                      f"|r| > {prune_threshold})")
                gaze_kept, gaze_pruning_report = remove_correlated_features(
                    gaze_sub[gaze_valid], gaze_corr,
                    df_raw=df, threshold=prune_threshold,
                    out_dir=None,
                )
                gaze_pruning_report.to_csv(out_dir / "inv_pruned_features_gaze.csv", index=False)
                print(f"  [OK] inv_pruned_features_gaze.csv")

                gaze_sub_for_export = gaze_sub
                gaze_valid_for_export = gaze_valid
                extra_gaze = [f for f in gaze_kept if f not in features_final]
            elif gaze_valid and not apply_pruning:
                # Sans pruning, garder toutes les features gaze valides
                gaze_sub_for_export = gaze_sub
                gaze_valid_for_export = gaze_valid
                extra_gaze = [f for f in gaze_valid if f not in features_final]

        if apply_pruning:
            msg = f"(exclues de la PCA globale car NaN > {max_missing*100:.0f}% sur PC+VR)"
        else:
            msg = f"(enrichissement VR-only pour analyse par famille)"
        print(f"  [INFO] {len(extra_gaze)} features gaze après {'pruning ' if apply_pruning else ''}ajoutées {msg}")

    features_for_modality = features_final + extra_gaze

    print("\n[ÉTAPE 9] Analyse PCA par famille INV (redondance des métriques)")
    modality_results = pca_by_inv_modality(df, features_for_modality, out_dir)

    # 9b. Export des redondances complètes par famille
    print("\n[ÉTAPE 9b] Redondances complètes par famille")
    _export_full_modality_redundancy(
        df=df,
        features_all=features_after_prep,
        gaze_sub=gaze_sub_for_export,
        gaze_valid=gaze_valid_for_export,
        threshold=prune_threshold,
        out_dir=out_dir,
    )

    # 10. Génération du résumé d'analyse
    print("\n[ÉTAPE 10] Génération du résumé d'analyse")
    build_analysis_summary(
        df=df,
        features_initial=features_initial,
        features_selected=features_after_prep,
        features_final=features_final,
        mode_name=mode_name,
        apply_pruning=apply_pruning,
        prune_threshold=prune_threshold,
        max_missing=max_missing,
        out_dir=out_dir,
    )

    # Résumé console
    print_summary(pca, features_final, cluster_df, n_retain, modality_results=modality_results)
    print(f"\n[DONE] Mode '{mode_name}' — Résultats dans : {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Point d'entrée principal.

    Exécute DEUX pipelines d'analyse distincts :
      1. with_pruning/   : PCA calculée sur les features après pruning des redondances
      2. without_pruning/ : PCA calculée sur toutes les features valides (redondances conservées)

    Les deux pipelines utilisent les mêmes filtres techniques (NaN, constantes).
    Seul le hard-pruning analytique (|r| > seuil) diffère.
    """
    ap = argparse.ArgumentParser(
        description="Analyse exploratoire INV : PCA + corrélations + clustering. "
                    "Produit DEUX ensembles de résultats : with_pruning/ et without_pruning/"
    )
    ap.add_argument(
        "--data",
        default=r"D:\Analyse_donnee\Longitudinale\results\INV\high_level_features_audit.csv",
        help="Fichier de features INV (CSV)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Dossier de sortie racine (auto si non spécifié)",
    )
    ap.add_argument("--max-missing", type=float, default=0.20,
                    help="Seuil max de valeurs manquantes par feature (défaut 0.20)")
    ap.add_argument("--min-cumvar", type=float, default=0.70,
                    help="Variance cumulée minimale pour sélection des composantes (défaut 0.70)")
    ap.add_argument("--prune-threshold", type=float, default=0.9,
                    help="Seuil |r| pour le hard pruning des features redondantes (défaut 0.80)")
    ap.add_argument("--mode", choices=["all", "vr-only"], default="all",
                    help="Mode d'analyse : 'all' (PC+VR, défaut) ou 'vr-only' (VR uniquement)")
    ap.add_argument("--min-row-completeness", type=float, default=0.60,
                    help="[vr-only] Proportion minimale de valeurs non manquantes par ligne (défaut 0.60)")
    ap.add_argument("--rotation", choices=["none", "varimax"], default="none",
                    help="Rotation des facteurs : 'none' (défaut) ou 'varimax'")
    ap.add_argument("--only-pruning-mode", choices=["with", "without", "both"], default="both",
                    help="Exécuter uniquement un mode : 'with' (pruning), 'without' (sans pruning), "
                         "ou 'both' (les deux, défaut)")
    ap.add_argument(
        "--exclude-groups",
        action="append",
        default=[],
        help="Groupes à exclure (ex: bim015,bim023). Peut être répété.",
    )
    args = ap.parse_args()

    data_path = Path(args.data)

    # Dossier de sortie racine
    if args.out is None:
        base_out = Path(r"D:\Analyse_donnee\Longitudinale\results")
        if args.mode == "vr-only":
            out_dir_root = base_out / "results_inv_structure_vr_only"
        else:
            out_dir_root = base_out / "results_inv_structure"
    else:
        out_dir_root = Path(args.out)
    out_dir_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALYSE INV — DOUBLE PIPELINE (WITH/WITHOUT PRUNING)")
    print("=" * 70)
    print(f"[LOAD] {data_path}")
    print(f"[MODE] {args.mode.upper()}")
    print(f"[PRUNING MODES] {args.only_pruning_mode}")
    print(f"[OUTPUT ROOT] {out_dir_root}")
    print("=" * 70)

    df = load_data(data_path)
    print(f"  {df.shape[0]} observations x {df.shape[1]} colonnes")

    # Filtrage VR si mode vr-only
    if args.mode == "vr-only":
        cond_col = None
        for cand in ["condition", "modalite", "modality"]:
            if cand in df.columns:
                cond_col = cand
                break
        if cond_col is None:
            print("[ERROR] Colonne condition/modalité introuvable pour filtrer VR.")
            return
        df[cond_col] = df[cond_col].astype(str).str.strip().str.upper()
        df = df[df[cond_col] == "VR"].copy()
        if df.empty:
            print("[ERROR] Aucune ligne VR après filtrage.")
            return
        print(f"  [VR-ONLY] {df.shape[0]} observations VR retenues")

    # Exclusion explicite de groupes (par group_id)
    def _parse_exclude_groups(raw_list: list[str]) -> tuple[set[str], set[str]]:
        tokens: list[str] = []
        for raw in raw_list:
            tokens.extend(re.split(r"[;,]", str(raw)))
        tokens = [t.strip().lower() for t in tokens if t and str(t).strip()]
        exact: set[str] = set()
        base: set[str] = set()
        for t in tokens:
            if re.search(r"_\d+$", t):
                exact.add(t)
            else:
                base.add(t)
        return exact, base

    exclude_exact, exclude_base = _parse_exclude_groups(args.exclude_groups or [])
    if exclude_exact or exclude_base:
        if "group_id" not in df.columns:
            print("[WARN] --exclude-groups ignoré : colonne group_id absente.")
        else:
            df["group_id"] = df["group_id"].astype(str).str.strip().str.lower()
            group_base = df["group_id"].str.replace(r"_\d+$", "", regex=True)
            mask = df["group_id"].isin(exclude_exact) | group_base.isin(exclude_base)
            if mask.any():
                removed = sorted(df.loc[mask, "group_id"].dropna().unique().tolist())
                df = df.loc[~mask].copy()
                print(f"  [EXCLUDE] {len(removed)} groupes exclus via --exclude-groups : {removed}")
            else:
                print("  [EXCLUDE] Aucun groupe correspondant à --exclude-groups.")

    if "condition" in df.columns:
        print(f"  Répartition : {dict(df['condition'].value_counts())}")

    # Colonnes de métadonnées
    meta_cols = [c for c in ["group_id", "condition", "scenario", "timepoint"] if c in df.columns]
    df_meta = df[meta_cols].copy()

    # --- Pipeline 1 : WITH PRUNING ---
    if args.only_pruning_mode in ("with", "both"):
        out_dir_pruning = out_dir_root / "with_pruning"
        run_inv_analysis_pipeline(
            df=df,
            df_meta=df_meta,
            out_dir=out_dir_pruning,
            mode_name="with_pruning",
            apply_pruning=True,
            max_missing=args.max_missing,
            min_cumvar=args.min_cumvar,
            prune_threshold=args.prune_threshold,
            rotation=args.rotation,
        )

    # --- Pipeline 2 : WITHOUT PRUNING ---
    if args.only_pruning_mode in ("without", "both"):
        out_dir_no_pruning = out_dir_root / "without_pruning"
        run_inv_analysis_pipeline(
            df=df,
            df_meta=df_meta,
            out_dir=out_dir_no_pruning,
            mode_name="without_pruning",
            apply_pruning=False,
            max_missing=args.max_missing,
            min_cumvar=args.min_cumvar,
            prune_threshold=args.prune_threshold,
            rotation=args.rotation,
        )

    # --- Fichier récapitulatif global ---
    print("\n" + "=" * 70)
    print("RÉCAPITULATIF GLOBAL")
    print("=" * 70)

    global_summary_lines = [
        "=" * 70,
        "ANALYSE INV — RÉCAPITULATIF DES DEUX MODES",
        "=" * 70,
        "",
        f"Fichier source : {data_path}",
        f"Mode d'analyse : {args.mode}",
        f"Groupes exclus : {', '.join(sorted(list(exclude_base)) + sorted(list(exclude_exact))) if (exclude_exact or exclude_base) else 'aucun'}",
        f"Dossier racine : {out_dir_root}",
        "",
        "SOUS-DOSSIERS GÉNÉRÉS :",
        "-" * 70,
    ]

    if args.only_pruning_mode in ("with", "both"):
        global_summary_lines.extend([
            "",
            "1. with_pruning/",
            f"   → PCA calculée sur les features APRÈS pruning des redondances",
            f"   → Seuil de pruning : |r| > {args.prune_threshold}",
            f"   → Utiliser ce mode pour une analyse avec variables indépendantes",
        ])

    if args.only_pruning_mode in ("without", "both"):
        global_summary_lines.extend([
            "",
            "2. without_pruning/",
            f"   → PCA calculée sur TOUTES les features valides (redondances conservées)",
            f"   → Pas de pruning analytique",
            f"   → Utiliser ce mode pour décrire la structure complète de l'espace des variables",
        ])

    global_summary_lines.extend([
        "",
        "=" * 70,
        "UTILISATION AVEC LE RAPPORT :",
        "=" * 70,
        "",
        "Pour générer le rapport avec pruning :",
        "  python ..\\rapport\\main.py --inv-analysis-mode pruning",
        "",
        "Pour générer le rapport sans pruning :",
        "  python ..\\rapport\\main.py --inv-analysis-mode no-pruning",
        "",
        "=" * 70,
    ])

    global_summary_path = out_dir_root / "ANALYSIS_MODES_README.txt"
    with open(global_summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(global_summary_lines))
    print(f"[OK] {global_summary_path}")

    print(f"\n[DONE] Analyse terminée.")
    print(f"  → Résultats avec pruning    : {out_dir_root / 'with_pruning'}")
    print(f"  → Résultats sans pruning    : {out_dir_root / 'without_pruning'}")
    print(f"  → Documentation             : {global_summary_path}")


if __name__ == "__main__":
    main()
