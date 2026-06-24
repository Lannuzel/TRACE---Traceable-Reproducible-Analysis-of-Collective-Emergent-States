"""
analyze_performance_effects.py
================================
Analyse statistique de l'effet du scénario et de la modalité sur la performance
de groupe dans des tâches collaboratives BIM (PC vs VR, S1 vs S2).

Pipeline :
  1. Chargement et détection automatique de la variable de performance
  2. Nettoyage et harmonisation des colonnes condition (modalite, scenario)
  3. Statistiques descriptives par condition
  4. A. Vérifications préalables (effectifs, normalité, homogénéité)
  5. B. Effet du scénario seul (t-test / Mann-Whitney)
  6. C. Effet de la modalité seule
  7. D. Modèle factoriel complet (modalité × scénario — OLS ANOVA)
  8. E. ANCOVA si covariable pertinente détectée dans les données
  9. Export CSV + figures + rapport console

Hypothèses méthodologiques :
  - Unité d'analyse : groupe (niveau macro, N≈18–21)
  - N petit → les tests paramétriques sont conservés à titre indicatif avec
    mention explicite des limites ; les alternatives non-paramétriques sont
    fournies en parallèle (Mann-Whitney, Kruskal-Wallis)
  - Tailles d'effet : Cohen's d pour comparaisons 2 groupes ; η² partiel pour ANOVA
  - ANCOVA : la covariable réduit la variance résiduelle ; elle ne remplace pas
    l'interprétation des effets principaux

Usage :
    python analyze_performance_effects.py
    python analyze_performance_effects.py --perf D:/…/results/performance_task/recap_scores_all.csv
                                          --out  D:/…/results/analyse_performance
    python analyze_performance_effects.py --covariate skill_mean
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal

warnings.filterwarnings("ignore")


def default_merged_dataset_path() -> str:
    candidates = [
        Path(r"D:\Analyse_donnee\Longitudinale\results\merged_dataset\with_pruning\merged_dataset_complete_all.csv"),
        Path(r"D:\Analyse_donnee\Longitudinale\results\merged_dataset\without_pruning\merged_dataset_complete_all.csv"),
        Path(r"D:\Analyse_donnee\Longitudinale\results\merged_dataset\merged_dataset_complete_all.csv"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def default_performance_path() -> str:
    candidates = [
        Path(r"D:\Analyse_donnee\Longitudinale\results\performance_task\recap_scores_all.csv"),
        Path(r"D:\Analyse_donnee\Longitudinale\results\recap_scores_all.csv"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Priorité de détection de la variable de performance (ordre décroissant)
PERF_COL_PRIORITY = [
    "Score_perf_tsk",
    "Score_perf_tsk_%",
    "Score_perf_tsk_z",
    "Score_final_%",
    "score_final",
    "score",
]

# Covariables candidates issues du dataset fusionné
COVARIATE_CANDIDATES = [
    "skill_mean",
    "skill_congruence_mean",
    "effort_norm",
    "effort_task_norm",
    "C_factor",
    "c_factor",
    "c_score",
    "coverage_correct_final_mean",
    "strategy_norm",
]

# Valeurs reconnues pour les catégories
SCENARIO_VALUES = {"S1", "S2"}
MODALITY_VALUES = {"PC", "VR"}

# Seuil p-value pour "significatif"
ALPHA = 0.05

# Groupes à exclure (score = 0 ou invalide)
# Aligné sur le reste du pipeline : on conserve bim065 et on exclut bim065_2.
EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}


# ---------------------------------------------------------------------------
# 1. Détection de la variable de performance
# ---------------------------------------------------------------------------

def detect_performance_column(df: pd.DataFrame) -> str | None:
    """
    Identifie la meilleure colonne de performance disponible selon la priorité
    définie dans PERF_COL_PRIORITY. Cherche également des correspondances
    partielles insensibles à la casse.

    Returns : nom exact de la colonne retenue, ou None.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # 1. Correspondance exacte dans la liste de priorité
    for cand in PERF_COL_PRIORITY:
        if cand in df.columns:
            print(f"  [PERF] Colonne retenue (correspondance exacte) : '{cand}'")
            return cand

    # 2. Correspondance insensible à la casse
    for cand in PERF_COL_PRIORITY:
        if cand.lower() in cols_lower:
            found = cols_lower[cand.lower()]
            print(f"  [PERF] Colonne retenue (casse insensible) : '{found}'")
            return found

    # 3. Correspondance partielle sur 'score'
    for col in df.columns:
        if "score" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            print(f"  [PERF] Colonne retenue (correspondance partielle 'score') : '{col}'")
            return col

    print("  [WARN] Aucune colonne de performance trouvée.")
    return None


# ---------------------------------------------------------------------------
# 2. Nettoyage des colonnes condition
# ---------------------------------------------------------------------------

def clean_condition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise les colonnes modalite et scenario :
      - normalise la casse : PC, VR, S1, S2
      - supprime espaces résiduels
      - remplace les valeurs non reconnues par NaN
      - renomme 'condition' → 'modalite' si besoin
    """
    df = df.copy()

    # Renommage condition → modalite
    if "modalite" not in df.columns and "condition" in df.columns:
        df = df.rename(columns={"condition": "modalite"})

    if "modalite" in df.columns:
        df["modalite"] = (
            df["modalite"].astype(str).str.strip().str.upper()
            .replace({"PC": "PC", "VR": "VR", "NAN": np.nan})
        )
        df.loc[~df["modalite"].isin(MODALITY_VALUES), "modalite"] = np.nan

    if "scenario" in df.columns:
        df["scenario"] = (
            df["scenario"].astype(str).str.strip().str.upper()
            .replace({"S1": "S1", "S2": "S2", "NAN": np.nan})
        )
        df.loc[~df["scenario"].isin(SCENARIO_VALUES), "scenario"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 3. Chargement et préparation
# ---------------------------------------------------------------------------

def load_and_prepare(
    perf_path: Path,
    merged_path: Path | None,
    perf_col_override: str | None,
) -> tuple[pd.DataFrame, str, str | None]:
    """
    Charge, nettoie et valide le dataset de performance.

    Returns :
      df        : DataFrame prêt pour analyse
      perf_col  : nom de la variable de performance retenue
      covariate : nom de la covariable détectée (ou None)
    """
    df = pd.read_csv(perf_path)
    print(f"[LOAD] {perf_path} → {df.shape[0]} lignes")

    # Fusionner avec le dataset complet si disponible (pour covariables)
    if merged_path and merged_path.exists():
        merged = pd.read_csv(merged_path)
        id_col = next((c for c in ["groupe", "group_id"] if c in df.columns and c in merged.columns), None)
        if id_col:
            extra_cols = [c for c in COVARIATE_CANDIDATES if c in merged.columns]
            if extra_cols:
                merged_sub = merged[[id_col] + extra_cols].drop_duplicates(subset=[id_col])
                df = df.merge(merged_sub, on=id_col, how="left")
                print(f"  [MERGE] Covariables intégrées depuis dataset fusionné : {extra_cols}")

    df = clean_condition_columns(df)

    # Détection de la colonne de performance
    perf_col = perf_col_override if perf_col_override else detect_performance_column(df)
    if perf_col is None or perf_col not in df.columns:
        raise ValueError(f"Variable de performance introuvable. Colonnes disponibles : {list(df.columns)}")

    # Nettoyage numérique
    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")

    # Exclusion groupes invalides (score nul ou groupe exclu)
    id_col = next((c for c in ["groupe", "group_id"] if c in df.columns), None)
    if id_col:
        before = len(df)
        df = df[~df[id_col].isin(EXCLUDED_GROUPS)]
        removed = before - len(df)
        if removed:
            print(f"  [EXCL] {removed} groupes exclus (EXCLUDED_GROUPS)")

    # Retirer lignes sans performance exploitable ou sans condition
    mask = (
        df[perf_col].notna()
        & (df[perf_col] > 0)  # score 0 = pas de données
        & df.get("modalite", pd.Series([True] * len(df))).notna()
        & df.get("scenario", pd.Series([True] * len(df))).notna()
    )
    n_before = len(df)
    df = df[mask].copy().reset_index(drop=True)
    print(f"  [FILTER] {n_before - len(df)} lignes retirées (NaN / score 0 / condition manquante)")
    print(f"  [OK] {len(df)} observations retenues pour l'analyse")

    # Détection covariable
    covariate = detect_covariate(df)

    return df, perf_col, covariate


def detect_covariate(df: pd.DataFrame) -> str | None:
    """
    Recherche une covariable pertinente parmi les colonnes disponibles.
    Vérifie qu'elle est numérique et a suffisamment de valeurs non-NaN (>= 50%).
    """
    for cand in COVARIATE_CANDIDATES:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            pct_valid = df[cand].notna().mean()
            if pct_valid >= 0.50:
                print(f"  [COVARIABLE] '{cand}' détectée ({pct_valid*100:.0f}% valeurs disponibles)")
                return cand
    print("  [COVARIABLE] Aucune covariable pertinente détectée → pas d'ANCOVA")
    return None


# ---------------------------------------------------------------------------
# 4. Statistiques descriptives
# ---------------------------------------------------------------------------

def compute_descriptives(df: pd.DataFrame, perf_col: str, group_by: list) -> pd.DataFrame:
    """
    Tableau descriptif : n, mean, sd, median, min, max par groupe.
    group_by : liste de colonnes de regroupement
    """
    desc = (
        df.groupby(group_by, dropna=False)[perf_col]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            min="min",
            max="max",
        )
        .reset_index()
    )
    for col in ["mean", "sd", "median", "min", "max"]:
        desc[col] = desc[col].round(3)
    return desc


# ---------------------------------------------------------------------------
# 5. Vérifications préalables
# ---------------------------------------------------------------------------

def check_assumptions(df: pd.DataFrame, perf_col: str, report: list):
    """
    Vérifie :
      - effectifs par cellule modalite × scenario
      - normalité résiduelle (Shapiro-Wilk si n ≤ 50)
      - homogénéité des variances (Levene)

    report : liste de chaînes pour le rapport console (mutée in-place)
    """
    report.append("\n--- A. VÉRIFICATIONS PRÉALABLES ---")

    if "modalite" in df.columns and "scenario" in df.columns:
        cell_counts = df.groupby(["modalite", "scenario"])[perf_col].count()
        report.append("Effectifs par cellule modalité × scénario :")
        for idx, n in cell_counts.items():
            flag = " ⚠ (n<4)" if n < 4 else ""
            report.append(f"  {idx[0]} × {idx[1]} : n={n}{flag}")
        if (cell_counts < 4).any():
            report.append("  → ATTENTION : certaines cellules ont n < 4. Les résultats sont exploratoires.")

    # Normalité (Shapiro-Wilk sur les résidus de la moyenne globale)
    vals = df[perf_col].dropna().values
    if 3 <= len(vals) <= 50:
        stat, p = stats.shapiro(vals)
        flag = " ✓" if p >= ALPHA else " ⚠ (non-normal)"
        report.append(f"Normalité (Shapiro-Wilk, n={len(vals)}) : W={stat:.3f}, p={p:.3f}{flag}")
        if p < ALPHA:
            report.append("  → Distribution non-normale détectée. Tests non-paramétriques fournis en parallèle.")
    else:
        report.append(f"Normalité : n={len(vals)} → test Shapiro-Wilk non applicable")

    # Homogénéité des variances (Levene entre modalités si disponible)
    if "modalite" in df.columns:
        groups = [g[perf_col].dropna().values for _, g in df.groupby("modalite") if len(g) >= 2]
        if len(groups) >= 2:
            stat, p = stats.levene(*groups)
            flag = " ✓" if p >= ALPHA else " ⚠ (variances hétérogènes)"
            report.append(f"Homogénéité variances (Levene, modalité) : F={stat:.3f}, p={p:.3f}{flag}")


# ---------------------------------------------------------------------------
# 6. Helpers statistiques
# ---------------------------------------------------------------------------

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan


def eta_squared(ss_effect: float, ss_total: float) -> float:
    return ss_effect / ss_total if ss_total > 0 else np.nan


def interpret_p(p: float, label: str) -> str:
    if np.isnan(p):
        return f"L'effet de {label} n'a pas pu être testé (données insuffisantes)."
    if p < 0.001:
        return f"L'effet de {label} est très significatif (p < 0.001)."
    if p < ALPHA:
        return f"L'effet de {label} est significatif (p = {p:.3f})."
    return f"Aucun effet significatif de {label} n'a été observé (p = {p:.3f})."


def ols_anova_table(df: pd.DataFrame, perf_col: str, factors: list) -> pd.DataFrame:
    """
    ANOVA type III via scipy — calcule manuellement pour rester sans dépendances lourdes.
    Pour 2 facteurs + interaction : utilise statsmodels si disponible, sinon dégradé.
    """
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        formula_terms = " * ".join(f"C({f})" for f in factors)
        formula = f"{perf_col} ~ {formula_terms}"
        model = smf.ols(formula, data=df).fit()
        table = anova_lm(model, typ=2)
        table = table.reset_index().rename(columns={"index": "term", "PR(>F)": "p_value", "F": "F_stat"})
        # Calcul η² partiel
        ss_residual = table.loc[table["term"] == "Residual", "sum_sq"].values[0]
        ss_total = table["sum_sq"].sum()
        table["eta2_partial"] = table["sum_sq"] / (table["sum_sq"] + ss_residual)
        table["eta2_partial"] = table["eta2_partial"].where(table["term"] != "Residual")
        return table, model
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# 7B. Effet du scénario
# ---------------------------------------------------------------------------

def run_scenario_effect(df: pd.DataFrame, perf_col: str, out_dir: Path, report: list) -> pd.DataFrame:
    """
    Test de l'effet du scénario (S1 vs S2) sur la performance.
    Paramétrique (t-test) + non-paramétrique (Mann-Whitney).
    """
    report.append("\n--- B. EFFET DU SCÉNARIO ---")
    rows = []

    if "scenario" not in df.columns:
        report.append("  [SKIP] Colonne 'scenario' absente.")
        return pd.DataFrame()

    scenarios = df["scenario"].dropna().unique()
    if len(scenarios) != 2:
        report.append(f"  [SKIP] {len(scenarios)} niveaux de scénario (attendu : 2).")
        return pd.DataFrame()

    s1_vals = df[df["scenario"] == "S1"][perf_col].dropna().values
    s2_vals = df[df["scenario"] == "S2"][perf_col].dropna().values
    report.append(f"  S1 : n={len(s1_vals)}, M={np.mean(s1_vals):.2f}, SD={np.std(s1_vals, ddof=1):.2f}")
    report.append(f"  S2 : n={len(s2_vals)}, M={np.mean(s2_vals):.2f}, SD={np.std(s2_vals, ddof=1):.2f}")

    # t-test (Welch)
    if len(s1_vals) >= 2 and len(s2_vals) >= 2:
        t_stat, p_t = stats.ttest_ind(s1_vals, s2_vals, equal_var=False)
        d = cohen_d(s1_vals, s2_vals)
        report.append(f"  t-test (Welch) : t={t_stat:.3f}, p={p_t:.3f}, Cohen's d={d:.3f}")
        report.append(f"  → {interpret_p(p_t, 'scénario')}")

        # Mann-Whitney
        u_stat, p_mw = mannwhitneyu(s1_vals, s2_vals, alternative="two-sided")
        report.append(f"  Mann-Whitney   : U={u_stat:.0f}, p={p_mw:.3f} (non-paramétrique)")

        rows.append({"test": "t-test Welch", "factor": "scenario", "level_a": "S1", "level_b": "S2",
                     "n_a": len(s1_vals), "n_b": len(s2_vals), "mean_a": np.mean(s1_vals).round(3),
                     "mean_b": np.mean(s2_vals).round(3), "stat": round(t_stat, 3),
                     "p_value": round(p_t, 4), "effect_size": round(d, 3), "effect_type": "Cohen's d"})
        rows.append({"test": "Mann-Whitney", "factor": "scenario", "level_a": "S1", "level_b": "S2",
                     "n_a": len(s1_vals), "n_b": len(s2_vals), "mean_a": np.mean(s1_vals).round(3),
                     "mean_b": np.mean(s2_vals).round(3), "stat": round(u_stat, 3),
                     "p_value": round(p_mw, 4), "effect_size": np.nan, "effect_type": "U"})
    else:
        report.append("  [SKIP] Données insuffisantes pour le test.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7C. Effet de la modalité
# ---------------------------------------------------------------------------

def run_modality_effect(df: pd.DataFrame, perf_col: str, out_dir: Path, report: list) -> pd.DataFrame:
    """
    Test de l'effet de la modalité (PC vs VR) sur la performance.
    """
    report.append("\n--- C. EFFET DE LA MODALITÉ ---")
    rows = []

    if "modalite" not in df.columns:
        report.append("  [SKIP] Colonne 'modalite' absente.")
        return pd.DataFrame()

    modalities = df["modalite"].dropna().unique()
    if len(modalities) != 2:
        report.append(f"  [SKIP] {len(modalities)} niveaux de modalité (attendu : 2).")
        return pd.DataFrame()

    pc_vals = df[df["modalite"] == "PC"][perf_col].dropna().values
    vr_vals = df[df["modalite"] == "VR"][perf_col].dropna().values
    report.append(f"  PC : n={len(pc_vals)}, M={np.mean(pc_vals):.2f}, SD={np.std(pc_vals, ddof=1):.2f}")
    report.append(f"  VR : n={len(vr_vals)}, M={np.mean(vr_vals):.2f}, SD={np.std(vr_vals, ddof=1):.2f}")

    if len(pc_vals) >= 2 and len(vr_vals) >= 2:
        t_stat, p_t = stats.ttest_ind(pc_vals, vr_vals, equal_var=False)
        d = cohen_d(pc_vals, vr_vals)
        report.append(f"  t-test (Welch) : t={t_stat:.3f}, p={p_t:.3f}, Cohen's d={d:.3f}")
        report.append(f"  → {interpret_p(p_t, 'modalité')}")

        u_stat, p_mw = mannwhitneyu(pc_vals, vr_vals, alternative="two-sided")
        report.append(f"  Mann-Whitney   : U={u_stat:.0f}, p={p_mw:.3f} (non-paramétrique)")

        rows.append({"test": "t-test Welch", "factor": "modalite", "level_a": "PC", "level_b": "VR",
                     "n_a": len(pc_vals), "n_b": len(vr_vals), "mean_a": np.mean(pc_vals).round(3),
                     "mean_b": np.mean(vr_vals).round(3), "stat": round(t_stat, 3),
                     "p_value": round(p_t, 4), "effect_size": round(d, 3), "effect_type": "Cohen's d"})
        rows.append({"test": "Mann-Whitney", "factor": "modalite", "level_a": "PC", "level_b": "VR",
                     "n_a": len(pc_vals), "n_b": len(vr_vals), "mean_a": np.mean(pc_vals).round(3),
                     "mean_b": np.mean(vr_vals).round(3), "stat": round(u_stat, 3),
                     "p_value": round(p_mw, 4), "effect_size": np.nan, "effect_type": "U"})
    else:
        report.append("  [SKIP] Données insuffisantes.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7D. Modèle factoriel complet
# ---------------------------------------------------------------------------

def run_factorial_model(df: pd.DataFrame, perf_col: str, out_dir: Path, report: list) -> pd.DataFrame:
    """
    Modèle factoriel : performance ~ modalite * scenario
    ANOVA type II via statsmodels si disponible.
    Rapporte effet principal modalité, effet principal scénario, interaction.
    """
    report.append("\n--- D. MODÈLE FACTORIEL (modalité × scénario) ---")

    if "modalite" not in df.columns or "scenario" not in df.columns:
        report.append("  [SKIP] Colonnes modalite ou scenario absentes.")
        return pd.DataFrame()

    df_clean = df[["modalite", "scenario", perf_col]].dropna()
    if len(df_clean) < 6:
        report.append(f"  [SKIP] N trop faible ({len(df_clean)}) pour un modèle factoriel.")
        return pd.DataFrame()

    table, model = ols_anova_table(df_clean, perf_col, ["modalite", "scenario"])

    if table is None:
        report.append("  [SKIP] statsmodels non disponible — installer avec : pip install statsmodels")
        return pd.DataFrame()

    report.append(f"  N total = {len(df_clean)}, R² = {model.rsquared:.3f}")
    report.append("  Tableau ANOVA (type II) :")

    for _, row in table.iterrows():
        term = row["term"]
        if term == "Residual":
            continue
        p = row.get("p_value", row.get("PR(>F)", np.nan))
        f = row.get("F_stat", row.get("F", np.nan))
        eta2 = row.get("eta2_partial", np.nan)
        sig = "**" if p < 0.01 else ("*" if p < ALPHA else "ns")
        report.append(f"    {term:<35} F={f:.3f}  p={p:.3f} {sig}  η²p={eta2:.3f}")

    # Interprétations automatiques
    for term_key, label in [
        ("C(modalite)", "modalité"),
        ("C(scenario)", "scénario"),
        ("C(modalite):C(scenario)", "interaction modalité × scénario"),
    ]:
        row = table[table["term"] == term_key]
        if not row.empty:
            p = row["p_value"].values[0]
            if term_key.startswith("C(modalite):"):
                if p < ALPHA:
                    report.append(f"  → L'interaction modalité × scénario est significative : "
                                  f"l'effet de la modalité dépend du scénario.")
                else:
                    report.append(f"  → Pas d'interaction significative modalité × scénario.")
            else:
                report.append(f"  → {interpret_p(p, label)}")

    return table


# ---------------------------------------------------------------------------
# 7E. ANCOVA
# ---------------------------------------------------------------------------

def run_ancova_if_possible(
    df: pd.DataFrame,
    perf_col: str,
    covariate: str | None,
    out_dir: Path,
    report: list,
) -> pd.DataFrame:
    """
    ANCOVA : performance ~ modalite * scenario + covariate
    La covariable sert à contrôler une source de variance pré-existante
    (ex : compétence individuelle). Elle ne remplace pas l'interprétation
    des effets expérimentaux (modalité, scénario).
    """
    report.append("\n--- E. ANCOVA ---")

    if covariate is None:
        report.append("  Aucune covariable pertinente détectée → ANCOVA non réalisée.")
        return pd.DataFrame()

    if "modalite" not in df.columns or "scenario" not in df.columns:
        report.append("  [SKIP] Colonnes condition absentes.")
        return pd.DataFrame()

    df_clean = df[["modalite", "scenario", perf_col, covariate]].dropna()
    if len(df_clean) < 8:
        report.append(f"  [SKIP] N={len(df_clean)} insuffisant pour ANCOVA (recommandé ≥ 8).")
        return pd.DataFrame()

    report.append(f"  Covariable : '{covariate}' (N={len(df_clean)} observations valides)")
    report.append(f"  Modèle : {perf_col} ~ modalite * scenario + {covariate}")

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        formula = f"{perf_col} ~ C(modalite) * C(scenario) + {covariate}"
        model = smf.ols(formula, data=df_clean).fit()
        table = anova_lm(model, typ=2)
        table = table.reset_index().rename(columns={"index": "term", "PR(>F)": "p_value", "F": "F_stat"})

        ss_residual = table.loc[table["term"] == "Residual", "sum_sq"].values[0]
        table["eta2_partial"] = table["sum_sq"] / (table["sum_sq"] + ss_residual)
        table["eta2_partial"] = table["eta2_partial"].where(table["term"] != "Residual")

        report.append(f"  R² = {model.rsquared:.3f}  (vs modèle sans covariable)")
        for _, row in table.iterrows():
            if row["term"] == "Residual":
                continue
            p = row.get("p_value", np.nan)
            f = row.get("F_stat", np.nan)
            eta2 = row.get("eta2_partial", np.nan)
            sig = "**" if p < 0.01 else ("*" if p < ALPHA else "ns")
            report.append(f"    {row['term']:<35} F={f:.3f}  p={p:.3f} {sig}  η²p={eta2:.3f}")

        return table
    except ImportError:
        report.append("  [SKIP] statsmodels non disponible.")
        return pd.DataFrame()
    except Exception as e:
        report.append(f"  [WARN] ANCOVA échouée : {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 7E. ANCOVA avec scénario comme covariable de contrôle
# ---------------------------------------------------------------------------

def run_ancova_scenario_control(
    df: pd.DataFrame,
    perf_col: str,
    out_dir: Path,
    report: list,
) -> pd.DataFrame:
    """
    Teste l'effet de la modalité après contrôle statistique du scénario.

    Contexte méthodologique :
    -------------------------
    Le design est DÉSÉQUILIBRÉ : VR est surreprésenté en S2 (7 vs 4 en S1),
    et S2 présente des performances VR plus faibles (M=32.7 vs 47.4 en S1).
    Sans ajustement, l'effet modalité est partiellement confondu avec le scénario.

    Ce modèle additif :
        performance ~ C(modalite) + C(scenario)
    estime l'effet de la modalité après avoir "retiré" la variance expliquée
    par le scénario. Le scénario est traité comme une covariable catégorielle
    (dummy-codée) — approche légitime même avec une variable binaire.

    Comparaison avec le modèle factoriel (D) :
    - Le modèle D (type II) test déjà chaque effet en contrôlant l'autre.
    - Ce modèle F rend cet ajustement explicite et donne des moyennes ajustées.
    - Il permet de rapporter un effet net de la modalité, plus robuste
      face au déséquilibre du design.

    Sorties :
    - Tableau ANOVA du modèle additif ajusté
    - Moyennes marginales estimées (EMM) par modalité, ajustées pour le scénario
    """
    report.append("\n--- E. ANCOVA — Modalité ajustée pour le scénario ---")
    report.append("  Justification : design déséquilibré (VR surreprésenté en S2 → confound possible)")
    report.append(f"  Modèle : {perf_col} ~ C(modalite) + C(scenario)  [sans interaction]")

    if "modalite" not in df.columns or "scenario" not in df.columns:
        report.append("  [SKIP] Colonnes condition absentes.")
        return pd.DataFrame()

    df_clean = df[["modalite", "scenario", perf_col]].dropna()
    if len(df_clean) < 6:
        report.append(f"  [SKIP] N trop faible ({len(df_clean)}).")
        return pd.DataFrame()

    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm

        # Modèle additif : effets principaux sans interaction
        formula = f"{perf_col} ~ C(modalite) + C(scenario)"
        model = smf.ols(formula, data=df_clean).fit()
        table = anova_lm(model, typ=2)
        table = table.reset_index().rename(
            columns={"index": "term", "PR(>F)": "p_value", "F": "F_stat"}
        )

        ss_residual = table.loc[table["term"] == "Residual", "sum_sq"].values[0]
        table["eta2_partial"] = table["sum_sq"] / (table["sum_sq"] + ss_residual)
        table["eta2_partial"] = table["eta2_partial"].where(table["term"] != "Residual")
        table["model"] = "additif_ajuste"

        report.append(f"  N = {len(df_clean)}, R² = {model.rsquared:.3f}")
        report.append("  Tableau ANOVA (type II — modèle additif ajusté) :")

        for _, row in table.iterrows():
            if row["term"] == "Residual":
                continue
            p = row.get("p_value", np.nan)
            f = row.get("F_stat", np.nan)
            eta2 = row.get("eta2_partial", np.nan)
            sig = "**" if p < 0.01 else ("*" if p < ALPHA else "ns")
            report.append(f"    {row['term']:<35} F={f:.3f}  p={p:.3f} {sig}  η²p={eta2:.3f}")

        # Moyennes marginales estimées (EMM) par modalité
        # = prédiction du modèle pour chaque modalité en fixant scenario à sa distribution moyenne
        modalities = sorted(df_clean["modalite"].unique())
        scenarios = df_clean["scenario"].unique()
        emm_rows = []
        for mod in modalities:
            # Prédire sur une grille balancée : toutes combinaisons scenario
            grid = pd.DataFrame({
                "modalite": [mod] * len(scenarios),
                "scenario": list(scenarios),
            })
            preds = model.predict(grid)
            emm = preds.mean()  # moyenne des prédictions = moyenne marginale ajustée
            emm_rows.append({"modalite": mod, "estimated_marginal_mean": round(emm, 3)})

        emm_df = pd.DataFrame(emm_rows)
        report.append("  Moyennes marginales estimées (ajustées pour le scénario) :")
        for _, r in emm_df.iterrows():
            report.append(f"    {r['modalite']} : {r['estimated_marginal_mean']:.2f}")

        # Comparaison : effet brut vs ajusté
        raw_pc = df_clean[df_clean["modalite"] == "PC"][perf_col].mean()
        raw_vr = df_clean[df_clean["modalite"] == "VR"][perf_col].mean()
        adj_pc = emm_df[emm_df["modalite"] == "PC"]["estimated_marginal_mean"].values[0]
        adj_vr = emm_df[emm_df["modalite"] == "VR"]["estimated_marginal_mean"].values[0]
        report.append(f"  Comparaison effet brut vs ajusté :")
        report.append(f"    PC brut={raw_pc:.2f}  ajusté={adj_pc:.2f}  (diff={adj_pc-raw_pc:+.2f})")
        report.append(f"    VR brut={raw_vr:.2f}  ajusté={adj_vr:.2f}  (diff={adj_vr-raw_vr:+.2f})")
        report.append(f"    Ecart ajusté PC-VR : {adj_pc - adj_vr:.2f} pts")

        # Interprétation de l'effet modalité ajusté
        row_mod = table[table["term"] == "C(modalite)"]
        if not row_mod.empty:
            p_mod = row_mod["p_value"].values[0]
            report.append(f"  → {interpret_p(p_mod, 'modalité (ajusté pour scénario)')}")
            if abs((adj_pc - adj_vr) - (raw_pc - raw_vr)) > 2:
                report.append("  → L'ajustement modifie sensiblement l'écart PC-VR : le design "
                               "déséquilibré confondait bien les deux facteurs.")
            else:
                report.append("  → L'ajustement modifie peu l'écart PC-VR : le confound scénario "
                               "est limité dans ce dataset.")

        # Sauvegarder EMM
        emm_df.to_csv(out_dir / "performance_emm_modalite_ajustee.csv", index=False)
        print("[OK] performance_emm_modalite_ajustee.csv")

        return table

    except ImportError:
        report.append("  [SKIP] statsmodels non disponible.")
        return pd.DataFrame()
    except Exception as e:
        report.append(f"  [WARN] ANCOVA scénario échouée : {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------

def save_performance_plots(df: pd.DataFrame, perf_col: str, out_dir: Path):
    """
    Génère :
      - violin + boxplot par scénario
      - violin + boxplot par modalité
      - violin + boxplot combiné modalité × scénario
    Chaque point individuel est superposé (jitter) pour transparence des données.
    """
    sns_available = _try_import_seaborn()
    palette = {"PC": "#4e79a7", "VR": "#e15759", "S1": "#59a14f", "S2": "#f28e2b"}

    def _add_jitter(ax, data_dict, positions, color_map=None):
        """Superpose les points individuels avec jitter."""
        for i, (label, vals) in enumerate(data_dict.items()):
            x_jitter = np.random.normal(positions[i], 0.04, size=len(vals))
            c = color_map.get(label, "#555") if color_map else "#555"
            ax.scatter(x_jitter, vals, color=c, s=28, alpha=0.75, zorder=3,
                       edgecolors="white", linewidths=0.5)

    # --- Plot 1 : par scénario ---
    if "scenario" in df.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        groups = {s: df[df["scenario"] == s][perf_col].dropna().values
                  for s in sorted(df["scenario"].dropna().unique())}
        positions = list(range(1, len(groups) + 1))
        bp = ax.violinplot([v for v in groups.values()], positions=positions,
                           showmedians=True, showextrema=True)
        for i, (key, pc_elem) in enumerate(zip(groups.keys(), bp["bodies"])):
            pc_elem.set_facecolor(palette.get(key, "#aaa"))
            pc_elem.set_alpha(0.6)
        _add_jitter(ax, groups, positions, palette)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()])
        ax.set_ylabel(perf_col)
        ax.set_title(f"Performance par scénario")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "performance_par_scenario.png", dpi=200)
        plt.close()
        print("[OK] performance_par_scenario.png")

    # --- Plot 2 : par modalité ---
    if "modalite" in df.columns:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        groups = {m: df[df["modalite"] == m][perf_col].dropna().values
                  for m in sorted(df["modalite"].dropna().unique())}
        positions = list(range(1, len(groups) + 1))
        bp = ax.violinplot([v for v in groups.values()], positions=positions,
                           showmedians=True, showextrema=True)
        for key, pc_elem in zip(groups.keys(), bp["bodies"]):
            pc_elem.set_facecolor(palette.get(key, "#aaa"))
            pc_elem.set_alpha(0.6)
        _add_jitter(ax, groups, positions, palette)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()])
        ax.set_ylabel(perf_col)
        ax.set_title("Performance par modalité")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "performance_par_modalite.png", dpi=200)
        plt.close()
        print("[OK] performance_par_modalite.png")

    # --- Plot 3 : modalité × scénario ---
    if "modalite" in df.columns and "scenario" in df.columns:
        scenarios = sorted(df["scenario"].dropna().unique())
        modalities = sorted(df["modalite"].dropna().unique())
        n_s = len(scenarios)
        fig, axes = plt.subplots(1, n_s, figsize=(4.5 * n_s, 5), sharey=True)
        if n_s == 1:
            axes = [axes]

        for ax, scen in zip(axes, scenarios):
            sub = df[df["scenario"] == scen]
            groups = {m: sub[sub["modalite"] == m][perf_col].dropna().values
                      for m in modalities}
            positions = list(range(1, len(modalities) + 1))
            bp = ax.violinplot([groups.get(m, np.array([])) for m in modalities],
                               positions=positions, showmedians=True, showextrema=True)
            for key, pc_elem in zip(modalities, bp["bodies"]):
                pc_elem.set_facecolor(palette.get(key, "#aaa"))
                pc_elem.set_alpha(0.6)
            _add_jitter(ax, {m: groups[m] for m in modalities}, positions, palette)
            ax.set_xticks(positions)
            ax.set_xticklabels([f"{m}\n(n={len(groups.get(m, []))})" for m in modalities])
            ax.set_title(f"{scen}")
            ax.grid(True, axis="y", alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel(perf_col)

        fig.suptitle("Performance par modalité × scénario", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_dir / "performance_par_condition.png", dpi=200)
        plt.close()
        print("[OK] performance_par_condition.png")


def _try_import_seaborn():
    try:
        import seaborn
        return seaborn
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Analyse statistique de l'effet scénario/modalité sur la performance"
    )
    ap.add_argument(
        "--perf",
        default=default_performance_path(),
        help="CSV de performance (priorité: performance_task/recap_scores_all.csv, puis chemin legacy)",
    )
    ap.add_argument(
        "--merged",
        default=default_merged_dataset_path(),
        help="Dataset fusionné (priorité: merged_dataset/with_pruning/, puis without_pruning/, puis chemin legacy)",
    )
    ap.add_argument(
        "--out",
        default=r"D:\Analyse_donnee\Longitudinale\results\analyse_performance",
        help="Dossier de sortie",
    )
    ap.add_argument("--perf-col", default=None, help="Forcer la colonne de performance (optionnel)")
    ap.add_argument("--covariate", default=None, help="Forcer la covariable (optionnel)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANALYSE DE PERFORMANCE — Effets scénario & modalité")
    print("=" * 60)

    # Chargement
    merged_path = Path(args.merged) if args.merged else None
    df, perf_col, covariate = load_and_prepare(
        perf_path=Path(args.perf),
        merged_path=merged_path,
        perf_col_override=args.perf_col,
    )
    if args.covariate:
        covariate = args.covariate if args.covariate in df.columns else covariate

    print(f"\n  Variable de performance retenue : '{perf_col}'")
    print(f"  Observations finales : {len(df)}")
    if covariate:
        print(f"  Covariable : '{covariate}'")

    report = [
        "=" * 60,
        "RAPPORT — Analyse de performance (scénario × modalité)",
        "=" * 60,
        f"Variable dépendante : {perf_col}",
        f"N observations      : {len(df)}",
        f"Covariable          : {covariate or 'aucune'}",
    ]

    # Descriptifs
    print("\n[ÉTAPE 1] Statistiques descriptives")
    desc_scen = pd.DataFrame()
    desc_mod = pd.DataFrame()
    desc_cond = pd.DataFrame()

    if "scenario" in df.columns:
        desc_scen = compute_descriptives(df, perf_col, ["scenario"])
        desc_scen.to_csv(out_dir / "performance_descriptifs_par_scenario.csv", index=False)
        print("[OK] performance_descriptifs_par_scenario.csv")
        report.append("\n--- Descriptifs par scénario ---")
        report.append(desc_scen.to_string(index=False))

    if "modalite" in df.columns:
        desc_mod = compute_descriptives(df, perf_col, ["modalite"])
        desc_mod.to_csv(out_dir / "performance_descriptifs_par_modalite.csv", index=False)
        print("[OK] performance_descriptifs_par_modalite.csv")
        report.append("\n--- Descriptifs par modalité ---")
        report.append(desc_mod.to_string(index=False))

    if "modalite" in df.columns and "scenario" in df.columns:
        desc_cond = compute_descriptives(df, perf_col, ["modalite", "scenario"])
        desc_cond.to_csv(out_dir / "performance_descriptifs_par_condition.csv", index=False)
        print("[OK] performance_descriptifs_par_condition.csv")
        report.append("\n--- Descriptifs par condition (modalité × scénario) ---")
        report.append(desc_cond.to_string(index=False))

    # A. Vérifications
    print("\n[ÉTAPE 2] Vérifications préalables")
    check_assumptions(df, perf_col, report)

    # B. Effet scénario
    print("\n[ÉTAPE 3] Effet du scénario")
    res_scen = run_scenario_effect(df, perf_col, out_dir, report)
    if not res_scen.empty:
        res_scen.to_csv(out_dir / "performance_modele_scenario.csv", index=False)
        print("[OK] performance_modele_scenario.csv")

    # C. Effet modalité
    print("\n[ÉTAPE 4] Effet de la modalité")
    res_mod = run_modality_effect(df, perf_col, out_dir, report)
    if not res_mod.empty:
        res_mod.to_csv(out_dir / "performance_modele_modalite.csv", index=False)
        print("[OK] performance_modele_modalite.csv")

    # D. Modèle factoriel
    print("\n[ÉTAPE 5] Modèle factoriel complet")
    res_fact = run_factorial_model(df, perf_col, out_dir, report)
    if isinstance(res_fact, pd.DataFrame) and not res_fact.empty:
        res_fact.to_csv(out_dir / "performance_modele_factoriel.csv", index=False)
        print("[OK] performance_modele_factoriel.csv")

    # E. ANCOVA covariable externe
    print("\n[ÉTAPE 6] ANCOVA (covariable externe)")
    res_ancova = run_ancova_if_possible(df, perf_col, covariate, out_dir, report)
    if isinstance(res_ancova, pd.DataFrame) and not res_ancova.empty:
        res_ancova.to_csv(out_dir / "performance_modele_ancova.csv", index=False)
        print("[OK] performance_modele_ancova.csv")

    # E. ANCOVA scénario comme contrôle (design déséquilibré)
    print("\n[ÉTAPE 6b] ANCOVA — Modalité ajustée pour le scénario")
    res_ancova_scen = run_ancova_scenario_control(df, perf_col, out_dir, report)
    if isinstance(res_ancova_scen, pd.DataFrame) and not res_ancova_scen.empty:
        res_ancova_scen.to_csv(out_dir / "performance_modele_ancova_scenario.csv", index=False)
        print("[OK] performance_modele_ancova_scenario.csv")

    # Figures
    print("\n[ÉTAPE 7] Figures")
    save_performance_plots(df, perf_col, out_dir)

    # Rapport
    report.append("\n" + "=" * 60)
    report.append("LIMITES MÉTHODOLOGIQUES")
    report.append("=" * 60)
    report.append(f"  - N total = {len(df)} (petit échantillon). Puissance statistique limitée.")
    report.append("  - Les résultats des tests paramétriques sont à interpréter avec prudence.")
    report.append("  - Les tests non-paramétriques (Mann-Whitney) sont fournis à titre complémentaire.")
    report.append("  - Unité d'analyse : groupe (pas participant). Pas d'imbrication modélisée.")
    if covariate is None:
        report.append("  - Aucune covariable pertinente trouvée dans les données.")

    report_text = "\n".join(report)
    print("\n" + report_text)

    report_path = out_dir / "rapport_analyse_performance.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] Rapport texte : {report_path}")
    print(f"[DONE] Résultats dans : {out_dir}")


if __name__ == "__main__":
    main()
