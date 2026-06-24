#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exploratory_pruning.py — Suppression exploratoire itérative d'items pour maximiser alpha.

Méthode (greedy forward-deletion) :
  - À chaque étape, on teste la suppression de chaque item restant.
  - On retire celui qui augmente le plus l'alpha de Cronbach.
    En cas d'égalité parfaite : on préfère l'item au r.drop le plus faible.
  - On s'arrête dès qu'aucune suppression n'améliore alpha, ou qu'il reste
    moins de `min_items` items (défaut 2).
  - La suppression n'est déclenchée que si l'alpha initial est inférieur à
    `alpha_threshold` (défaut : ALPHA_ACCEPTABILITY_THRESHOLD = 0.70).
    Les dimensions avec α ≥ seuil sont considérées acceptables et ignorées.

Remarque : ce module est EXPLORATOIRE. Il ne remplace pas les résultats
principaux (cronbach_alpha_questionnaire.csv, stats_items_questionnaire.csv).
Les sorties sont dans des fichiers séparés (exploratory_*.csv).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DIMENSION_LABELS, ALPHA_ACCEPTABILITY_THRESHOLD, RDROP_THRESHOLD
from .reliability import _cronbach_alpha, _item_total_correlation


# ---------------------------------------------------------------------------
# Fonctions publiques exposées (alpha, r.drop, alpha-if-deleted)
# Ces wrappers rendent explicites les primitives déjà présentes dans
# reliability.py, pour faciliter les appels ponctuels depuis d'autres modules.
# ---------------------------------------------------------------------------

def compute_alpha(data: np.ndarray) -> float | None:
    """Alpha de Cronbach pour une matrice (n_subjects × k_items)."""
    return _cronbach_alpha(data)


def compute_rdrop(data: np.ndarray, col_idx: int) -> float | None:
    """
    r.drop (corrected item-total correlation) de l'item `col_idx`.
    = corrélation entre l'item et la somme de tous les autres items.
    """
    return _item_total_correlation(data, col_idx)


def compute_alpha_if_deleted(data: np.ndarray, col_idx: int) -> float | None:
    """Alpha de Cronbach si l'item `col_idx` est supprimé."""
    remaining = np.delete(data, col_idx, axis=1)
    if remaining.shape[1] < 2:
        return None
    return _cronbach_alpha(remaining)


# ---------------------------------------------------------------------------
# Suppression exploratoire pour une dimension
# ---------------------------------------------------------------------------

def _exploratory_prune_dimension(
    data: np.ndarray,
    col_names: list[str],
    min_items: int = 2,
) -> dict:
    """
    Suppression itérative pour une seule dimension.

    Paramètres
    ----------
    data      : matrice (n_subjects × k_items) des items valides (variance > 0)
    col_names : liste des noms de colonnes (même ordre que les colonnes de data)
    min_items : nombre minimum d'items à conserver

    Retourne
    --------
    dict avec :
      alpha_initial   : alpha de Cronbach avant tout retrait
      alpha_optimise  : alpha après retrait(s) exploratoires
      items_retires   : liste de dict décrivant chaque retrait
                        {code, step, rdrop_at_removal, alpha_before, alpha_after}
      items_restants  : liste des codes conservés
    """
    # Indices courants dans la matrice `data` (partent de 0..k-1)
    remaining_idx = list(range(data.shape[1]))
    alpha_initial = _cronbach_alpha(data)
    alpha_current = alpha_initial
    items_removed = []
    step = 0

    while len(remaining_idx) > min_items:
        candidates = []

        for loc, col_idx in enumerate(remaining_idx):
            # Indices de tous les items sauf celui-ci
            candidate_idx = [i for j, i in enumerate(remaining_idx) if j != loc]
            if len(candidate_idx) < min_items:
                continue

            sub = data[:, candidate_idx]
            a_new = _cronbach_alpha(sub)
            if a_new is None:
                continue

            # r.drop de l'item dans le sous-ensemble courant
            rdrop = _item_total_correlation(data[:, remaining_idx], loc)
            candidates.append((loc, col_idx, a_new, rdrop))

        # Garder uniquement les suppressions qui améliorent alpha
        alpha_ref = alpha_current if alpha_current is not None else -1.0
        improving = [c for c in candidates if c[2] > alpha_ref]

        if not improving:
            break  # aucune amélioration possible → arrêt

        # Tri : meilleur alpha_new en premier ; r.drop le plus faible en cas d'égalité
        improving.sort(key=lambda c: (-c[2], c[3] if c[3] is not None else 999.0))
        loc, col_idx, alpha_new, rdrop = improving[0]

        step += 1
        items_removed.append({
            "code": col_names[col_idx],
            "step": step,
            "rdrop_at_removal": round(rdrop, 3) if rdrop is not None else None,
            "alpha_before": round(alpha_current, 3) if alpha_current is not None else None,
            "alpha_after": round(alpha_new, 3),
        })

        # Retirer cet item de la liste courante (pop via index local)
        remaining_idx.pop(loc)
        alpha_current = alpha_new

    return {
        "alpha_initial": round(alpha_initial, 3) if alpha_initial is not None else None,
        "alpha_optimise": round(alpha_current, 3) if alpha_current is not None else None,
        "items_retires": items_removed,
        "items_restants": [col_names[i] for i in remaining_idx],
    }


# ---------------------------------------------------------------------------
# Suppression exploratoire pour toutes les dimensions
# ---------------------------------------------------------------------------

def exploratory_prune_all(
    df_long: pd.DataFrame,
    min_items: int = 2,
    alpha_threshold: float = ALPHA_ACCEPTABILITY_THRESHOLD,
    rdrop_threshold: float = RDROP_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lance la suppression exploratoire pour chaque dimension du df_long.

    La suppression n'est tentée que si l'alpha initial de la dimension est
    inférieur à `alpha_threshold`. Les dimensions avec α ≥ seuil sont
    considérées acceptables : elles apparaissent dans le résumé avec
    n_items_retires=0 et le statut "acceptable".

    Dans tous les cas, les items avec r.drop < rdrop_threshold sont listés
    dans la colonne `items_signales`, quelle que soit la santé de la dimension.

    Paramètres
    ----------
    alpha_threshold : seuil d'acceptabilité pour déclencher la suppression (défaut 0.70)
    rdrop_threshold : seuil de signalement des items faibles (défaut RDROP_THRESHOLD=0.20)
    rdrop_threshold : seuil r.drop pour le signalement (défaut RDROP_THRESHOLD=0.20)

    Retourne
    --------
    summary_df : une ligne par dimension, colonnes :
                 dimension, label, n_items, alpha_initial, alpha_optimise,
                 gain_alpha, n_items_retires, items_signales, items_retires, statut
                 - items_signales : TOUS les items avec r.drop < rdrop_threshold
                 - items_retires  : items réellement supprimés (uniquement si α < threshold)
    trace_df   : une ligne par item retiré, colonnes :
                 dimension, label, code, step, rdrop_at_removal,
                 alpha_before, alpha_after
    """
    dims = sorted(df_long["dimension"].dropna().unique())
    summary_rows = []
    trace_rows = []

    for dim in dims:
        sub = df_long.loc[
            (df_long["dimension"] == dim) & df_long["reponse_num"].notna(),
            ["Participant", "code", "reponse_num"],
        ]
        wide = sub.pivot_table(index="Participant", columns="code", values="reponse_num")

        # Exclure items sans variabilité (impossibles pour alpha)
        valid_cols = wide.var()[wide.var() > 0].index.tolist()
        n_items = len(valid_cols)

        if n_items < 2:
            summary_rows.append({
                "dimension": dim,
                "label": DIMENSION_LABELS.get(dim, dim),
                "n_items": n_items,
                "alpha_initial": None,
                "alpha_optimise": None,
                "gain_alpha": None,
                "n_items_retires": 0,
                "items_signales": "",
                "items_retires": "",
                "statut": "insuffisant",
            })
            continue

        data = wide[valid_cols].values
        alpha_initial = _cronbach_alpha(data)
        alpha_initial_r = round(alpha_initial, 3) if alpha_initial is not None else None

        # ── Signalement : items avec r.drop < rdrop_threshold (toutes dimensions) ──
        # Indépendant du seuil alpha — liste systématiquement les items à surveiller.
        flagged_codes = []
        for idx, col in enumerate(valid_cols):
            rdrop = _item_total_correlation(data, idx)
            if rdrop is None or rdrop < rdrop_threshold:
                flagged_codes.append(col)
        items_signales_str = "; ".join(flagged_codes)

        # ── Condition centrale : ne tenter la suppression que si α < seuil ──
        # Si l'alpha est déjà acceptable (≥ alpha_threshold), on signale mais ne retire rien.
        if alpha_initial is not None and alpha_initial >= alpha_threshold:
            summary_rows.append({
                "dimension": dim,
                "label": DIMENSION_LABELS.get(dim, dim),
                "n_items": n_items,
                "alpha_initial": alpha_initial_r,
                "alpha_optimise": alpha_initial_r,
                "gain_alpha": 0.0,
                "n_items_retires": 0,
                "items_signales": items_signales_str,
                "items_retires": "",
                "statut": f"acceptable (α ≥ {alpha_threshold})",
            })
            continue

        # α < seuil ou None (ex. négatif) → on lance la suppression greedy
        res = _exploratory_prune_dimension(data, valid_cols, min_items=min_items)

        codes_retires = [r["code"] for r in res["items_retires"]]
        gain = (
            round(res["alpha_optimise"] - res["alpha_initial"], 3)
            if (res["alpha_optimise"] is not None and res["alpha_initial"] is not None)
            else None
        )

        summary_rows.append({
            "dimension": dim,
            "label": DIMENSION_LABELS.get(dim, dim),
            "n_items": n_items,
            "alpha_initial": res["alpha_initial"],
            "alpha_optimise": res["alpha_optimise"],
            "gain_alpha": gain,
            "n_items_retires": len(codes_retires),
            "items_signales": items_signales_str,
            "items_retires": "; ".join(codes_retires),
            "statut": "optimisé" if codes_retires else "déjà optimal",
        })

        for r in res["items_retires"]:
            trace_rows.append({
                "dimension": dim,
                "label": DIMENSION_LABELS.get(dim, dim),
                **r,
            })

    summary_df = pd.DataFrame(summary_rows)
    trace_df = (
        pd.DataFrame(trace_rows)
        if trace_rows
        else pd.DataFrame(columns=[
            "dimension", "label", "code", "step",
            "rdrop_at_removal", "alpha_before", "alpha_after",
        ])
    )
    return summary_df, trace_df


def apply_exploratory_pruning(
    df_long: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applique les suppressions calculées par `exploratory_prune_all` au df_long.

    Pour chaque dimension, retire les items identifiés comme nuisibles à l'alpha.
    Les items retirés sont listés dans summary_df["items_retires"] (séparés par ";").

    Retourne un df_long épuré (même structure, moins les items retirés).
    """
    # Collecter tous les codes à supprimer depuis la colonne items_retires
    codes_to_remove: set[str] = set()
    for items_str in summary_df["items_retires"].dropna():
        if items_str.strip():
            for code in items_str.split(";"):
                code = code.strip()
                if code:
                    codes_to_remove.add(code)

    if not codes_to_remove:
        return df_long.copy()

    return df_long[~df_long["code"].isin(codes_to_remove)].copy()


# ---------------------------------------------------------------------------
# Affichage console
# ---------------------------------------------------------------------------
def print_exploratory_summary(
    summary_df: pd.DataFrame,
    alpha_threshold: float = ALPHA_ACCEPTABILITY_THRESHOLD,
) -> None:
    """
    Affiche un tableau de synthèse par dimension dans la console :
    statut, alpha_initial, n_items, items signalés, alpha après retrait exploratoire.
    """
    print("\n" + "=" * 88)
    print(
        f"  ANALYSE EXPLORATOIRE — Suppression itérative d'items (greedy alpha-max)"
        f"  [seuil α = {alpha_threshold}]"
    )
    print("=" * 88)
    header = (
        f"{'Dim':<6} {'Label':<34} {'n':<4} "
        f"{'α init':<8} {'α opt':<8} {'Δα':<7} {'Statut':<22} {'Items retirés'}"
    )
    print(header)
    print("-" * 88)

    for _, row in summary_df.iterrows():
        alpha_i = f"{row['alpha_initial']:.3f}"  if pd.notna(row["alpha_initial"])  else "  N/A "
        alpha_o = f"{row['alpha_optimise']:.3f}" if pd.notna(row["alpha_optimise"]) else "  N/A "
        gain    = f"{row['gain_alpha']:+.3f}"    if pd.notna(row["gain_alpha"])      else "  N/A "
        statut  = str(row.get("statut", ""))[:21]

        # Items retirés (suppression effective)
        items_ret = row["items_retires"] if row.get("items_retires") else "—"
        ret_short = "; ".join(
            c.split("[")[-1].rstrip("]").split(".")[0]
            if "[" in c else c.split(".")[-1]
            for c in items_ret.split(";")
        ) if items_ret != "—" else "—"

        # Items signalés r.drop faible (toutes dimensions)
        items_sig = row.get("items_signales", "") or ""
        sig_short = "; ".join(
            c.split("[")[-1].rstrip("]").split(".")[0]
            if "[" in c else c.split(".")[-1]
            for c in items_sig.split(";")
        ) if items_sig else "—"

        label = str(row["label"])[:33]
        print(
            f"{row['dimension']:<6} {label:<34} {row['n_items']:<4} "
            f"{alpha_i:<8} {alpha_o:<8} {gain:<7} {statut:<22} ret={ret_short} | sig={sig_short}"
        )
    print("=" * 88)
    print("  Note : les suppressions sont EXPLORATOIRES et ne modifient pas les")
    print("  résultats principaux. Voir exploratory_*.csv pour le détail.")
    print("=" * 88 + "\n")

    # Signale les dimensions avec alpha initial négatif (probable problème d'inversion)
    neg_alpha_dims = summary_df[
        summary_df["alpha_initial"].notna() & (summary_df["alpha_initial"] < 0)
    ]
    if not neg_alpha_dims.empty:
        print("  ⚠  AVERTISSEMENT — Dimensions avec alpha initial NÉGATIF :")
        print("     Cause probable : inversion incorrecte dans INVERT_SHORT (config.py).")
        print("     Un alpha négatif indique que certains items mesurent dans la direction")
        print("     opposée aux autres. Vérifiez les items suivants :")
        for _, row in neg_alpha_dims.iterrows():
            print(f"     • {row['dimension']} ({row['label']}) : α = {row['alpha_initial']}")
        print()
