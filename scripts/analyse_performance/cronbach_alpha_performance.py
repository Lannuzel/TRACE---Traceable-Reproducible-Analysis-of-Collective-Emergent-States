#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cronbach_alpha_performance.py
=============================
Calcule l'alpha de Cronbach sur les métriques M1..M4 du score final de tâche.

Sortie:
  - tableau (lignes = groupes, colonnes = M1..M4 + alpha_cronbach)
  - tableau variable-level (alpha_if_deleted par variable)
  - CSV exporté
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_M_COLS = [
    "M1_consignes_%",
    "M2_nombre_%",
    "M3_precision_%",
    "M4_temps_%",
]


def cronbach_alpha(df_items: pd.DataFrame) -> float | None:
    """Alpha de Cronbach pour un DataFrame (n_groupes x k_items)."""
    k = df_items.shape[1]
    if k < 2:
        return None
    item_vars = df_items.var(axis=0, ddof=1)
    total_scores = df_items.sum(axis=1)
    total_var = total_scores.var(ddof=1)
    if pd.isna(total_var) or total_var <= 0:
        return None
    alpha = (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)
    return float(alpha) if np.isfinite(alpha) else None


def cronbach_alpha_standardized(df_items: pd.DataFrame) -> float | None:
    """Alpha standardisé basé sur la matrice de corrélation inter-items."""
    k = df_items.shape[1]
    if k < 2:
        return None
    corr = df_items.corr()
    if corr.isna().all().all():
        return None
    r_bar = corr.where(~np.eye(k, dtype=bool)).stack().mean()
    if pd.isna(r_bar):
        return None
    den = 1.0 + (k - 1.0) * r_bar
    if den == 0:
        return None
    alpha_std = (k * r_bar) / den
    return float(alpha_std) if np.isfinite(alpha_std) else None


def _item_total_correlation(df_items: pd.DataFrame, col: str) -> float | None:
    item = df_items[col]
    rest = df_items.drop(columns=[col]).sum(axis=1)
    mask = item.notna() & rest.notna()
    if mask.sum() < 3:
        return None
    r = np.corrcoef(item[mask], rest[mask])[0, 1]
    return float(r) if np.isfinite(r) else None


def variable_reliability_table(df_items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df_items.columns:
        alpha_if_deleted = cronbach_alpha(df_items.drop(columns=[col]))
        r_drop = _item_total_correlation(df_items, col)
        rows.append(
            {
                "variable": col,
                "variance": round(float(df_items[col].var(ddof=1)), 4),
                "r_drop": round(r_drop, 4) if r_drop is not None else np.nan,
                "alpha_if_deleted": round(alpha_if_deleted, 4) if alpha_if_deleted is not None else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)


def _safe_to_csv(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return output_path
    except PermissionError:
        fallback = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def _detect_group_col(df: pd.DataFrame) -> str:
    for c in ("groupe", "group_id", "group"):
        if c in df.columns:
            return c
    raise ValueError("Colonne groupe introuvable (attendu: groupe/group_id/group).")


def _detect_m_cols(df: pd.DataFrame) -> list[str]:
    if all(c in df.columns for c in DEFAULT_M_COLS):
        return DEFAULT_M_COLS
    fallback = [c for c in ("M1", "M2", "M3", "M4") if c in df.columns]
    if len(fallback) == 4:
        return fallback
    raise ValueError("Colonnes M1..M4 introuvables dans le fichier d'entrée.")


def build_output_table(
    df: pd.DataFrame, keep_zero_final: bool
) -> tuple[pd.DataFrame, float | None, float | None, pd.DataFrame]:
    group_col = _detect_group_col(df)
    m_cols = _detect_m_cols(df)

    work = df.copy()
    for c in m_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if "Score_perf_tsk" in work.columns:
        work["Score_perf_tsk"] = pd.to_numeric(work["Score_perf_tsk"], errors="coerce")
        if not keep_zero_final:
            work = work[work["Score_perf_tsk"] > 0].copy()

    work = work.dropna(subset=[group_col] + m_cols).copy()
    alpha = cronbach_alpha(work[m_cols])
    alpha_std = cronbach_alpha_standardized(work[m_cols])
    variable_stats = variable_reliability_table(work[m_cols])

    out_cols = [group_col] + m_cols
    if "Score_perf_tsk" in work.columns:
        out_cols.append("Score_perf_tsk")
    out = work[out_cols].sort_values(group_col).reset_index(drop=True)
    out["alpha_cronbach"] = round(alpha, 4) if alpha is not None else np.nan
    out["alpha_standardise"] = round(alpha_std, 4) if alpha_std is not None else np.nan
    return out, alpha, alpha_std, variable_stats


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    candidates = [
        project_root / "results" / "performance_task" / "recap_scores_all.csv",
        project_root / "results" / "recap_scores_all.csv",
    ]
    default_input = next((p for p in candidates if p.exists()), candidates[0])
    default_output = project_root / "results" / "analyse_performance" / "cronbach_alpha_performance.csv"
    default_output_vars = project_root / "results" / "analyse_performance" / "cronbach_alpha_performance_variables.csv"

    parser = argparse.ArgumentParser(description="Alpha de Cronbach sur M1..M4 (score tâche).")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="CSV d'entrée (priorité: results/performance_task/recap_scores_all.csv, puis chemin legacy)",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="CSV de sortie")
    parser.add_argument("--output-vars", type=Path, default=default_output_vars,
                        help="CSV de sortie des stats variables (alpha_if_deleted)")
    parser.add_argument("--keep-zero-final", action="store_true",
                        help="Garde les groupes avec Score_perf_tsk=0 (par défaut: exclus)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    out, alpha, alpha_std, variable_stats = build_output_table(df, keep_zero_final=args.keep_zero_final)

    saved_main = _safe_to_csv(out, args.output)
    saved_vars = _safe_to_csv(variable_stats, args.output_vars)

    print("\nTableau (groupes en lignes):")
    print(out.to_string(index=False))
    if alpha is None:
        print("\nAlpha de Cronbach: non calculable")
    else:
        print(f"\nAlpha de Cronbach (global M1..M4): {alpha:.4f}")
    if alpha_std is None:
        print("Alpha standardisé: non calculable")
    else:
        print(f"Alpha standardisé (M1..M4): {alpha_std:.4f}")
    print("\nAlpha sur les variables (alpha_if_deleted):")
    print(variable_stats.to_string(index=False))
    print(f"[OK] CSV exporté: {saved_main}")
    print(f"[OK] CSV variables exporté: {saved_vars}")


if __name__ == "__main__":
    main()

