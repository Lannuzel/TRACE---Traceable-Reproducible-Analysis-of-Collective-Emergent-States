#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlation_metrics_final.py
============================
Matrice de corrélation entre les métriques M1..M4 et le score final.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLS = [
    "M1_consignes_%",
    "M2_nombre_%",
    "M3_precision_%",
    "M4_temps_%",
    "Score_perf_tsk",
]


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    candidates = [
        project_root / "results" / "performance_task" / "recap_scores_all.csv",
        project_root / "results" / "recap_scores_all.csv",
    ]
    default_input = next((p for p in candidates if p.exists()), candidates[0])
    default_output = project_root / "results" / "analyse_performance" / "correlation_metrics_final.csv"

    parser = argparse.ArgumentParser(description="Matrice de corrélation M1..M4 vs score final.")
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="CSV d'entrée (priorité: results/performance_task/recap_scores_all.csv, puis chemin legacy)",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="CSV de sortie")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson",
                        help="Méthode de corrélation")
    parser.add_argument("--keep-zero-final", action="store_true",
                        help="Garde les lignes avec score final = 0 (par défaut: exclus)")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.input}")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = [c for c in DEFAULT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    work = df[DEFAULT_COLS].copy()
    for c in DEFAULT_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if not args.keep_zero_final:
        work = work[work["Score_perf_tsk"] > 0]
    work = work.dropna()

    corr = work.corr(method=args.method).round(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(args.output, encoding="utf-8-sig")

    print(f"\nMatrice de corrélation ({args.method}) :")
    print(corr.to_string())
    print(f"\n[OK] CSV exporté: {args.output}")


if __name__ == "__main__":
    main()

