#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiple_regression_perf.py
===========================
Régressions linéaires multiples sur le dataset fusionné.

Objectif:
- Tester la performance de groupe prédite par C-factor, RME (mean/max/min), modalité.
- Deux VDs: Perf_mean et Perf_Z.
- Deux modèles par VD:
  1) Simple       : VD ~ C + RME + modalité
  2) Interactions : VD ~ C*modalité + RME*modalité (C et RME centrés)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


EXCLUDED_GROUPS = {"bim002", "bim032", "bim065", "bim075"}

DV_CANDIDATES = {
    "Perf_mean": ["Perf_mean", "Score_perf_tsk", "Score_perf_tsk_mean", "score_final"],
    "Perf_Z": ["Perf_Z", "Score_perf_tsk_z", "score_z", "performance_z"],
}

PREDICTOR_CANDIDATES = {
    "C": ["c_score", "C_factor", "c_factor", "C"],
    "RME_mean": ["rme_mean", "RME_mean"],
    "RME_max": ["rme_max", "RME_max"],
    "RME_min": ["rme_min", "RME_min"],
    "modalite": ["modalite", "condition", "modality"],
    "group_id": ["group_id", "groupe", "group"],
}


def detect_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
    for cand in candidates:
        low = cand.lower()
        if low in cols_lower:
            return cols_lower[low]
    raise ValueError(f"Colonne introuvable pour '{label}'. Candidats: {list(candidates)}")


def prepare_analysis_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    col_map = {
        "dv_perf_mean": detect_column(df, DV_CANDIDATES["Perf_mean"], "Perf_mean"),
        "dv_perf_z": detect_column(df, DV_CANDIDATES["Perf_Z"], "Perf_Z"),
        "c_score": detect_column(df, PREDICTOR_CANDIDATES["C"], "C-factor"),
        "rme_mean": detect_column(df, PREDICTOR_CANDIDATES["RME_mean"], "RME mean"),
        "rme_max": detect_column(df, PREDICTOR_CANDIDATES["RME_max"], "RME max"),
        "rme_min": detect_column(df, PREDICTOR_CANDIDATES["RME_min"], "RME min"),
        "modalite": detect_column(df, PREDICTOR_CANDIDATES["modalite"], "modalité"),
    }

    group_col = None
    try:
        group_col = detect_column(df, PREDICTOR_CANDIDATES["group_id"], "group_id")
    except ValueError:
        pass

    keep_cols = list(dict.fromkeys(list(col_map.values()) + ([group_col] if group_col else [])))
    d = df[keep_cols].copy()

    rename_map = {v: k for k, v in col_map.items()}
    if group_col:
        rename_map[group_col] = "group_id"
    d = d.rename(columns=rename_map)

    # Nettoyage modalité
    d["modalite"] = d["modalite"].astype(str).str.strip().str.upper()
    d.loc[~d["modalite"].isin(["PC", "VR"]), "modalite"] = np.nan
    d["modalite"] = pd.Categorical(d["modalite"], categories=["VR", "PC"])

    # Numérisation prédicteurs / VD
    num_cols = ["dv_perf_mean", "dv_perf_z", "c_score", "rme_mean", "rme_max", "rme_min"]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Exclusions groupes corrompus
    if "group_id" in d.columns:
        d = d[~d["group_id"].astype(str).isin(EXCLUDED_GROUPS)].copy()

    return d, col_map


def model_to_table(model, dv_label: str, model_name: str) -> pd.DataFrame:
    ci = model.conf_int()
    rows = []
    for term in model.params.index:
        rows.append(
            {
                "dv": dv_label,
                "model": model_name,
                "term": term,
                "beta": float(model.params[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "n_used": int(model.nobs),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
            }
        )
    return pd.DataFrame(rows)


def run_regression_for_dv(df_base: pd.DataFrame, dv_col: str, dv_label: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    req = [dv_col, "c_score", "rme_mean", "rme_max", "rme_min", "modalite"]
    d = df_base[req].dropna().copy()
    if dv_col == "dv_perf_mean":
        d = d[d[dv_col] > 0].copy()

    # Modèle simple
    formula_simple = f"{dv_col} ~ c_score + rme_mean + rme_max + rme_min + C(modalite)"
    model_simple = smf.ols(formula_simple, data=d).fit()

    # Modèle interactions (variables centrées)
    for c in ["c_score", "rme_mean", "rme_max", "rme_min"]:
        d[f"{c}_c"] = d[c] - d[c].mean()
    formula_inter = (
        f"{dv_col} ~ c_score_c*C(modalite) + "
        f"rme_mean_c*C(modalite) + rme_max_c*C(modalite) + rme_min_c*C(modalite)"
    )
    model_inter = smf.ols(formula_inter, data=d).fit()

    t_simple = model_to_table(model_simple, dv_label=dv_label, model_name="simple")
    t_inter = model_to_table(model_inter, dv_label=dv_label, model_name="interactions")

    # Résumé lisible pour rapport
    def pick_p(model, term: str) -> str:
        if term in model.pvalues.index:
            return f"{model.params[term]:+.3f} (p={model.pvalues[term]:.4f})"
        return "n/a"

    lines = [
        f"\n[{dv_label}] N={int(model_simple.nobs)}",
        f"  - Modèle simple: R²={model_simple.rsquared:.3f}, R²aj={model_simple.rsquared_adj:.3f}",
        f"    * Effet principal C: {pick_p(model_simple, 'c_score')}",
        f"    * Effet principal RME mean: {pick_p(model_simple, 'rme_mean')}",
        f"    * Effet principal RME max: {pick_p(model_simple, 'rme_max')}",
        f"    * Effet principal RME min: {pick_p(model_simple, 'rme_min')}",
        f"    * Effet principal modalité (PC vs VR): {pick_p(model_simple, 'C(modalite)[T.PC]')}",
        f"  - Modèle interactions: R²={model_inter.rsquared:.3f}, R²aj={model_inter.rsquared_adj:.3f}",
        f"    * Interaction C×modalité: {pick_p(model_inter, 'c_score_c:C(modalite)[T.PC]')}",
        f"    * Interaction RMEmean×modalité: {pick_p(model_inter, 'rme_mean_c:C(modalite)[T.PC]')}",
        f"    * Interaction RMEmax×modalité: {pick_p(model_inter, 'rme_max_c:C(modalite)[T.PC]')}",
        f"    * Interaction RMEmin×modalité: {pick_p(model_inter, 'rme_min_c:C(modalite)[T.PC]')}",
    ]
    return t_simple, t_inter, lines


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    candidates = [
        project_root / "results" / "merged_dataset" / "with_pruning" / "merged_dataset_complete_all.csv",
        project_root / "results" / "merged_dataset" / "without_pruning" / "merged_dataset_complete_all.csv",
        project_root / "results" / "merged_dataset" / "merged_dataset_complete_all.csv",
    ]
    default_input = next((p for p in candidates if p.exists()), candidates[0])
    default_out = project_root / "results" / "analyse_performance"

    ap = argparse.ArgumentParser(description="Régression multiple Perf_mean / Perf_Z sur dataset fusionné.")
    ap.add_argument("--input", type=Path, default=default_input, help="CSV dataset fusionné")
    ap.add_argument("--out-dir", type=Path, default=default_out, help="Dossier de sortie")
    args = ap.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(args.input, encoding="utf-8-sig")
    df_base, col_map = prepare_analysis_df(df_raw)

    t1_simple, t1_inter, lines1 = run_regression_for_dv(df_base, "dv_perf_mean", "Perf_mean")
    t2_simple, t2_inter, lines2 = run_regression_for_dv(df_base, "dv_perf_z", "Perf_Z")

    # Exports
    t1_simple.to_csv(args.out_dir / "regression_perf_mean_simple.csv", index=False, encoding="utf-8-sig")
    t1_inter.to_csv(args.out_dir / "regression_perf_mean_interactions.csv", index=False, encoding="utf-8-sig")
    t2_simple.to_csv(args.out_dir / "regression_perf_z_simple.csv", index=False, encoding="utf-8-sig")
    t2_inter.to_csv(args.out_dir / "regression_perf_z_interactions.csv", index=False, encoding="utf-8-sig")

    all_rows = pd.concat([t1_simple, t1_inter, t2_simple, t2_inter], ignore_index=True)
    all_rows.to_csv(args.out_dir / "regression_perf_all_models.csv", index=False, encoding="utf-8-sig")

    summary_path = args.out_dir / "regression_perf_summary.txt"
    summary = [
        "Régression linéaire multiple — dataset fusionné",
        f"Input: {args.input}",
        f"Colonnes utilisées: {col_map}",
        "",
        *lines1,
        "",
        *lines2,
        "",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("[OK] Exports regression:")
    print(f"  - {args.out_dir / 'regression_perf_mean_simple.csv'}")
    print(f"  - {args.out_dir / 'regression_perf_mean_interactions.csv'}")
    print(f"  - {args.out_dir / 'regression_perf_z_simple.csv'}")
    print(f"  - {args.out_dir / 'regression_perf_z_interactions.csv'}")
    print(f"  - {args.out_dir / 'regression_perf_all_models.csv'}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()

