#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalcule des variantes du c-factor à partir de la population parente Riedl
(2021), tout en conservant une variante normalisée sur l'échantillon.

Entrées attendues
-----------------
- `c_scores_with_tasks_allowed.csv`
- `c_scores_with_tasks_all.csv`
- `full.csv` du dossier `Riedl_2021_QuantifyingCI`

Sorties principales
-------------------
- `c_scores_allowed_pop.csv`
- `c_scores_all_pop.csv`
- `task_reference_stats.csv`

Notes méthodologiques
---------------------
- Le mapping entre tâches locales et tâches de `full.csv` est explicite et auditable.
- `c_factor_pop` n'est plus obtenu en refittant une PCA sur l'échantillon local
  après z-scoring parent. Les groupes locaux sont désormais projetés sur la PC1
  apprise dans la population parente Riedl.
- Les tâches locales sans équivalent strict dans `full.csv` restent documentées
  dans `task_reference_stats.csv`, mais ne peuvent pas contribuer à la projection
  sur les loadings parentaux.
- Le fichier de sortie garde `c_score` comme alias de `c_factor_pop` pour rester
  compatible avec le pipeline historique qui attend une colonne `c_score`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


TASK_MAPPING: dict[str, str | None] = {
    # Les tâches ci-dessous ont un équivalent raisonnablement comparable dans
    # la population parente Riedl 2021.
    "Brainstrorming_word_P_N": "BrainstormWords1",
    "MatrixSolvingN1_FR": "MatrixSolving1",
    "Sudoku_FR": "Sudoku1",
    "TypingText_FR": "TypingText1",
    # Les tâches suivantes n'ont pas d'équivalent strictement compatible :
    # on conserve donc une normalisation locale échantillon.
    "Brainstrorming_equation": None,
    "MemoryGrid1_FR": None,
    "MemoryGrid2_FR": None,
    "MemoryGrid3_FR": None,
}

PARENT_TASK_COLUMNS = [
    "BrainstormObject1",
    "BrainstormWords1",
    "MatrixSolving1",
    "MemoryPicture1",
    "Sudoku1",
    "TypingNumbers1",
    "TypingText1",
    "UnscrambleWords1",
]

ID_COLS = {"group_id", "c_score", "rme_mean", "rme_max", "rme_min"}


def _task_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLS]


def _safe_series_stats(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, 0
    mean_val = float(values.mean())
    sd_val = float(values.std(ddof=0))
    return mean_val, sd_val, int(values.shape[0])


def _build_reference_stats(
    wide_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    scope: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    stats_map: dict[str, dict[str, Any]] = {}

    for task_col in _task_columns(wide_df):
        mapped = TASK_MAPPING.get(task_col)
        if mapped is not None and mapped in full_df.columns:
            ref_series = full_df[mapped]
            mean_pop, sd_pop, n_pop = _safe_series_stats(ref_series)
            ref_source = "Riedl2021"
            flag = "mapped_to_parent_population"
            if not np.isfinite(sd_pop) or sd_pop == 0:
                mean_pop, sd_pop, n_pop = _safe_series_stats(wide_df[task_col])
                ref_source = "sample"
                flag = "fallback_sample_parent_sd_invalid"
        else:
            mean_pop, sd_pop, n_pop = _safe_series_stats(wide_df[task_col])
            ref_source = "sample"
            flag = "fallback_sample_no_parent_match"

        rows.append(
            {
                "scope": scope,
                "task_sample": task_col,
                "task_full_csv": mapped if mapped is not None else "",
                "reference_pop": ref_source,
                "mean_pop": mean_pop,
                "sd_pop": sd_pop,
                "n_pop": n_pop,
                "flag": flag,
            }
        )
        stats_map[task_col] = rows[-1]

    return pd.DataFrame(rows), stats_map


def _zscore_against_reference(wide_df: pd.DataFrame, ref_stats: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = pd.DataFrame(index=wide_df.index)
    for task_col in _task_columns(wide_df):
        vals = pd.to_numeric(wide_df[task_col], errors="coerce")
        mean_pop = float(ref_stats[task_col]["mean_pop"])
        sd_pop = float(ref_stats[task_col]["sd_pop"])
        if not np.isfinite(sd_pop) or sd_pop == 0:
            out[task_col] = np.nan
        else:
            out[task_col] = (vals - mean_pop) / sd_pop
    return out


def _sample_zscores(wide_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=wide_df.index)
    for task_col in _task_columns(wide_df):
        vals = pd.to_numeric(wide_df[task_col], errors="coerce")
        mean_val, sd_val, _ = _safe_series_stats(vals)
        if not np.isfinite(sd_val) or sd_val == 0:
            out[task_col] = np.nan
        else:
            out[task_col] = (vals - mean_val) / sd_val
    return out


def _parent_mapped_pairs(
    wide_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Retourne les couples (tâche locale, tâche parent) utilisables pour la PCA parentale."""
    pairs: list[tuple[str, str]] = []
    for task_col in _task_columns(wide_df):
        mapped = TASK_MAPPING.get(task_col)
        if mapped is None:
            continue
        if mapped not in full_df.columns:
            continue
        mean_pop, sd_pop, _ = _safe_series_stats(full_df[mapped])
        if not np.isfinite(sd_pop) or sd_pop == 0:
            continue
        pairs.append((task_col, mapped))
    return pairs


def _fit_parent_pca_model(
    reference_wide_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Apprend la PC1 directement sur la population parente Riedl.

    La PCA est ajustée sur la batterie parentale complète disponible dans
    `full.csv`. Les groupes locaux sont ensuite projetés sur cette composante ;
    les tâches parentales sans équivalent local sont imputées au niveau moyen
    de la population parente lors de la projection (z = 0).
    """
    mapped_pairs = _parent_mapped_pairs(reference_wide_df, full_df)
    parent_task_cols: list[str] = []
    for parent_task in PARENT_TASK_COLUMNS:
        if parent_task not in full_df.columns:
            continue
        mean_pop, sd_pop, _ = _safe_series_stats(full_df[parent_task])
        if not np.isfinite(sd_pop) or sd_pop == 0:
            continue
        parent_task_cols.append(parent_task)

    if len(parent_task_cols) < 2:
        raise ValueError(
            "Impossible d'ajuster une PCA parentale: moins de deux tâches "
            "utilisables dans la population parente."
        )

    parent_z = pd.DataFrame(index=full_df.index)
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    n_pops: dict[str, int] = {}

    for parent_task in parent_task_cols:
        raw = pd.to_numeric(full_df[parent_task], errors="coerce")
        mean_pop, sd_pop, n_pop = _safe_series_stats(raw)
        means[parent_task] = mean_pop
        sds[parent_task] = sd_pop
        n_pops[parent_task] = n_pop
        parent_z[parent_task] = (raw - mean_pop) / sd_pop

    parent_missing_before = int(parent_z.isna().sum().sum())
    parent_total_cells = int(parent_z.shape[0] * parent_z.shape[1]) if parent_z.shape[0] and parent_z.shape[1] else 0
    parent_imputed = parent_z.fillna(0.0)

    pca = PCA(n_components=1)
    pca.fit(parent_imputed.to_numpy(dtype=float))

    weights = pd.Series(pca.components_[0], index=parent_z.columns, name="parent_pc1_weight")
    loadings = pd.Series(
        pca.components_[0] * math.sqrt(float(pca.explained_variance_[0])),
        index=parent_z.columns,
        name="parent_pc1_loading",
    )

    parent_to_local = {full_task: local_task for local_task, full_task in mapped_pairs}
    local_to_parent = {local_task: full_task for local_task, full_task in mapped_pairs}

    return {
        "pca": pca,
        "mapped_pairs": mapped_pairs,
        "local_task_cols": [local_task for local_task, _ in mapped_pairs],
        "parent_task_cols": list(parent_z.columns),
        "parent_to_local": parent_to_local,
        "local_to_parent": local_to_parent,
        "means": means,
        "sds": sds,
        "n_pops": n_pops,
        "weights": weights,
        "loadings": loadings,
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "parent_missing_before": parent_missing_before,
        "parent_total_cells": parent_total_cells,
        "parent_imputed_pct": (parent_missing_before / parent_total_cells) if parent_total_cells > 0 else np.nan,
        "sign": 1.0,
        "orientation_corr": np.nan,
        "orientation_source": "",
    }


def _zscore_with_parent_model(
    wide_df: pd.DataFrame,
    parent_model: dict[str, Any],
) -> pd.DataFrame:
    """Z-score les tâches locales avec les moyennes/SD de la population parente."""
    out = pd.DataFrame(index=wide_df.index)
    for parent_task in parent_model["parent_task_cols"]:
        local_task = parent_model["parent_to_local"].get(parent_task)
        if local_task is not None and local_task in wide_df.columns:
            vals = pd.to_numeric(wide_df.get(local_task), errors="coerce")
        else:
            vals = pd.Series(np.nan, index=wide_df.index, dtype=float)
        mean_pop = float(parent_model["means"][parent_task])
        sd_pop = float(parent_model["sds"][parent_task])
        out[parent_task] = (vals - mean_pop) / sd_pop
    return out


def _orient_parent_model(
    parent_model: dict[str, Any],
    orientation_wide_df: pd.DataFrame,
    *,
    reference: pd.Series | None,
    orientation_source: str,
) -> dict[str, Any]:
    """
    Fige une orientation de signe unique pour la PC1 parentale.

    On la choisit une seule fois à partir du jeu `all`, afin que les loadings
    exportés et les scores projetés restent cohérents entre sorties.
    """
    oriented = dict(parent_model)
    z_orient = _zscore_with_parent_model(orientation_wide_df, oriented).fillna(0.0)
    raw_scores = pd.Series(
        oriented["pca"].transform(z_orient.to_numpy(dtype=float)).ravel(),
        index=orientation_wide_df.index,
        name="pc1",
    )

    sign = 1.0
    corr = np.nan
    if reference is not None:
        aligned = pd.concat([raw_scores, pd.to_numeric(reference, errors="coerce")], axis=1).dropna()
        if aligned.shape[0] >= 3:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
            if pd.notna(corr) and corr < 0:
                sign = -1.0

    oriented["sign"] = sign
    oriented["orientation_corr"] = corr
    oriented["orientation_source"] = orientation_source
    oriented["weights"] = parent_model["weights"] * sign
    oriented["loadings"] = parent_model["loadings"] * sign
    return oriented


def _project_parent_pc1(
    wide_df: pd.DataFrame,
    parent_model: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """Projette l'échantillon local sur la PC1 apprise dans la population parente."""
    z_parent = _zscore_with_parent_model(wide_df, parent_model)
    missing_before = int(z_parent.isna().sum().sum())
    total_cells = int(z_parent.shape[0] * z_parent.shape[1]) if z_parent.shape[0] and z_parent.shape[1] else 0
    imputed = z_parent.fillna(0.0)
    raw_scores = parent_model["pca"].transform(imputed.to_numpy(dtype=float)).ravel()
    scores = pd.Series(parent_model["sign"] * raw_scores, index=wide_df.index, name="pc1")
    meta = {
        "n_rows": int(imputed.shape[0]),
        "n_cols": int(imputed.shape[1]),
        "missing_before": missing_before,
        "total_cells": total_cells,
        "imputed_pct": (missing_before / total_cells) if total_cells > 0 else np.nan,
        "used_parent_tasks": list(parent_model["parent_task_cols"]),
        "observed_local_tasks": list(parent_model["local_task_cols"]),
        "imputed_parent_only_tasks": [
            task for task in parent_model["parent_task_cols"]
            if task not in set(parent_model["parent_to_local"])
        ],
        "excluded_local_tasks": [col for col in _task_columns(wide_df) if col not in set(parent_model["local_task_cols"])],
        "explained_variance_ratio": float(parent_model["explained_variance_ratio"]),
        "parent_orientation_sign": float(parent_model["sign"]),
        "parent_orientation_corr": parent_model["orientation_corr"],
        "parent_orientation_source": parent_model["orientation_source"],
    }
    return scores, meta


def _impute_column_means(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    total_cells = int(out.shape[0] * out.shape[1]) if out.shape[0] and out.shape[1] else 0
    missing_before = int(out.isna().sum().sum())
    imputed_cols: dict[str, int] = {}

    for col in out.columns:
        series = pd.to_numeric(out[col], errors="coerce")
        missing_col = int(series.isna().sum())
        if missing_col == 0:
            out[col] = series
            continue
        mean_val = series.mean()
        if np.isfinite(mean_val):
            out[col] = series.fillna(mean_val)
            imputed_cols[col] = missing_col
        else:
            out[col] = series.fillna(0.0)
            imputed_cols[col] = missing_col

    meta = {
        "n_rows": int(out.shape[0]),
        "n_cols": int(out.shape[1]),
        "missing_before": missing_before,
        "total_cells": total_cells,
        "imputed_pct": (missing_before / total_cells) if total_cells > 0 else np.nan,
        "imputed_columns": imputed_cols,
    }
    return out, meta


def _orient_scores(scores: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    if reference is None:
        return scores
    aligned = pd.concat([scores, reference], axis=1).dropna()
    if aligned.shape[0] < 3:
        return scores
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    if pd.notna(corr) and corr < 0:
        return -scores
    return scores


def _pca_pc1(zscores: pd.DataFrame, reference: pd.Series | None = None) -> tuple[pd.Series, PCA, dict[str, Any]]:
    imputed, meta = _impute_column_means(zscores)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(imputed.to_numpy(dtype=float)).ravel()
    scores = pd.Series(pc1, index=zscores.index, name="pc1")
    scores = _orient_scores(scores, reference)
    meta["explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
    return scores, pca, meta


def _annotate_task_reference_stats(
    task_reference_stats: pd.DataFrame,
    parent_model: dict[str, Any],
) -> pd.DataFrame:
    """Ajoute les poids/loadings parentaux et les flags d'utilisation au tableau d'audit."""
    out = task_reference_stats.copy()
    mapped_dict = dict(parent_model["mapped_pairs"])
    out["used_for_parent_pca"] = out["task_sample"].isin(parent_model["local_task_cols"]).astype(int)
    out["parent_pca_full_task"] = out["task_sample"].map(mapped_dict)
    out["parent_pc1_weight"] = out["parent_pca_full_task"].map(parent_model["weights"].to_dict())
    out["parent_pc1_loading"] = out["parent_pca_full_task"].map(parent_model["loadings"].to_dict())
    out["parent_pca_explained_variance_ratio"] = float(parent_model["explained_variance_ratio"])
    out["parent_pca_orientation_sign"] = float(parent_model["sign"])
    out["parent_pca_orientation_corr"] = parent_model["orientation_corr"]
    out["parent_pca_orientation_source"] = parent_model["orientation_source"]
    return out


def _export_parent_loadings(
    parent_model: dict[str, Any],
    out_csv: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for task in parent_model["parent_task_cols"]:
        rows.append(
            {
                "task_sample": parent_model["parent_to_local"].get(task, ""),
                "task_full_csv": task,
                "parent_pc1_weight": float(parent_model["weights"][task]),
                "parent_pc1_loading": float(parent_model["loadings"][task]),
                "mean_pop": float(parent_model["means"][task]),
                "sd_pop": float(parent_model["sds"][task]),
                "n_pop": int(parent_model["n_pops"][task]),
                "explained_variance_ratio": float(parent_model["explained_variance_ratio"]),
                "orientation_sign": float(parent_model["sign"]),
                "orientation_corr_with_legacy": parent_model["orientation_corr"],
                "orientation_source": parent_model["orientation_source"],
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8", float_format="%.6f")


def _compute_rank_delta(pop_scores: pd.Series, sample_scores: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"c_factor_pop": pop_scores, "c_factor_sample": sample_scores})
    df["rank_pop"] = df["c_factor_pop"].rank(ascending=False, method="min")
    df["rank_sample"] = df["c_factor_sample"].rank(ascending=False, method="min")
    df["rank_delta"] = df["rank_sample"] - df["rank_pop"]
    return df


def process_one_scope(
    wide_df: pd.DataFrame,
    out_csv: Path,
    full_df: pd.DataFrame,
    *,
    input_path: Path,
    parent_model: dict[str, Any],
    scope: str,
    log_lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "group_id" not in wide_df.columns:
        raise ValueError(f"`group_id` manquant dans {input_path}")

    ref_df, ref_stats = _build_reference_stats(wide_df, full_df, scope=scope)
    z_sample = _sample_zscores(wide_df)

    legacy_ref = pd.to_numeric(wide_df.get("c_score"), errors="coerce") if "c_score" in wide_df.columns else None
    c_pop, impute_pop = _project_parent_pc1(wide_df, parent_model)
    c_sample, _, impute_sample = _pca_pc1(z_sample, reference=legacy_ref)
    c_pop.name = "c_factor_pop"
    c_sample.name = "c_factor_sample"

    corr = pd.concat([c_pop, c_sample], axis=1).corr(method="spearman").iloc[0, 1]
    rank_df = _compute_rank_delta(c_pop, c_sample)

    out_df = wide_df.copy()
    out_df["c_factor_pop"] = c_pop.values
    out_df["c_factor_sample"] = c_sample.values
    out_df["c_score"] = c_pop.values
    out_df["rank_pop"] = rank_df["rank_pop"].values
    out_df["rank_sample"] = rank_df["rank_sample"].values
    out_df["rank_delta"] = rank_df["rank_delta"].values
    out_df.to_csv(out_csv, index=False, encoding="utf-8", float_format="%.6f")

    wide_out = out_csv.with_name(out_csv.stem + "_with_tasks.csv")
    out_df.to_csv(wide_out, index=False, encoding="utf-8", float_format="%.6f")

    log_lines.append(f"[{scope}] input={input_path}")
    log_lines.append(f"[{scope}] output={out_csv}")
    log_lines.append(f"[{scope}] n_groups={len(out_df)}")
    log_lines.append(f"[{scope}] c_factor_pop_vs_sample_spearman={corr:.6f}" if pd.notna(corr) else f"[{scope}] c_factor_pop_vs_sample_spearman=nan")
    log_lines.append(f"[{scope}] imputation_pop={json.dumps(impute_pop, ensure_ascii=False)}")
    log_lines.append(f"[{scope}] imputation_sample={json.dumps(impute_sample, ensure_ascii=False)}")

    return out_df, ref_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute parent-population normalized c-factors.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Racine `results/` du projet Longitudinale.")
    parser.add_argument("--reference-full-csv", type=Path, required=True, help="Chemin vers `full.csv` Riedl 2021.")
    parser.add_argument("--allowed-wide", type=Path, default=None, help="Chemin vers `c_scores_with_tasks_allowed.csv`.")
    parser.add_argument("--all-wide", type=Path, default=None, help="Chemin vers `c_scores_with_tasks_all.csv`.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    tci_dir = results_dir / "TCI"
    tci_dir.mkdir(parents=True, exist_ok=True)

    allowed_wide = args.allowed_wide or (tci_dir / "c_scores_with_tasks_allowed.csv")
    all_wide = args.all_wide or (tci_dir / "c_scores_with_tasks_all.csv")

    if not allowed_wide.exists():
        raise FileNotFoundError(f"Fichier introuvable: {allowed_wide}")
    if not all_wide.exists():
        raise FileNotFoundError(f"Fichier introuvable: {all_wide}")
    if not args.reference_full_csv.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.reference_full_csv}")

    full_df = pd.read_csv(args.reference_full_csv, encoding="utf-8")
    allowed_df = pd.read_csv(allowed_wide, encoding="utf-8")
    all_df = pd.read_csv(all_wide, encoding="utf-8")

    parent_model = _fit_parent_pca_model(all_df, full_df)
    legacy_all = pd.to_numeric(all_df.get("c_score"), errors="coerce") if "c_score" in all_df.columns else None
    parent_model = _orient_parent_model(
        parent_model,
        all_df,
        reference=legacy_all,
        orientation_source="all_wide_legacy_c_score",
    )

    log_lines = [
        "compute_cfactor_population.py",
        f"reference_full_csv={args.reference_full_csv}",
        f"parent_pca_tasks={json.dumps(parent_model['parent_task_cols'], ensure_ascii=False)}",
        f"parent_pca_observed_local_tasks={json.dumps(parent_model['local_task_cols'], ensure_ascii=False)}",
        f"parent_pca_mapped_pairs={json.dumps(parent_model['mapped_pairs'], ensure_ascii=False)}",
        f"parent_pca_explained_variance_ratio={parent_model['explained_variance_ratio']:.6f}",
        f"parent_pca_orientation_sign={parent_model['sign']:.1f}",
        f"parent_pca_orientation_corr={parent_model['orientation_corr']}",
        f"parent_pca_orientation_source={parent_model['orientation_source']}",
        f"parent_pca_imputed_pct={parent_model['parent_imputed_pct']}",
    ]

    process_one_scope(
        allowed_df,
        tci_dir / "c_scores_allowed_pop.csv",
        full_df,
        input_path=allowed_wide,
        parent_model=parent_model,
        scope="allowed",
        log_lines=log_lines,
    )
    process_one_scope(
        all_df,
        tci_dir / "c_scores_all_pop.csv",
        full_df,
        input_path=all_wide,
        parent_model=parent_model,
        scope="all",
        log_lines=log_lines,
    )

    # Export combiné des stats de référence pour audit.
    task_reference_stats_allowed, _ = _build_reference_stats(allowed_df, full_df, scope="allowed")
    task_reference_stats_all, _ = _build_reference_stats(all_df, full_df, scope="all")
    task_reference_stats = pd.concat([task_reference_stats_allowed, task_reference_stats_all], ignore_index=True)
    task_reference_stats = _annotate_task_reference_stats(task_reference_stats, parent_model)
    task_reference_stats.to_csv(tci_dir / "task_reference_stats.csv", index=False, encoding="utf-8", float_format="%.6f")
    _export_parent_loadings(parent_model, tci_dir / "c_factor_parent_loadings.csv")

    (tci_dir / "cfactor_population_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"[OK] c_scores_allowed_pop.csv -> {tci_dir / 'c_scores_allowed_pop.csv'}")
    print(f"[OK] c_scores_all_pop.csv     -> {tci_dir / 'c_scores_all_pop.csv'}")
    print(f"[OK] task_reference_stats.csv -> {tci_dir / 'task_reference_stats.csv'}")
    print(f"[OK] c_factor_parent_loadings.csv -> {tci_dir / 'c_factor_parent_loadings.csv'}")


if __name__ == "__main__":
    main()
