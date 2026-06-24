#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_modalite.py — Analyse questionnaire stratifiée par Scénario × Modalité.

Correspond à analyse_questionnaire_scenario_modalite.R :
- Lecture du fichier survey (xlsx) avec métadonnées (Session, Modalité, Scénario, Groupe)
- Calcul de scores par dimension / participant / scénario / modalité
- Descriptifs + plots Scénario × Modalité
- Modèles statistiques (mixed-effects si possible, sinon OLS)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import (
    ALL_LEVELS,
    CODE_MAP,
    DIMENSION_LABELS,
    INVERT_SHORT,
)
from .io_read import read_survey, get_meta_cols


def _extract_dimension_bracket(colname: str) -> str | None:
    """Extrait dimension depuis format G1Q00001[COR01]."""
    m = re.search(r"(G1Q00001|G2Q00001)\[([A-Z]{3})\d{2}\]", colname)
    return m.group(2) if m else None


def _extract_item_short_bracket(colname: str) -> str | None:
    """Extrait code court depuis format G1Q00001[COR03]."""
    m = re.search(r"(G1Q00001|G2Q00001)\[([A-Z]{3}\d{2})\]", colname)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def _get_item_cols_bracket(df: pd.DataFrame) -> list[str]:
    """Colonnes items au format bracket G1Q00001[...] / G2Q00001[...]."""
    return [c for c in df.columns if re.match(r"^G[12]Q00001\[", c)]


def _build_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame long depuis le fichier brut avec métadonnées.
    """
    meta_map = get_meta_cols(df_raw)

    # Métadonnées
    meta = pd.DataFrame()
    meta["row_id"] = range(len(df_raw))

    # Participant
    id_col = [c for c in df_raw.columns if "ID de la réponse" in str(c)]
    meta["Participant"] = df_raw[id_col[0]].astype(str) if id_col else df_raw.index.astype(str)

    for std_name, real_col in meta_map.items():
        if real_col is not None:
            meta[std_name] = df_raw[real_col].astype(str).str.strip()
        else:
            meta[std_name] = pd.NA

    # Normalisation
    if "Scenario" in meta.columns:
        meta["Scenario"] = meta["Scenario"].str.upper().str.replace(r"\s+", "", regex=True)
        meta["Scenario"] = meta["Scenario"].apply(
            lambda x: "S1" if "S1" in str(x) else ("S2" if "S2" in str(x) else x)
        )
    if "Modalite" in meta.columns:
        meta["Modalite"] = meta["Modalite"].str.upper().str.replace(r"\s+", "", regex=True)
        meta["Modalite"] = meta["Modalite"].apply(
            lambda x: "VR" if "VR" in str(x) else ("PC" if "PC" in str(x) else x)
        )
    if "Groupe" in meta.columns:
        meta["Groupe"] = meta["Groupe"].str.lower().str.strip()

    # Rôle
    role_col = [c for c in df_raw.columns if c.startswith("G3Q00008")]
    if role_col:
        meta["Role"] = df_raw[role_col[0]].astype(str)
    else:
        meta["Role"] = pd.NA

    # Items
    item_cols = _get_item_cols_bracket(df_raw)
    if not item_cols:
        raise ValueError("Aucune colonne item G1Q00001[...] / G2Q00001[...] trouvée.")

    items = df_raw[item_cols].copy()
    items["row_id"] = range(len(items))

    long = items.melt(id_vars=["row_id"], var_name="item_col", value_name="reponse_txt")
    long = long.merge(meta, on="row_id", how="left")

    long["dimension"] = long["item_col"].apply(_extract_dimension_bracket)
    long["item_short"] = long["item_col"].apply(_extract_item_short_bracket)

    # Recodage
    long["reponse_num"] = long["reponse_txt"].apply(
        lambda v: (
            float(v) if pd.notna(v) and re.fullmatch(r"[1-9]", str(v).strip())
            else (float(CODE_MAP[v]) if pd.notna(v) and v in CODE_MAP else None)
        )
    )

    # Inversion
    mask_inv = long["item_short"].isin(INVERT_SHORT) & long["reponse_num"].notna()
    long.loc[mask_inv, "reponse_num"] = 10 - long.loc[mask_inv, "reponse_num"]

    long = long[long["dimension"].notna() & long["reponse_num"].notna()].copy()
    long.drop(columns=["row_id"], inplace=True)

    return long


def _compute_scores(long: pd.DataFrame) -> pd.DataFrame:
    """Score moyen par participant × scénario × modalité × dimension."""
    grp = ["Participant", "Groupe", "Session", "Scenario", "Modalite", "Role", "dimension"]
    scores = (
        long.groupby([c for c in grp if c in long.columns])
        .agg(n_items=("item_col", "nunique"), score=("reponse_num", "mean"))
        .reset_index()
    )
    scores["dimension_label"] = scores["dimension"].map(DIMENSION_LABELS).fillna(scores["dimension"])
    return scores


def _compute_descriptives(scores: pd.DataFrame) -> pd.DataFrame:
    """Descriptives par dimension × scénario × modalité."""
    sub = scores.dropna(subset=["Scenario", "Modalite"])
    desc = (
        sub.groupby(["dimension", "dimension_label", "Scenario", "Modalite"])
        .agg(n=("score", "count"), mean=("score", "mean"), sd=("score", "std"))
        .reset_index()
    )
    desc["se"] = desc["sd"] / np.sqrt(desc["n"])
    desc["ic95"] = desc.apply(
        lambda r: sp_stats.t.ppf(0.975, r["n"] - 1) * r["se"] if r["n"] > 1 else 0,
        axis=1,
    )
    desc["mean"] = desc["mean"].round(3)
    desc["sd"] = desc["sd"].round(3)
    return desc


def _fit_models(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Teste l'effet Scénario × Modalité par dimension.
    Tente un modèle mixte (statsmodels MixedLM), sinon OLS.
    """
    dims = sorted(scores["dimension"].dropna().unique())
    rows = []

    for dim in dims:
        dat = scores[
            (scores["dimension"] == dim)
            & scores["score"].notna()
            & scores["Scenario"].notna()
            & scores["Modalite"].notna()
        ].copy()

        if len(dat) < 8 or dat["Scenario"].nunique() < 2 or dat["Modalite"].nunique() < 2:
            for term in ["Scenario", "Modalite", "Scenario:Modalite"]:
                rows.append({
                    "dimension": dim,
                    "dimension_label": DIMENSION_LABELS.get(dim, dim),
                    "model": "insuffisant",
                    "term": term,
                    "p.value": None,
                })
            continue

        # Encode facteurs
        dat["Scenario_c"] = (dat["Scenario"] == "S2").astype(float)
        dat["Modalite_c"] = (dat["Modalite"] == "VR").astype(float)
        dat["Interaction"] = dat["Scenario_c"] * dat["Modalite_c"]

        model_type = "lm"
        pvals = {}

        # Tentative modèle mixte
        try:
            import statsmodels.formula.api as smf

            if "Groupe" in dat.columns and dat["Groupe"].nunique() >= 2:
                md = smf.mixedlm(
                    "score ~ Scenario_c * Modalite_c",
                    data=dat,
                    groups=dat["Groupe"],
                )
                mdf = md.fit(reml=False)
                model_type = "lmer"
                for param, term in [
                    ("Scenario_c", "Scenario"),
                    ("Modalite_c", "Modalite"),
                    ("Scenario_c:Modalite_c", "Scenario:Modalite"),
                ]:
                    if param in mdf.pvalues.index:
                        pvals[term] = float(mdf.pvalues[param])
                    else:
                        pvals[term] = None
        except Exception:
            pass

        # Fallback OLS
        if not pvals:
            try:
                import statsmodels.api as sm

                X = dat[["Scenario_c", "Modalite_c", "Interaction"]]
                X = sm.add_constant(X)
                ols = sm.OLS(dat["score"], X).fit()
                model_type = "lm"
                for col, term in [
                    ("Scenario_c", "Scenario"),
                    ("Modalite_c", "Modalite"),
                    ("Interaction", "Scenario:Modalite"),
                ]:
                    if col in ols.pvalues.index:
                        pvals[term] = float(ols.pvalues[col])
            except Exception:
                model_type = "error"

        for term in ["Scenario", "Modalite", "Scenario:Modalite"]:
            rows.append({
                "dimension": dim,
                "dimension_label": DIMENSION_LABELS.get(dim, dim),
                "model": model_type,
                "term": term,
                "p.value": pvals.get(term),
            })

    return pd.DataFrame(rows)


def _plot_scenario_modalite(desc: pd.DataFrame, outdir: Path):
    """Barplot Scénario × Modalité par dimension."""
    fig, axes_grid = plt.subplots(3, 2, figsize=(10, 12))
    axes_flat = axes_grid.flatten()
    dims = sorted(desc["dimension_label"].unique())

    for idx, dim_label in enumerate(dims):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        df_dim = desc[desc["dimension_label"] == dim_label]

        scenarios = sorted(df_dim["Scenario"].unique())
        modalities = sorted(df_dim["Modalite"].unique())
        x = np.arange(len(scenarios))
        width = 0.35

        for i, mod in enumerate(modalities):
            sub = df_dim[df_dim["Modalite"] == mod].set_index("Scenario")
            vals = [sub.loc[s, "mean"] if s in sub.index else 0 for s in scenarios]
            errs = [sub.loc[s, "ic95"] if s in sub.index else 0 for s in scenarios]
            ax.bar(x + i * width, vals, width, yerr=errs, label=mod, capsize=4)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(scenarios)
        ax.set_title(dim_label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Score moyen (1-9)")
        ax.legend()

    for idx in range(len(dims), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Questionnaire — Scénario × Modalité", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "plot_means_ic95_scenario_modalite.pdf")
    plt.close(fig)


def run_scenario_modalite(data_file: str | Path, outdir: str | Path):
    """
    Pipeline complet d'analyse scénario × modalité.
    Produit CSV + PDF dans outdir.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = read_survey(data_file)
    long = _build_long(df_raw)
    scores = _compute_scores(long)
    desc = _compute_descriptives(scores)
    models = _fit_models(scores)

    # Exports CSV
    scores.to_csv(outdir / "scores_dimension_par_participant.csv", index=False, encoding="utf-8")
    desc.to_csv(outdir / "descriptifs_scenario_modalite.csv", index=False, encoding="utf-8")
    models.to_csv(outdir / "modeles_scenario_modalite_par_dimension.csv", index=False, encoding="utf-8")

    # Pivot moyennes
    pivot = desc[["dimension_label", "Scenario", "Modalite", "mean"]].pivot_table(
        index="dimension_label", columns=["Scenario", "Modalite"], values="mean"
    )
    pivot.to_csv(outdir / "table_moyennes_pivot.csv", encoding="utf-8")

    # Plot
    _plot_scenario_modalite(desc, outdir)

    print(f"[OK] Analyse scénario × modalité terminée. Sorties dans : {outdir}")
