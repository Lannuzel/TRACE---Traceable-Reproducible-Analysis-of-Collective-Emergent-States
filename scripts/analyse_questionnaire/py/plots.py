#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py — Exports PDF : distributions items par dimension + moyennes±IC95 par rôle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .config import DIMENSION_LABELS
from .role_tests import _pval_roles


def _make_df_dim(df_long: pd.DataFrame, dim: str) -> pd.DataFrame | None:
    """
    Filtre les données d'une dimension pour les plots.
    Utilise reponse_num déjà calculé (avec inversions) depuis df_long
    plutôt que de recalculer depuis le texte (ce qui perdrait les inversions).
    """
    sub = df_long[
        (df_long["dimension"] == dim)
        & df_long["reponse_num"].notna()
    ].copy()
    return sub if not sub.empty else None


def export_item_distributions(df_long: pd.DataFrame, outdir: str | Path):
    """Crée un PDF avec les distributions des items facetées par question, par dimension."""
    outdir = Path(outdir)
    dims = sorted(df_long["dimension"].dropna().unique())

    with PdfPages(outdir / "analyse_perf.pdf") as pdf:
        for dim in dims:
            df_dim = _make_df_dim(df_long, dim)
            if df_dim is None:
                continue

            label = DIMENSION_LABELS.get(dim, dim)
            questions = sorted(df_dim["question_wrapped"].dropna().unique())
            n_q = len(questions)
            if n_q == 0:
                continue

            ncols = 2
            nrows = max(1, (n_q + ncols - 1) // ncols)
            fig, axes_raw = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows),
                                         squeeze=False)
            fig.suptitle(
                f"{label}\n(items inversés inclus — axe : 1=Fortement en désaccord, 9=Fortement en accord)",
                fontsize=11, fontweight="bold"
            )
            axes = axes_raw.flatten()

            for idx, q in enumerate(questions):
                ax = axes[idx]
                q_data = df_dim[df_dim["question_wrapped"] == q]["reponse_num"]
                ax.hist(q_data.dropna(), bins=np.arange(0.5, 10.5, 1), edgecolor="black", linewidth=0.5)
                mean_val = q_data.mean()
                ax.axvline(mean_val, color="red", linestyle="--", linewidth=0.8)
                ax.set_xticks(range(1, 10))
                # On utilise des chiffres (1-9) pour éviter le débordement des étiquettes
                # longues en texte Likert sur l'axe x ; l'échelle est rappelée en titre global.
                ax.set_xticklabels(range(1, 10), fontsize=8)
                ax.set_title(q, fontsize=8)
                ax.set_ylabel("")

            # Masquer les axes vides
            for idx in range(n_q, len(axes)):
                axes[idx].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


def export_role_means(df_long: pd.DataFrame, outdir: str | Path):
    """Crée un PDF avec les barres de moyennes ± IC95 par rôle, par dimension."""
    outdir = Path(outdir)
    # Filtre les Role manquants ou résidus de conversion str (nan/<NA>)
    sub = df_long[
        df_long["reponse_num"].notna()
        & df_long["Role"].notna()
        & ~df_long["Role"].isin(["nan", "<NA>", "None", ""])
    ].copy()

    stats_role = (
        sub.groupby(["dimension", "Role"])["reponse_num"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats_role["se"] = stats_role["std"] / np.sqrt(stats_role["count"])
    stats_role["ic95"] = stats_role.apply(
        lambda row: sp_stats.t.ppf(0.975, row["count"] - 1) * row["se"]
        if row["count"] > 1 else 0,
        axis=1,
    )

    # P-values
    dims = sorted(sub["dimension"].dropna().unique())
    pvals = {}
    for dim in dims:
        pvals[dim] = _pval_roles(df_long[df_long["dimension"] == dim])

    with PdfPages(outdir / "plots_dimensions_roles_perf.pdf") as pdf:
        for dim in dims:
            df_dim = stats_role[stats_role["dimension"] == dim]
            if df_dim.empty:
                continue

            label = DIMENSION_LABELS.get(dim, dim)
            p = pvals.get(dim)
            p_str = f"{p:.4f}" if p is not None else "NA"

            fig, ax = plt.subplots(figsize=(8, 5))
            roles = df_dim["Role"].values
            means = df_dim["mean"].values
            ic95s = df_dim["ic95"].values

            bars = ax.bar(roles, means, width=0.6, edgecolor="black", linewidth=0.5)
            ax.errorbar(roles, means, yerr=ic95s, fmt="none", ecolor="black",
                        capsize=4, linewidth=0.5)
            ax.axhline(means.mean(), linestyle="--", color="gray", linewidth=0.8)
            ax.set_title(f"{label}\np-value (rôles) = {p_str}", fontsize=12)
            ax.set_ylabel("Score moyen (1-9)")
            ax.set_xlabel("")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
