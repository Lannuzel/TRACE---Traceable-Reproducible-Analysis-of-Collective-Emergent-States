r"""
c_factor.py (v2.0 – refactored)
================================

Calcul du c-factor (PCA sur tâches TCI), corrélations RME, visualisations.

Le script peut fonctionner sur :
- les groupes `allowed` (whitelist historique) ;
- ou `all`, c'est-à-dire tous les groupes détectés dans les fichiers TaskScore.

Exemple :
```powershell
python TCI.py .\data\data_TCI\**\TaskScore_*.csv \
              --missing mean \
              --out c_scores.csv \
              --out-wide c_scores_with_tasks.csv \
              --rme-task-corr rme_task_correlations.csv \
              --heatmap --scatter-tasks
```
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ajoute le dossier parent (scripts/) au path pour importer common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
from typing import Iterable, List
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as _scipy_pearsonr

# Groupes autorisés pour l'analyse TCI historique
ALLOWED_GROUPS = {
    "bim073", "bim081", "bim057",
    "bim066", "bim067", "bim068",
    "bim006", "bim010", "bim007",
    "bim025", "bim015", "bim060",
}


def apply_group_scope(matrix: pd.DataFrame, group_scope: str) -> pd.DataFrame:
    """Applique le périmètre d'analyse demandé aux groupes TCI."""
    scope = str(group_scope).strip().lower()
    if scope == "all":
        return matrix.copy()
    if scope == "allowed":
        return matrix.loc[matrix.index.isin(ALLOWED_GROUPS)].copy()
    raise ValueError(f"Unknown group scope {group_scope!r}")


###############################################################################
#                       Utility: expand wildcards & folders                   #
###############################################################################

def expand_paths(path_patterns: List[str]) -> List[Path]:
    """Return a list of concrete CSV Paths from CLI patterns or folders (Windows-proof)."""
    expanded: List[Path] = []

    for pat in path_patterns:
        pat_norm = pat.strip().strip('"').strip("'")
        p = Path(pat_norm)

        if p.is_dir():
            expanded.extend(p.rglob("TaskScore_*.csv"))
            continue

        if p.exists() and p.is_file():
            expanded.append(p)
            continue

        if "**" in pat_norm:
            before, after = pat_norm.split("**", 1)
            root = Path(before.replace("\\", "/")).resolve()
            filename_pat = Path(after.replace("\\", "/")).name

            if root.exists() and root.is_dir():
                expanded.extend(root.rglob(filename_pat))
            continue

        if any(ch in pat_norm for ch in "*?[]"):
            pat_posix = pat_norm.replace("\\", "/")
            expanded.extend(Path().glob(pat_posix))
            continue

        expanded.append(p)

    unique = list(dict.fromkeys(expanded))

    if not unique:
        raise RuntimeError(
            "Aucun fichier CSV trouvé. Vérifie :\n"
            "- que le dossier racine existe\n"
            "- que les fichiers s'appellent bien TaskScore_*.csv\n"
            "- que tu pointes sur le bon disque/chemin"
        )

    return unique

###############################################################################
#                           Lecture et pré-traitements                        #
###############################################################################

def read_tci_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1)
    expected = {"StudyPrefix", "SessionId", "TaskName", "SoloTask", "TotalScore"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")

    df["group_id"] = _bim_from_path(path).lower()
    return df

_BIM_RE = re.compile(r"^bim\d+$", re.IGNORECASE)

def _bim_from_path(path: Path) -> str:
    for parent in path.parents:
        if _BIM_RE.match(parent.name):
            return parent.name
    raise ValueError(f"Impossible de trouver un dossier bimXXX dans le chemin: {path}")


###############################################################################
#                       Matrice groupes × tâches (team)                       #
###############################################################################

def build_group_task_matrix(frames: List[pd.DataFrame]) -> pd.DataFrame:
    concat = pd.concat(frames, ignore_index=True)
    team_tasks = concat[concat["SoloTask"] == False].copy()
    matrix = team_tasks.pivot_table(
        index="group_id",
        columns="TaskName",
        values="TotalScore",
        aggfunc="mean",
    )
    return matrix.dropna(axis="columns", how="all").dropna(axis="rows", how="all")

###############################################################################
#                        Gestion des valeurs manquantes                       #
###############################################################################

def handle_missing(matrix: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    if strategy == "drop":
        return matrix.dropna(axis="rows")
    elif strategy in {"mean", "median"}:
        impute_func = np.nanmean if strategy == "mean" else np.nanmedian
        filled = matrix.copy()
        for col in filled.columns:
            filled[col] = filled[col].fillna(impute_func(filled[col].values))
        return filled
    else:
        raise ValueError(f"Unknown missing strategy {strategy}")

###############################################################################
#                           Calcul du C-factor (PCA)                          #
###############################################################################

def compute_c_factor(matrix: pd.DataFrame, scale_tasks: bool = True):
    """Compute c on the **imputed raw task means** (matrix), after optional scaling."""
    X = matrix.copy()
    scaler = None
    if scale_tasks:
        scaler = StandardScaler().fit(X)
        X = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    pca = PCA(n_components=1).fit(X)
    scores = pd.Series(pca.transform(X).flatten(), index=X.index, name="c_score")
    return scores, pca, scaler

###############################################################################
#               Extraction des statistiques RME au niveau groupe             #
###############################################################################

def extract_group_rme_stats(frames: List[pd.DataFrame], motifs: Iterable[str]) -> pd.DataFrame:
    patterns = [m.lower() for m in motifs]
    rows = []
    for df in frames:
        mask = df["TaskName"].str.lower().apply(lambda s: any(p in s for p in patterns))
        rows.append(df[mask])
    if not any(len(r) for r in rows):
        raise RuntimeError("Aucune ligne RME détectée dans les CSV fournis.")
    rme_concat = pd.concat(rows, ignore_index=True)
    return (
        rme_concat.groupby("group_id")["TotalScore"]
        .agg(rme_mean="mean", rme_max="max", rme_min="min")
    )

###############################################################################
#                     Profil visuel détaillé par groupe                       #
###############################################################################

def plot_group_profiles(task_matrix: pd.DataFrame, c_scores: pd.Series, rme_mean: pd.Series,
                        path: str = "group_profiles.png") -> None:
    groups = task_matrix.index
    tasks = list(task_matrix.columns)
    n_groups = len(groups)
    fig_height = max(3, n_groups * 2.5)
    fig_width = max(8, len(tasks) * 0.7)
    fig, axes = plt.subplots(n_groups, 1, figsize=(fig_width, fig_height), sharex=True)
    if n_groups == 1:
        axes = [axes]

    x = np.arange(len(tasks))
    for ax, group in zip(axes, groups):
        scores = task_matrix.loc[group].values
        ax.bar(x, scores, color="skyblue", label="Task perf.")
        ax.axhline(c_scores[group], color="red", linestyle="--", label="C-factor")
        ax.axhline(rme_mean[group], color="green", linestyle=":", label="RME mean")
        ax.set_title(group)
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=min(0, scores.min() - 1))
        ax.legend(loc="upper right")

    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

###############################################################################
#                     Corrélations brutes TÂCHES ↔ RME (table/plots)          #
###############################################################################

def _pearson_r_p(x_arr, y_arr):
    """Pearson r and p-value using scipy, with NaN handling."""
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean, y_clean = x_arr[mask], y_arr[mask]
    if len(x_clean) < 3:
        return float("nan"), float("nan")
    r, p = _scipy_pearsonr(x_clean, y_clean)
    return float(r), float(p)


def compute_task_rme_correlations(task_matrix: pd.DataFrame, rme_stats: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with Pearson r and p between each raw task mean and RME stats."""
    common_idx = task_matrix.index.intersection(rme_stats.index)
    if len(common_idx) < 2:
        raise RuntimeError("Pas assez de groupes communs pour corrélations tâches–RME.")

    rows = []
    for task in task_matrix.columns:
        x = task_matrix.loc[common_idx, task].astype(float).values
        rec = {"task": task, "n": int(len(common_idx))}
        for col in ["rme_mean", "rme_max", "rme_min"]:
            y = rme_stats.loc[common_idx, col].astype(float).values
            r, p = _pearson_r_p(x, y)
            rec[f"r_{col}"] = r
            rec[f"p_{col}"] = p
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("task").sort_values("r_rme_mean", ascending=False)
    return df


def save_corr_heatmap(corr_df: pd.DataFrame, path: str = "rme_task_corr_heatmap.png") -> None:
    """Save a heatmap (r-values only) for tasks × {RME_mean, RME_max, RME_min}."""
    r_only = corr_df[["r_rme_mean", "r_rme_max", "r_rme_min"]]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(r_only))))
    im = ax.imshow(r_only.values, aspect="auto")
    ax.set_yticks(range(len(r_only)))
    ax.set_yticklabels(r_only.index)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["RME_mean", "RME_max", "RME_min"])
    ax.set_title("Corrélations (r) tâches brutes ↔ RME")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_task_scatterplots(task_matrix: pd.DataFrame, rme_stats: pd.DataFrame,
                           out_dir: Path, rme_col: str = "rme_mean") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    common_idx = task_matrix.index.intersection(rme_stats.index)
    for task in task_matrix.columns:
        fig, ax = plt.subplots()
        ax.scatter(rme_stats.loc[common_idx, rme_col], task_matrix.loc[common_idx, task])
        ax.set_xlabel(f"{rme_col}")
        ax.set_ylabel(task)
        ax.set_title(f"{task} vs {rme_col}")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{task}_{rme_col}.png", dpi=300)
        plt.close(fig)

###############################################################################
#                                Interface CLI                               #
###############################################################################

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute c-factor, RME stats and visualisations.")
    p.add_argument("paths", nargs="+", help="CSV paths, wildcards or folders")
    p.add_argument("--out", default="c_scores.csv", type=Path, help="Output CSV path (c + RME)")
    p.add_argument("--out-wide", dest="out_wide", default=None, type=Path,
                   help="Output wide CSV (c + RME + raw task means). If omitted, auto name.")
    p.add_argument("--rme-task-corr", dest="rme_task_corr", default=None, type=Path,
                   help="CSV path to save task-RME correlation table. If omitted, auto name.")
    p.add_argument("--missing", choices=["drop", "mean", "median"], default="mean",
                   help="Missing-value strategy (default: mean)")
    p.add_argument("--rme-motifs", dest="rme_motifs",
                   default="RME,RMET,Eyes", help="Comma-separated substrings for RME tasks")
    p.add_argument("--scatter", action="store_true",
                   help="Save scatterplot of c-score vs RME_mean")
    p.add_argument("--heatmap", action="store_true",
                   help="Save heatmap of task-RME correlations")
    p.add_argument("--scatter-tasks", action="store_true",
                   help="Save per-task scatterplots (raw task vs RME_mean)")
    p.add_argument("--profile", action="store_true",
                   help="Save per-group performance profile")
    p.add_argument(
        "--group-scope",
        choices=["allowed", "all"],
        default="allowed",
        help=(
            "Périmètre des groupes à analyser : "
            "`allowed` = whitelist historique, `all` = tous les groupes détectés."
        ),
    )
    return p.parse_args(argv)

###############################################################################
#                                   Main                                     #
###############################################################################

def main(argv=None):
    args = parse_args(argv)

    # ---------- Expand paths ----------
    csv_paths = expand_paths(args.paths)
    print(f"Found {len(csv_paths)} TaskScore CSV files")

    # ---------- Read data ----------
    frames = [read_tci_csv(p) for p in csv_paths]

    # ---------- C-factor ----------
    task_matrix = handle_missing(build_group_task_matrix(frames), args.missing)

    task_matrix = apply_group_scope(task_matrix, args.group_scope)

    print(f"Group scope: {args.group_scope}")
    print("Groups after filter:", sorted(task_matrix.index.tolist()))
    print("N groups after filter:", len(task_matrix))
    print("Task matrix shape:", task_matrix.shape)
    print("Remaining NaN by column:\n", task_matrix.isna().sum())

    if task_matrix.shape[0] < 2 or task_matrix.isna().any().any():
        raise RuntimeError("Données insuffisantes ou valeurs manquantes pour PCA.")
    c_scores, pca, _ = compute_c_factor(task_matrix)

    # ---------- RME stats ----------
    motifs = [m.strip() for m in args.rme_motifs.split(",") if m.strip()]
    rme_stats = extract_group_rme_stats(frames, motifs)
    rme_stats = apply_group_scope(rme_stats, args.group_scope)

    merged = pd.concat([c_scores, rme_stats], axis=1, join="inner").dropna()
    merged.to_csv(args.out, index_label="group_id", float_format="%.5f")

    print(f"C-factor computed for {len(merged)} groups -> {args.out}")
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")

    # ---------- Correlations c <-> RME ----------
    if merged.shape[0] >= 2:
        for col in ["rme_mean", "rme_max", "rme_min"]:
            r, p = _pearson_r_p(
                merged["c_score"].values.astype(float),
                merged[col].values.astype(float),
            )
            print(f"Pearson r(c, {col.upper()}) = {r:.2f}  (p = {p:.3g}, n = {merged.shape[0]})")

    # ---------- Wide export with raw task means ----------
    wide_out = args.out_wide
    if wide_out is None:
        wide_out = args.out.parent / f"{args.out.stem}_with_tasks{args.out.suffix}"
    wide_df = merged.join(task_matrix.loc[merged.index])
    wide_df.to_csv(wide_out, index_label="group_id", float_format="%.5f")
    print(f"Wide table (raw task means + c + RME) saved -> {wide_out}")

    # ---------- Corrélations brutes tâches <-> RME ----------
    corr_df = compute_task_rme_correlations(task_matrix.loc[merged.index], rme_stats.loc[merged.index])
    corr_out = args.rme_task_corr or (args.out.parent / f"{args.out.stem}_rme_task_correlations.csv")
    corr_df.to_csv(corr_out, index_label="task", float_format="%.5f")
    print(f"Task-RME correlation table saved -> {corr_out}")

    top5 = corr_df.sort_values("r_rme_mean", ascending=False).head(5)
    print("Top 5 tâches corrélées (r) avec RME_mean:\n" + top5["r_rme_mean"].to_string())

    # ---------- Visualisations ----------
    if args.scatter and merged.shape[0] >= 2:
        _save_scatter(merged)
        print("Scatter saved to rme_vs_c.png")

    if args.heatmap:
        save_corr_heatmap(corr_df, Path("rme_task_corr_heatmap"))
        print("Heatmap saved to rme_task_corr_heatmap.png")

    if args.scatter_tasks:
        save_task_scatterplots(task_matrix.loc[merged.index], rme_stats.loc[merged.index], Path("scatter_rme_tasks"))
        print("Per-task scatters saved to ./scatter_rme_tasks/")

    if args.profile and merged.shape[0] >= 1:
        plot_group_profiles(task_matrix.loc[merged.index], c_scores.loc[merged.index],
                            rme_stats.loc[merged.index, "rme_mean"], Path("group_profiles"))
        print("Group profiles saved to group_profiles.png")


###############################################################################
#                           Auxiliary statistics plots                        #
###############################################################################

def _save_scatter(df: pd.DataFrame, path: str = "rme_vs_c.png") -> None:
    fig, ax = plt.subplots()
    ax.scatter(df["rme_mean"], df["c_score"])
    ax.set_xlabel("RME moyen de groupe")
    ax.set_ylabel("C-factor")
    ax.set_title("C-factor vs RME moyen")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
