"""
Helpers de rendu pour la section INV / PCA du rapport.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer, Table, TableStyle

from pathlib import Path as _Path

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))
_scripts_dir = _Path(__file__).resolve().parents[3]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from config import (
    COHESION_SCORE_COL,
    EXCLUDE_PREFIXES,
    EXCLUDE_SUFFIXES,
    ID_COLS,
    is_excluded_inv_feature,
)
from config.inv_features_config import (
    infer_family_from_name,
    INV_FEATURES,
    PRUNING_PROTECTED_PAIRS,
    REDUNDANCY_CORR_THRESHOLD,
    REGRESSION_FORCE_INCLUDE,
)
from py.pca_regression import render_pca_regression_section

LEGACY_AUDIO_EXCLUDE_PREFIXES = ("int_",)


def _render_inv_modality_pca_subsection(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    inv_dir: Path,
    md_table_fn: Callable[..., str],
    pdf_table_fn: Callable[..., Any],
    apply_pruning: bool = False,
    inv_pruned_features: list[str] | None = None,
    max_rows_md: int = 50,
    max_rows_pdf: int = 50,
):
    """
    Sous-section : PCA par famille INV (audio / face / gaze).
    Détecte la redondance interne à chaque famille de métriques.
    """
    var_family_png = inv_dir / "pca_variance_by_inv_modality.png"
    if not var_family_png.exists():
        return

    subtitle = "Analyse PCA par famille INV (redondance interne)"
    lines.append(f"\n#### {subtitle}\n")
    pdf_elems.append(Spacer(1, 0.1 * inch))
    pdf_elems.append(Paragraph(subtitle, styles["Heading4"]))
    pdf_elems.append(Spacer(1, 0.05 * inch))

    note = (
        "Cette analyse réalise une PCA séparée pour chaque famille de features INV "
        "(audio, face, gaze). Un PC1 élevé indique que plusieurs métriques d'une "
        "même famille mesurent essentiellement la même chose (redondance interne). "
        "Le seuil de redondance est |loading| > 0.60 sur la même composante."
    )
    pdf_elems.append(Paragraph(note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    lines.append(f"![]({var_family_png.name})\n_Variance expliquée par PC1/PC2 selon la famille INV_\n\n")
    pdf_elems.append(Image(str(var_family_png), width=5.5 * inch, height=3.0 * inch))
    pdf_elems.append(Paragraph(
        "<i>PC1 élevé = forte redondance interne dans la famille de métriques</i>",
        styles["Normal"],
    ))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    red_csv = inv_dir / "inv_modality_redundancy.csv"
    if red_csv.exists():
        try:
            red_df = pd.read_csv(red_csv)
            if not red_df.empty:
                is_pairwise_schema = {"feature_a", "feature_b", "corr_pearson"}.issubset(set(red_df.columns))

                if apply_pruning and inv_pruned_features is not None and is_pairwise_schema:
                    before_n = len(red_df)
                    red_df = red_df[
                        red_df["feature_a"].isin(inv_pruned_features)
                        & red_df["feature_b"].isin(inv_pruned_features)
                    ]
                    n_filtered = before_n - len(red_df)
                    if n_filtered > 0:
                        print(f"  [PRUNING] Redondances: {n_filtered} paires filtrées (features prunées)")

                if red_df.empty:
                    lines.append("_Aucune redondance résiduelle après pruning des features._\n")
                    pdf_elems.append(Paragraph(
                        "<i>Aucune redondance résiduelle après pruning des features.</i>",
                        styles["Normal"],
                    ))
                else:
                    summary_rows = []
                    all_modalities = sorted(red_df["modality"].unique().tolist()) if "modality" in red_df.columns else []

                    expected_modalities = {"audio", "face", "gaze"}
                    missing_mods = expected_modalities - set(all_modalities)
                    if missing_mods:
                        diag_note = f"_Note : Modalites {', '.join(sorted(missing_mods))} non présentes dans les redondances détectées._"
                        lines.append(diag_note + "\n")
                        pdf_elems.append(Paragraph(diag_note, styles["Normal"]))

                    for mod, grp in red_df.groupby("modality"):
                        if is_pairwise_schema:
                            feat_set = set(grp["feature_a"].dropna().tolist()) | set(grp["feature_b"].dropna().tolist())
                            summary_rows.append({
                                "famille": mod,
                                "paires redondantes": int(len(grp)),
                                "features impliquées": int(len(feat_set)),
                            })
                        else:
                            summary_rows.append({
                                "famille": mod,
                                "composantes concernées": grp["component"].nunique(),
                                "features redondantes": grp["feature"].nunique(),
                            })

                    summ_df = pd.DataFrame(summary_rows)
                    lines.append("**Redondances détectées par famille :**\n")
                    lines.append(md_table_fn(summ_df))
                    pdf_elems.append(Paragraph("Redondances détectées par famille INV :", styles["Heading4"]))
                    pdf_elems.append(Spacer(1, 0.05 * inch))
                    pdf_elems.append(pdf_table_fn(summ_df))
                    pdf_elems.append(Spacer(1, 0.08 * inch))

                    if is_pairwise_schema:
                        lines.append(f"**Détail des paires redondantes (|r| > {REDUNDANCY_CORR_THRESHOLD:.2f}) :**\n")
                        detail_title = f"Détail des paires redondantes (|r| > {REDUNDANCY_CORR_THRESHOLD:.2f}) :"
                    else:
                        lines.append("**Détail des features redondantes (|loading| > 0.60) :**\n")
                        detail_title = "Détail des features redondantes (|loading| > 0.60) :"

                    lines.append(md_table_fn(red_df, max_rows=max_rows_md))
                    pdf_elems.append(Paragraph(detail_title, styles["Heading4"]))
                    pdf_elems.append(pdf_table_fn(red_df, max_rows=max_rows_pdf))
                    pdf_elems.append(Spacer(1, 0.08 * inch))

                    corr_csv = inv_dir / "inv_modality_redundancy_correlations.csv"
                    if corr_csv.exists() and not is_pairwise_schema:
                        corr_df = pd.read_csv(corr_csv)
                        if not corr_df.empty:
                            lines.append("**Corrélations entre features redondantes :**\n")
                            lines.append(md_table_fn(corr_df, max_rows=max_rows_md))
                            pdf_elems.append(Paragraph("Corrélations entre features redondantes :", styles["Heading4"]))
                            pdf_elems.append(pdf_table_fn(corr_df, max_rows=max_rows_pdf))
                            pdf_elems.append(Spacer(1, 0.08 * inch))
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Redondances non disponibles : {e})", styles["Normal"]))

    for modality in ["audio", "face", "gaze"]:
        scree_path = inv_dir / f"pca_scree_{modality}.png"
        if scree_path.exists():
            lines.append(f"![]({scree_path.name})\n_Scree plot — famille {modality}_\n\n")
            pdf_elems.append(Image(str(scree_path), width=5.5 * inch, height=2.8 * inch))
            pdf_elems.append(Paragraph(
                f"<i>PCA famille {modality.capitalize()} — variance expliquée et scree plot</i>",
                styles["Normal"],
            ))
            pdf_elems.append(Spacer(1, 0.08 * inch))

    pdf_elems.append(Spacer(1, 0.1 * inch))


def generate_pca_projections_by_variables(
    inv_dir: Path,
    fig_dir: Path,
    merged_data: pd.DataFrame,
    inv_face: pd.DataFrame | None = None,
    inv_speech: pd.DataFrame | None = None,
    inv_gaze: pd.DataFrame | None = None,
    pca_rotation: str = "none",
) -> list[tuple[Path, str]]:
    """
    Génère les projections PCA colorées par variables continues.
    """
    from sklearn.preprocessing import StandardScaler

    generated: list[tuple[Path, str]] = []

    rotation = str(pca_rotation).lower().strip()
    loadings_csv = None
    if rotation == "varimax":
        candidate_list = [
            "pca_loadings_varimax.csv",
            "pca_loadings_raw.csv",
            "pca_loadings.csv",
            "pca_loadings_full_table.csv",
        ]
    else:
        candidate_list = [
            "pca_loadings.csv",
            "pca_loadings_raw.csv",
            "pca_loadings_varimax.csv",
            "pca_loadings_full_table.csv",
        ]
    for candidate in candidate_list:
        candidate_path = inv_dir / candidate
        if candidate_path.exists():
            loadings_csv = candidate_path
            print(f"  [OK] Fichier loadings trouvé : {candidate}")
            break

    if loadings_csv is None:
        print(f"  [SKIP] Pas de loadings PCA trouvés dans {inv_dir}")
        print("        Fichiers cherchés : pca_loadings_raw.csv, pca_loadings_varimax.csv, pca_loadings.csv, pca_loadings_full_table.csv")
        return generated

    try:
        try:
            loadings_df = pd.read_csv(loadings_csv, index_col=0)
        except Exception:
            loadings_df = pd.read_csv(loadings_csv)
            if loadings_df.shape[1] > 0:
                loadings_df = loadings_df.set_index(loadings_df.columns[0])
    except Exception as e:
        print(f"  [WARN] Erreur lecture loadings : {e}")
        return generated

    if loadings_df.empty or loadings_df.shape[1] == 0:
        print("  [SKIP] Fichier loadings vide")
        return generated

    feature_names = loadings_df.index.tolist()
    print(f"  [INFO] {len(feature_names)} features dans loadings, {loadings_df.shape[1]} composantes")

    inv_dfs = []
    if inv_face is not None and not inv_face.empty:
        inv_dfs.append(inv_face)
    if inv_speech is not None and not inv_speech.empty:
        inv_dfs.append(inv_speech)
    if inv_gaze is not None and not inv_gaze.empty:
        inv_dfs.append(inv_gaze)

    if not inv_dfs:
        print("  [SKIP] Aucune donnée INV disponible pour projection")
        return generated

    inv_merged = inv_dfs[0]
    for df in inv_dfs[1:]:
        merge_cols = [c for c in ["group_id", "modalite", "scenario"] if c in df.columns and c in inv_merged.columns]
        if merge_cols:
            inv_merged = inv_merged.merge(df, on=merge_cols, how="outer", suffixes=("", "_dup"))
            inv_merged = inv_merged[[c for c in inv_merged.columns if not c.endswith("_dup")]]

    available_features = [f for f in feature_names if f in inv_merged.columns]
    if len(available_features) < 3:
        missing_features = [f for f in feature_names if f not in inv_merged.columns]
        print(f"  [SKIP] Seulement {len(available_features)}/{len(feature_names)} features disponibles")
        print(f"        Features manquantes : {missing_features[:5]}...")
        return generated

    group_col = "group_id" if "group_id" in inv_merged.columns else None
    if group_col is None:
        print("  [SKIP] Pas de colonne group_id")
        return generated

    X = inv_merged[available_features].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    loadings_subset = loadings_df.loc[available_features]
    n_components = min(loadings_subset.shape[1], 2)
    if n_components < 2:
        print("  [SKIP] Moins de 2 composantes")
        return generated

    pc_scores = X_scaled @ loadings_subset.iloc[:, :2].values
    scores_df = pd.DataFrame({
        "group_id": inv_merged[group_col].values,
        "PC1": pc_scores[:, 0],
        "PC2": pc_scores[:, 1],
    })

    if merged_data is not None and not merged_data.empty and "group_id" in merged_data.columns:
        scores_agg = scores_df.groupby("group_id")[["PC1", "PC2"]].mean().reset_index()
        merged_copy = merged_data.copy()

        cohesion_cols_social = [c for c in merged_copy.columns if "cohesion" in c.lower() and "social" in c.lower()]
        cohesion_cols_task = [c for c in merged_copy.columns if "cohesion" in c.lower() and "task" in c.lower()]
        cohesion_cols_comm = [c for c in merged_copy.columns if "communication" in c.lower() or "comm_" in c.lower()]
        all_cohesion = cohesion_cols_social + cohesion_cols_task + cohesion_cols_comm
        if len(all_cohesion) >= 2:
            merged_copy["COHESION_calc"] = merged_copy[all_cohesion].mean(axis=1)
        else:
            cohesion_questionnaire_cols = [c for c in ["SOC", "TSK", "COM"] if c in merged_copy.columns]
            if len(cohesion_questionnaire_cols) >= 2:
                merged_copy["COHESION_calc"] = merged_copy[cohesion_questionnaire_cols].mean(axis=1)

        projection_specs = [
            ("Performance", ["Score_perf_tsk"]),
            (COHESION_SCORE_COL, [
                COHESION_SCORE_COL,
                "COHESION_score",
                "COHESION",
                "cohesion_total",
                "Cohesion_total",
                "COHESION_calc",
            ]),
            ("COR", ["COR"]),
            ("CRE", ["CRE"]),
            ("SPE", ["SPE"]),
            ("C-score", ["c_score", "C_score", "cscore"]),
            ("RME", ["rme_mean", "RME_mean", "rme_total", "RME"]),
        ]

        for var_label, possible_cols in projection_specs:
            var_col = next((col for col in possible_cols if col in merged_copy.columns), None)
            if var_col is None:
                continue

            plot_df = scores_agg.merge(
                merged_copy[["group_id", var_col]].drop_duplicates(),
                on="group_id",
                how="inner",
            )
            if len(plot_df) < 5 or plot_df[var_col].isna().all():
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                plot_df["PC1"],
                plot_df["PC2"],
                c=plot_df[var_col],
                cmap="viridis",
                s=80,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label(var_label, fontsize=11)
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.set_title(f"Projection PCA colorée par {var_label}", fontsize=13)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.4)

            safe_label = unicodedata.normalize("NFKD", var_label).encode("ascii", "ignore").decode("ascii")
            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", safe_label).strip("_")
            filename = f"pca_projection_by_{safe_label}.png"
            out_path = fig_dir / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            generated.append((out_path, var_label))
            print(f"  [OK] Projection PCA par {var_label} ({var_col}) : {out_path.name}")

    return generated


def _append_projection_single(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    proj_path: Path,
    label: str,
) -> None:
    lines.append(f"![]({proj_path.name})\n_Projection colorée par {label}_\n\n")
    pdf_elems.append(Image(str(proj_path), width=4.8 * inch, height=3.4 * inch))
    pdf_elems.append(Paragraph(f"<i>Projection colorée par {label}</i>", styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))


def _append_projection_triplet(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    title: str,
    projections: list[tuple[Path, str]],
) -> None:
    if not projections:
        return

    lines.append(f"**{title}**\n\n")
    for proj_path, label in projections:
        lines.append(f"![]({proj_path.name})\n_Projection colorée par {label}_\n\n")

    pdf_elems.append(Paragraph(title, styles["Heading4"]))
    pdf_elems.append(Spacer(1, 0.04 * inch))

    cells: list[Any] = []
    for proj_path, label in projections[:3]:
        cells.append([
            Image(str(proj_path), width=2.1 * inch, height=1.6 * inch),
            Paragraph(f"<i>{label}</i>", styles["Normal"]),
        ])
    while len(cells) < 3:
        cells.append("")

    table = Table([cells], colWidths=[2.2 * inch, 2.2 * inch, 2.2 * inch])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    pdf_elems.append(table)
    pdf_elems.append(Spacer(1, 0.08 * inch))


def render_inv_pca_section(
    lines: list[str],
    pdf_elems: list,
    styles: Any,
    results_dir: Path,
    fig_dir: Path,
    md_table_fn: Callable[..., str],
    pdf_table_fn: Callable[..., Any],
    section_num: str = "5",
    inv_pruned_features: list[str] | None = None,
    all_pruned_info: pd.DataFrame | None = None,
    apply_pruning: bool = False,
    inv_subdir: str = "results_inv_structure",
    merged_data: pd.DataFrame | None = None,
    inv_face: pd.DataFrame | None = None,
    inv_speech: pd.DataFrame | None = None,
    inv_gaze: pd.DataFrame | None = None,
    pca_rotation: str = "none",
    max_rows_md: int = 120,
    max_rows_pdf: int = 40,
    show_pruning_audit: bool = True,
):
    """
    Intègre dans le rapport PDF les résultats de l'analyse factorielle des INV.
    """
    inv_dir = results_dir / inv_subdir
    title = f"{section_num} Analyse factorielle des INV (PCA + clustering)"
    lines.append(f"### {title}\n")
    pdf_elems.append(Paragraph(title, styles["Heading2"]))
    pdf_elems.append(Spacer(1, 0.08 * inch))

    if not inv_dir.exists():
        msg = f"Dossier {inv_subdir} absent ({inv_dir}). Exécuter : python analyse_inv/analyze_inv_structure.py"
        lines.append(f"_{msg}_\n\n")
        pdf_elems.append(Paragraph(msg, styles["Normal"]))
        return

    features_used_csv = inv_dir / "inv_features_used.csv"
    features_used_df: pd.DataFrame | None = None
    if features_used_csv.exists():
        try:
            features_used_df = pd.read_csv(features_used_csv)
            if features_used_df is not None and not features_used_df.empty and "feature" in features_used_df.columns:
                features_used_df = features_used_df[["feature"]].dropna().copy()
                features_used_df = features_used_df[
                    ~features_used_df["feature"].astype(str).apply(is_excluded_inv_feature)
                ].drop_duplicates().copy()
                features_used_df["family"] = (
                    features_used_df["feature"].astype(str).map(infer_family_from_name).fillna("autre")
                )
                features_used_df = features_used_df.sort_values(["family", "feature"]).reset_index(drop=True)
        except Exception as e:
            print(f"  [WARN] Impossible de charger inv_features_used.csv : {e}")
            features_used_df = None

    # Exclusions techniques (autres que la corrélation) pour l'audit des variables retirées
    technical_exclusions: list[dict[str, str]] = []
    _audit_csv = results_dir / "INV" / "high_level_features_audit.csv"
    if not _audit_csv.exists():
        _audit_csv = results_dir.parent / "INV" / "high_level_features_audit.csv"
    try:
        if _audit_csv.exists():
            _df_audit = pd.read_csv(_audit_csv)
            if "vr_only" in inv_subdir.lower() and "condition" in _df_audit.columns:
                _df_audit = _df_audit[_df_audit["condition"].astype(str).str.upper() == "VR"]

            max_missing = 0.20
            summary_csv = inv_dir / "analysis_summary.csv"
            if summary_csv.exists():
                try:
                    _summary = pd.read_csv(summary_csv)
                    if (
                        _summary is not None
                        and not _summary.empty
                        and "technical_filter_max_nan_threshold" in _summary.columns
                    ):
                        max_missing = float(_summary.loc[0, "technical_filter_max_nan_threshold"])
                except Exception:
                    pass

            features_used_set = set()
            if features_used_df is not None and not features_used_df.empty:
                features_used_set = set(features_used_df["feature"].astype(str).tolist())

            numeric_cols = [
                c for c in _df_audit.columns if pd.api.types.is_numeric_dtype(_df_audit[c])
            ]
            excluded_reasons: dict[str, str] = {}

            def _set_reason(col: str, reason: str):
                if col not in excluded_reasons:
                    excluded_reasons[col] = reason

            for col in numeric_cols:
                if col in ID_COLS:
                    _set_reason(col, "colonne d'identifiant (ID_COLS)")
                    continue
                if any(str(col).endswith(suf) for suf in EXCLUDE_SUFFIXES):
                    _set_reason(col, "suffixe exclu (règle technique)")
                    continue
                if any(str(col).startswith(pre) for pre in EXCLUDE_PREFIXES):
                    _set_reason(col, "préfixe exclu (règle technique)")
                    continue
                if any(str(col).startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES):
                    _set_reason(col, "legacy audio (int_*)")
                    continue
                if is_excluded_inv_feature(str(col)):
                    _set_reason(col, "exclue par configuration (is_excluded_inv_feature)")

            candidates = [
                c
                for c in numeric_cols
                if c not in excluded_reasons
                and c not in ID_COLS
                and not any(str(c).endswith(suf) for suf in EXCLUDE_SUFFIXES)
                and not any(str(c).startswith(pre) for pre in EXCLUDE_PREFIXES)
                and not any(str(c).startswith(pre) for pre in LEGACY_AUDIO_EXCLUDE_PREFIXES)
                and not is_excluded_inv_feature(str(c))
            ]
            if candidates:
                miss_ratio = _df_audit[candidates].isnull().mean()
                for col in candidates:
                    if miss_ratio.get(col, 0) > max_missing:
                        _set_reason(col, f"taux de NA > {max_missing:.0%}")

                kept_for_impute = [
                    c for c in candidates if c not in excluded_reasons
                    and miss_ratio.get(c, 0) <= max_missing
                ]
                if kept_for_impute:
                    try:
                        from sklearn.impute import SimpleImputer
                        _imp = SimpleImputer(strategy="median")
                        _vals = _imp.fit_transform(_df_audit[kept_for_impute])
                        _stds = np.std(_vals, axis=0)
                        for col, std in zip(kept_for_impute, _stds):
                            if std <= 1e-6:
                                _set_reason(col, "constante après imputation")
                    except Exception as e:
                        print(f"  [WARN] Exclusions techniques (constantes) : {e}")

            for col, reason in excluded_reasons.items():
                if col in features_used_set:
                    continue
                technical_exclusions.append({
                    "Statut": "Supprimée (technique)",
                    "Type exclusion": "Technique",
                    "Périmètre": "Technique",
                    "Famille": infer_family_from_name(col) or "autre",
                    "Feature": col,
                    "Raison (corrélée avec)": reason,
                })
    except Exception as e:
        print(f"  [WARN] Exclusions techniques non disponibles : {e}")

    if all_pruned_info is not None and not all_pruned_info.empty:
        dropped = all_pruned_info[all_pruned_info["kept"] == 0]
        kept = all_pruned_info[all_pruned_info["kept"] == 1]
        n_total = len(all_pruned_info)
        n_kept = (
            int(features_used_df["feature"].nunique())
            if features_used_df is not None and not features_used_df.empty
            else int(kept["feature"].dropna().astype(str).nunique())
        )
        n_dropped = len(dropped)
        scope_label = "VR-only" if "vr_only" in inv_subdir.lower() else "PC+VR"
        pruning_applied = apply_pruning and inv_pruned_features is not None

        if pruning_applied:
            pruning_status = (
                f"**Mode d'analyse : WITH PRUNING** (`--inv-analysis-mode pruning`, scope {scope_label})\n\n"
                f"La PCA présentée ici a été calculée sur **{n_kept} features uniques après pruning** "
                f"des variables redondantes (seuil |r| > {REDUNDANCY_CORR_THRESHOLD:.2f}). {n_dropped} décisions de suppression "
                "ont été documentées dans les tableaux ci-dessous."
            )
            pruning_status_pdf = (
                f"<b>Mode d'analyse : WITH PRUNING</b> (<tt>--inv-analysis-mode pruning</tt>, scope {scope_label})<br/>"
                f"La PCA présentée ici a été calculée sur <b>{n_kept} features uniques après pruning</b> "
                f"des variables redondantes (seuil |r| &gt; {REDUNDANCY_CORR_THRESHOLD:.2f}). {n_dropped} décisions de suppression "
                "ont été documentées dans les tableaux ci-dessous."
            )
            pruning_table_title_md = "Features effectivement supprimées avant la PCA :"
            pruning_table_title_pdf = "<i>Features effectivement supprimées avant la PCA :</i>"
            pruning_table_note_md = (
                "_Le tableau ci-dessous liste uniquement les variables réellement retirées "
                "du jeu de features utilisé pour la PCA._"
            )
            pruning_table_note_pdf = (
                "<i>Le tableau ci-dessous liste uniquement les variables réellement retirées "
                "du jeu de features utilisé pour la PCA.</i>"
            )
            status_label = "Supprimée"
        else:
            pruning_status = (
                f"**Mode d'analyse : WITHOUT PRUNING** (`--inv-analysis-mode no-pruning`, scope {scope_label})\n\n"
                f"La PCA présentée ici a été calculée sur **{n_total} features valides sans pruning analytique**. "
                "Les redondances sont conservées afin de décrire la structure complète de l'espace des variables.\n\n"
                f"_Diagnostic : {n_dropped} features seraient redondantes (|r| > {REDUNDANCY_CORR_THRESHOLD:.2f})._"
            )
            pruning_status_pdf = (
                f"<b>Mode d'analyse : WITHOUT PRUNING</b> (<tt>--inv-analysis-mode no-pruning</tt>, scope {scope_label})<br/>"
                f"La PCA présentée ici a été calculée sur <b>{n_total} features valides sans pruning analytique</b>. "
                "Les redondances sont conservées afin de décrire la structure complète de l'espace des variables.<br/>"
                f"<i>Diagnostic : {n_dropped} features seraient redondantes (|r| &gt; {REDUNDANCY_CORR_THRESHOLD:.2f}).</i>"
            )
            pruning_table_title_md = "Diagnostic de redondance non appliqué :"
            pruning_table_title_pdf = "<i>Diagnostic de redondance non appliqué :</i>"
            pruning_table_note_md = (
                "_Le tableau ci-dessous simule la règle de pruning pour diagnostic, "
                "mais aucune des variables listées n'a été retirée de la PCA affichée._"
            )
            pruning_table_note_pdf = (
                "<i>Le tableau ci-dessous simule la règle de pruning pour diagnostic, "
                "mais aucune des variables listées n'a été retirée de la PCA affichée.</i>"
            )
            status_label = "Diagnostic"

        if technical_exclusions:
            if pruning_applied:
                pruning_table_note_md = (
                    "_Le tableau ci-dessous liste les variables retirées avant la PCA, "
                    "y compris les exclusions techniques (NA, constantes, règles de filtrage) "
                    "et les suppressions par pruning._"
                )
                pruning_table_note_pdf = (
                    "<i>Le tableau ci-dessous liste les variables retirées avant la PCA, "
                    "y compris les exclusions techniques (NA, constantes, règles de filtrage) "
                    "et les suppressions par pruning.</i>"
                )
            else:
                pruning_table_note_md = (
                    "_Le tableau ci-dessous combine les exclusions techniques réellement appliquées "
                    "(NA, constantes, règles de filtrage) et un diagnostic de redondance "
                    "qui n'a pas été appliqué à la PCA affichée._"
                )
                pruning_table_note_pdf = (
                    "<i>Le tableau ci-dessous combine les exclusions techniques réellement appliquées "
                    "(NA, constantes, règles de filtrage) et un diagnostic de redondance "
                    "qui n'a pas été appliqué à la PCA affichée.</i>"
                )

        lines.append(f"> {pruning_status}\n\n")
        pdf_elems.append(Paragraph(pruning_status_pdf, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

        if show_pruning_audit and (n_dropped > 0 or technical_exclusions):
            scope_map = {
                "inv_pruned_features": "PCA globale",
                "inv_pruned_features_gaze": "Complément gaze",
            }
            lines.append(f"{pruning_table_title_md}\n\n")
            lines.append(f"{pruning_table_note_md}\n\n")

            drop_rows: list[dict[str, str]] = []
            if n_dropped > 0:
                drop_table = dropped[["feature", "reason"]].copy()
                drop_table.insert(0, "Statut", status_label)
                drop_table.insert(1, "Type exclusion", "Corrélation")
                if "source" in dropped.columns:
                    drop_table.insert(2, "Périmètre", dropped["source"].map(scope_map).fillna(dropped["source"]))
                drop_table["Famille"] = drop_table["feature"].astype(str).map(infer_family_from_name).fillna("autre")
                drop_table = drop_table.sort_values(["Famille", "feature"]).reset_index(drop=True)
                # Réordonner colonnes : Statut, Famille, (Périmètre), Feature, Raison
                col_order = ["Statut", "Type exclusion"]
                if "Périmètre" in drop_table.columns:
                    col_order.append("Périmètre")
                col_order += ["Famille", "feature", "reason"]
                drop_table = drop_table[col_order]
                drop_table.columns = list(drop_table.columns[:-2]) + ["Feature", "Raison (corrélée avec)"]
                drop_rows.extend(drop_table.to_dict(orient="records"))

            if technical_exclusions:
                pruned_features = {row["Feature"] for row in drop_rows if "Feature" in row}
                for row in technical_exclusions:
                    if row["Feature"] not in pruned_features:
                        drop_rows.append(row)

            drop_df = pd.DataFrame(drop_rows)
            if not drop_df.empty:
                def _extract_abs_corr(val: str) -> float:
                    if not isinstance(val, str):
                        return np.nan
                    match = re.search(r"\|r\|=([0-9.]+)", val)
                    if not match:
                        return np.nan
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return np.nan

                drop_df["_abs_corr"] = drop_df.apply(
                    lambda row: _extract_abs_corr(row.get("Raison (corrélée avec)", ""))
                    if row.get("Type exclusion") == "Corrélation"
                    else np.nan,
                    axis=1,
                )
                drop_df = drop_df.sort_values(
                    ["Type exclusion", "Famille", "_abs_corr", "Feature"],
                    ascending=[True, True, False, True],
                    na_position="last",
                ).reset_index(drop=True)
                drop_df = drop_df.drop(columns=["_abs_corr"])
                max_rows_table = int(len(drop_df))
                lines.append(md_table_fn(drop_df, max_rows=max_rows_table))
                lines.append("\n")
                pdf_elems.append(Paragraph(pruning_table_title_pdf, styles["Normal"]))
                pdf_elems.append(Paragraph(pruning_table_note_pdf, styles["Normal"]))
                pdf_elems.append(pdf_table_fn(drop_df, max_rows=max_rows_table))
                pdf_elems.append(Spacer(1, 0.15 * inch))

        # — Variables gardées en force malgré une corrélation potentiellement élevée —
        if show_pruning_audit:
            _protected_vars: set[str] = set()
            for _pair in PRUNING_PROTECTED_PAIRS:
                _protected_vars.update(_pair)
            _force_include_vars: set[str] = set(REGRESSION_FORCE_INCLUDE)
            _all_force_kept = _protected_vars | _force_include_vars
            if all_pruned_info is not None and not all_pruned_info.empty:
                _candidates = set(all_pruned_info["feature"].tolist())
                _force_kept_shown = sorted(_all_force_kept & _candidates)
            else:
                _force_kept_shown = sorted(_all_force_kept)
            if _force_kept_shown:
                _desc_map_fk = {f: cfg.get("description", "") for f, cfg in INV_FEATURES.items()}
                _fk_rows = []
                for _fv in _force_kept_shown:
                    _reason = ""
                    for _pair in PRUNING_PROTECTED_PAIRS:
                        if _fv in _pair:
                            _other = next(iter(_pair - {_fv}), "")
                            _reason = f"Paire protégée avec {_other}"
                            break
                    if _fv in _force_include_vars and not _reason:
                        _reason = "REGRESSION_FORCE_INCLUDE"
                    elif _fv in _force_include_vars:
                        _reason += " + REGRESSION_FORCE_INCLUDE"
                    _fk_rows.append({
                        "Statut": "Gardée (force)",
                        "Famille": infer_family_from_name(_fv) or "autre",
                        "Feature": _fv,
                        "Description": _desc_map_fk.get(_fv, ""),
                        "Motif": _reason,
                    })
                _fk_df = pd.DataFrame(_fk_rows).sort_values(["Famille", "Feature"]).reset_index(drop=True)
                _fk_title_md = "Variables conservées en force (protégées du pruning) :"
                _fk_title_pdf = "<i>Variables conservées en force (protégées du pruning) :</i>"
                _fk_note_md = (
                    "_Ces variables auraient pu être retirées par le pruning automatique mais "
                    "sont protégées explicitement (paire protégée ou REGRESSION_FORCE_INCLUDE)._"
                )
                _fk_note_pdf = (
                    "<i>Ces variables auraient pu être retirées par le pruning automatique mais "
                    "sont protégées explicitement (paire protégée ou REGRESSION_FORCE_INCLUDE).</i>"
                )
                lines.append(f"{_fk_title_md}\n\n")
                lines.append(f"{_fk_note_md}\n\n")
                lines.append(md_table_fn(_fk_df, max_rows=50))
                lines.append("\n")
                pdf_elems.append(Paragraph(_fk_title_pdf, styles["Normal"]))
                pdf_elems.append(Paragraph(_fk_note_pdf, styles["Normal"]))
                pdf_elems.append(pdf_table_fn(_fk_df, max_rows=50))
                pdf_elems.append(Spacer(1, 0.15 * inch))

        if features_used_df is not None and not features_used_df.empty:
            _desc_map = {f: cfg.get("description", "") for f, cfg in INV_FEATURES.items()}
            features_used_df["description"] = features_used_df["feature"].map(_desc_map).fillna("")

            # Stats descriptives depuis high_level_features_audit.csv
            _audit_csv = results_dir / "INV" / "high_level_features_audit.csv"
            if not _audit_csv.exists():
                _audit_csv = results_dir.parent / "INV" / "high_level_features_audit.csv"
            if _audit_csv.exists():
                try:
                    _df_stats = pd.read_csv(_audit_csv)
                    if "vr_only" in inv_subdir.lower() and "condition" in _df_stats.columns:
                        _df_stats = _df_stats[_df_stats["condition"].str.upper() == "VR"]
                    _feats = [f for f in features_used_df["feature"] if f in _df_stats.columns]
                    if _feats:
                        _stats = _df_stats[_feats].agg(["count", "mean", "std", "min", "max"])
                        features_used_df["n"]    = features_used_df["feature"].map(_stats.loc["count"].round(0).astype(int).to_dict()).fillna("")
                        features_used_df["mean"] = features_used_df["feature"].map(_stats.loc["mean"].round(3).to_dict()).fillna("")
                        features_used_df["std"]  = features_used_df["feature"].map(_stats.loc["std"].round(3).to_dict()).fillna("")
                        features_used_df["min"]  = features_used_df["feature"].map(_stats.loc["min"].round(3).to_dict()).fillna("")
                        features_used_df["max"]  = features_used_df["feature"].map(_stats.loc["max"].round(3).to_dict()).fillna("")
                except Exception as e:
                    print(f"  [WARN] Stats descriptives PCA : {e}")

            used_table = features_used_df.rename(columns={
                "feature": "Feature", "family": "Famille", "description": "Description",
                "n": "N", "mean": "Moy", "std": "ET", "min": "Min", "max": "Max",
            })
            used_table_title_md = "Features effectivement utilisées dans la PCA :"
            used_table_title_pdf = "<i>Features effectivement utilisées dans la PCA :</i>"
            used_table_note_md = (
                "_Cette liste correspond au jeu final de variables réellement injecté dans la PCA "
                "(loadings, projections et clustering)._"
            )
            used_table_note_pdf = (
                "<i>Cette liste correspond au jeu final de variables réellement injecté dans la PCA "
                "(loadings, projections et clustering).</i>"
            )
            lines.append(f"{used_table_title_md}\n\n")
            lines.append(f"{used_table_note_md}\n\n")
            lines.append(md_table_fn(used_table, max_rows=80))
            lines.append("\n")
            pdf_elems.append(Paragraph(used_table_title_pdf, styles["Normal"]))
            pdf_elems.append(Paragraph(used_table_note_pdf, styles["Normal"]))
            pdf_elems.append(pdf_table_fn(used_table, max_rows=80))
            pdf_elems.append(Spacer(1, 0.15 * inch))

            # — Matrice de corrélation des variables après pruning —
            corr_pruned_csv = inv_dir / "inv_correlation_matrix_pruned.csv"
            corr_pruned_png = inv_dir / "corr_matrix_inv_pruned.png"
            if corr_pruned_csv.exists():
                try:
                    corr_df = pd.read_csv(corr_pruned_csv, index_col=0)
                    corr_df = corr_df.round(2)
                    pdf_elems.append(Paragraph("Matrice de corrélation des variables après pruning :", styles["Heading4"]))
                    pdf_elems.append(Spacer(1, 0.05 * inch))
                    if corr_pruned_png.exists():
                        pdf_elems.append(Image(str(corr_pruned_png), width=6.3 * inch, height=5.5 * inch))
                        pdf_elems.append(Paragraph(
                            f"<i>Corrélations de Pearson — variables retenues après pruning (|r| > {REDUNDANCY_CORR_THRESHOLD:.2f})</i>",
                            styles["Normal"],
                        ))
                    else:
                        pdf_elems.append(pdf_table_fn(corr_df.reset_index().rename(columns={"index": ""}), max_rows=80))
                    pdf_elems.append(Spacer(1, 0.15 * inch))
                except Exception as e:
                    print(f"  [WARN] Matrice corrélation pruned : {e}")

    note = (
        "La PCA réduit les features INV (audio, gaze, face) en composantes latentes "
        "orthogonales maximisant la variance. Le critère de rétention est l'ANALYSE "
        "PARALLÈLE DE HORN (1965 ; percentile 95 sur 1000 tirages aléatoires de même "
        "dimension) : seules les composantes dont l'eigenvalue dépasse le seuil aléatoire "
        "sont retenues. À p > n (ici 17 features pour 12 groupes VR), le critère de Kaiser "
        "(eigenvalue > 1) N'EST PAS valide — il surestime le nombre de composantes car la "
        "variance totale est répartie sur min(n−1, p) axes — et n'est reporté qu'à titre "
        "indicatif. Le clustering hiérarchique regroupe les features fortement corrélées. "
        "Imputation (audit) : un groupe VR sans features audio complètes est imputé (médiane) "
        "pour atteindre n=12 ; les composantes à dominante audio reposent donc sur 11 observations "
        "audio réelles + 1 imputée, à interpréter avec la prudence correspondante."
    )
    pdf_elems.append(Paragraph(note, styles["Normal"]))
    pdf_elems.append(Spacer(1, 0.1 * inch))

    if features_used_df is not None and not features_used_df.empty:
        n_features_actual = len(features_used_df)
        family_counts = features_used_df["family"].value_counts().to_dict()
        fam_str = ", ".join(f"{n} {fam}" for fam, n in sorted(family_counts.items()))
        note_features = f"Analyse basée sur {n_features_actual} features utilisées ({fam_str})"
        lines.append(note_features + "\n\n")
        pdf_elems.append(Paragraph(note_features, styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))
        missing_families = {"audio", "face", "gaze"} - set(family_counts.keys())
        if missing_families:
            miss_note = (
                f"Note : Famille(s) {', '.join(sorted(missing_families))} non présente(s) dans l'analyse."
            )
            lines.append(miss_note + "\n\n")
            pdf_elems.append(Paragraph(miss_note, styles["Normal"]))

    ev_csv = inv_dir / "pca_explained_variance.csv"
    if ev_csv.exists():
        try:
            ev = pd.read_csv(ev_csv)
            ev_show = ev[ev["eigenvalue"] >= 1.0].copy() if "eigenvalue" in ev.columns else ev.head(7)
            for col in ["eigenvalue", "variance_ratio", "cumulative_variance"]:
                if col in ev_show.columns:
                    ev_show[col] = ev_show[col].round(3)
            n_comp = len(ev_show)
            cum_var = ev_show["cumulative_variance"].max() if "cumulative_variance" in ev_show.columns else None
            summary = f"{n_comp} composantes retenues (eigenvalue ≥ 1)" + (f", variance cumulée = {cum_var*100:.1f}%" if cum_var else "")
            lines.append(f"**{summary}**\n\n")
            pdf_elems.append(Paragraph(f"<b>{summary}</b>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            lines.append("**Variance expliquée par composante :**\n")
            lines.append(md_table_fn(ev_show, max_rows=10))
            pdf_elems.append(Paragraph("Variance expliquée par composante :", styles["Heading4"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(pdf_table_fn(ev_show, max_rows=10))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Variance expliquée non disponible : {e})", styles["Normal"]))

    # Tableau de l'analyse parallèle de Horn (critère de rétention retenu)
    horn_csv = inv_dir / "pca_horn_parallel_analysis.csv"
    if horn_csv.exists():
        try:
            horn = pd.read_csv(horn_csv)
            for col in ["eigenvalue_observed", "horn_threshold_p95"]:
                if col in horn.columns:
                    horn[col] = horn[col].round(3)
            n_horn = int(horn["passes_horn"].sum()) if "passes_horn" in horn.columns else 0
            horn_summary = (
                f"Analyse parallèle de Horn : {n_horn} composante(s) retenue(s) "
                "(eigenvalue observée > seuil aléatoire p95). Critère principal de rétention à p > n."
            )
            # C8 : détection des composantes borderline (marge relative < 5 % au seuil p95).
            _borderline = []
            if {"eigenvalue_observed", "horn_threshold_p95", "component", "passes_horn"}.issubset(horn.columns):
                for _, _r in horn.iterrows():
                    if not bool(_r["passes_horn"]):
                        continue
                    _thr = float(_r["horn_threshold_p95"])
                    _obs = float(_r["eigenvalue_observed"])
                    if _thr > 0 and (_obs - _thr) / _thr < 0.05:
                        _borderline.append((str(_r["component"]), (_obs - _thr) / _thr * 100.0))
            lines.append(f"**{horn_summary}**\n\n")
            lines.append(md_table_fn(horn, max_rows=12))
            pdf_elems.append(Paragraph(f"<b>{horn_summary}</b>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.05 * inch))
            pdf_elems.append(pdf_table_fn(horn, max_rows=12))
            if _borderline:
                _bl_str = ", ".join(f"{c} (+{m:.2f} %)" for c, m in _borderline)
                horn_caution = (
                    f"⚠ Composante(s) BORDERLINE : {_bl_str} dépassent le seuil p95 de moins de 5 % — "
                    "leur rétention est fragile (elles pourraient basculer sous un autre tirage aléatoire). "
                    "L'interprétation substantielle doit être ancrée sur PC1–PC2 (marges confortables) ; "
                    "PC3–PC4 sont à considérer comme exploratoires."
                )
                lines.append(f"\n_{horn_caution}_\n\n")
                pdf_elems.append(Spacer(1, 0.03 * inch))
                pdf_elems.append(Paragraph(horn_caution, styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Analyse de Horn non disponible : {e})", styles["Normal"]))

    scree_png = inv_dir / "pca_scree_plot.png"
    if scree_png.exists():
        lines.append(f"![]({scree_png.name})\n_Scree plot — la ligne rouge (Kaiser eigenvalue=1) est indicative ; la rétention suit Horn_\n\n")
        pdf_elems.append(Image(str(scree_png), width=4.5 * inch, height=3.0 * inch))
        pdf_elems.append(Paragraph("<i>Scree plot — ligne rouge Kaiser (indicative, invalide à p>n) ; rétention par analyse de Horn</i>", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    component_palette = [
        colors.HexColor("#e8f1fb"),
        colors.HexColor("#eaf7ea"),
        colors.HexColor("#fff3d9"),
        colors.HexColor("#f6e8fb"),
        colors.HexColor("#fdecea"),
        colors.HexColor("#e9f7f6"),
        colors.HexColor("#f1f1f1"),
    ]

    # Composantes retenues = celles qui passent l'analyse parallèle de Horn.
    # (Fallback sur Kaiser eigenvalue≥1 seulement si le CSV Horn est absent — indicatif.)
    kaiser_comps: list[str] = []
    horn_csv_sel = inv_dir / "pca_horn_parallel_analysis.csv"
    if horn_csv_sel.exists():
        try:
            _horn_sel = pd.read_csv(horn_csv_sel)
            if "passes_horn" in _horn_sel.columns and "component" in _horn_sel.columns:
                kaiser_comps = _horn_sel[_horn_sel["passes_horn"].astype(bool)]["component"].astype(str).tolist()
        except Exception:
            pass
    if not kaiser_comps and ev_csv.exists():
        try:
            ev_tmp = pd.read_csv(ev_csv)
            kaiser_comps = ev_tmp[ev_tmp["eigenvalue"] >= 1.0]["component"].astype(str).tolist()
        except Exception:
            pass

    def _render_loadings_block(
        csv_path: Path,
        label: str,
        label_pdf: str,
        full_csv_path: Path | None = None,
    ) -> None:
        """Affiche top-4 + tableau complet pour un jeu de loadings (raw ou varimax)."""
        if not csv_path.exists():
            return
        try:
            ld = pd.read_csv(csv_path, index_col=0)
            ld.index.name = "feature"
            ld = ld.reset_index()
            feature_col = "feature"
            ld = ld[~ld[feature_col].astype(str).apply(is_excluded_inv_feature)].copy()
            comp_cols = [c for c in ld.columns if c.startswith("PC")]
            if kaiser_comps:
                comp_cols = [c for c in comp_cols if c in kaiser_comps]

            # — Top 4 —
            top_rows = []
            for pc in comp_cols[:7]:
                if pc not in ld.columns:
                    continue
                top_f = ld[[feature_col, pc]].copy()
                top_f["abs"] = top_f[pc].abs()
                top_f = top_f.nlargest(4, "abs")
                for _, r in top_f.iterrows():
                    top_rows.append({"Composante": pc, "Feature": r[feature_col], "Loading": round(r[pc], 3)})

            if top_rows:
                top_ld = pd.DataFrame(top_rows)
                lines.append(f"**{label} — Top 4 features par composante :**\n\n")
                lines.append(md_table_fn(top_ld, max_rows=30))
                lines.append("\n")
                pdf_elems.append(Paragraph(f"{label_pdf} — Top 4 features par composante :", styles["Heading4"]))
                pdf_elems.append(Spacer(1, 0.05 * inch))

                top_ld_pdf = top_ld.copy()
                top_ld_pdf["Loading"] = top_ld_pdf["Loading"].map(lambda x: f"{x:.3f}")
                table_rows_pdf = [list(top_ld_pdf.columns)] + top_ld_pdf.values.tolist()
                pdf_tbl = Table(table_rows_pdf, colWidths=[1.0 * inch, 3.7 * inch, 1.0 * inch])
                style_cmds = [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (2, 1), (2, -1), "CENTER"),
                ]
                comp_to_color = {
                    pc: component_palette[i % len(component_palette)]
                    for i, pc in enumerate(top_ld["Composante"].drop_duplicates().tolist())
                }
                for row_idx, comp in enumerate(top_ld["Composante"].tolist(), start=1):
                    style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), comp_to_color[comp]))
                pdf_tbl.setStyle(TableStyle(style_cmds))
                pdf_elems.append(pdf_tbl)
                pdf_elems.append(Spacer(1, 0.1 * inch))

            # — Tableau complet —
            # Préférer full_csv_path si fourni, sinon reconstruire depuis csv_path
            src_full = full_csv_path if (full_csv_path and full_csv_path.exists()) else csv_path
            full_ld = pd.read_csv(src_full, index_col=0)
            full_ld.index.name = "Variable"
            full_ld = full_ld.reset_index()
            if "Variable" in full_ld.columns:
                full_ld = full_ld[~full_ld["Variable"].astype(str).apply(is_excluded_inv_feature)].copy()
            pc_cols_full = [c for c in full_ld.columns if c.startswith("PC")]
            if kaiser_comps:
                pc_cols_full = [c for c in pc_cols_full if c in kaiser_comps]
            show_cols = ["Variable"] + pc_cols_full
            show_df = full_ld[[c for c in show_cols if c in full_ld.columns]].copy()
            for c in pc_cols_full:
                if c in show_df.columns:
                    show_df[c] = pd.to_numeric(show_df[c], errors="coerce").round(3)
            if not show_df.empty:
                lines.append(f"**{label} — Tableau complet (variables × composantes) :**\n\n")
                lines.append(md_table_fn(show_df, max_rows=50))
                lines.append("\n")
                pdf_elems.append(Paragraph(f"{label_pdf} — Tableau complet (variables × composantes) :", styles["Heading4"]))
                pdf_elems.append(Spacer(1, 0.05 * inch))
                pdf_elems.append(pdf_table_fn(show_df, max_rows=50))
                pdf_elems.append(Spacer(1, 0.1 * inch))

        except Exception as e:
            pdf_elems.append(Paragraph(f"({label} non disponibles : {e})", styles["Normal"]))

    rotation = str(pca_rotation).lower().strip()
    if rotation == "varimax":
        varimax_csv = inv_dir / "pca_loadings_varimax.csv"
        if varimax_csv.exists():
            _render_loadings_block(
                csv_path=varimax_csv,
                label="Loadings PCA (rotation varimax)",
                label_pdf="Loadings PCA (rotation varimax)",
                full_csv_path=inv_dir / "pca_loadings_full_table.csv",
            )
        heatmap_varimax = inv_dir / "pca_loadings_heatmap_varimax.png"
        if heatmap_varimax.exists():
            lines.append(f"![]({heatmap_varimax.name})\n_Heatmap des loadings après rotation varimax_\n\n")
            pdf_elems.append(Paragraph("Heatmap loadings après rotation varimax :", styles["Heading4"]))
            pdf_elems.append(Image(str(heatmap_varimax), width=6.0 * inch, height=3.8 * inch))
            pdf_elems.append(Paragraph("<i>Rotation varimax : structure simplifiée des composantes</i>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.1 * inch))
    else:
        raw_csv = inv_dir / "pca_loadings.csv"
        if not raw_csv.exists():
            raw_csv = inv_dir / "pca_loadings_raw.csv"
        if raw_csv.exists():
            _render_loadings_block(
                csv_path=raw_csv,
                label="Loadings PCA (sans rotation)",
                label_pdf="Loadings PCA (sans rotation)",
                full_csv_path=inv_dir / "pca_loadings_full_table.csv",
            )

        heatmap_png = inv_dir / "pca_loadings_heatmap.png"
        if heatmap_png.exists():
            lines.append(f"![]({heatmap_png.name})\n_Heatmap des loadings sans rotation (composantes × features)_\n\n")
            pdf_elems.append(Image(str(heatmap_png), width=6.0 * inch, height=3.8 * inch))
            pdf_elems.append(Paragraph("<i>Heatmap des loadings PCA sans rotation (composantes × features)</i>", styles["Normal"]))
            pdf_elems.append(Spacer(1, 0.1 * inch))

    cl_csv = inv_dir / "feature_clusters.csv"
    dendro_png = inv_dir / "feature_dendrogram.png"
    n_clusters = "?"
    if cl_csv.exists():
        try:
            cl = pd.read_csv(cl_csv)
            n_clusters = cl["cluster"].nunique() if "cluster" in cl.columns else "?"
        except Exception:
            pass

    pruning_label = "WITH PRUNING" if "with_pruning" in inv_subdir else ("WITHOUT PRUNING" if "without_pruning" in inv_subdir else "MODE NON SPÉCIFIÉ")
    scope_label = "VR-only" if "vr_only" in inv_subdir.lower() else "PC+VR"
    cluster_caption = f"Dendrogramme du clustering hiérarchique agglomératif ({n_clusters} clusters, {pruning_label}, {scope_label})"

    if dendro_png.exists():
        lines.append(f"**{cluster_caption} :**\n\n")
        lines.append(f"![]({dendro_png.name})\n_{cluster_caption}_\n\n")
        pdf_elems.append(Paragraph(f"{cluster_caption} :", styles["Heading4"]))
        pdf_elems.append(Image(str(dendro_png), width=6.3 * inch, height=4.8 * inch))
        pdf_elems.append(Paragraph(f"<i>{cluster_caption}</i>", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))
    elif cl_csv.exists():
        try:
            cl = pd.read_csv(cl_csv)
            lines.append(f"**{n_clusters} clusters de features détectés (clustering hiérarchique) :**\n")
            lines.append(md_table_fn(cl, max_rows=40))
            pdf_elems.append(Paragraph(f"{n_clusters} clusters de features (clustering hiérarchique agglomératif) :", styles["Heading4"]))
            pdf_elems.append(pdf_table_fn(cl, max_rows=40))
            pdf_elems.append(Spacer(1, 0.1 * inch))
        except Exception as e:
            pdf_elems.append(Paragraph(f"(Clusters non disponibles : {e})", styles["Normal"]))

    proj_png = inv_dir / "pca_projection_groups.png"
    if proj_png.exists():
        lines.append(f"![]({proj_png.name})\n_Projection PC1 vs PC2 par groupe_\n\n")
        pdf_elems.append(Paragraph("Projection PC1 vs PC2 par groupe :", styles["Heading4"]))
        pdf_elems.append(Image(str(proj_png), width=5.0 * inch, height=3.5 * inch))
        pdf_elems.append(Paragraph("<i>Projection des groupes dans l'espace PC1 × PC2</i>", styles["Normal"]))
        pdf_elems.append(Spacer(1, 0.1 * inch))

    if merged_data is not None and inv_face is not None:
        print("  [PCA] Génération des projections par variables continues...")
        generated_projections = generate_pca_projections_by_variables(
            inv_dir=inv_dir,
            fig_dir=fig_dir,
            merged_data=merged_data,
            inv_face=inv_face,
            inv_speech=inv_speech,
            inv_gaze=inv_gaze,
            pca_rotation=pca_rotation,
        )
    else:
        generated_projections = []

    proj_by_vars = [
        ("pca_projection_by_Performance.png", "Performance"),
        ("pca_projection_by_Cohesion_questionnaire_score.png", COHESION_SCORE_COL),
        ("pca_projection_by_COR.png", "COR"),
        ("pca_projection_by_CRE.png", "CRE"),
        ("pca_projection_by_SPE.png", "SPE"),
        ("pca_projection_by_C_score.png", "C-score"),
        ("pca_projection_by_C-score.png", "C-score"),
        ("pca_projection_by_RME.png", "RME"),
        # Legacy fallbacks from older reports
        ("pca_projection_by_Cohésion.png", COHESION_SCORE_COL),
        ("pca_projection_by_COHESION.png", COHESION_SCORE_COL),
    ]

    projections_found = [(proj_path, label) for proj_path, label in generated_projections if proj_path.exists()]
    found_labels = {lbl for _, lbl in projections_found}
    for filename, label in proj_by_vars:
        if label in found_labels or label.split()[0] in [l.split()[0] for l in found_labels]:
            continue
        proj_path = inv_dir / filename
        if proj_path.exists():
            projections_found.append((proj_path, label))

    if projections_found:
        lines.append("**Projections PCA colorées par variables :**\n\n")
        pdf_elems.append(Paragraph("Projections PCA colorées par variables :", styles["Heading4"]))
        pdf_elems.append(Spacer(1, 0.05 * inch))
        proj_map = {label: proj_path for proj_path, label in projections_found}

        for label in ["Performance", "C-score", "RME", COHESION_SCORE_COL]:
            proj_path = proj_map.get(label)
            if proj_path is not None:
                _append_projection_single(lines, pdf_elems, styles, proj_path, label)

        tms_triplet = [
            (proj_map[label], label)
            for label in ["COR", "CRE", "SPE"]
            if label in proj_map
        ]
        if tms_triplet:
            _append_projection_triplet(
                lines,
                pdf_elems,
                styles,
                "Dimensions TMS (COR, CRE, SPE)",
                tms_triplet,
            )

    var_mod_png = inv_dir / "pca_variance_by_modality.png"
    if var_mod_png.exists():
        lines.append(f"![]({var_mod_png.name})\n_Variance expliquée par composante — PC vs VR_\n\n")
        pdf_elems.append(Paragraph("Comparaison PC vs VR — variance expliquée par composante :", styles["Heading4"]))
        pdf_elems.append(Image(str(var_mod_png), width=5.5 * inch, height=3.2 * inch))
        pdf_elems.append(Paragraph(
            "<i>La structure factorielle peut différer selon la modalité "
            "(PC vs VR), suggérant des patterns d'interaction différents.</i>",
            styles["Normal"],
        ))
        pdf_elems.append(Spacer(1, 0.15 * inch))

    _render_inv_modality_pca_subsection(
        lines,
        pdf_elems,
        styles,
        inv_dir,
        md_table_fn=md_table_fn,
        pdf_table_fn=pdf_table_fn,
        apply_pruning=apply_pruning,
        inv_pruned_features=inv_pruned_features,
        max_rows_md=min(max_rows_md, 50),
        max_rows_pdf=50,
    )

    pdf_elems.append(Spacer(1, 0.1 * inch))

    # — Section régressions PCA —
    render_pca_regression_section(
        lines=lines,
        pdf_elems=pdf_elems,
        styles=styles,
        results_dir=results_dir,
        inv_subdir=inv_subdir,
        section_num=f"{section_num}.5",
        pca_rotation=pca_rotation,
    )
