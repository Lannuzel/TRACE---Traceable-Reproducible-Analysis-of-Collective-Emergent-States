#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Orchestration du pipeline d'analyse des questionnaires TMS / Cohésion.

Nomenclature des modes :
  --mode global   : analyse globale sur l'ensemble des participants
                    (fiabilité, descriptifs, tests rôle, plots)
                    → dossier "global/" ; fichiers nommés *_questionnaire.csv
                    Note : le terme "TMS" dans les anciens noms de fichiers
                    désignait "Transactive Memory System", l'un des deux blocs
                    du questionnaire. On l'a remplacé par "questionnaire" pour
                    englober aussi la Cohésion.

  --mode scenario : analyse stratifiée Scénario × Modalité
                    → dossier "analyse/"

  --mode all      : les deux (défaut)

Usage :
    # Analyse complète :
    python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire --mode all

    # Avec application des suppressions exploratoires (recalcul alpha sur jeu épuré) :
    python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire --mode all --apply-pruning

    # Avec seuil alpha personnalisé (ne traite que les dimensions avec α < seuil, défaut 0.70) :
    python main.py --data ../../data_e2/results-survey.xlsx --out ../../results/questionnaire --mode all --apply-pruning --alpha-threshold 0.80
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permet l'import relatif quand exécuté comme script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd

from py.g3_context import (
    build_free_comments,
    extract_participant_profile,
    summarize_participant_profile,
)
from py.io_read import read_survey
from py.transform import extract_item_short, make_long_survey
from py.reliability import cronbach_by_dimension, item_stats_by_dimension
from py.descriptives import descriptives_by_dimension
from py.role_tests import summary_by_role
from py.plots import export_item_distributions, export_role_means
from py.item_pruning import prune_by_rdrop
from py.config import RDROP_THRESHOLD, ALPHA_ACCEPTABILITY_THRESHOLD
from py.exploratory_pruning import (
    exploratory_prune_all,
    apply_exploratory_pruning,
    print_exploratory_summary,
)
from py.scenario_modalite import run_scenario_modalite, _build_long, _compute_scores


MANUAL_EXPLORATORY_PRUNING_KEEP: dict[str, set[str]] = {
    # Override analytique : pour CRE, on ne conserve que la suppression de CRE04.
    "CRE": {"G1Q00001.CRE04"},
}


def _apply_manual_exploratory_pruning_overrides(
    summary: pd.DataFrame,
    trace: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique des overrides analytiques aux suppressions exploratoires.

    Le pruning greedy reste calculé normalement, puis on restreint certaines
    dimensions à une liste explicite d'items effectivement supprimés.
    """
    if summary is None or summary.empty or trace is None or trace.empty:
        return summary, trace

    summary_out = summary.copy()
    trace_out = trace.copy()

    trace_out["dimension"] = trace_out["dimension"].astype(str).str.upper().str.strip()
    summary_out["dimension"] = summary_out["dimension"].astype(str).str.upper().str.strip()
    trace_out["_code_short"] = trace_out["code"].apply(extract_item_short)

    for dim, allowed_codes in MANUAL_EXPLORATORY_PRUNING_KEEP.items():
        dim_mask = trace_out["dimension"] == dim
        dim_trace = trace_out.loc[dim_mask].copy()
        if dim_trace.empty:
            continue

        kept_trace = dim_trace[dim_trace["_code_short"].isin(allowed_codes)].copy()
        if kept_trace.empty:
            print(
                f"[WARN] Override pruning {dim}: aucun item parmi "
                f"{sorted(allowed_codes)} n'a été trouvé dans la trace exploratoire."
            )
            continue

        kept_trace = kept_trace.sort_values("step").reset_index(drop=True)
        kept_trace["step"] = range(1, len(kept_trace) + 1)

        trace_out = pd.concat(
            [trace_out.loc[~dim_mask], kept_trace],
            ignore_index=True,
        )

        summary_mask = summary_out["dimension"] == dim
        if summary_mask.any():
            alpha_initial = pd.to_numeric(
                summary_out.loc[summary_mask, "alpha_initial"],
                errors="coerce",
            ).iloc[0]
            alpha_optimise = pd.to_numeric(
                kept_trace["alpha_after"],
                errors="coerce",
            ).dropna()
            alpha_final = alpha_optimise.iloc[-1] if not alpha_optimise.empty else np.nan
            removed_codes = kept_trace["code"].astype(str).tolist()

            summary_out.loc[summary_mask, "alpha_optimise"] = round(float(alpha_final), 3) if pd.notna(alpha_final) else np.nan
            if pd.notna(alpha_initial) and pd.notna(alpha_final):
                summary_out.loc[summary_mask, "gain_alpha"] = round(float(alpha_final - alpha_initial), 3)
            else:
                summary_out.loc[summary_mask, "gain_alpha"] = np.nan
            summary_out.loc[summary_mask, "n_items_retires"] = int(len(removed_codes))
            summary_out.loc[summary_mask, "items_retires"] = "; ".join(removed_codes)
            summary_out.loc[summary_mask, "statut"] = "optimisé" if removed_codes else "déjà optimal"

            kept_short = [code for code in kept_trace["_code_short"].dropna().astype(str).tolist()]
            print(f"[INFO] Override pruning {dim}: items retenus pour suppression = {kept_short}")

    if "_code_short" in trace_out.columns:
        trace_out = trace_out.drop(columns=["_code_short"])

    order = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]
    rank = {k: i for i, k in enumerate(order)}
    summary_out = summary_out.sort_values(
        by=["dimension"],
        key=lambda s: s.map(rank).fillna(999),
    ).reset_index(drop=True)
    trace_out = trace_out.sort_values(
        by=["dimension", "step"],
        key=lambda s: s.map(rank).fillna(999) if s.name == "dimension" else s,
    ).reset_index(drop=True)

    return summary_out, trace_out


def _write_exploratory_report(
    summary: pd.DataFrame, trace: pd.DataFrame, outdir: Path
) -> None:
    """
    Génère un rapport texte (exploratory_report.txt) résumant les suppressions
    exploratoires par dimension : items retirés, alpha avant/après.
    """
    lines = [
        "=" * 70,
        "RAPPORT — Analyse exploratoire de fiabilité (suppression d'items)",
        "=" * 70,
        "",
        "Méthode : suppression itérative greedy — à chaque étape, l'item dont",
        "la suppression maximise le plus l'alpha de Cronbach est retiré.",
        "Critère d'arrêt : aucune suppression n'améliore l'alpha, ou moins de",
        "2 items restants. Ces résultats sont EXPLORATOIRES.",
        "",
    ]

    for _, row in summary.iterrows():
        dim = row["dimension"]
        label = row["label"]
        statut = row.get("statut", "")
        lines.append(f"Dimension : {dim} — {label}  [{statut}]")
        lines.append(f"  Items initiaux     : {row['n_items']}")
        lines.append(
            f"  Alpha initial      : {row['alpha_initial'] if pd.notna(row['alpha_initial']) else 'N/A'}"
        )
        lines.append(
            f"  Alpha optimisé     : {row['alpha_optimise'] if pd.notna(row['alpha_optimise']) else 'N/A'}"
        )
        gain_str = (
            f"{row['gain_alpha']:+.3f}" if pd.notna(row["gain_alpha"]) else "N/A"
        )
        lines.append(f"  Gain alpha (Δ)     : {gain_str}")

        # Items signalés (r.drop faible) — toutes dimensions
        items_sig = row.get("items_signales", "") or ""
        if items_sig:
            lines.append(f"  Items r.drop faible (signalés, non supprimés si α acceptable) :")
            for c in items_sig.split(";"):
                lines.append(f"    • {c.strip()}")
        else:
            lines.append("  Items r.drop faible : aucun")

        # Items retirés (suppression effective — uniquement si α < seuil)
        if row["n_items_retires"] == 0:
            lines.append("  → Aucun item retiré")
        else:
            lines.append(f"  Items retirés ({row['n_items_retires']}) :")
            dim_trace = trace[trace["dimension"] == dim].sort_values("step")
            for _, t in dim_trace.iterrows():
                rdrop_str = (
                    f"{t['rdrop_at_removal']:.3f}"
                    if pd.notna(t["rdrop_at_removal"])
                    else "N/A"
                )
                lines.append(
                    f"    Étape {int(t['step'])} : {t['code']}"
                    f"  (r.drop={rdrop_str}"
                    f", α {t['alpha_before']} → {t['alpha_after']})"
                )
        lines.append("")

    lines.append("=" * 70)

    # Avertissement inversion : dimensions avec alpha initial négatif
    neg = summary[summary["alpha_initial"].notna() & (summary["alpha_initial"] < 0)]
    if not neg.empty:
        lines.append("")
        lines.append("⚠  AVERTISSEMENT — Dimensions avec alpha initial négatif :")
        lines.append("   Cause probable : inversion incorrecte dans INVERT_SHORT (config.py).")
        lines.append("   Un alpha négatif signifie que des items vont dans la direction opposée.")
        for _, row in neg.iterrows():
            lines.append(f"   • {row['dimension']} ({row['label']}) : α_initial = {row['alpha_initial']}")
        lines.append("")

    lines.append(
        "Note : items_dropped.csv (seuil fixe r.drop) et ce rapport (greedy exploratoire)"
        " sont deux méthodes distinctes et peuvent lister des items différents."
    )
    lines.append("=" * 70)

    report_path = outdir / "exploratory_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Rapport exploratoire : {report_path.name}")


def _export_numeric_questionnaire_long(df_long: pd.DataFrame, out_path: Path) -> None:
    """
    Exporte les réponses item-level recodées en numérique pour audit.

    Le fichier conserve :
    - la réponse texte d'origine ;
    - la valeur numérique brute avant inversion ;
    - l'indicateur `item_inverse` ;
    - la valeur finale `reponse_num` utilisée dans les analyses.
    """
    if df_long is None or df_long.empty:
        pd.DataFrame().to_csv(out_path, index=False, encoding="utf-8")
        return

    export = df_long.copy()
    export = export[export["dimension"].notna()].copy()
    export["reponse_num_finale"] = pd.to_numeric(export["reponse_num"], errors="coerce")
    export["reponse_num_brut"] = pd.to_numeric(export["reponse_num_brut"], errors="coerce")

    keep_cols = [
        "Participant",
        "Groupe",
        "Session",
        "Scenario",
        "Modalite",
        "Role",
        "dimension",
        "code_short",
        "code",
        "question",
        "reponse",
        "reponse_num_brut",
        "item_inverse",
        "reponse_num_finale",
    ]
    export = export[[c for c in keep_cols if c in export.columns]].copy()
    export.to_csv(out_path, index=False, encoding="utf-8")


def run_global(data_file: str, outdir: str, apply_pruning: bool = False,
               alpha_threshold: float = ALPHA_ACCEPTABILITY_THRESHOLD):
    """Pipeline global : fiabilité, descriptifs, rôles, plots, pruning."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Lecture de {data_file}...")
    raw = read_survey(data_file)

    print("[INFO] Extraction du profil participant (G3) et des commentaires libres...")
    profile_df = extract_participant_profile(raw)
    profile_summary_df, profile_counts_df = summarize_participant_profile(profile_df)
    comments_df, comments_theme_summary_df = build_free_comments(profile_df)

    profile_df.to_csv(outdir / "participant_profile_responses.csv", index=False, encoding="utf-8")
    profile_summary_df.to_csv(outdir / "participant_profile_summary.csv", index=False, encoding="utf-8")
    profile_counts_df.to_csv(outdir / "participant_profile_category_counts.csv", index=False, encoding="utf-8")
    comments_df.to_csv(outdir / "free_comments_long.csv", index=False, encoding="utf-8")
    comments_theme_summary_df.to_csv(outdir / "free_comments_theme_summary.csv", index=False, encoding="utf-8")

    print("[INFO] Transformation wide → long...")
    long = make_long_survey(raw)
    numeric_long_path = outdir / "questionnaire_reponses_numeriques_long.csv"
    _export_numeric_questionnaire_long(long, numeric_long_path)
    print(f"  → Export numérique item-level : {numeric_long_path.name}")

    # Fiabilité
    print("[INFO] Calcul Cronbach alpha par dimension...")
    alpha_table = cronbach_by_dimension(long)
    item_table = item_stats_by_dimension(long)

    alpha_table.to_csv(outdir / "cronbach_alpha_questionnaire.csv", index=False, encoding="utf-8")
    item_table.to_csv(outdir / "stats_items_questionnaire.csv", index=False, encoding="utf-8")

    # Descriptifs + rôles
    print("[INFO] Descriptifs par dimension + synthèse par rôle...")
    desc_dim = descriptives_by_dimension(long)
    role_tbl = summary_by_role(long)

    desc_dim.to_csv(outdir / "desc_dim_questionnaire.csv", index=False, encoding="utf-8")
    role_tbl.to_csv(outdir / "stats_par_dimension_role_perf.csv", index=False, encoding="utf-8")

    # Plots
    print("[INFO] Génération des plots PDF...")
    export_item_distributions(long, outdir)
    export_role_means(long, outdir)

    # Pruning (méthode 1 : seuil fixe r.drop < RDROP_THRESHOLD)
    # Attention : cette méthode ne supprime PAS les dimensions où TOUS les items
    # sont sous le seuil (il resterait 0 items). Ces dimensions sont signalées séparément.
    print("[INFO] Item pruning (r.drop, seuil fixe)...")
    pruning_res = prune_by_rdrop(long, rdrop_threshold=RDROP_THRESHOLD,
                                 alpha_threshold=alpha_threshold)
    # Exporte TOUS les items signalés (r.drop faible), avec colonne `supprime`
    pruning_res["all_flagged"].to_csv(
        outdir / "items_signales.csv", index=False, encoding="utf-8"
    )
    n_dropped = len(pruning_res["items_to_drop"])
    n_flagged = len(pruning_res["all_flagged"])
    print(f"  → {n_flagged} item(s) signalé(s) (r.drop < {RDROP_THRESHOLD}) → items_signales.csv")
    if n_dropped > 0:
        print(f"  → {n_dropped} item(s) effectivement supprimé(s) (α < {alpha_threshold})")
    else:
        print(f"  → Aucun item supprimé (toutes les dimensions ont α ≥ {alpha_threshold} ou pas assez d'items)")

    # Avertissement : dimensions entièrement défaillantes (alpha négatif probable)
    # Ces dimensions ne peuvent pas être corrigées par simple seuillage.
    # Cause fréquente : configuration d'inversion incorrecte dans config.py (INVERT_SHORT).
    if pruning_res["dims_all_bad"]:
        print(
            f"\n  ⚠  AVERTISSEMENT — Dimensions critiques (tous les items sous le seuil r.drop) :"
        )
        for dim in pruning_res["dims_all_bad"]:
            row = pruning_res["dim_info"].loc[dim]
            print(
                f"     {dim} : {int(row['n_bad'])}/{int(row['n_items'])} items"
                f" avec r.drop < {RDROP_THRESHOLD}. Alpha probablement négatif."
                f" Vérifiez INVERT_SHORT dans config.py."
            )
        print()

    print("  ─── Méthode 2 : suppression itérative exploratoire (greedy alpha-max) ───")

    # Analyse exploratoire : suppression itérative pour maximiser alpha
    # - items_signales.csv : TOUS les items r.drop < seuil (signalement universel)
    # - exploratory_*.csv  : suppression greedy uniquement si α < alpha_threshold
    print("[INFO] Analyse exploratoire (suppression itérative d'items)...")
    exp_summary, exp_trace = exploratory_prune_all(
        long, alpha_threshold=alpha_threshold, rdrop_threshold=RDROP_THRESHOLD
    )
    exp_summary, exp_trace = _apply_manual_exploratory_pruning_overrides(
        exp_summary,
        exp_trace,
    )

    exp_summary.to_csv(
        outdir / "exploratory_summary.csv", index=False, encoding="utf-8"
    )
    exp_trace.to_csv(
        outdir / "exploratory_trace_items.csv", index=False, encoding="utf-8"
    )

    # Rapport texte : résumé des suppressions exploratoires par dimension
    _write_exploratory_report(exp_summary, exp_trace, outdir)

    # Synthèse console
    print_exploratory_summary(exp_summary, alpha_threshold=alpha_threshold)

    # ── Application optionnelle des suppressions exploratoires ──────────────
    # Activée avec --apply-pruning. Recalcule alpha, statistiques items et
    # descriptifs sur le jeu épuré, et les exporte dans le sous-dossier pruned/.
    # Les fichiers originaux (sur tous les items) ne sont PAS modifiés.
    if apply_pruning:
        print("[INFO] Application des suppressions exploratoires (--apply-pruning)...")
        long_pruned = apply_exploratory_pruning(long, exp_summary)
        n_removed = long["code"].nunique() - long_pruned["code"].nunique()

        pruned_dir = outdir / "pruned"
        pruned_dir.mkdir(exist_ok=True)

        alpha_pruned = cronbach_by_dimension(long_pruned)
        items_pruned = item_stats_by_dimension(long_pruned)
        desc_pruned  = descriptives_by_dimension(long_pruned)
        role_pruned  = summary_by_role(long_pruned)
        pruned_numeric_long_path = pruned_dir / "questionnaire_reponses_numeriques_long_pruned.csv"
        _export_numeric_questionnaire_long(long_pruned, pruned_numeric_long_path)

        alpha_pruned.to_csv(pruned_dir / "cronbach_alpha_pruned.csv", index=False, encoding="utf-8")
        items_pruned.to_csv(pruned_dir / "stats_items_pruned.csv",    index=False, encoding="utf-8")
        desc_pruned.to_csv( pruned_dir / "desc_dim_pruned.csv",       index=False, encoding="utf-8")
        role_pruned.to_csv( pruned_dir / "stats_role_pruned.csv",     index=False, encoding="utf-8")

        # Tableau de comparaison alpha avant/après pour chaque dimension
        compare = exp_summary[["dimension", "label", "n_items", "alpha_initial",
                                "alpha_optimise", "gain_alpha", "n_items_retires",
                                "items_retires"]].copy()
        compare = compare.rename(columns={
            "alpha_initial":  "alpha_avant",
            "alpha_optimise": "alpha_apres",
        })
        compare.to_csv(pruned_dir / "alpha_comparison.csv", index=False, encoding="utf-8")

        # Affichage console de la comparaison
        print(f"\n  Items supprimés : {n_removed} code(s) unique(s) retiré(s)")
        print(f"  ─── Comparaison alpha avant / après pruning exploratoire ───")
        print(f"  {'Dim':<6} {'α avant':<10} {'α après':<10} {'Δα':<8} Items retirés")
        print(f"  {'─'*6} {'─'*10} {'─'*10} {'─'*8} {'─'*30}")
        for _, row in compare.iterrows():
            av = f"{row['alpha_avant']:.3f}"  if pd.notna(row["alpha_avant"])  else "N/A"
            ap = f"{row['alpha_apres']:.3f}"  if pd.notna(row["alpha_apres"])  else "N/A"
            ga = f"{row['gain_alpha']:+.3f}"  if pd.notna(row["gain_alpha"])   else "N/A"
            it = row["items_retires"] if row["items_retires"] else "—"
            # Tronque les codes longs pour l'affichage
            it_short = "; ".join(
                c.split("[")[-1].rstrip("]").split(".")[0]
                if "[" in c else c.split(".")[-1]
                for c in it.split(";")
            ) if it != "—" else "—"
            print(f"  {row['dimension']:<6} {av:<10} {ap:<10} {ga:<8} {it_short}")
        print(f"\n  → Sorties pruned dans : {pruned_dir}")

        # ── Recalcul des scores par participant avec les items prunés ──────
        # Reconstruit le long format scénario (avec Session, Scenario, Modalite)
        # et retire les mêmes items, puis recalcule les scores moyens.
        print("[INFO] Recalcul des scores par participant avec items prunés...")
        codes_to_remove: set = set()
        for items_str in exp_summary["items_retires"].dropna():
            if str(items_str).strip():
                for code in str(items_str).split(";"):
                    code = code.strip()
                    if code:
                        codes_to_remove.add(code)

        long_scenario = _build_long(raw)
        if codes_to_remove:
            long_scenario_pruned = long_scenario[
                ~long_scenario["item_col"].isin(codes_to_remove)
            ].copy()
        else:
            long_scenario_pruned = long_scenario.copy()

        scores_pruned = _compute_scores(long_scenario_pruned)
        scores_path = pruned_dir / "scores_dimension_par_participant_pruned.csv"
        scores_pruned.to_csv(scores_path, index=False, encoding="utf-8")
        print(f"  → Scores prunés : {scores_path.name}")
        print(f"  → Export numérique pruné : {pruned_numeric_long_path.name}")
        print(f"    ({len(codes_to_remove)} item(s) retiré(s), "
              f"{scores_pruned['n_items'].min()}-{scores_pruned['n_items'].max()} items/dim)\n")

    print(f"[OK] Analyse globale terminée. Sorties dans : {outdir}")


def main():
    ap = argparse.ArgumentParser(
        description="Analyse des questionnaires TMS / Cohésion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="Chemin vers le fichier questionnaire (xlsx ou csv)")
    ap.add_argument("--out", required=True, help="Dossier de sortie")
    ap.add_argument(
        "--mode",
        choices=["global", "scenario", "all"],
        default="all",
        help="Mode d'analyse : global (questionnaire), scenario (Scénario×Modalité), ou all (défaut)",
    )
    ap.add_argument(
        "--alpha-threshold",
        type=float,
        default=ALPHA_ACCEPTABILITY_THRESHOLD,
        metavar="SEUIL",
        help=(
            f"Seuil d'acceptabilité de l'alpha de Cronbach (défaut : {ALPHA_ACCEPTABILITY_THRESHOLD}). "
            "La suppression exploratoire n'est déclenchée que pour les dimensions "
            "dont l'alpha initial est inférieur à ce seuil."
        ),
    )
    ap.add_argument(
        "--apply-pruning",
        action="store_true",
        default=False,
        help=(
            "Applique les suppressions exploratoires (greedy alpha-max) et recalcule "
            "alpha/items/descriptifs sur le jeu épuré. "
            "Les résultats sont exportés dans pruned/ sans modifier les fichiers principaux."
        ),
    )
    args = ap.parse_args()

    if args.mode in ("global", "all"):
        out_global = Path(args.out) / "global" if args.mode == "all" else Path(args.out)
        run_global(args.data, str(out_global),
                   apply_pruning=args.apply_pruning,
                   alpha_threshold=args.alpha_threshold)

    if args.mode in ("scenario", "all"):
        out_scenario = Path(args.out) / "analyse" if args.mode == "all" else Path(args.out)
        run_scenario_modalite(args.data, str(out_scenario))

    print("\n[DONE] Pipeline questionnaire terminé.")


if __name__ == "__main__":
    main()
