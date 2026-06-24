#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_performance.py — Évalue la performance des groupes (M1..M4 + score final).

Parcourt automatiquement les dossiers performance_task/performance_VR /
performance_task/performance_PC × S1 / S2 sous le répertoire de résultats,
ou accepte des chemins explicites.

Examples
--------
# Évaluer toutes les modalités et scénarios :
python run_performance.py --results-dir ../results

# VR uniquement :
python run_performance.py --results-dir ../results --modality VR

# PC, scénario S2 :
python run_performance.py --results-dir ../results --modality PC --scenario S2

# Passer des dossiers explicites :
python run_performance.py --roots ../results/performance_task/performance_VR/S1 ../results/performance_task/performance_PC/S2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import subprocess

SCRIPT_DIR = Path(__file__).resolve().parent
SCORE_SCRIPT = SCRIPT_DIR / "evaluation_performance_scenario" / "score_groupes.py"


def _resolve_path(p: str | None) -> str | None:
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())


def discover_roots(results_dir: Path, modality: str = "all", scenario: str = "all"):
    """
    Discover performance root directories under results_dir.

    Layout attendu (préféré) :
        results_dir/performance_task/performance_VR/S1/
        results_dir/performance_task/performance_VR/S2/
        results_dir/performance_task/performance_PC/S1/
        results_dir/performance_task/performance_PC/S2/

    Compatibilité legacy :
        results_dir/performance_VR/S1/
        results_dir/performance_VR/S2/
        results_dir/performance_PC/S1/
        results_dir/performance_PC/S2/
    """
    modalities = ["VR", "PC"] if modality == "all" else [modality.upper()]
    scenarios = ["S1", "S2"] if scenario == "all" else [scenario.upper()]
    roots = []
    base_dirs = [results_dir / "performance_task", results_dir]
    for m in modalities:
        for s in scenarios:
            for base_dir in base_dirs:
                p = base_dir / f"performance_{m}" / s
                if p.is_dir() and p not in roots:
                    roots.append(p)
                    break
    return roots


def infer_default_out(results_dir: str | None, roots: list[str]) -> str:
    """
    Détermine un chemin de sortie par défaut cohérent avec le layout
    performance_task.
    """
    if results_dir:
        return str((Path(_resolve_path(results_dir)) / "performance_task" / "recap_scores_all.csv").resolve())

    for root_str in roots:
        root = Path(root_str)
        for parent in [root] + list(root.parents):
            parent_name = parent.name.lower()
            if parent_name == "performance_task":
                return str((parent / "recap_scores_all.csv").resolve())
            if parent_name in {"performance_pc", "performance_vr"}:
                return str((parent.parent / "performance_task" / "recap_scores_all.csv").resolve())

    return str(Path("recap_scores_all.csv").resolve())


def main():
    ap = argparse.ArgumentParser(
        description="Évalue la performance de tous les groupes (M1–M4 + score final).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--results-dir",
        type=str,
        help="Répertoire results/ contenant performance_task/performance_VR/ et performance_task/performance_PC/ (fallback legacy supporté)",
    )
    source.add_argument(
        "--roots",
        nargs="+",
        type=str,
        help="Chemins explicites vers les dossiers de groupes",
    )

    ap.add_argument("--modality", choices=["VR", "PC", "all"], default="all",
                     help="Filtrer par modalité (défaut : all)")
    ap.add_argument("--scenario", choices=["S1", "S2", "all"], default="all",
                     help="Filtrer par scénario (défaut : all)")
    ap.add_argument(
        "--out",
        default=None,
        help="CSV de sortie (défaut: results/performance_task/recap_scores_all.csv si --results-dir)",
    )
    ap.add_argument("--tol", type=float, default=0.8, help="Tolérance (m) pour M3")
    ap.add_argument("--tref", type=float, default=1500.0, help="Temps de référence (s) pour M4")

    args = ap.parse_args()

    if args.roots:
        roots = [_resolve_path(r) for r in args.roots if r]
    else:
        resolved_results_dir = Path(_resolve_path(args.results_dir))
        roots_list = discover_roots(
            resolved_results_dir, modality=args.modality, scenario=args.scenario
        )
        if not roots_list:
            print(f"[WARN] Aucun dossier performance trouvé dans {resolved_results_dir}")
            print(f"       (modality={args.modality}, scenario={args.scenario})")
            sys.exit(1)
        roots = [str(r.resolve()) for r in roots_list]
        print(f"[INFO] results_dir resolved to: {resolved_results_dir}")
        print(f"[INFO] {len(roots)} dossier(s) détecté(s) :")
        for r in roots:
            print(f"       - {r}")

    out_csv = _resolve_path(args.out) if args.out else infer_default_out(args.results_dir, roots)

    cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--roots", *roots,
        "--out", out_csv,
        "--tol", str(args.tol),
        "--tref", str(args.tref),
    ]

    print(f"[INFO] out_csv resolved to: {out_csv}")

    print(f"\n{'='*60}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
