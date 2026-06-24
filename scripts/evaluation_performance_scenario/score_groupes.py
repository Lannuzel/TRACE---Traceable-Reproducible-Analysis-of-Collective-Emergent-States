#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_groupes.py — Score de performance par groupe (M1..M4 + final).

Supporte :
  - Mode single-root (--root) : un seul dossier contenant les groupes.
  - Mode multi-root  (--roots) : plusieurs dossiers
    (performance_task/performance_VR/S1, performance_task/performance_PC/S2, ...).

Pondérations : M1=0.50, M2=0.00, M3=0.50, M4=0.00 (depuis common.constants.PERF_WEIGHTS).
"""

import sys
from pathlib import Path

# Ajoute le dossier parent (scripts/) au path pour importer common
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import math
import re

from common.constants import PERF_WEIGHTS


# ---------------------------------------------------------------------------
#  Utilitaires I/O
# ---------------------------------------------------------------------------

def as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "vrai", "yes", "oui", "y", "t"}


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_first_file(dirpath: Path, candidates):
    """Retourne le premier fichier existant parmi *candidates* dans *dirpath*."""
    for name in candidates:
        p = dirpath / name
        if p.exists() and p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
#  Chargement des données d'un groupe
# ---------------------------------------------------------------------------

def load_consignes(consigne_dir: Path):
    """
    Retourne (n_rules_passed, n_rules_total).
    Supporte JSON (consignes_report.json) ou CSV (R1..R7).
    """
    json_path = find_first_file(consigne_dir, ["consignes_report.json", "consigne.json"])
    if json_path:
        data = read_json(json_path)
        n_passed = data.get("n_rules_passed")
        n_total = data.get("n_rules_total")
        if isinstance(n_passed, int) and isinstance(n_total, int) and n_total > 0:
            return n_passed, n_total
        rules = data.get("rules", {})
        if isinstance(rules, dict) and rules:
            vals = [as_bool(v) for v in rules.values()]
            return sum(vals), len(vals)

    csv_path = find_first_file(consigne_dir, ["consignes_report.csv", "consigne.csv"])
    if csv_path:
        rows = read_csv_rows(csv_path)
        if not rows:
            return 0, 0
        cols = rows[0].keys()
        rule_cols = [c for c in cols if c.strip().lower().startswith(("r1", "r2", "r3", "r4", "r5", "r6", "r7", "rule"))]
        if len(rows) > 1 and "rule" in {k.lower() for k in cols} and "ok" in {k.lower() for k in cols}:
            n_passed = sum(as_bool(r.get("ok")) for r in rows)
            return n_passed, len(rows)
        if rule_cols:
            vals = [as_bool(rows[0].get(c)) for c in rule_cols]
            return sum(vals), len(vals)

    return 0, 0


def load_performance(perf_dir: Path):
    """
    Retourne un dict {n_expected, n_placed, duration_s, n_matched, sum_distance_m}.
    Supporte JSON, CSV agrégé et CSV de distances par paire.
    """
    json_path = find_first_file(perf_dir, ["performance_summary.json", "performance.json"])
    if json_path:
        d = read_json(json_path)
        return {
            "n_expected": int(d.get("n_expected", 0) or 0),
            "n_placed": int(d.get("n_placed", 0) or 0),
            "duration_s": safe_float(d.get("duration_s", None)),
            "n_matched": int(d.get("n_matched", 0) or 0),
            "sum_distance_m": safe_float(d.get("sum_distance_m", None)),
        }

    csv_agg = find_first_file(perf_dir, ["performance_summary.csv", "performance.csv"])
    if csv_agg:
        rows = read_csv_rows(csv_agg)
        d = rows[0] if rows else {}
        return {
            "n_expected": int(safe_float(d.get("n_expected", 0)) or 0),
            "n_placed": int(safe_float(d.get("n_placed", 0)) or 0),
            "duration_s": safe_float(d.get("duration_s", None)),
            "n_matched": int(safe_float(d.get("n_matched", 0)) or 0),
            "sum_distance_m": safe_float(d.get("sum_distance_m", None)),
        }

    csv_pairs = find_first_file(perf_dir, ["matches.csv", "assignments.csv", "distances.csv"])
    if csv_pairs:
        rows = read_csv_rows(csv_pairs)
        distances = []
        for r in rows:
            for key in ("distance", "dist_m", "d"):
                if key in r and r[key] != "":
                    distances.append(safe_float(r[key], 0.0))
                    break
        return {
            "n_expected": 0,
            "n_placed": len(rows),
            "duration_s": None,
            "n_matched": len(distances),
            "sum_distance_m": sum(d for d in distances if d is not None),
        }

    return {"n_expected": 0, "n_placed": 0, "duration_s": None,
            "n_matched": 0, "sum_distance_m": None}


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------

def score_simple_plus(
    n_rules_passed, n_rules_total,
    n_placed, n_expected,
    n_matched, sum_distance_m, duration_s,
    tol_m=0.8, tref_s=1500.0,
):
    """
    M1 – Consignes (0–100) : 100 * pass/total
    M2 – Nombre   (0–100) : F1 sur le comptage (gère sur/sous-placement)
    M3 – Précision (0–100) : décroissance exponentielle 100*exp(-MAE/tol)
    M4 – Temps    (0–100) : 100*(1 - clamp(duration/tref, 0..1))

    Score_perf_tsk = pondération depuis PERF_WEIGHTS.
    """
    n_rules_passed = 0 if n_rules_passed is None else float(n_rules_passed)
    n_rules_total = 0 if n_rules_total is None else float(n_rules_total)
    n_placed = 0 if n_placed is None else int(n_placed)
    n_expected = 0 if n_expected is None else int(n_expected)
    n_matched = 0 if n_matched is None else int(n_matched)

    # M1
    M1 = 100.0 * (n_rules_passed / n_rules_total) if n_rules_total > 0 else 0.0
    M1 = max(0.0, min(100.0, M1))

    # M2 (F1 comptage)
    tp = max(n_matched, 0)
    fp = max(n_placed - tp, 0)
    fn = max(n_expected - tp, 0)
    den = 2 * tp + fp + fn
    M2 = 100.0 * (2 * tp / den) if den > 0 else 0.0
    M2 = max(0.0, min(100.0, M2))

    # M3 (exponentielle)
    if tp > 0 and sum_distance_m is not None:
        mae = float(sum_distance_m) / float(tp)
        tau = float(tol_m) if tol_m and tol_m > 0 else 1.0
        M3 = 100.0 * math.exp(-mae / tau)
    else:
        mae = None
        M3 = 0.0
    M3 = max(0.0, min(100.0, M3))

    # M4 (temps : plus rapide = meilleur, clippé au plafond)
    t_max = float(tref_s)
    if duration_s is not None:
        d = max(0.0, min(float(duration_s), t_max))
        M4 = 100.0 * (1.0 - d / t_max) if t_max > 0 else 0.0
    else:
        M4 = 0.0
    M4 = max(0.0, min(100.0, M4))

    # Score final
    w = PERF_WEIGHTS
    final = w["M1_consignes"] * M1 + w["M2_nombre"] * M2 + w["M3_precision"] * M3 + w["M4_temps"] * M4
    final = max(0.0, min(100.0, final))

    return M1, M2, M3, M4, final


# ---------------------------------------------------------------------------
#  Découverte automatique de groupes et labels
# ---------------------------------------------------------------------------

_SCEN_RE = re.compile(r"(?:^|\\|/)(S\d+)(?:\\|/|$)", re.IGNORECASE)
_MOD_RE = re.compile(r"(?:^|\\|/)(PC|VR)(?:\\|/|$)", re.IGNORECASE)


def infer_modalite_scenario(path: Path):
    """Devine (modalité, scénario) à partir du chemin."""
    s = str(path)

    scen = None
    m = _SCEN_RE.search(s)
    if m:
        scen = m.group(1).upper()

    mod = None
    sl = s.lower()
    if "performance_vr" in sl:
        mod = "VR"
    elif "performance_pc" in sl:
        mod = "PC"
    else:
        m2 = _MOD_RE.search(s)
        if m2:
            mod = m2.group(1).upper()

    return mod, scen


def iter_group_dirs(root: Path):
    """Itère sur les sous-dossiers bimXXX (ou bimXXX_N) d'un répertoire."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and re.match(r"^bim\d{3}(?:_\d+)?$", p.name, re.IGNORECASE)
    )


# ---------------------------------------------------------------------------
#  Traitement d'un groupe
# ---------------------------------------------------------------------------

def score_one_group(gdir, tol_m, tref_s, modalite="", scenario=""):
    """Score un groupe et renvoie un dict de résultats (ou None si pas de données)."""
    gdir = Path(gdir)
    cons_dir = gdir / "consigne"
    perf_dir = gdir / "performance"

    if not (cons_dir.exists() or perf_dir.exists()):
        return None

    n_rules_passed, n_rules_total = load_consignes(cons_dir)
    perf = load_performance(perf_dir)

    M1, M2, M3, M4, final = score_simple_plus(
        n_rules_passed=n_rules_passed,
        n_rules_total=n_rules_total,
        n_placed=perf.get("n_placed", 0),
        n_expected=perf.get("n_expected", 0),
        n_matched=perf.get("n_matched", 0),
        sum_distance_m=perf.get("sum_distance_m"),
        duration_s=perf.get("duration_s"),
        tol_m=tol_m,
        tref_s=tref_s,
    )

    mae = None
    if perf.get("n_matched", 0) and perf.get("sum_distance_m") is not None:
        mae = perf["sum_distance_m"] / perf["n_matched"]

    raw_duration = perf.get("duration_s")
    duration_export = min(float(raw_duration), tref_s) if raw_duration is not None else ""

    return {
        "modalite": modalite,
        "scenario": scenario,
        "groupe": gdir.name,
        "n_rules_passed": n_rules_passed,
        "n_rules_total": n_rules_total,
        "M1_consignes_%": round(M1, 2),
        "n_expected": perf.get("n_expected", 0),
        "n_placed": perf.get("n_placed", 0),
        "M2_nombre_%": round(M2, 2),
        "n_matched": perf.get("n_matched", 0),
        "sum_distance_m": round(perf.get("sum_distance_m", 0.0), 4)
            if perf.get("sum_distance_m") is not None else "",
        "MAE_m": round(mae, 4) if mae is not None else "",
        "tol_m": tol_m,
        "M3_precision_%": round(M3, 2),
        "duration_s": round(duration_export, 2) if duration_export != "" else "",
        "tref_s": tref_s,
        "M4_temps_%": round(M4, 2),
        "Score_perf_tsk": round(final, 2),
    }


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score global (PC/VR x S1/S2) à partir des dossiers résultats."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="Dossier unique contenant les sous-dossiers de groupes")
    group.add_argument(
        "--roots", nargs="+",
        help="Plusieurs dossiers roots (ex: .../VR_performance/S1 .../PC_performance/S2)",
    )
    ap.add_argument("--out", default="recap_scores.csv", help="CSV de sortie")
    ap.add_argument("--tol", type=float, default=0.8, help="Tolérance (m) pour M3")
    ap.add_argument("--tref", type=float, default=1500.0, help="Temps de référence (s) max pour M4")
    args = ap.parse_args()

    roots = [Path(args.root)] if args.root else [Path(r) for r in args.roots]

    rows_out = []
    for root in roots:
        mod_guess, scen_guess = infer_modalite_scenario(root)
        for gdir in iter_group_dirs(root):
            row = score_one_group(
                gdir, tol_m=args.tol, tref_s=args.tref,
                modalite=mod_guess or "", scenario=scen_guess or "",
            )
            if row is not None:
                rows_out.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"[OK] Récapitulatif écrit : {out_path} ({len(rows_out)} lignes)")
    else:
        print("[INFO] Aucune donnée exploitable trouvée.")


if __name__ == "__main__":
    main()
