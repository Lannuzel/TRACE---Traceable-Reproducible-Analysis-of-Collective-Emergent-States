#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario S2 (Revit) — M1 uniquement : Rule conformity (consignes).
On compare un fichier participant vs une solution/correction Revit.

Hypothèses (cohérentes avec tes exports) :
- Solution et participant ont : CenterX_m, CenterY_m, CenterZ_m, SizeX_m, SizeY_m, SizeZ_m, ElementName
- On matche spatialement chaque objet attendu (solution) à un objet participant (Hungarian ou greedy),
  puis on vérifie les contraintes "consignes" (type/forme/dimensions/épaisseur).

Sorties :
- JSON : règles globales + score M1 + diagnostics
- CSV  : diagnostic par réservation attendue (match, deltas, pass/fail)
- PNG  : overlay optionnel (XZ)

Usage :
python eval_consignes_s2_reservations.py ^
  --participant D:\...\bim057.csv ^
  --solution    D:\...\correction_s2.csv ^
  --pos_tol     0.25 ^
  --size_abs    0.01 ^
  --size_rel    0.05 ^
  --out_json    consignes_report.json ^
  --out_csv     consignes_diag.csv ^
  --plot        consignes_overlay.png
"""

import argparse, os, glob, json, re
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------- IO helpers --------------------------

def resolve_path_or_glob(p: str, pick: str = "newest") -> str:
    if os.path.isfile(p):
        return p
    m = glob.glob(p)
    if not m:
        raise FileNotFoundError(f"Aucun fichier pour : {p}")
    if len(m) == 1 or pick == "first":
        return m[0]
    m.sort(key=lambda s: os.path.getmtime(s), reverse=True)
    return m[0]

def read_csv_semicolon(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())

REQ_COLS = ["CenterX_m","CenterY_m","CenterZ_m","SizeX_m","SizeY_m","SizeZ_m","ElementName"]

def load_revit_export(path: str) -> pd.DataFrame:
    df = normalize_cols(read_csv_semicolon(path))
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {os.path.basename(path)}: {missing}. "
                         f"Colonnes présentes: {df.columns.tolist()}")
    # cast numerics
    for c in ["CenterX_m","CenterY_m","CenterZ_m","SizeX_m","SizeY_m","SizeZ_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ElementName"] = df["ElementName"].astype(str)
    df = df.dropna(subset=["CenterX_m","CenterY_m","CenterZ_m"])
    return df.reset_index(drop=True)

# -------------------------- Parsing "consigne" tokens --------------------------

_HOST_PAT = re.compile(r"\b(SOL|MUR)\b", re.IGNORECASE)
_SHAPE_PAT = re.compile(r"\b(CIRCULAIRE|RECTANGULAIRE)\b", re.IGNORECASE)

def parse_host_shape(name: str) -> Tuple[str, str]:
    """
    Retourne (host, shape) parmi:
    host ∈ {"SOL","MUR","UNK"} ; shape ∈ {"CIRCULAIRE","RECTANGULAIRE","UNK"}
    """
    s = (name or "").upper()
    mh = _HOST_PAT.search(s)
    ms = _SHAPE_PAT.search(s)
    host = mh.group(1).upper() if mh else "UNK"
    shape = ms.group(1).upper() if ms else "UNK"
    return host, shape

# -------------------------- Matching (solution -> participant) --------------------------

def pairwise_dist_3d(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    # S: (nS,3), P:(nP,3) => (nS,nP)
    return np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))

def greedy_match(D: np.ndarray) -> List[Tuple[int,int,float]]:
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    used_s, used_p = set(), set()
    pairs = []
    for si, pi in order:
        si = int(si); pi = int(pi)
        if si in used_s or pi in used_p:
            continue
        pairs.append((si, pi, float(D[si, pi])))
        used_s.add(si); used_p.add(pi)
        if len(used_s) == D.shape[0] or len(used_p) == D.shape[1]:
            break
    return pairs

def hungarian_match(D: np.ndarray) -> List[Tuple[int,int,float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        if D.size == 0:
            return []
        nS, nP = D.shape
        if nS == nP:
            r, c = linear_sum_assignment(D)
            return [(int(ri), int(ci), float(D[ri, ci])) for ri, ci in zip(r, c)]
        n = max(nS, nP)
        big = float(D.max() * 1000.0) if D.size else 1e9
        C = np.full((n, n), big, dtype=float)
        C[:nS, :nP] = D
        r, c = linear_sum_assignment(C)
        out = []
        for ri, ci in zip(r, c):
            if ri < nS and ci < nP:
                out.append((int(ri), int(ci), float(D[ri, ci])))
        return out
    except Exception:
        return greedy_match(D)

# -------------------------- Rule checks (per object) --------------------------

def within_tol(val: float, ref: float, abs_tol: float, rel_tol: float) -> bool:
    if np.isnan(val) or np.isnan(ref):
        return False
    return abs(val - ref) <= max(abs_tol, rel_tol * abs(ref))

def plan_dims_ok(p: pd.Series, s: pd.Series, shape_expected: str,
                 abs_tol: float, rel_tol: float) -> Tuple[bool, Dict[str, float]]:
    px, py = float(p["SizeX_m"]), float(p["SizeY_m"])
    sx, sy = float(s["SizeX_m"]), float(s["SizeY_m"])

    # Pour "circulaire", X≈Y ; on compare par dimension triée (robuste aux permutations)
    if shape_expected == "CIRCULAIRE":
        p_sorted = sorted([px, py])
        s_sorted = sorted([sx, sy])
        ok0 = within_tol(p_sorted[0], s_sorted[0], abs_tol, rel_tol)
        ok1 = within_tol(p_sorted[1], s_sorted[1], abs_tol, rel_tol)
        return bool(ok0 and ok1), {
            "p_size_min": p_sorted[0], "p_size_max": p_sorted[1],
            "s_size_min": s_sorted[0], "s_size_max": s_sorted[1],
        }

    # Rectangulaire: on accepte permutation X/Y (très utile en Revit)
    ok_xy = within_tol(px, sx, abs_tol, rel_tol) and within_tol(py, sy, abs_tol, rel_tol)
    ok_yx = within_tol(px, sy, abs_tol, rel_tol) and within_tol(py, sx, abs_tol, rel_tol)
    return bool(ok_xy or ok_yx), {
        "p_size_x": px, "p_size_y": py,
        "s_size_x": sx, "s_size_y": sy,
        "perm_used": ("xy" if ok_xy else ("yx" if ok_yx else "none"))
    }

def thickness_ok(p: pd.Series, s: pd.Series, abs_tol: float, rel_tol: float) -> Tuple[bool, Dict[str, float]]:
    pz = float(p["SizeZ_m"]); sz = float(s["SizeZ_m"])
    ok = within_tol(pz, sz, abs_tol, rel_tol)
    return bool(ok), {"p_thickness": pz, "s_thickness": sz, "delta": abs(pz - sz)}

# -------------------------- Plot --------------------------

def make_plot_xz(S: pd.DataFrame, P: pd.DataFrame, out_png: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9,5), dpi=130)
    ax.scatter(S["CenterX_m"], S["CenterZ_m"], s=55, marker="o", alpha=0.85, label="Solution")
    ax.scatter(P["CenterX_m"], P["CenterZ_m"], s=55, marker="x", alpha=0.90, label="Participant")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CenterX_m"); ax.set_ylabel("CenterZ_m")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="S2 — M1 Rule conformity (consignes) sur exports Revit.")
    ap.add_argument("--participant", required=True, help="CSV participant")
    ap.add_argument("--solution", required=True, help="CSV solution/correction")
    ap.add_argument("--pos_tol", type=float, default=0.25, help="Tolérance de matching (m) centre-à-centre")
    ap.add_argument("--size_abs", type=float, default=0.01, help="Tolérance absolue dimensions (m)")
    ap.add_argument("--size_rel", type=float, default=0.05, help="Tolérance relative dimensions (fraction)")
    ap.add_argument("--allow_extra", type=int, default=0, help="Nombre d'objets en trop autorisés")
    ap.add_argument("--assignment", choices=["hungarian","greedy"], default="hungarian")
    ap.add_argument("--out_json", default="consignes_report.json")
    ap.add_argument("--out_csv", default="consignes_diag.csv")
    ap.add_argument("--plot", default=None, help="PNG overlay XZ (optionnel)")
    ap.add_argument("--pick", choices=["newest","first"], default="newest")
    args = ap.parse_args()

    sol_path = resolve_path_or_glob(args.solution, "first")
    par_path = resolve_path_or_glob(args.participant, args.pick)

    S = load_revit_export(sol_path)
    P = load_revit_export(par_path)

    S_xyz = S[["CenterX_m","CenterY_m","CenterZ_m"]].to_numpy(float)
    P_xyz = P[["CenterX_m","CenterY_m","CenterZ_m"]].to_numpy(float)

    D = pairwise_dist_3d(S_xyz, P_xyz) if (len(S) and len(P)) else np.zeros((len(S), len(P)), float)

    pairs = hungarian_match(D) if args.assignment == "hungarian" else greedy_match(D)

    # Pour chaque attendu (solution), garder au plus 1 match
    match_for_s: Dict[int, Tuple[int, float]] = {}
    used_p = set()
    for si, pi, dist in pairs:
        if si in match_for_s or pi in used_p:
            continue
        match_for_s[si] = (pi, dist)
        used_p.add(pi)

    # Diagnostics par item attendu
    rows = []
    n_present = 0
    n_host_ok = 0
    n_shape_ok = 0
    n_plan_ok = 0
    n_thick_ok = 0

    for si in range(len(S)):
        srow = S.iloc[si]
        exp_host, exp_shape = parse_host_shape(srow["ElementName"])

        pi, dist = match_for_s.get(si, (None, None))
        present = (pi is not None) and (dist is not None) and (dist <= args.pos_tol)

        if present:
            n_present += 1
            prow = P.iloc[int(pi)]
            got_host, got_shape = parse_host_shape(prow["ElementName"])

            host_ok = (exp_host == "UNK") or (got_host == exp_host)
            shape_ok = (exp_shape == "UNK") or (got_shape == exp_shape)

            plan_ok, plan_diag = plan_dims_ok(
                prow, srow, shape_expected=(exp_shape if exp_shape != "UNK" else "RECTANGULAIRE"),
                abs_tol=args.size_abs, rel_tol=args.size_rel
            )
            thick_ok, thick_diag = thickness_ok(prow, srow, abs_tol=args.size_abs, rel_tol=args.size_rel)

            n_host_ok += int(host_ok)
            n_shape_ok += int(shape_ok)
            n_plan_ok += int(plan_ok)
            n_thick_ok += int(thick_ok)
        else:
            host_ok = shape_ok = plan_ok = thick_ok = False
            plan_diag = {}
            thick_diag = {}
            got_host = got_shape = "NA"

        rows.append({
            "s_index": si,
            "s_ElementName": srow["ElementName"],
            "s_host": exp_host,
            "s_shape": exp_shape,
            "s_CenterX_m": float(srow["CenterX_m"]),
            "s_CenterY_m": float(srow["CenterY_m"]),
            "s_CenterZ_m": float(srow["CenterZ_m"]),
            "s_SizeX_m": float(srow["SizeX_m"]),
            "s_SizeY_m": float(srow["SizeY_m"]),
            "s_SizeZ_m": float(srow["SizeZ_m"]),
            "p_index": -1 if pi is None else int(pi),
            "p_ElementName": "" if pi is None else str(P.iloc[int(pi)]["ElementName"]),
            "p_host": got_host,
            "p_shape": got_shape,
            "match_dist_m": None if dist is None else float(dist),
            "present_ok": bool(present),
            "host_ok": bool(host_ok),
            "shape_ok": bool(shape_ok),
            "plan_dims_ok": bool(plan_ok),
            "thickness_ok": bool(thick_ok),
            **{f"plan_{k}": v for k, v in plan_diag.items()},
            **{f"thick_{k}": v for k, v in thick_diag.items()},
        })

    diag_df = pd.DataFrame(rows)

    # Objets en trop = participants non utilisés dans un match "present"
    matched_p = set(diag_df.loc[diag_df["present_ok"], "p_index"].tolist())
    matched_p.discard(-1)
    n_extra = int(len(P) - len(matched_p))

    # --------- Règles globales (consignes) ---------
    # R1 : toutes les réservations attendues présentes (matching spatial)
    R1 = (n_present == len(S))
    # R2 : pas (ou peu) d'extra
    R2 = (n_extra <= args.allow_extra)
    # R3 : host SOL/MUR correct pour toutes les présentes
    R3 = bool(diag_df.loc[diag_df["present_ok"], "host_ok"].all()) if n_present > 0 else False
    # R4 : forme CIRC/RECT correcte
    R4 = bool(diag_df.loc[diag_df["present_ok"], "shape_ok"].all()) if n_present > 0 else False
    # R5 : dimensions plan correctes
    R5 = bool(diag_df.loc[diag_df["present_ok"], "plan_dims_ok"].all()) if n_present > 0 else False
    # R6 : épaisseur correcte
    R6 = bool(diag_df.loc[diag_df["present_ok"], "thickness_ok"].all()) if n_present > 0 else False

    rules = {
        "R1_all_expected_present": R1,
        "R2_no_extra_objects": R2,
        "R3_host_type_SOL_vs_MUR": R3,
        "R4_shape_CIRC_vs_RECT": R4,
        "R5_plan_dimensions": R5,
        "R6_thickness_equals_host": R6,
    }

    # --------- Score M1 (0–100) : contraintes atomiques ---------
    # On compte des contraintes "par réservation attendue" :
    # - présence
    # - host
    # - shape
    # - plan dims
    # - thickness
    total_constraints = int(len(S) * 5)
    satisfied = 0
    satisfied += int(diag_df["present_ok"].sum())
    satisfied += int(diag_df["host_ok"].sum())
    satisfied += int(diag_df["shape_ok"].sum())
    satisfied += int(diag_df["plan_dims_ok"].sum())
    satisfied += int(diag_df["thickness_ok"].sum())

    # Option : pénaliser les extras (consigne implicite "ne pas créer de réservations inutiles")
    # Ici on pénalise au niveau global via R2, sans modifier le score atomique.
    M1 = 100.0 * (satisfied / total_constraints) if total_constraints > 0 else 0.0

    report = {
        "meta": {
            "scenario": "S2_reservations",
            "participant": os.path.abspath(par_path),
            "solution": os.path.abspath(sol_path),
            "pos_tol_m": args.pos_tol,
            "size_abs_m": args.size_abs,
            "size_rel": args.size_rel,
            "allow_extra": args.allow_extra,
            "assignment": args.assignment,
        },
        "counts": {
            "n_expected": int(len(S)),
            "n_participant": int(len(P)),
            "n_present": int(n_present),
            "n_extra": int(n_extra),
        },
        "rules": rules,
        "M1_rule_conformity_score_0_100": float(M1),
        "constraints": {
            "total": int(total_constraints),
            "satisfied": int(satisfied),
            "details": {
                "present_ok": int(diag_df["present_ok"].sum()),
                "host_ok": int(diag_df["host_ok"].sum()),
                "shape_ok": int(diag_df["shape_ok"].sum()),
                "plan_dims_ok": int(diag_df["plan_dims_ok"].sum()),
                "thickness_ok": int(diag_df["thickness_ok"].sum()),
            }
        }
    }

    # Save
    diag_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.plot:
        title = f"M1={M1:.1f} | present={n_present}/{len(S)} | extra={n_extra}"
        make_plot_xz(S, P, args.plot, title=title)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", os.path.abspath(args.out_json))
    print("Saved:", os.path.abspath(args.out_csv))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
