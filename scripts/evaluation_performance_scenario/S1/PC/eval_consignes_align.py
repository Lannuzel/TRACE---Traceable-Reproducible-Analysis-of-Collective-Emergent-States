#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évalue R1, R2, R3, R4, R5, R6, R7 (R2/R5/R7 = tests d'alignement des bords).
- R3 (espacement entre rangées) et R4 (intra-rangée) sont *inférés à partir de la solution*.
- R2 = alignement de la rangée avant (Z max)   entre participant et solution
- R5 = alignement de la colonne droite (X max) entre participant et solution
- R7 = alignement de la colonne gauche (X min) entre participant et solution

Sorties :
- JSON : OK/NON + diagnostics (valeurs inférées + tolérances + deltas)
- CSV  : positions (après calibration optionnelle) + index de rangée
- PNG  : overlay (optionnel)
"""

import argparse, os, glob, re, json, math
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- IO ----------------

def resolve_path_or_glob(p: str, pick: str = "newest") -> str:
    if os.path.isfile(p): return p
    m = glob.glob(p)
    if not m: raise FileNotFoundError(f"Aucun fichier pour : {p}")
    if len(m) == 1 or pick == "first": return m[0]
    m.sort(key=lambda s: os.path.getmtime(s), reverse=True)
    return m[0]

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())

def parse_freeform(df: pd.DataFrame) -> pd.DataFrame:
    """Parse des logs 't, name, (x,y,z)' et garde l'état final (t max)."""
    recs=[]
    for _, row in df.iterrows():
        line = " | ".join(str(v) for v in row.values if pd.notna(v))
        m = re.search(
            r'(?P<t>\d+(?:[.,]\d+)?)\s*,\s*(?P<name>[^,]+),\s*\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<z>-?\d+(?:\.\d+)?)\s*\)',
            line
        )
        if m:
            recs.append({
                "time": float(m.group("t").replace(",", ".")),
                "name": m.group("name").strip(),
                "PositionX": float(m.group("x")),
                "PositionY": float(m.group("y")),
                "PositionZ": float(m.group("z")),
            })
    if not recs:
        return pd.DataFrame(columns=["time","name","PositionX","PositionY","PositionZ"])
    dff = pd.DataFrame.from_records(recs)
    tmax = dff["time"].max()
    return dff[dff["time"]==tmax].reset_index(drop=True)

def load_points(path: str, axes=("PositionX","PositionZ")) -> pd.DataFrame:
    df = normalize_cols(read_table(path))
    if set(axes).issubset(df.columns):
        out = df[list(axes)].copy()
        out["name"] = df["name"] if "name" in df.columns else ""
        return out
    parsed = parse_freeform(df)
    if parsed.empty: raise ValueError("Fichier participant non lisible (ni structuré, ni log libre).")
    out = parsed[list(axes)].copy()
    out["name"] = parsed["name"] if "name" in parsed.columns else ""
    return out

def load_solution(path: str, axes=("PositionX","PositionZ")) -> pd.DataFrame:
    df = normalize_cols(read_table(path))
    for a in axes:
        if a not in df.columns:
            raise ValueError(f"Solution: axe manquant {a}. Colonnes: {df.columns.tolist()}")
    return df[list(axes)].copy()

# -------- calibration (affine 2D : échelle + translation) --------

def pairwise_dists(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sqrt(((S[:,None,:]-P[None,:,:])**2).sum(axis=2))

def greedy_match(S: np.ndarray, P: np.ndarray) -> List[Tuple[int,int]]:
    D = pairwise_dists(S,P)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    used_s, used_p = set(), set(); pairs=[]
    for si,pi in order:
        si=int(si); pi=int(pi)
        if si in used_s or pi in used_p: continue
        pairs.append((si,pi)); used_s.add(si); used_p.add(pi)
        if len(used_s)==len(S) or len(used_p)==len(P): break
    return pairs

def hungarian_match(S: np.ndarray, P: np.ndarray) -> List[Tuple[int,int]]:
    try:
        from scipy.optimize import linear_sum_assignment
        D = pairwise_dists(S,P)
        if D.size==0: return []
        nS, nP = D.shape
        if nS==nP:
            r,c = linear_sum_assignment(D); return list(zip(r.tolist(), c.tolist()))
        n=max(nS,nP); big=(D.max()*1000) if D.size else 1e9
        C=np.full((n,n),big,float); C[:nS,:nP]=D
        r,c=linear_sum_assignment(C); mask=(r<nS)&(c<nP)
        return list(zip(r[mask].tolist(), c[mask].tolist()))
    except Exception:
        return greedy_match(S,P)

def estimate_transform(P: np.ndarray, S: np.ndarray) -> Tuple[float,float,float,float]:
    if P.shape[0] < 2: return 1.0, 0.0, 1.0, 0.0
    px, pz = P[:,0], P[:,1]
    A = np.vstack([px, np.ones_like(px)]).T
    sx, tx = np.linalg.lstsq(A, S[:,0], rcond=None)[0]
    A = np.vstack([pz, np.ones_like(pz)]).T
    sz, tz = np.linalg.lstsq(A, S[:,1], rcond=None)[0]
    return float(sx), float(tx), float(sz), float(tz)

def apply_transform(points: pd.DataFrame, axes, T: Dict[str,Any]) -> pd.DataFrame:
    sx = float(T.get("scale",{}).get("x",1.0)); sz = float(T.get("scale",{}).get("z",1.0))
    tx = float(T.get("offset",{}).get("x",0.0)); tz = float(T.get("offset",{}).get("z",0.0))
    d = points.copy()
    d[axes[0]] = sx*d[axes[0]] + tx
    d[axes[1]] = sz*d[axes[1]] + tz
    return d

def estimate_translation(P: np.ndarray, S: np.ndarray) -> Tuple[float,float]:
    """Calcule tx,tz tels que P + t ≈ S (au sens des moindres carrés)."""
    if P.size == 0:
        return 0.0, 0.0
    t = S.mean(axis=0) - P.mean(axis=0)
    return float(t[0]), float(t[1])

# def auto_calibrate(P_df: pd.DataFrame, S_df: pd.DataFrame, axes=("PositionX","PositionZ")) -> Dict[str,Any]:
#     P = P_df[list(axes)].to_numpy(float)
#     S = S_df[list(axes)].to_numpy(float)
#     if P.size == 0 or S.size == 0:
#         return {"scale":{"x":1.0,"z":1.0}, "offset":{"x":0.0,"z":0.0}}

#     tx = float(S[:,0].mean() - P[:,0].mean())
#     tz = float(S[:,1].mean() - P[:,1].mean())
#     return {"scale":{"x":1.0,"z":1.0}, "offset":{"x":tx,"z":tz}}


def auto_calibrate(P_df: pd.DataFrame, S_df: pd.DataFrame, axes=("PositionX","PositionZ"), tol_level: float = 0.25) -> Dict[str,Any]:
    """
    Translation seule, robuste aux colonnes/rangées en trop :
    on aligne les bords (min des centres) plutôt que la moyenne.
    """
    Px = P_df[axes[0]].to_numpy(float)
    Pz = P_df[axes[1]].to_numpy(float)
    Sx = S_df[axes[0]].to_numpy(float)
    Sz = S_df[axes[1]].to_numpy(float)

    if Px.size == 0 or Sx.size == 0:
        return {"scale":{"x":1.0,"z":1.0}, "offset":{"x":0.0,"z":0.0}}

    # centres de colonnes/rangées (robuste)
    cx_P = cluster_levels(Px, tol=tol_level)
    cz_P = cluster_levels(Pz, tol=tol_level)
    cx_S = cluster_levels(Sx, tol=tol_level)
    cz_S = cluster_levels(Sz, tol=tol_level)

    # fallback si clustering échoue
    if cx_P.size == 0 or cx_S.size == 0:
        tx = float(np.mean(Sx) - np.mean(Px))
    else:
        tx = float(np.min(cx_S) - np.min(cx_P))  # aligne la colonne gauche

    if cz_P.size == 0 or cz_S.size == 0:
        tz = float(np.mean(Sz) - np.mean(Pz))
    else:
        tz = float(np.min(cz_S) - np.min(cz_P))  # aligne la rangée "arrière" (Z min)

    return {"scale":{"x":1.0,"z":1.0}, "offset":{"x":tx,"z":tz}}

# ----------- inférence des espacements depuis la solution -----------

def cluster_levels(vals: np.ndarray, tol: float = 0.25) -> np.ndarray:
    v = np.sort(np.asarray(vals, float))
    if v.size == 0: return np.array([])
    clusters = [[v[0]]]; center = v[0]
    for x in v[1:]:
        if abs(x - center) <= tol:
            clusters[-1].append(x); center = np.mean(clusters[-1])
        else:
            clusters.append([x]); center = x
    centers = np.array([np.mean(c) for c in clusters], float); centers.sort()
    return centers

def infer_spacings_from_solution(S_df: pd.DataFrame, tol_level: float = 0.25) -> Dict[str,float]:
    z_centers = cluster_levels(S_df["PositionZ"].to_numpy(), tol_level)  # rangées
    x_centers = cluster_levels(S_df["PositionX"].to_numpy(), tol_level)  # colonnes
    row_spacing   = float(np.median(np.diff(z_centers))) if z_centers.size > 1 else None
    intra_spacing = float(np.median(np.diff(x_centers))) if x_centers.size > 1 else None
    return {"row_spacing": row_spacing, "intra_spacing": intra_spacing,
            "z_centers": z_centers.tolist(), "x_centers": x_centers.tolist()}

# ---------- attribution rangées (pour R1/R3/R4/R6) ----------

def assign_rows(points: pd.DataFrame, step: float, axis: str = "PositionZ", tol: float = 0.25) -> np.ndarray:
    v = points[axis].to_numpy(float)
    if len(v)==0 or step is None or step<=0: return np.full(len(points), np.nan)
    vmin=v.min(); phis=np.linspace(vmin, vmin+step, 61)
    best_phi=None; best_res=1e18
    for phi in phis:
        k=np.rint((v-phi)/step); snap=phi+k*step
        res=np.mean(np.abs(v-snap))
        if res<best_res: best_res=res; best_phi=phi
    k=np.rint((v-best_phi)/step); snap=best_phi+k*step
    return np.where(np.abs(v - snap) <= tol, k.astype(int), np.nan)

# ------------------------------ règles ------------------------------

def approx_equal(val, target, abs_tol=0.15, rel_tol=0.10) -> bool:
    if target is None: return False
    return abs(val - target) <= max(abs_tol, rel_tol*abs(target))

def rule_R1_rows_layout(P: pd.DataFrame, step_z: float, tol_align: float = 0.25):
    rows = assign_rows(P, step_z, "PositionZ", tol_align)
    n_rows = np.unique(rows[~np.isnan(rows)]).size
    return bool(n_rows >= 2), {"n_rows": int(n_rows)}

def rule_R3_row_spacing(P: pd.DataFrame, expected_step: float, tol_align: float = 0.25,
                        abs_tol: float = 0.15, rel_tol: float = 0.10):
    rows = assign_rows(P, expected_step, "PositionZ", tol_align)
    df = P.assign(row=rows).dropna(subset=["row"])
    if df["row"].nunique() < 2:
        return False, {"expected": expected_step, "observed": None}
    centroids = df.groupby("row")["PositionZ"].mean().sort_values().to_numpy()
    gaps = np.diff(centroids)
    obs = float(np.median(gaps))
    return approx_equal(obs, expected_step, abs_tol, rel_tol), {"expected": expected_step, "observed_all": gaps.tolist(), "observed_median": obs}

def rule_R4_intrarow_spacing(P: pd.DataFrame, expected_gap: float, step_z: float, tol_align: float = 0.25,
                             abs_tol: float = 0.15, rel_tol: float = 0.10):
    rows = assign_rows(P, step_z, "PositionZ", tol_align)
    df = P.assign(row=rows).dropna(subset=["row"])
    gaps=[]
    for _, g in df.groupby("row"):
        xs = np.sort(g["PositionX"].to_numpy())
        if xs.size >= 2: gaps.extend(np.diff(xs).tolist())
    if not gaps:
        return False, {"expected": expected_gap, "observed": None}
    obs = float(np.median(gaps))
    return approx_equal(obs, expected_gap, abs_tol, rel_tol), {"expected": expected_gap, "observed_all": gaps, "observed_median": obs}

def rule_R6_equal_tables_per_row(P: pd.DataFrame, step_z: float, tol_align: float = 0.25):
    rows = assign_rows(P, step_z, "PositionZ", tol_align)
    s = pd.Series(rows); counts = s.dropna().value_counts().sort_index().astype(int).to_numpy()
    if counts.size == 0:
        return False, {"counts": None}
    return bool(np.all(counts == counts[0])), {"counts": counts.tolist()}

# ----------- nouvelles règles d’alignement R2, R5, R7 -----------

def centers_from_points(df: pd.DataFrame, axis: str, tol: float = 0.25) -> np.ndarray:
    return cluster_levels(df[axis].to_numpy(), tol)

def rule_R2_align_front_row(P: pd.DataFrame, S: pd.DataFrame, tol_z: float):
    """Aligne la rangée 'avant' (Z max)."""
    cz_S = centers_from_points(S, "PositionZ", tol=tol_z)
    cz_P = centers_from_points(P, "PositionZ", tol=tol_z)
    if cz_S.size==0 or cz_P.size==0: return False, {"solution_rows": cz_S.tolist(), "participant_rows": cz_P.tolist()}
    zS_front = float(np.max(cz_S)); zP_front = float(np.max(cz_P))
    delta = abs(zP_front - zS_front)
    return bool(delta <= tol_z), {"z_solution_front": zS_front, "z_participant_front": zP_front, "delta": delta, "tol_z": tol_z}

def rule_R5_align_right_column(P: pd.DataFrame, S: pd.DataFrame, tol_x: float):
    """Aligne la colonne 'droite' (X max)."""
    cx_S = centers_from_points(S, "PositionX", tol=tol_x)
    cx_P = centers_from_points(P, "PositionX", tol=tol_x)
    if cx_S.size==0 or cx_P.size==0: return False, {"solution_cols": cx_S.tolist(), "participant_cols": cx_P.tolist()}
    xS_right = float(np.max(cx_S)); xP_right = float(np.max(cx_P))
    delta = abs(xP_right - xS_right)
    return bool(delta <= tol_x), {"x_solution_right": xS_right, "x_participant_right": xP_right, "delta": delta, "tol_x": tol_x}

def rule_R7_align_left_column(P: pd.DataFrame, S: pd.DataFrame, tol_x: float):
    cx_S = centers_from_points(S, "PositionX", tol=tol_x)
    cx_P = centers_from_points(P, "PositionX", tol=tol_x)
    if cx_S.size == 0 or cx_P.size == 0:
        return False, {
            "solution_cols": cx_S.tolist(),
            "participant_cols": cx_P.tolist(),
            "message": "Impossible d'estimer les colonnes (centres vides)."
        }
    xS_left = float(np.min(cx_S))
    xP_left = float(np.min(cx_P))
    diff = xP_left - xS_left
    ok = diff >= -tol_x
    return ok, {
        "x_solution_left": xS_left,
        "x_participant_left": xP_left,
        "diff": diff,
        "tol_x": tol_x,
        "criterion": "xP_left >= xS_left - tol_x"
    }
# ------------------------------ plot ------------------------------

def make_plot(P: pd.DataFrame, S: pd.DataFrame, out_png: str, report: dict):
    fig, ax = plt.subplots(figsize=(9,5), dpi=130)
    ax.scatter(S["PositionX"], S["PositionZ"], s=60, marker='o', label="Solution (réf.)", alpha=0.85)
    ax.scatter(P["PositionX"], P["PositionZ"], s=60, marker='x', label="Participant", alpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PositionX (m)"); ax.set_ylabel("PositionZ (m)")
    txt = [f"{k}: {'OK' if v else 'NON'}" for k,v in report["rules"].items()]
    ax.text(1.02, 1.0, "\n".join(txt), transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", alpha=0.15))
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

# ------------------------------ main ------------------------------

def main():
    ap = argparse.ArgumentParser(description="R1..R7 avec inférence et alignements de bords (R2/R5/R7).")
    ap.add_argument("--participant", required=True, help="CSV/XLSX ou motif glob")
    ap.add_argument("--solution",    required=True, help="Fichier solution (XLSX/CSV)")
    ap.add_argument("--no_calib", action="store_true", help="Désactiver la calibration affine P->S")
    ap.add_argument("--tol_level", type=float, default=0.25, help="Tolérance pour regrouper niveaux (solution & bords)")
    ap.add_argument("--row_align", type=float, default=0.25, help="Tolérance d'accrochage aux rangées (participant)")
    ap.add_argument("--abs_tol",   type=float, default=0.15)
    ap.add_argument("--rel_tol",   type=float, default=0.10)
    ap.add_argument("--out_json",  default="consignes_report.json")
    ap.add_argument("--out_csv",   default="consignes_diag.csv")
    ap.add_argument("--plot",      default=None, help="PNG overlay (optionnel)")
    ap.add_argument("--allow_scale", action="store_true",
                help="Autoriser la mise à l'échelle dans la calibration (VR). Par défaut: translation seule (Revit).")
    ap.add_argument("--pick",      choices=["newest","first"], default="newest")
    args = ap.parse_args()

    # lecture
    S = load_solution(resolve_path_or_glob(args.solution, "first"))
    P = load_points(resolve_path_or_glob(args.participant, args.pick))

    # calibration optionnelle
    if not args.no_calib:
        T = auto_calibrate(P, S, tol_level=args.tol_level)
        print("[Calibration] Transform suggérée:", json.dumps(T, indent=2, ensure_ascii=False))
        P = apply_transform(P, ("PositionX","PositionZ"), T)


    # inférence des espacements attendus depuis la solution
    infer = infer_spacings_from_solution(S, tol_level=args.tol_level)
    step_z = infer["row_spacing"]      # attendu R3 (rangées / Z)
    gap_x  = infer["intra_spacing"]    # attendu R4 (colonnes / X)

    # tolérance d'accrochage aux rangées : doit tolérer les décalages réels
    row_align_eff = max(args.row_align, 0.10 * step_z) if step_z is not None else args.row_align

    # tolérances d’alignement déduites
    tol_R2 = max(0.20, 0.25*(step_z if step_z is not None else 0.0))
    tol_X  = max(0.20, 0.25*(gap_x  if gap_x  is not None else 0.0))

    rules = {}
    diags = {
        "inferred_from_solution": {
            "row_spacing_Z": step_z,
            "intra_spacing_X": gap_x,
            "tol_R2_rows_Z": tol_R2,
            "tol_columns_X": tol_X
        }
    }

    # R1
    ok, info = rule_R1_rows_layout(P, step_z, row_align_eff)
    rules["R1_rows_layout"] = ok; diags["R1"] = info

    # R2 (alignement rangée avant Z max)
    ok, info = rule_R2_align_front_row(P, S, tol_R2)
    rules["R2_front_row_alignment"] = ok; diags["R2"] = info

    # R3
    ok, info = rule_R3_row_spacing(P, step_z, row_align_eff, args.abs_tol, args.rel_tol)
    rules["R3_row_spacing"] = ok; diags["R3"] = info

    # R4
    ok, info = rule_R4_intrarow_spacing(P, gap_x, step_z, row_align_eff, args.abs_tol, args.rel_tol)
    rules["R4_intrarow_spacing"] = ok; diags["R4"] = info

    # R5 (alignement colonne droite X max)
    ok, info = rule_R5_align_right_column(P, S, tol_X)
    rules["R5_right_column_alignment"] = ok; diags["R5"] = info

    # R6
    ok, info = rule_R6_equal_tables_per_row(P, step_z, row_align_eff)
    rules["R6_equal_tables_per_row"] = ok; diags["R6"] = info

    # R7 (alignement colonne gauche X min)
    ok, info = rule_R7_align_left_column(P, S, tol_X)
    rules["R7_left_column_alignment"] = ok; diags["R7"] = info

    report = {
        "rules": rules,
        "diagnostics": diags,
        "n_rules_passed": int(sum(rules.values())),
        "n_rules_total": 7
    }

    # sorties
    P.assign(row=assign_rows(P, step_z, "PositionZ", row_align_eff)).to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if args.plot:
        make_plot(P, S, args.plot, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", os.path.abspath(args.out_json))
    print("Saved:", os.path.abspath(args.out_csv))
    if args.plot: print("Saved FIG:", os.path.abspath(args.plot))

if __name__ == "__main__":
    main()
