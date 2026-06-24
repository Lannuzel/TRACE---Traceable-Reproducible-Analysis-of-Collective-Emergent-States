#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scenario S2 (VR) — M1 uniquement : Rule conformity (consignes).
On compare un fichier participant VR (ReservationPositionData.csv) vs une solution/correction VR (VR_S2.csv).

- Solution VR_S2.csv contient souvent une entête "meta" puis une vraie table "name;position.x;...;scaleX;scaleY;scaleZ".
- Participant VR: lignes freeform (décimales virgule) + MARKER, on extrait les objets au dernier timestamp.
- Matching: Hungarian (si SciPy dispo) ou greedy, basé sur position (x,y,z) et tolérance pos_tol.
- Règles: host (Mur/Sol), shape (Cylinder/Square), dimensions plan (scaleX/scaleZ), thickness (scaleY).

Sorties :
- JSON : règles globales + score M1 + diagnostics
- CSV  : diagnostic par réservation attendue (match, deltas, pass/fail)
- PNG  : overlay optionnel (XZ)

Usage :
python eval_consignes_s2.py ^
  --participant D:\...\ReservationPositionData.csv ^
  --solution    D:\...\VR_S2.csv ^
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


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())


def _to_float_series(x: pd.Series) -> pd.Series:
    # gère virgule décimale
    return pd.to_numeric(
        x.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def read_vr_solution_smart(path: str) -> pd.DataFrame:
    """
    VR_S2.csv peut commencer par:
      Time; ID ; PositionCorrectionLog
    puis plus bas:
      name; position.x ; position.y; position.z; ...; scaleX ; scaleY;  scaleZ
    On détecte l'entête 'name;...position.x...'
    """
    # tentative simple
    try:
        df = pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
        df = normalize_cols(df)
        cols = [c.strip().lower() for c in df.columns.astype(str)]
        if "position.x" in cols and "scalex" in cols:
            return df
    except Exception:
        pass

    # scan header réel
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines[:500]):
        low = line.lower().strip()
        if low.startswith("name;") and ("position.x" in low) and (";" in low):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Entête VR solution introuvable dans {os.path.basename(path)}")

    df = pd.read_csv(path, sep=";", skiprows=header_idx, engine="python", encoding="utf-8-sig")
    return normalize_cols(df)


# -------------------------- VR participant parsing --------------------------

_NUM_COMMA = re.compile(r"-?\d+,\d+")

def _parse_vr_freeform_line(line: str) -> Optional[dict]:
    """
    Parse une ligne VR freeform du type:
    "2270,851, CylinderMurV2(Clone), 5,10, 2,83, 3,81, -0,50, 0,50, ... , 0,32, 0,20, 0,32"

    -> time, name, position.x/y/z, rotation (opt), scaleX/Y/Z (opt)
    """
    s = (line or "").strip()
    if not s:
        return None

    m_time = _NUM_COMMA.match(s)
    if not m_time:
        return None

    t_str = m_time.group(0)
    rest = s[m_time.end():].lstrip(" ,")

    m_name = re.match(r"([^,]+)", rest)
    if not m_name:
        return None
    name = m_name.group(1).strip()
    rest2 = rest[m_name.end():]

    nums = _NUM_COMMA.findall(rest2)
    if len(nums) < 3:
        return None

    vals = [float(n.replace(",", ".")) for n in nums]

    out = {
        "time": float(t_str.replace(",", ".")),
        "name": name,
        "position.x": vals[0] if len(vals) > 0 else np.nan,
        "position.y": vals[1] if len(vals) > 1 else np.nan,
        "position.z": vals[2] if len(vals) > 2 else np.nan,
    }
    if len(vals) >= 7:
        out.update({
            "rotation.x": vals[3],
            "rotation.y": vals[4],
            "rotation.z": vals[5],
            "rotation.w": vals[6],
        })
    if len(vals) >= 10:
        out.update({
            "scaleX": vals[7],
            "scaleY": vals[8],
            "scaleZ": vals[9],
        })
    return out


def load_vr_participant_final(path: str) -> pd.DataFrame:
    """
    Lit le CSV participant VR (ReservationPositionData) et retourne les objets au dernier timestamp.

    Important: si aucune ligne freeform trouvée -> retourne DF vide (0 placements), sans erreur.
    """
    # lecture "smart" : parfois plusieurs colonnes ; on concatène les cellules par ligne
    try:
        df_raw = pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
    except Exception:
        # fallback ultra brut
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            df_raw = pd.DataFrame({"_raw": f.read().splitlines()})

    df_raw = normalize_cols(df_raw)

    records = []
    for _, row in df_raw.iterrows():
        parts = [str(v) for v in row.values if pd.notna(v)]
        if not parts:
            continue
        line = " ".join(parts)
        rec = _parse_vr_freeform_line(line)
        if rec is not None:
            records.append(rec)

    if not records:
        # cas "header mais aucune ligne après" => 0 placements
        return pd.DataFrame(columns=["time","name","position.x","position.y","position.z","scaleX","scaleY","scaleZ"])

    df = pd.DataFrame.from_records(records)
    tmax = df["time"].max()
    out = df[df["time"] == tmax].copy().reset_index(drop=True)

    # normaliser les types
    for c in ["position.x","position.y","position.z","scaleX","scaleY","scaleZ"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["name"] = out["name"].astype(str)
    return out


# -------------------------- Parsing "consigne" tokens --------------------------

# Ex: "1_CylinderMurV2" (solution) ; "CylinderMurV2(Clone)" (participant)
_HOST_PAT = re.compile(r"\b(MUR|SOL)\b", re.IGNORECASE)
_SHAPE_PAT = re.compile(r"\b(CYLINDER|SQUARE|CIRC|RECT)\b", re.IGNORECASE)

def _normalize_obj_name(name: str) -> str:
    s = (name or "").strip()
    s = re.sub(r"^\d+_", "", s)              # "1_" prefix
    s = s.replace("(Clone)", "").strip()
    return s

def parse_host_shape(name: str) -> Tuple[str, str]:
    """
    host ∈ {"SOL","MUR","UNK"}
    shape ∈ {"CIRCULAIRE","RECTANGULAIRE","UNK"}

    - host: détecté via "Mur"/"Sol"
    - shape: Cylinder -> circulaire ; SquareHole -> rectangulaire
    """
    s = _normalize_obj_name(name).upper()

    mh = _HOST_PAT.search(s)
    host = mh.group(1).upper() if mh else "UNK"

    # shape inference
    if "CYLINDER" in s:
        shape = "CIRCULAIRE"
    elif "SQUARE" in s:
        shape = "RECTANGULAIRE"
    else:
        shape = "UNK"

    return host, shape


# -------------------------- Matching (solution -> participant) --------------------------

def pairwise_dist_3d(S: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))

def greedy_match(D: np.ndarray) -> List[Tuple[int,int,float]]:
    if D.size == 0:
        return []
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
    # VR: plan = scaleX & scaleZ
    px, pz = float(p.get("scaleX", np.nan)), float(p.get("scaleZ", np.nan))
    sx, sz = float(s.get("scaleX", np.nan)), float(s.get("scaleZ", np.nan))

    if shape_expected == "CIRCULAIRE":
        p_sorted = sorted([px, pz])
        s_sorted = sorted([sx, sz])
        ok0 = within_tol(p_sorted[0], s_sorted[0], abs_tol, rel_tol)
        ok1 = within_tol(p_sorted[1], s_sorted[1], abs_tol, rel_tol)
        return bool(ok0 and ok1), {
            "p_plan_min": p_sorted[0], "p_plan_max": p_sorted[1],
            "s_plan_min": s_sorted[0], "s_plan_max": s_sorted[1],
        }

    ok_xz = within_tol(px, sx, abs_tol, rel_tol) and within_tol(pz, sz, abs_tol, rel_tol)
    ok_zx = within_tol(px, sz, abs_tol, rel_tol) and within_tol(pz, sx, abs_tol, rel_tol)
    return bool(ok_xz or ok_zx), {
        "p_plan_x": px, "p_plan_z": pz,
        "s_plan_x": sx, "s_plan_z": sz,
        "perm_used": ("xz" if ok_xz else ("zx" if ok_zx else "none")),
    }

def thickness_ok(p: pd.Series, s: pd.Series, abs_tol: float, rel_tol: float) -> Tuple[bool, Dict[str, float]]:
    # VR: thickness/height = scaleY
    py = float(p.get("scaleY", np.nan))
    sy = float(s.get("scaleY", np.nan))
    ok = within_tol(py, sy, abs_tol, rel_tol)
    return bool(ok), {"p_thickness": py, "s_thickness": sy, "delta": abs(py - sy)}


# -------------------------- Plot --------------------------

def make_plot_xz(S: pd.DataFrame, P: pd.DataFrame, out_png: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(9,5), dpi=130)
    ax.scatter(S["position.x"], S["position.z"], s=55, marker="o", alpha=0.85, label="Solution")
    ax.scatter(P["position.x"], P["position.z"], s=55, marker="x", alpha=0.90, label="Participant")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("position.x"); ax.set_ylabel("position.z")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser(description="S2 (VR) — M1 Rule conformity (consignes).")
    ap.add_argument("--participant", required=True, help="CSV participant VR (ReservationPositionData.csv)")
    ap.add_argument("--solution", required=True, help="CSV solution/correction VR (VR_S2.csv)")
    ap.add_argument("--pos_tol", type=float, default=0.25, help="Tolérance matching (m) centre-à-centre (3D)")
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

    # --- Load solution (VR correction) ---
    S = read_vr_solution_smart(sol_path)
    S = normalize_cols(S)
    for c in ["position.x","position.y","position.z","scaleX","scaleY","scaleZ"]:
        if c not in S.columns:
            raise ValueError(f"Colonne manquante dans solution VR: '{c}'. Colonnes: {S.columns.tolist()}")
        S[c] = _to_float_series(S[c])

    S["name"] = S["name"].astype(str).str.strip()

    # --- Load participant (VR final objects) ---
    P = load_vr_participant_final(par_path)
    if len(P) > 0:
        # assurer colonnes présentes
        for c in ["position.x","position.y","position.z"]:
            if c not in P.columns:
                P[c] = np.nan
        for c in ["scaleX","scaleY","scaleZ"]:
            if c not in P.columns:
                P[c] = np.nan

    # distances
    S_xyz = S[["position.x","position.y","position.z"]].to_numpy(float)
    P_xyz = P[["position.x","position.y","position.z"]].to_numpy(float) if len(P) else np.zeros((0,3), float)

    D = pairwise_dist_3d(S_xyz, P_xyz) if (len(S) and len(P)) else np.zeros((len(S), len(P)), float)
    pairs = hungarian_match(D) if args.assignment == "hungarian" else greedy_match(D)

    # match unique par solution
    match_for_s: Dict[int, Tuple[int, float]] = {}
    used_p = set()
    for si, pi, dist in pairs:
        if si in match_for_s or pi in used_p:
            continue
        match_for_s[si] = (pi, dist)
        used_p.add(pi)

    rows = []
    n_present = 0

    for si in range(len(S)):
        srow = S.iloc[si]
        exp_host, exp_shape = parse_host_shape(srow["name"])

        pi, dist = match_for_s.get(si, (None, None))
        present = (pi is not None) and (dist is not None) and (dist <= args.pos_tol)

        if present:
            n_present += 1
            prow = P.iloc[int(pi)]
            got_host, got_shape = parse_host_shape(prow.get("name", ""))

            host_ok = (exp_host == "UNK") or (got_host == exp_host)
            shape_ok = (exp_shape == "UNK") or (got_shape == exp_shape)

            plan_ok, plan_diag = plan_dims_ok(
                prow, srow,
                shape_expected=(exp_shape if exp_shape != "UNK" else "RECTANGULAIRE"),
                abs_tol=args.size_abs, rel_tol=args.size_rel
            )
            thick_ok, thick_diag = thickness_ok(prow, srow, abs_tol=args.size_abs, rel_tol=args.size_rel)
        else:
            host_ok = shape_ok = plan_ok = thick_ok = False
            plan_diag = {}
            thick_diag = {}
            got_host = got_shape = "NA"

        rows.append({
            "s_index": si,
            "s_name": srow["name"],
            "s_host": exp_host,
            "s_shape": exp_shape,
            "s_position_x": float(srow["position.x"]),
            "s_position_y": float(srow["position.y"]),
            "s_position_z": float(srow["position.z"]),
            "s_scaleX": float(srow["scaleX"]),
            "s_scaleY": float(srow["scaleY"]),
            "s_scaleZ": float(srow["scaleZ"]),
            "p_index": -1 if pi is None else int(pi),
            "p_name": "" if pi is None or len(P) == 0 else str(P.iloc[int(pi)].get("name", "")),
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

    matched_p = set(diag_df.loc[diag_df["present_ok"], "p_index"].tolist()) if len(diag_df) else set()
    matched_p.discard(-1)
    n_extra = int(len(P) - len(matched_p)) if len(P) else 0

    # --------- Règles globales ---------
    R1 = (n_present == len(S))
    R2 = (n_extra <= args.allow_extra)
    R3 = bool(diag_df.loc[diag_df["present_ok"], "host_ok"].all()) if n_present > 0 else False
    R4 = bool(diag_df.loc[diag_df["present_ok"], "shape_ok"].all()) if n_present > 0 else False
    R5 = bool(diag_df.loc[diag_df["present_ok"], "plan_dims_ok"].all()) if n_present > 0 else False
    R6 = bool(diag_df.loc[diag_df["present_ok"], "thickness_ok"].all()) if n_present > 0 else False

    rules = {
        "R1_all_expected_present": R1,
        "R2_no_extra_objects": R2,
        "R3_host_type_SOL_vs_MUR": R3,
        "R4_shape_CIRC_vs_RECT": R4,
        "R5_plan_dimensions": R5,
        "R6_thickness_equals_host": R6,
    }

    # --------- Score M1 (0–100) ---------
    total_constraints = int(len(S) * 5)
    satisfied = 0
    satisfied += int(diag_df["present_ok"].sum()) if len(diag_df) else 0
    satisfied += int(diag_df["host_ok"].sum()) if len(diag_df) else 0
    satisfied += int(diag_df["shape_ok"].sum()) if len(diag_df) else 0
    satisfied += int(diag_df["plan_dims_ok"].sum()) if len(diag_df) else 0
    satisfied += int(diag_df["thickness_ok"].sum()) if len(diag_df) else 0

    M1 = 100.0 * (satisfied / total_constraints) if total_constraints > 0 else 0.0

    report = {
        "meta": {
            "scenario": "S2_VR_reservations",
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
                "present_ok": int(diag_df["present_ok"].sum()) if len(diag_df) else 0,
                "host_ok": int(diag_df["host_ok"].sum()) if len(diag_df) else 0,
                "shape_ok": int(diag_df["shape_ok"].sum()) if len(diag_df) else 0,
                "plan_dims_ok": int(diag_df["plan_dims_ok"].sum()) if len(diag_df) else 0,
                "thickness_ok": int(diag_df["thickness_ok"].sum()) if len(diag_df) else 0,
            }
        }
    }

    diag_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.plot:
        title = f"M1={M1:.1f} | present={n_present}/{len(S)} | extra={n_extra}"
        make_plot_xz(S, P if len(P) else pd.DataFrame(columns=["position.x","position.z"]), args.plot, title=title)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", os.path.abspath(args.out_json))
    print("Saved:", os.path.abspath(args.out_csv))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()