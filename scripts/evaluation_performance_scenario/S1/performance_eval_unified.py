#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified performance evaluation (PC/Revit + VR/Unity) for object placement.

Key points:
- PC/Revit exports are in "Revit space" (large coords). VR logs are in Unity world space.
- To compare them, we bring everything into a common space.
  By default: common space = Unity world.
  Revit -> Unity conversion: X_unity = X_revit / scale_x ; Z_unity = Z_revit / scale_z
  (We ignore position/rotation as requested; only scale is applied.)
- Robust translation calibration (median) after an initial assignment; then re-assign.

Outputs:
- per-object CSV
- summary JSON
- optional overlay PNG

Also includes optional M2 variants to penalize "too many chairs":
- M2_min:      min(n_placed, n_expected)/n_expected
- M2_precision: tp/(tp+fp)
- M2_f1:       2*tp/(2*tp+fp+fn)

Author: Tristan pipeline
"""
from __future__ import annotations

import os, re, glob, json, math, argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ IO helpers ------------------------------
def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # CSV: try ; then autodetect
    try:
        return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())

def _resolve_path_or_glob(p: str, pick: str = "newest") -> str:
    if os.path.isfile(p):
        return p
    matches = glob.glob(p)
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {p}")
    if len(matches) == 1 or pick == "first":
        return matches[0]
    matches = sorted(matches, key=lambda fp: os.path.getmtime(fp), reverse=True)
    return matches[0]


# ------------------------------ Parsing (VR freeform) ------------------------------
_FREEFORM_RE = re.compile(
    r'(?P<t>\d+(?:[.,]\d+)?)\s*,\s*(?P<name>[^,]+),\s*\(\s*'
    r'(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<z>-?\d+(?:\.\d+)?)\s*\)'
)

def parse_freeform_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse rows like: '2283,4, TablePref(Clone), (3.99, 0.00, 2.32)' and keep final timestamp."""
    records = []
    for _, row in df.iterrows():
        line = " | ".join(str(v) for v in row.values if pd.notna(v))
        m = _FREEFORM_RE.search(line)
        if not m:
            continue
        t = float(m.group("t").replace(",", "."))
        name = m.group("name").strip()
        x = float(m.group("x")); y = float(m.group("y")); z = float(m.group("z"))
        records.append({"time": t, "name": name, "PositionX": x, "PositionY": y, "PositionZ": z})

    if not records:
        return pd.DataFrame(columns=["time", "name", "PositionX", "PositionY", "PositionZ"])

    df_rec = pd.DataFrame.from_records(records)
    t_max = df_rec["time"].max()
    return df_rec[df_rec["time"] == t_max].reset_index(drop=True)


def extract_duration_seconds(
    df_structured: pd.DataFrame,
    df_freeform: Optional[pd.DataFrame],
    max_duration_s: float = 1500.0  # 25 minutes
) -> Optional[float]:

    duration = None

    # 1) VR freeform
    if df_freeform is not None and not df_freeform.empty and "time" in df_freeform.columns:
        vals = pd.to_numeric(
            df_freeform["time"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
        if vals.notna().any():
            duration = float(vals.max())

    # 2) Revit structured duration
    if duration is None:
        for col in ["Duration(s)", "Duration", "duration_s", "duration"]:
            if col in df_structured.columns:
                vals = pd.to_numeric(
                    df_structured[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
                if vals.notna().any():
                    duration = float(vals.max())
                    break

    # 3) fallback timestamps
    if duration is None:
        for col in ["time", "Time", "timestamp", "Timestamp"]:
            if col in df_structured.columns:
                vals = pd.to_numeric(
                    df_structured[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
                if vals.notna().any():
                    duration = float(vals.max())
                    break

    # 4) sécurité expérimentale (plafond à 25 min)
    if duration is not None:
        if duration <= 0:
            return max_duration_s
        if duration > max_duration_s:
            return max_duration_s
        return duration

    return None


# ------------------------------ Assignment ------------------------------
def greedy_nearest_neighbor(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    if S.size == 0 or P.size == 0:
        return []
    D = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    used_s, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    matches = []
    for si, pi in order:
        si = int(si); pi = int(pi)
        if si in used_s or pi in used_p:
            continue
        used_s.add(si); used_p.add(pi)
        matches.append((si, pi, float(D[si, pi])))
        if len(used_s) == len(S) or len(used_p) == len(P):
            break
    return matches

def hungarian_assignment(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        nS, nP = S.shape[0], P.shape[0]
        D = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
        if D.size == 0:
            return []
        if nS == nP:
            r, c = linear_sum_assignment(D)
            return [(int(i), int(j), float(D[i, j])) for i, j in zip(r, c)]
        n = max(nS, nP)
        big = float(D.max() * 1000) if D.size else 1e9
        C = np.full((n, n), big, dtype=float)
        C[:nS, :nP] = D
        r, c = linear_sum_assignment(C)
        mask = (r < nS) & (c < nP)
        r = r[mask]; c = c[mask]
        return [(int(i), int(j), float(D[i, j])) for i, j in zip(r, c)]
    except Exception:
        return greedy_nearest_neighbor(S, P)


# ------------------------------ Space conversion ------------------------------
def apply_scale_revit_to_unity(df: pd.DataFrame, axes=("PositionX", "PositionZ"), scale_x=25.7, scale_z=25.95) -> pd.DataFrame:
    """
    Convert Revit-space coords to Unity-space coords using ONLY scale:
      X_unity = X_revit / scale_x
      Z_unity = Z_revit / scale_z
    """
    out = df.copy()
    out[axes[0]] = out[axes[0]].astype(float) / float(scale_x)
    out[axes[1]] = out[axes[1]].astype(float) / float(scale_z)
    return out


# ------------------------------ Robust translation calibration ------------------------------
def _median_translation_from_matches(S: np.ndarray, P: np.ndarray, matches: List[Tuple[int, int, float]]) -> Tuple[float, float]:
    if not matches:
        return 0.0, 0.0
    dx = []
    dz = []
    for si, pi, _ in matches:
        dx.append(float(S[si, 0] - P[pi, 0]))
        dz.append(float(S[si, 1] - P[pi, 1]))
    return float(np.median(dx)), float(np.median(dz))

def calibrate_translation_iterative(
    S: np.ndarray,
    P: np.ndarray,
    n_iter: int = 2,
    assignment: str = "hungarian"
) -> Tuple[np.ndarray, Dict[str, Any], List[Tuple[int, int, float]]]:
    """
    Iterative:
      - initial tx,tz from median difference of coordinates (rough)
      - assign
      - update tx,tz as median over matched diffs
      - repeat, then final assign
    """
    if S.size == 0 or P.size == 0:
        meta = {"calibration": "translation", "calib_method": "median", "tx": 0.0, "tz": 0.0}
        return P, meta, []

    # rough init (median of coords)
    tx = float(np.median(S[:, 0]) - np.median(P[:, 0]))
    tz = float(np.median(S[:, 1]) - np.median(P[:, 1]))

    P_adj = P.copy()
    P_adj[:, 0] += tx
    P_adj[:, 1] += tz

    for _ in range(max(1, int(n_iter))):
        if assignment == "hungarian":
            matches = hungarian_assignment(S, P_adj)
        else:
            matches = greedy_nearest_neighbor(S, P_adj)

        dtx, dtz = _median_translation_from_matches(S, P_adj, matches)
        # Note: dtx/dtz are residuals because P_adj already shifted
        tx += dtx
        tz += dtz
        P_adj[:, 0] += dtx
        P_adj[:, 1] += dtz

    # final assignment
    if assignment == "hungarian":
        matches = hungarian_assignment(S, P_adj)
    else:
        matches = greedy_nearest_neighbor(S, P_adj)

    meta = {
        "calibration": "translation",
        "calib_method": "median",
        "tx": tx,
        "tz": tz,
    }
    return P_adj, meta, matches


# ------------------------------ Core compute ------------------------------
def load_points_any(path_or_glob: str, axes=("PositionX", "PositionZ"), pick="newest") -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], bool]:
    """
    Returns:
      df_raw (normalized), points_df (axes + name), parsed_freeform_df (or None), used_freeform
    """
    path = _resolve_path_or_glob(path_or_glob, pick=pick)
    raw = _normalize_cols(_read_table(path))

    if set(axes).issubset(raw.columns):
        pts = raw[list(axes)].copy()
        name_col = [c for c in raw.columns if c.lower() in ("name", "object", "prefab", "id", "label", "elementname")]
        pts["name"] = raw[name_col[0]].astype(str) if name_col else ""
        return raw, pts, None, False

    parsed = parse_freeform_positions(raw)
    if parsed.empty:
        raise ValueError(f"Could not parse participant file: {path}")
    pts = parsed[list(axes)].copy()
    pts["name"] = parsed["name"].astype(str) if "name" in parsed.columns else ""
    return raw, pts, parsed, True


def compute_performance_unified(
    solution_path: str,
    participant_path_or_glob: str,
    tolerance_m: float = 0.25,
    axes=("PositionX", "PositionZ"),
    pick="newest",
    assignment="hungarian",
    calib="translation",  # none|translation
    # Revit->Unity scale (only used if --participant_space revit or --solution_space revit)
    unity_scale_x: float = 25.7,
    unity_scale_z: float = 25.95,
    solution_space: str = "unity",      # unity|revit
    participant_space: str = "auto",    # auto|unity|revit
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, pd.DataFrame, List[Tuple[int, int, float]]]:

    # --- Load solution (assumed structured) ---
    sol_raw = _normalize_cols(_read_table(solution_path))
    for a in axes:
        if a not in sol_raw.columns:
            raise ValueError(f"Solution missing column {a}. Found: {sol_raw.columns.tolist()}")

    sol_points = sol_raw[list(axes)].copy()
    if "ChairRef" in sol_raw.columns:
        sol_points["solution_id"] = sol_raw["ChairRef"].astype(str)
    else:
        sol_points["solution_id"] = (np.arange(1, len(sol_points) + 1)).astype(str)

    # --- Load participant (structured OR freeform) ---
    par_raw, par_points, parsed_freeform, used_freeform = load_points_any(
        participant_path_or_glob, axes=axes, pick=pick
    )
    par_points["participant_id"] = (np.arange(1, len(par_points) + 1)).astype(str)

    # --- Decide participant space if auto ---
    if participant_space == "auto":
        # Heuristic: Revit often has huge coords (hundreds+). Unity tends to be smaller.
        # If median abs(X) is > 200 or abs(Z) > 200 -> treat as revit.
        mx = float(np.median(np.abs(pd.to_numeric(par_points[axes[0]], errors="coerce").fillna(0.0))))
        mz = float(np.median(np.abs(pd.to_numeric(par_points[axes[1]], errors="coerce").fillna(0.0))))
        participant_space = "revit" if (mx > 200 or mz > 200) else "unity"

    # --- Bring both to common space: Unity ---
    meta_space = {
        "common_space": "unity",
        "solution_space_in": solution_space,
        "participant_space_in": participant_space,
        "unity_scale_x": float(unity_scale_x),
        "unity_scale_z": float(unity_scale_z),
    }

    if solution_space == "revit":
        sol_points = apply_scale_revit_to_unity(sol_points, axes=axes, scale_x=unity_scale_x, scale_z=unity_scale_z)
        meta_space["solution_space_applied"] = "revit_to_unity_scale_div"
    else:
        meta_space["solution_space_applied"] = "unity_no_scale"

    if participant_space == "revit":
        par_points = apply_scale_revit_to_unity(par_points, axes=axes, scale_x=unity_scale_x, scale_z=unity_scale_z)
        meta_space["participant_space_applied"] = "revit_to_unity_scale_div"
    else:
        meta_space["participant_space_applied"] = "unity_no_scale"

    # numpy arrays in common space
    S = sol_points[list(axes)].to_numpy(float)
    P = par_points[list(axes)].to_numpy(float)

    # --- Calibration (translation) in common space ---
    calib_meta: Dict[str, Any] = {"calibration": "none"}
    matches: List[Tuple[int, int, float]] = []

    if calib == "translation":
        P_cal, calib_meta, matches = calibrate_translation_iterative(S, P, n_iter=2, assignment=assignment)
        # write back adjusted coords
        par_points.loc[:, axes[0]] = P_cal[:, 0]
        par_points.loc[:, axes[1]] = P_cal[:, 1]
        P = P_cal
    else:
        # direct matching
        if assignment == "hungarian":
            matches = hungarian_assignment(S, P)
        else:
            matches = greedy_nearest_neighbor(S, P)

    # --- Per-object report ---
    rows = []
    for si, pi, dist in matches:
        sp = sol_points.iloc[si]
        pp = par_points.iloc[pi]
        rows.append({
            "solution_id": sp.get("solution_id", str(si + 1)),
            "solution_x": float(sp[axes[0]]),
            "solution_z": float(sp[axes[1]]),
            "participant_id": pp["participant_id"],
            "participant_name": pp.get("name", ""),
            "participant_x": float(pp[axes[0]]),
            "participant_z": float(pp[axes[1]]),
            "distance_m": float(dist),
            "within_tolerance": bool(dist <= tolerance_m),
        })
    report = pd.DataFrame(rows)

    # --- Metrics ---
    n_expected = int(len(sol_points))
    n_placed = int(len(par_points))
    tp = int(len(report))  # matched
    fn = int(max(n_expected - tp, 0))
    fp = int(max(n_placed - tp, 0))

    sum_distance_m = float(report["distance_m"].sum()) if not report.empty else 0.0
    duration_s = extract_duration_seconds(par_raw, parsed_freeform if used_freeform else None)

    # M2 variants
    M2_min = 100.0 * (min(n_placed, n_expected) / n_expected) if n_expected > 0 else 0.0
    den_prec = tp + fp
    M2_precision = 100.0 * (tp / den_prec) if den_prec > 0 else 0.0
    den_f1 = (2 * tp + fp + fn)
    M2_f1 = 100.0 * (2 * tp / den_f1) if den_f1 > 0 else 0.0

    summary = {
        "n_expected": n_expected,
        "n_placed": n_placed,
        "count_diff": int(n_placed - n_expected),
        "abs_count_diff": int(abs(n_placed - n_expected)),
        "sum_distance_m": sum_distance_m,
        "duration_s": duration_s,
        "tolerance_m": float(tolerance_m),
        "n_matched": tp,
        "n_missing": fn,
        "n_extra": fp,
        "assignment": "hungarian" if assignment == "hungarian" else "greedy",
        **meta_space,
        **calib_meta,
        "M2_variants": {
            "M2_min_%": M2_min,
            "M2_precision_%": M2_precision,
            "M2_f1_%": M2_f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
    }

    return report, summary, par_points, sol_points, matches


# ------------------------------ Plot ------------------------------
def draw_tolerance_circles(ax, sol_points: pd.DataFrame, axes=("PositionX", "PositionZ"), radius=0.25):
    try:
        import matplotlib.patches as patches
        for _, r in sol_points.iterrows():
            circ = patches.Circle((float(r[axes[0]]), float(r[axes[1]])), radius=radius,
                                  fill=False, alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)
    except Exception:
        pass

def make_overlay_plot(
    sol_points: pd.DataFrame,
    par_points: pd.DataFrame,
    matches: List[Tuple[int, int, float]],
    axes=("PositionX", "PositionZ"),
    out_path="overlay.png",
    tolerance=None,
    metrics: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.scatter(sol_points[axes[0]], sol_points[axes[1]], marker="o", s=60, label="Solution (ref)")
    ax.scatter(par_points[axes[0]], par_points[axes[1]], marker="x", s=70, label="Participant")

    for si, pi, _ in matches:
        xs = [float(sol_points.iloc[si][axes[0]]), float(par_points.iloc[pi][axes[0]])]
        ys = [float(sol_points.iloc[si][axes[1]]), float(par_points.iloc[pi][axes[1]])]
        ax.plot(xs, ys, linewidth=0.9, alpha=0.7)

    if tolerance and tolerance > 0:
        draw_tolerance_circles(ax, sol_points, axes=axes, radius=float(tolerance))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axes[0] + " (m)")
    ax.set_ylabel(axes[1] + " (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)

    if metrics:
        txt = "\n".join([
            f"expected={metrics.get('n_expected')}  placed={metrics.get('n_placed')}  matched={metrics.get('n_matched')}",
            f"extra={metrics.get('n_extra')}  missing={metrics.get('n_missing')}",
            f"sum_dist={metrics.get('sum_distance_m'):.3f} m  duration={metrics.get('duration_s')} s",
            f"space={metrics.get('common_space')}  calib={metrics.get('calibration')}  assign={metrics.get('assignment')}",
        ])
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Unified PC(Revit)/VR(Unity) performance evaluation with Revit->Unity scale + robust translation.")
    ap.add_argument("--solution", required=True, help="Solution file (.csv/.xlsx) with PositionX, PositionZ")
    ap.add_argument("--participant", required=True, help="Participant path OR glob")
    ap.add_argument("--tolerance", type=float, default=0.25, help="Tolerance in meters (within_tolerance + optional circles)")
    ap.add_argument("--axes", default="PositionX,PositionZ", help="Comma-separated axes (default: PositionX,PositionZ)")
    ap.add_argument("--pick", default="newest", choices=["newest", "first"], help="When glob matches many files, pick which")
    ap.add_argument("--assignment", default="hungarian", choices=["hungarian", "greedy"], help="Assignment strategy")
    ap.add_argument("--calib", default="translation", choices=["none", "translation"], help="Calibration in common space")
    ap.add_argument("--unity_scale_x", type=float, default=25.700000762939454, help="Unity scale X used on the Revit model")
    ap.add_argument("--unity_scale_z", type=float, default=25.950000762939454, help="Unity scale Z used on the Revit model")
    ap.add_argument("--solution_space", default="unity", choices=["unity", "revit"], help="Space of solution coordinates")
    ap.add_argument("--participant_space", default="auto", choices=["auto", "unity", "revit"], help="Space of participant coordinates")
    ap.add_argument("--out_csv", default="performance_per_object.csv", help="Per-object CSV output")
    ap.add_argument("--out_json", default="performance_summary.json", help="Summary JSON output")
    ap.add_argument("--plot", default=None, help="Optional output path for overlay PNG")
    ap.add_argument("--plot_title", default=None, help="Optional overlay title")
    args = ap.parse_args()

    axes = tuple([a.strip() for a in args.axes.split(",") if a.strip()])
    report, summary, par_points, sol_points, matches = compute_performance_unified(
        solution_path=args.solution,
        participant_path_or_glob=args.participant,
        tolerance_m=args.tolerance,
        axes=axes,
        pick=args.pick,
        assignment=args.assignment,
        calib=args.calib,
        unity_scale_x=args.unity_scale_x,
        unity_scale_z=args.unity_scale_z,
        solution_space=args.solution_space,
        participant_space=args.participant_space,
    )

    report.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        make_overlay_plot(sol_points, par_points, matches, axes=axes,
                          out_path=args.plot, tolerance=args.tolerance,
                          metrics=summary, title=args.plot_title)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved CSV:", os.path.abspath(args.out_csv))
    print("Saved JSON:", os.path.abspath(args.out_json))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
