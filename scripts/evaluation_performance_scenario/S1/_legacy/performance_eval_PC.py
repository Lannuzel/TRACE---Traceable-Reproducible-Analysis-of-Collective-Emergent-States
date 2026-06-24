#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance evaluation for object placement (participant vs solution)

Fixes:
- IMPORTANT: calibration (translation) is applied BEFORE assignment, so distances are correct.
- Robust translation estimation using clustered X/Z centers + median delta.
- Duration extraction supports 'Duration(s)' (Revit export) + time/timestamp variants.
- Plot uses already-calibrated points (no double translation).

Author: Analyse E1
"""
from __future__ import annotations

import os, re, glob, json, argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ IO helpers ------------------------------
def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # NOTE: read_excel has no 'sep' argument; keeping it minimal.
        return pd.read_excel(path, engine="openpyxl")
    # CSV
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
    except Exception:
        # fallback: let pandas sniff (still with python engine)
        return pd.read_csv(path, encoding="utf-8-sig", engine="python")


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


# ------------------------------ Parsing ------------------------------
_FREEFORM_RE = re.compile(
    r'(?P<t>\d+(?:[.,]\d+)?)\s*,\s*(?P<name>[^,]+),\s*\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<z>-?\d+(?:\.\d+)?)\s*\)'
)


def parse_freeform_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse rows like: '2283,4, TablePref(Clone), (3.99, 0.00, 2.32)' and keep final timestamp."""
    records = []
    for _, row in df.iterrows():
        line = " | ".join(str(v) for v in row.values if pd.notna(v))
        m = _FREEFORM_RE.search(line)
        if not m:
            continue
        records.append({
            "time": float(m.group("t").replace(",", ".")),
            "name": m.group("name").strip(),
            "PositionX": float(m.group("x")),
            "PositionY": float(m.group("y")),
            "PositionZ": float(m.group("z")),
        })
    if not records:
        return pd.DataFrame(columns=["time", "name", "PositionX", "PositionY", "PositionZ"])
    df_rec = pd.DataFrame.from_records(records)
    t_max = float(df_rec["time"].max())
    return df_rec[df_rec["time"] == t_max].reset_index(drop=True)


def extract_duration_seconds(
    df_structured: pd.DataFrame,
    df_freeform: Optional[pd.DataFrame]
) -> Optional[float]:
    # 1) freeform logs
    if df_freeform is not None and not df_freeform.empty and "time" in df_freeform.columns:
        vals = pd.to_numeric(df_freeform["time"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if vals.notna().any():
            return float(vals.max())

    # 2) explicit duration columns (Revit export etc.)
    for col in ["Duration(s)", "Duration", "duration_s", "duration"]:
        if col in df_structured.columns:
            vals = pd.to_numeric(df_structured[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if vals.notna().any():
                return float(vals.max())

    # 3) generic timestamps
    for col in ["time", "Time", "timestamp", "Timestamp"]:
        if col in df_structured.columns:
            vals = pd.to_numeric(df_structured[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
            if vals.notna().any():
                return float(vals.max())

    return None


# ------------------------------ Assignment ------------------------------
def greedy_nearest_neighbor(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    if len(S) == 0 or len(P) == 0:
        return []
    dists = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    matches = []
    used_s, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
    for si, pi in order:
        si = int(si); pi = int(pi)
        if si in used_s or pi in used_p:
            continue
        matches.append((si, pi, float(dists[si, pi])))
        used_s.add(si); used_p.add(pi)
        if len(used_s) == len(S) or len(used_p) == len(P):
            break
    return matches


def hungarian_assignment(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    """Use SciPy if available; else fallback to greedy."""
    try:
        from scipy.optimize import linear_sum_assignment
        nS, nP = S.shape[0], P.shape[0]
        D = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
        if D.size == 0:
            return []
        if nS == nP:
            row_ind, col_ind = linear_sum_assignment(D)
        else:
            n = max(nS, nP)
            big = float(D.max() * 1000) if D.size else 1e9
            C = np.full((n, n), big, dtype=float)
            C[:nS, :nP] = D
            row_ind, col_ind = linear_sum_assignment(C)
            mask = (row_ind < nS) & (col_ind < nP)
            row_ind, col_ind = row_ind[mask], col_ind[mask]
        return [(int(i), int(j), float(D[i, j])) for i, j in zip(row_ind, col_ind)]
    except Exception:
        return greedy_nearest_neighbor(S, P)


# ------------------------------ Calibration (robust translation) ------------------------------
def _cluster_centers(vals: np.ndarray, tol: float = 0.35) -> np.ndarray:
    """1D clustering for regularly spaced grids; returns sorted cluster means."""
    v = np.sort(np.asarray(vals, float))
    if v.size == 0:
        return np.array([], dtype=float)
    clusters = [[v[0]]]
    center = v[0]
    for x in v[1:]:
        if abs(x - center) <= tol:
            clusters[-1].append(x)
            center = float(np.mean(clusters[-1]))
        else:
            clusters.append([x])
            center = x
    centers = np.array([np.mean(c) for c in clusters], dtype=float)
    centers.sort()
    return centers


def estimate_translation_by_centers(
    S: np.ndarray,
    P: np.ndarray,
    tol_cluster: float = 0.35
) -> Tuple[float, float, str]:
    """
    Robust tx,tz using clustered X and Z centers:
    tx = median( cxS[:k] - cxP[:k] ) where centers are sorted
    tz = median( czS[:k] - czP[:k] )
    """
    if S.size == 0 or P.size == 0:
        return 0.0, 0.0, "none"

    cxS = _cluster_centers(S[:, 0], tol=tol_cluster)
    cxP = _cluster_centers(P[:, 0], tol=tol_cluster)
    czS = _cluster_centers(S[:, 1], tol=tol_cluster)
    czP = _cluster_centers(P[:, 1], tol=tol_cluster)

    # fallback if clustering fails
    if cxS.size == 0 or cxP.size == 0 or czS.size == 0 or czP.size == 0:
        tx = float(np.median(S[:, 0]) - np.median(P[:, 0]))
        tz = float(np.median(S[:, 1]) - np.median(P[:, 1]))
        return tx, tz, "median_fallback"

    kx = min(cxS.size, cxP.size)
    kz = min(czS.size, czP.size)

    tx = float(np.median(cxS[:kx] - cxP[:kx]))
    tz = float(np.median(czS[:kz] - czP[:kz]))
    return tx, tz, "median_centers"


# ------------------------------ Core ------------------------------
def compute_performance(
    solution_path: str,
    participant_path_or_glob: str,
    tolerance_m: float = 0.25,
    axes: Tuple[str, str] = ("PositionX", "PositionZ"),
    pick: str = "newest",
    assignment: str = "hungarian",
    calib: str = "translation",  # "none" | "translation"
    calib_cluster_tol: float = 0.35,
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, List[Tuple[int, int, float]]]:

    part_path = _resolve_path_or_glob(participant_path_or_glob, pick=pick)

    # --- solution ---
    sol_raw = _normalize_cols(_read_table(solution_path))
    for a in axes:
        if a not in sol_raw.columns:
            raise ValueError(f"Solution missing column: {a}. Found: {sol_raw.columns.tolist()}")

    sol = sol_raw.copy()
    if "ChairRef" in sol.columns:
        sol["solution_id"] = sol["ChairRef"].astype(str)
    else:
        sol["solution_id"] = (np.arange(1, len(sol) + 1)).astype(str)

    sol_points = sol[list(axes) + ["solution_id"]].copy()

    # --- participant ---
    par_raw = _normalize_cols(_read_table(part_path))
    used_freeform = False

    if set(axes).issubset(par_raw.columns):
        par_points = par_raw[list(axes)].copy()
        name_col = [c for c in par_raw.columns if c.lower() in ("name", "object", "prefab", "id", "label")]
        par_points["name"] = par_raw[name_col[0]].astype(str) if name_col else ""
        parsed_freeform = None
    else:
        parsed_freeform = parse_freeform_positions(par_raw)
        if parsed_freeform.empty:
            raise ValueError("Participant file could not be parsed (neither structured nor free-form).")
        used_freeform = True
        par_points = parsed_freeform[list(axes)].copy()
        par_points["name"] = parsed_freeform["name"].astype(str) if "name" in parsed_freeform.columns else ""

    par_points["participant_id"] = (np.arange(1, len(par_points) + 1)).astype(str)

    # --- calibration BEFORE assignment ---
    S = sol_points[list(axes)].to_numpy(float)
    P = par_points[list(axes)].to_numpy(float)

    tx = tz = 0.0
    calib_method = "none"
    if calib == "translation" and len(S) and len(P):
        tx, tz, calib_method = estimate_translation_by_centers(S, P, tol_cluster=calib_cluster_tol)
        par_points[axes[0]] = par_points[axes[0]] + tx
        par_points[axes[1]] = par_points[axes[1]] + tz
        P = par_points[list(axes)].to_numpy(float)

    # --- assignment on calibrated points ---
    if assignment == "hungarian":
        matches = hungarian_assignment(S, P)
        assign_name = "hungarian"
    else:
        matches = greedy_nearest_neighbor(S, P)
        assign_name = "greedy"

    # --- per-object report (DISTANCES ARE NOW CONSISTENT) ---
    rows = []
    for si, pi, dist in matches:
        sp = sol_points.iloc[si]
        pp = par_points.iloc[pi]
        rows.append({
            "solution_id": sp["solution_id"],
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

    # --- metrics ---
    n_expected = int(len(sol_points))
    n_placed = int(len(par_points))
    count_diff = n_placed - n_expected
    abs_count_diff = abs(count_diff)
    sum_distance_m = float(report["distance_m"].sum()) if not report.empty else 0.0
    duration_s = extract_duration_seconds(par_raw, parsed_freeform if used_freeform else None)

    summary = {
        "n_expected": n_expected,
        "n_placed": n_placed,
        "count_diff": count_diff,
        "abs_count_diff": abs_count_diff,
        "sum_distance_m": sum_distance_m,
        "duration_s": duration_s,
        "tolerance_m": tolerance_m,
        "n_matched": int(len(report)),
        "n_missing": int(max(n_expected - len(report), 0)),
        "n_extra": int(max(n_placed - len(report), 0)),
        "assignment": assign_name,
        "calibration": calib,
        "calib_method": calib_method,
        "tx": float(tx),
        "tz": float(tz),
    }

    return report, summary, par_points, sol_points, matches


# ------------------------------ Plot ------------------------------
def draw_tolerance_circles(ax, sol_points: pd.DataFrame, axes=("PositionX", "PositionZ"), radius=0.25):
    try:
        import matplotlib.patches as patches
        for _, r in sol_points.iterrows():
            circ = patches.Circle(
                (float(r[axes[0]]), float(r[axes[1]])),
                radius=radius, fill=False, alpha=0.25, linewidth=0.8
            )
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
    metrics: dict | None = None,
    title: str | None = None
):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.scatter(sol_points[axes[0]], sol_points[axes[1]], marker='o', s=60, label='Solution (référence)')
    ax.scatter(par_points[axes[0]], par_points[axes[1]], marker='x', s=70, label='Participant (final)')

    # lines for matches
    for si, pi, _ in matches:
        xs = [float(sol_points.iloc[si][axes[0]]), float(par_points.iloc[pi][axes[0]])]
        ys = [float(sol_points.iloc[si][axes[1]]), float(par_points.iloc[pi][axes[1]])]
        ax.plot(xs, ys, linewidth=0.9, alpha=0.7)

    if tolerance and tolerance > 0:
        draw_tolerance_circles(ax, sol_points, axes=axes, radius=tolerance)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(axes[0] + " (m)")
    ax.set_ylabel(axes[1] + " (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if title:
        ax.set_title(title)

    if metrics:
        txt = "\n".join([
            f"n_expected={metrics.get('n_expected')}  n_placed={metrics.get('n_placed')}",
            f"sum_dist={metrics.get('sum_distance_m'):.3f} m  MAE={(metrics.get('sum_distance_m')/max(1,metrics.get('n_matched',1))):.3f} m",
            f"tx={metrics.get('tx'):.3f}  tz={metrics.get('tz'):.3f}",
            f"duration={metrics.get('duration_s')} s  assign={metrics.get('assignment')}",
        ])
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compute participant performance (count diff, sum distance, duration) with Hungarian assignment."
    )
    ap.add_argument("--solution", required=True, help="Solution file (.xlsx/.csv) with PositionX, PositionZ")
    ap.add_argument("--participant", required=True, help="Participant file path OR glob")
    ap.add_argument("--tolerance", type=float, default=0.25, help="Tolerance in meters")
    ap.add_argument("--axes", default="PositionX,PositionZ", help="Comma-separated axes (default: PositionX,PositionZ)")
    ap.add_argument("--pick", default="newest", choices=["newest", "first"], help="When glob matches many files, which to pick")
    ap.add_argument("--assignment", default="hungarian", choices=["hungarian", "greedy"], help="Assignment strategy")
    ap.add_argument("--out_csv", default="performance_per_object.csv", help="Per-object CSV output")
    ap.add_argument("--out_json", default="performance_summary.json", help="Summary JSON output")
    ap.add_argument("--plot", default=None, help="Optional output path for overlay figure")
    ap.add_argument("--plot_title", default=None, help="Optional plot title")
    ap.add_argument("--calib", choices=["none", "translation"], default="translation",
                    help="Calibration 2D. PC/Revit: translation recommended. VR: none or translation.")
    ap.add_argument("--calib_cluster_tol", type=float, default=0.35,
                    help="Clustering tolerance (m) for center-based translation estimation.")
    args = ap.parse_args()

    axes = tuple([a.strip() for a in args.axes.split(",") if a.strip()])
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two columns, e.g. PositionX,PositionZ")

    report, summary, par_points, sol_points, matches = compute_performance(
        solution_path=args.solution,
        participant_path_or_glob=args.participant,
        tolerance_m=args.tolerance,
        axes=axes, pick=args.pick,
        assignment=args.assignment,
        calib=args.calib,
        calib_cluster_tol=args.calib_cluster_tol,
    )

    report.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        make_overlay_plot(
            sol_points, par_points, matches,
            axes=axes, out_path=args.plot,
            tolerance=args.tolerance,
            metrics=summary,
            title=args.plot_title
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved CSV:", os.path.abspath(args.out_csv))
    print("Saved JSON:", os.path.abspath(args.out_json))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
