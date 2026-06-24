#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance evaluation for object placement (participant vs solution)
Adapté S2 (Revit) :
- Supporte CenterX_m/CenterY_m/CenterZ_m (+ mapping auto depuis PositionX/Y/Z)
- Axes par défaut: CenterX_m,CenterY_m (plan Revit)
- solution_id: ElementId si disponible
- duration_s: lit Duration(s) si présent
- calibration optionnelle (OFF par défaut pour Revit)
"""
from __future__ import annotations

import os, re, glob, json, argparse
from typing import List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ IO helpers ------------------------------

def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        # (note: read_excel n'a pas sep= ; je garde simple)
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path, sep=";", engine="python", encoding="utf-8-sig")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip())

def _resolve_path_or_glob(p: str, pick: str="newest") -> str:
    if os.path.isfile(p):
        return p
    matches = glob.glob(p)
    if not matches:
        raise FileNotFoundError(f"No file matched pattern: {p}")
    if len(matches) == 1:
        return matches[0]
    if pick == "newest":
        matches = sorted(matches, key=lambda fp: os.path.getmtime(fp), reverse=True)
        return matches[0]
    return matches[0]


# ------------------------------ Parsing ------------------------------

def parse_freeform_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Parse lignes type VR: '2283,4, Obj, (x, y, z)' et garde t max."""
    records = []
    for _, row in df.iterrows():
        line = " | ".join(str(v) for v in row.values if pd.notna(v))
        m = re.search(
            r'(?P<t>\d+(?:[.,]\d+)?)\s*,\s*(?P<name>[^,]+),\s*\(\s*(?P<x>-?\d+(?:\.\d+)?)\s*,\s*(?P<y>-?\d+(?:\.\d+)?)\s*,\s*(?P<z>-?\d+(?:\.\d+)?)\s*\)',
            line
        )
        if m:
            t = float(m.group("t").replace(",", "."))
            name = m.group("name").strip()
            x = float(m.group("x")); y = float(m.group("y")); z = float(m.group("z"))
            records.append({"time": t, "name": name, "PositionX": x, "PositionY": y, "PositionZ": z})
    if not records:
        return pd.DataFrame(columns=["time","name","PositionX","PositionY","PositionZ"])
    df_rec = pd.DataFrame.from_records(records)
    t_max = df_rec["time"].max()
    return df_rec[df_rec["time"] == t_max].reset_index(drop=True)

def extract_duration_seconds(df_structured: pd.DataFrame, df_freeform: pd.DataFrame | None) -> float | None:
    # Freeform logs
    if df_freeform is not None and not df_freeform.empty and "time" in df_freeform.columns:
        try:
            return float(pd.to_numeric(
                df_freeform["time"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            ).max())
        except Exception:
            pass

    # Revit / structurés : durée explicite
    for col in ["Duration(s)", "Duration", "duration_s", "duration"]:
        if col in df_structured.columns:
            try:
                vals = pd.to_numeric(
                    df_structured[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
                if vals.notna().any():
                    return float(vals.max())
            except Exception:
                pass

    # Fallback timestamp
    for col in ["time", "Time", "timestamp", "Timestamp"]:
        if col in df_structured.columns:
            try:
                vals = pd.to_numeric(
                    df_structured[col].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
                if vals.notna().any():
                    return float(vals.max())
            except Exception:
                pass

    return None


# ------------------------------ Axis mapping (Revit <-> VR) ------------------------------

_AXIS_ALIASES = {
    "PositionX": ["PositionX", "CenterX_m", "X_m", "X"],
    "PositionY": ["PositionY", "CenterY_m", "Y_m", "Y"],
    "PositionZ": ["PositionZ", "CenterZ_m", "Z_m", "Z"],
}

def _resolve_axis(df: pd.DataFrame, requested: str) -> str:
    """
    requested: ex "PositionX" ou "CenterX_m"
    - si la colonne existe telle quelle, ok
    - sinon si requested est PositionX/Y/Z, chercher alias CenterX_m etc.
    """
    if requested in df.columns:
        return requested
    # mapping générique si on demande PositionX/Y/Z
    if requested in _AXIS_ALIASES:
        for cand in _AXIS_ALIASES[requested]:
            if cand in df.columns:
                return cand
    raise ValueError(f"Axis '{requested}' not found. Columns: {df.columns.tolist()}")

def _get_points(df: pd.DataFrame, axes: Tuple[str, str]) -> pd.DataFrame:
    ax0 = _resolve_axis(df, axes[0])
    ax1 = _resolve_axis(df, axes[1])
    out = df[[ax0, ax1]].copy()
    out.columns = [axes[0], axes[1]]  # normalise au nom demandé
    return out


# ------------------------------ Assignment ------------------------------

def greedy_nearest_neighbor(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    if len(S) == 0 or len(P) == 0:
        return []
    dists = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
    matches = []
    used_s, used_p = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(dists, axis=None), dists.shape))[0]
    for si, pi in order:
        if si in used_s or pi in used_p:
            continue
        matches.append((int(si), int(pi), float(dists[si, pi])))
        used_s.add(int(si)); used_p.add(int(pi))
        if len(used_s) == len(S) or len(used_p) == len(P):
            break
    return matches

def hungarian_assignment(S: np.ndarray, P: np.ndarray) -> List[Tuple[int, int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment
        nS, nP = S.shape[0], P.shape[0]
        D = np.sqrt(((S[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
        if nS == nP:
            row_ind, col_ind = linear_sum_assignment(D)
            return [(int(i), int(j), float(D[i, j])) for i, j in zip(row_ind, col_ind)]
        n = max(nS, nP)
        big = float(D.max() * 1000.0) if D.size else 1e9
        C = np.full((n, n), big, dtype=float)
        C[:nS, :nP] = D
        row_ind, col_ind = linear_sum_assignment(C)
        mask = (row_ind < nS) & (col_ind < nP)
        row_ind, col_ind = row_ind[mask], col_ind[mask]
        return [(int(i), int(j), float(D[i, j])) for i, j in zip(row_ind, col_ind)]
    except Exception:
        return greedy_nearest_neighbor(S, P)


# ------------------------------ Core ------------------------------

def compute_performance(
    solution_path: str,
    participant_path_or_glob: str,
    tolerance_m: float = 0.25,
    axes: Tuple[str, str] = ("CenterX_m", "CenterY_m"),
    pick: str = "newest",
    assignment: str = "hungarian",
    calib: str = "none",  # "none" | "translation"
):
    part_path = _resolve_path_or_glob(participant_path_or_glob, pick=pick)

    sol_raw = _normalize_cols(_read_table(solution_path))
    par_raw = _normalize_cols(_read_table(part_path))

    # --- solution points ---
    sol_points_xy = _get_points(sol_raw, axes)
    # id solution
    if "ElementId" in sol_raw.columns:
        sol_points_xy["solution_id"] = sol_raw["ElementId"].astype(str)
    elif "ChairRef" in sol_raw.columns:
        sol_points_xy["solution_id"] = sol_raw["ChairRef"].astype(str)
    else:
        sol_points_xy["solution_id"] = (np.arange(1, len(sol_points_xy)+1)).astype(str)

    # --- participant points (structured or freeform) ---
    used_freeform = False
    parsed_freeform = None

    # Si l'un des axes n'existe pas, on tente freeform (VR)
    try:
        par_points_xy = _get_points(par_raw, axes)
        name_col = [c for c in par_raw.columns if c.lower() in ("name","object","prefab","id","label","elementname")]
        par_points_xy["name"] = par_raw[name_col[0]].astype(str) if name_col else ""
    except Exception:
        parsed_freeform = parse_freeform_positions(par_raw)
        if parsed_freeform.empty:
            raise ValueError("Participant file could not be parsed (neither structured axes nor free-form).")
        # remap axes (souvent PositionX/PositionZ)
        par_points_xy = _get_points(parsed_freeform, axes)
        par_points_xy["name"] = parsed_freeform["name"].astype(str) if "name" in parsed_freeform.columns else ""
        used_freeform = True

    par_points_xy["participant_id"] = (np.arange(1, len(par_points_xy)+1)).astype(str)

    # arrays
    S = sol_points_xy[list(axes)].to_numpy(float)
    P = par_points_xy[list(axes)].to_numpy(float)

    # --- calibration (optionnelle) ---
    if calib == "translation" and len(S) and len(P):
        # translation au barycentre (2D)
        t0 = float(S[:, 0].mean() - P[:, 0].mean())
        t1 = float(S[:, 1].mean() - P[:, 1].mean())
        par_points_xy[axes[0]] = par_points_xy[axes[0]] + t0
        par_points_xy[axes[1]] = par_points_xy[axes[1]] + t1
        P = par_points_xy[list(axes)].to_numpy(float)

    # assignment
    if assignment == "hungarian":
        matches = hungarian_assignment(S, P)
    else:
        matches = greedy_nearest_neighbor(S, P)

    # per-object report
    rows = []
    for si, pi, dist in matches:
        sp = sol_points_xy.iloc[si]
        pp = par_points_xy.iloc[pi]
        rows.append({
            "solution_id": sp["solution_id"],
            f"solution_{axes[0]}": float(sp[axes[0]]),
            f"solution_{axes[1]}": float(sp[axes[1]]),
            "participant_id": pp["participant_id"],
            "participant_name": pp.get("name",""),
            f"participant_{axes[0]}": float(pp[axes[0]]),
            f"participant_{axes[1]}": float(pp[axes[1]]),
            "distance_m": float(dist),
            "within_tolerance": bool(dist <= tolerance_m),
        })
    report = pd.DataFrame(rows)

    # metrics
    n_expected = int(len(sol_points_xy))
    n_placed = int(len(par_points_xy))
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
        "assignment": "hungarian" if assignment == "hungarian" else "greedy",
        "axes": list(axes),
        "calibration": calib,
    }
    return report, summary, par_points_xy, sol_points_xy, matches


# ------------------------------ Plot ------------------------------

def draw_tolerance_circles(ax, sol_points: pd.DataFrame, axes=("CenterX_m","CenterY_m"), radius=0.25):
    try:
        import matplotlib.patches as patches
        for _, r in sol_points.iterrows():
            circ = patches.Circle((float(r[axes[0]]), float(r[axes[1]])),
                                  radius=radius, fill=False, alpha=0.25, linewidth=0.8)
            ax.add_patch(circ)
    except Exception:
        pass

def make_overlay_plot(sol_points: pd.DataFrame,
                      par_points: pd.DataFrame,
                      matches: List[Tuple[int,int,float]],
                      axes=("CenterX_m","CenterY_m"),
                      out_path="overlay.png",
                      tolerance=None,
                      metrics: dict | None=None,
                      title: str | None=None):
    fig, ax = plt.subplots(figsize=(8,6), dpi=120)
    ax.scatter(sol_points[axes[0]], sol_points[axes[1]], marker='o', s=60, label='Solution (référence)')
    ax.scatter(par_points[axes[0]], par_points[axes[1]], marker='x', s=70, label='Participant (final)')

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
            f"count_diff={metrics.get('count_diff')}  sum_dist={metrics.get('sum_distance_m'):.3f} m",
            f"duration={metrics.get('duration_s')} s  assign={metrics.get('assignment')}",
            f"axes={','.join(metrics.get('axes', []))}  calib={metrics.get('calibration')}",
        ])
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute placement performance (Revit/VR) with Hungarian assignment.")
    ap.add_argument("--solution", required=True, help="Solution file (.xlsx/.csv)")
    ap.add_argument("--participant", required=True, help="Participant file path OR glob")
    ap.add_argument("--tolerance", type=float, default=0.25, help="Tolerance in meters")
    ap.add_argument("--axes", default="CenterX_m,CenterY_m",
                    help="Comma-separated axes. Revit: CenterX_m,CenterY_m. VR: PositionX,PositionZ etc.")
    ap.add_argument("--pick", default="newest", choices=["newest","first"])
    ap.add_argument("--assignment", default="hungarian", choices=["hungarian","greedy"])
    ap.add_argument("--calib", default="none", choices=["none","translation"],
                    help="Calibration 2D. Revit: none (default). VR: translation si besoin.")
    ap.add_argument("--out_csv", default="performance_per_object.csv")
    ap.add_argument("--out_json", default="performance_summary.json")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--plot_title", default=None)
    args = ap.parse_args()

    axes = tuple([a.strip() for a in args.axes.split(",") if a.strip()])
    if len(axes) != 2:
        raise ValueError("--axes doit contenir exactement 2 axes (ex: CenterX_m,CenterY_m)")

    report, summary, par_points, sol_points, matches = compute_performance(
        args.solution, args.participant,
        tolerance_m=args.tolerance,
        axes=axes,
        pick=args.pick,
        assignment=args.assignment,
        calib=args.calib,
    )

    report.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        make_overlay_plot(sol_points, par_points, matches, axes=axes, out_path=args.plot,
                          tolerance=args.tolerance, metrics=summary, title=args.plot_title)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved CSV:", os.path.abspath(args.out_csv))
    print("Saved JSON:", os.path.abspath(args.out_json))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
