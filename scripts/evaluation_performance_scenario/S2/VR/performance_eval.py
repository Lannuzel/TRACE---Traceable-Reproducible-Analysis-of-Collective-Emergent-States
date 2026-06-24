#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VR Performance evaluation for object placement (participant vs solution) - S2

Adapted from performance_eval.py (PC/Revit) to VR logs:
- Reads VR correction/solution file (e.g., VR_S2.csv) with ';' separator and comma decimals.
- Reads VR participant ReservationPositionData.csv:
    * top marker rows + a "time,name, position.x ..." header line
    * data lines where BOTH delimiter and decimals are commas (e.g., "2270,851, CylinderMurV2(Clone), 5,10, ...")
  -> parsed robustly via regex capturing numbers like -?\d+,\d+.

Metrics:
- n_expected, n_placed, count_diff, abs_count_diff
- sum_distance_m over matched pairs
- duration_s = max timestamp found (parsed freeform preferred)
- within_tolerance per matched pair

CLI example:
python performance_eval_vr.py --solution VR_S2.csv --participant "D:/.../*_ReservationPositionData.csv" --axes "position.x,position.z" --tolerance 0.25 --plot overlay.png
"""

from __future__ import annotations
import os, re, glob, json, argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ IO helpers ------------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # strip + collapse spaces
    def norm(c: str) -> str:
        c = str(c).strip()
        c = re.sub(r"\s+", " ", c)
        return c
    return df.rename(columns=norm)


def _resolve_path_or_glob(p: str, pick: str = "newest") -> str:
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


def _read_csv_smart(path: str) -> pd.DataFrame:
    """
    Smart CSV reader for VR correction files (with meta header) and VR logs.
    Key point: even if sep=';' read succeeds, we verify the header is the *data* header.
    """
    # 1) try normal
    try:
        df = pd.read_csv(path)
        # If it already contains the expected VR columns, keep it
        cols = [c.strip() for c in df.columns.astype(str)]
        if any(c.lower() == "position.x" for c in cols) or any("centerx" in c.lower() for c in cols):
            return df
    except Exception:
        pass

    # 2) try sep=';' (may succeed on meta header -> must validate)
    try:
        df = pd.read_csv(path, sep=";", engine="python")
        cols = [c.strip().lower() for c in df.columns.astype(str)]
        # Accept only if it's a *data* table header (e.g. contains position.x)
        if ("position.x" in cols) or ("position.z" in cols) or ("centerx_m" in cols):
            return df
        # Otherwise: keep going to header scan
    except Exception:
        df = None  # just to be explicit

    # 3) scan lines to find the real data header row
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    header_idx = None
    for i, line in enumerate(lines[:500]):
        low = line.lower().strip()

        # Strict detection for VR_S2 correction table header:
        # must start with "name;" and contain "position.x" (not just "PositionCorrectionLog")
        if low.startswith("name;") and ("position.x" in low) and (";" in low):
            header_idx = i
            break

    if header_idx is None:
        # last resort: return raw lines
        return pd.DataFrame({"_raw": lines})

    return pd.read_csv(
        path,
        sep=";",
        skiprows=header_idx,
        engine="python",
        skip_blank_lines=True,
    )


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # csv-like
    return _read_csv_smart(path)


def _to_float_series(x: pd.Series) -> pd.Series:
    # handles " -13,56" and "0,71" etc
    return pd.to_numeric(x.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


# ------------------------------ Parsing ------------------------------
_NUM_COMMA = re.compile(r"-?\d+,\d+")

def _parse_vr_freeform_line(line: str) -> Optional[dict]:
    """
    Parse a VR freeform line such as:
    "2270,851, CylinderMurV2(Clone), 5,10, 2,83, 3,81, -0,50, 0,50, 0,50, 0,50, 0,32, 0,20, 0,32"
    Strategy:
      - capture first number like \d+,\d+ as time
      - capture name until next comma
      - capture subsequent comma-decimal numbers as floats (expect at least 10 values: pos(3)+rot(4)+scale(3))
    """
    s = line.strip()
    if not s:
        return None

    m_time = _NUM_COMMA.match(s)
    if not m_time:
        return None

    t_str = m_time.group(0)
    rest = s[m_time.end():].lstrip(" ,")

    # name = until next comma
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
    # optional: keep rot/scale if present
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


def parse_vr_freeform_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract final (last timestamp) object positions from a VR participant file.

    Works even if df is a weird 3-column ';' file with data embedded in a single cell.
    """
    records = []

    for _, row in df.iterrows():
        # concatenate non-NA cells to rebuild the full line
        parts = [str(v) for v in row.values if pd.notna(v)]
        if not parts:
            continue
        line = " ".join(parts)
        rec = _parse_vr_freeform_line(line)
        if rec is not None:
            records.append(rec)

    if not records:
        return pd.DataFrame(columns=["time", "name", "position.x", "position.y", "position.z"])

    df_rec = pd.DataFrame.from_records(records)
    tmax = df_rec["time"].max()
    return df_rec[df_rec["time"] == tmax].reset_index(drop=True)


def extract_duration_seconds(df_structured: pd.DataFrame, df_freeform: Optional[pd.DataFrame]) -> Optional[float]:
    # Prefer parsed freeform timestamps (they represent the real task time)
    if df_freeform is not None and not df_freeform.empty and "time" in df_freeform.columns:
        vals = pd.to_numeric(df_freeform["time"], errors="coerce")
        if vals.notna().any():
            return float(vals.max())

    # Fallback: try any time-like column in structured df
    for col in ["time", "Time", "timestamp", "Timestamp"]:
        if col in df_structured.columns:
            vals = _to_float_series(df_structured[col])
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
        if si in used_s or pi in used_p:
            continue
        matches.append((int(si), int(pi), float(dists[si, pi])))
        used_s.add(si)
        used_p.add(pi)
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
        else:
            n = max(nS, nP)
            big = (D.max() * 1000) if D.size else 1e9
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
    axes: Tuple[str, str] = ("position.x", "position.z"),
    pick: str = "newest",
    assignment: str = "hungarian",
):
    part_path = _resolve_path_or_glob(participant_path_or_glob, pick=pick)

    # Solution
    sol_raw = _read_table(solution_path)
    sol = _normalize_cols(sol_raw)

    # strip weird spaces in VR_S2 headers (e.g., " position.x ")
    sol.columns = [c.strip() for c in sol.columns]

    for a in axes:
        if a not in sol.columns:
            raise ValueError(f"Solution missing column '{a}'. Found: {sol.columns.tolist()}")

    if "name" in sol.columns:
        sol["solution_id"] = sol["name"].astype(str).str.strip()
    else:
        sol["solution_id"] = (np.arange(1, len(sol) + 1)).astype(str)

    sol_points = sol[[axes[0], axes[1], "solution_id"]].copy()
    sol_points[axes[0]] = _to_float_series(sol_points[axes[0]])
    sol_points[axes[1]] = _to_float_series(sol_points[axes[1]])

    sol_points = sol_points.dropna(subset=[axes[0], axes[1]]).reset_index(drop=True)

    # Participant
    par_raw = _read_table(part_path)
    par = _normalize_cols(par_raw)
    par.columns = [c.strip() for c in par.columns]

    used_freeform = False
    parsed_freeform = None

    if set(axes).issubset(par.columns):
        # already structured
        par_points = par[list(axes)].copy()
        name_col = [c for c in par.columns if c.lower() in ("name", "object", "prefab", "id", "label")]
        par_points["name"] = par[name_col[0]].astype(str).str.strip() if name_col else ""
        par_points[axes[0]] = _to_float_series(par_points[axes[0]])
        par_points[axes[1]] = _to_float_series(par_points[axes[1]])
        par_points = par_points.dropna(subset=[axes[0], axes[1]]).reset_index(drop=True)
    else:
        # VR ReservationPositionData freeform
        parsed_freeform = parse_vr_freeform_positions(par)

        if parsed_freeform.empty:
            # Cas valide : aucun objet placé après le header/MARKER -> on considère 0 placements
            par_points = pd.DataFrame(columns=[axes[0], axes[1], "name"])
            used_freeform = True
        else:
            par_points = parsed_freeform[[axes[0], axes[1], "name"]].copy()
            used_freeform = True

    par_points["participant_id"] = (np.arange(1, len(par_points) + 1)).astype(str)

    # Assignment
    S = sol_points[[axes[0], axes[1]]].to_numpy(float)
    P = par_points[[axes[0], axes[1]]].to_numpy(float)

    if assignment == "hungarian":
        matches = hungarian_assignment(S, P)
    else:
        matches = greedy_nearest_neighbor(S, P)

    # Per-object table
    rows = []
    for si, pi, dist in matches:
        sp = sol_points.iloc[si]
        pp = par_points.iloc[pi]
        rows.append(
            {
                "solution_id": sp["solution_id"],
                "solution_x": float(sp[axes[0]]),
                "solution_z": float(sp[axes[1]]),
                "participant_id": pp["participant_id"],
                "participant_name": str(pp.get("name", "")),
                "participant_x": float(pp[axes[0]]),
                "participant_z": float(pp[axes[1]]),
                "distance_m": float(dist),
                "within_tolerance": bool(dist <= tolerance_m),
            }
        )
    report = pd.DataFrame(rows)

    # Metrics
    n_expected = int(len(sol_points))
    n_placed = int(len(par_points))
    count_diff = n_placed - n_expected
    abs_count_diff = abs(count_diff)
    sum_distance_m = float(report["distance_m"].sum()) if not report.empty else 0.0
    duration_s = extract_duration_seconds(par, parsed_freeform if used_freeform else None)

    summary = {
        "solution_file": os.path.abspath(solution_path),
        "participant_file": os.path.abspath(part_path),
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
    }
    return report, summary, par_points, sol_points, matches


# ------------------------------ Plot ------------------------------
def draw_tolerance_circles(ax, sol_points: pd.DataFrame, axes=("position.x", "position.z"), radius=0.25):
    try:
        import matplotlib.patches as patches
        for _, r in sol_points.iterrows():
            circ = patches.Circle(
                (float(r[axes[0]]), float(r[axes[1]])),
                radius=radius,
                fill=False,
                alpha=0.25,
                linewidth=0.8,
            )
            ax.add_patch(circ)
    except Exception:
        pass


def make_overlay_plot(
    sol_points: pd.DataFrame,
    par_points: pd.DataFrame,
    matches: List[Tuple[int, int, float]],
    axes=("position.x", "position.z"),
    out_path="overlay.png",
    tolerance: Optional[float] = None,
    metrics: Optional[dict] = None,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.scatter(sol_points[axes[0]], sol_points[axes[1]], marker="o", s=60, label="Solution (référence)")
    ax.scatter(par_points[axes[0]], par_points[axes[1]], marker="x", s=70, label="Participant (final)")

    for si, pi, _ in matches:
        xs = [float(sol_points.iloc[si][axes[0]]), float(par_points.iloc[pi][axes[0]])]
        ys = [float(sol_points.iloc[si][axes[1]]), float(par_points.iloc[pi][axes[1]])]
        ax.plot(xs, ys, linewidth=0.9, alpha=0.7)

    if tolerance and tolerance > 0:
        draw_tolerance_circles(ax, sol_points, axes=axes, radius=tolerance)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{axes[0]} (m)")
    ax.set_ylabel(f"{axes[1]} (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if title:
        ax.set_title(title)

    if metrics:
        txt = "\n".join(
            [
                f"n_expected={metrics.get('n_expected')}  n_placed={metrics.get('n_placed')}",
                f"count_diff={metrics.get('count_diff')}  sum_dist={metrics.get('sum_distance_m'):.3f} m",
                f"duration={metrics.get('duration_s')} s  assign={metrics.get('assignment')}",
            ]
        )
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2),
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute VR participant performance vs solution (S2) using Hungarian assignment if available.")
    ap.add_argument("--solution", required=True, help="VR solution/correction CSV (e.g., VR_S2.csv)")
    ap.add_argument("--participant", required=True, help="Participant file path OR glob (e.g., .../*_ReservationPositionData.csv)")
    ap.add_argument("--tolerance", type=float, default=0.25, help="Tolerance in meters")
    ap.add_argument("--axes", default="position.x,position.z", help="Comma-separated axes (default: position.x,position.z)")
    ap.add_argument("--pick", default="newest", choices=["newest", "first"], help="When glob matches many files, which to pick")
    ap.add_argument("--assignment", default="hungarian", choices=["hungarian", "greedy"], help="Assignment strategy")
    ap.add_argument("--out_csv", default="performance_per_object.csv", help="Per-object CSV output")
    ap.add_argument("--out_json", default="performance_summary.json", help="Summary JSON output")
    ap.add_argument("--plot", default=None, help="Optional output path for overlay figure")
    ap.add_argument("--plot_title", default=None, help="Optional plot title")
    args = ap.parse_args()

    axes = tuple([a.strip() for a in args.axes.split(",") if a.strip()])
    if len(axes) != 2:
        raise ValueError("axes must contain exactly 2 columns, e.g. 'position.x,position.z'")

    report, summary, par_points, sol_points, matches = compute_performance(
        args.solution,
        args.participant,
        tolerance_m=args.tolerance,
        axes=axes,  # type: ignore[arg-type]
        pick=args.pick,
        assignment=args.assignment,
    )

    report.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.plot:
        make_overlay_plot(
            sol_points,
            par_points,
            matches,
            axes=axes,  # type: ignore[arg-type]
            out_path=args.plot,
            tolerance=args.tolerance,
            metrics=summary,
            title=args.plot_title,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved CSV:", os.path.abspath(args.out_csv))
    print("Saved JSON:", os.path.abspath(args.out_json))
    if args.plot:
        print("Saved FIG:", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()