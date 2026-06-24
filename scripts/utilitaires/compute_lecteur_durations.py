#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_lecteur_durations.py
---------------------------------
Parcourt récursivement un dossier
→ cherche tous les 'lecteur.mp4'
→ extrait durée avec ffprobe
→ récupère groupe (bimXXX) + scénario (S1/S2)
→ export CSV : scenario + groupe + durée

Usage:
python compute_lecteur_durations.py "D:/data_e2"
"""

import subprocess
import csv
import sys
from pathlib import Path


def get_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def format_hms(seconds: float) -> str:
    if seconds < 0:
        return "NA"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02}:{m:02}:{s:02}"


def extract_group_and_scenario(video: Path):
    group = None
    scenario = None

    for part in video.parts:
        # groupe
        if group is None and part.startswith("bim"):
            group = part

        # scénario
        if scenario is None and part in ("S1", "S2"):
            scenario = part

        if group and scenario:
            break

    return group, scenario


def main(root):
    root = Path(root)

    rows = []

    for video in root.rglob("lecteur.mp4"):
        group, scenario = extract_group_and_scenario(video)

        duration = get_duration(video)

        rows.append([
            scenario,
            group,
            round(duration, 3),
            format_hms(duration),
            str(video)
        ])

        print(f"{scenario} | {group} -> {format_hms(duration)}")

    # tri : S1 avant S2, puis bimXXX
    def sort_key(r):
        scen = r[0] if r[0] in ("S1", "S2") else "S9"
        grp = r[1] or "bim999"
        return (scen, grp)

    rows.sort(key=sort_key)

    out_csv = root / "lecteur_durations.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["scenario", "group", "duration_seconds", "duration_hms", "path"])
        writer.writerows(rows)

    print(f"\nFichier créé : {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_lecteur_durations.py <root_folder>")
        sys.exit(1)

    main(sys.argv[1])
