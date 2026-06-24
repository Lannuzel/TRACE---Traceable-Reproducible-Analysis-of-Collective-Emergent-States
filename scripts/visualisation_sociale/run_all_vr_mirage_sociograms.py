#!/usr/bin/env python3
"""
Lance mirage_sociogram.py sur tous les groupes VR detectes sous une racine.

Le script cible est appele avec --group-dir, ce qui permet a
mirage_sociogram.py d'inferer automatiquement la modalite, le scenario et le
timepoint depuis le chemin du groupe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common.io_utils import find_groups, is_vr_group  # noqa: E402
from common.metadata import extract_condition, extract_scenario, extract_timepoint  # noqa: E402


def discover_vr_groups(data_dir: Path) -> list[Path]:
    groups = []
    for group_dir in find_groups(data_dir):
        if not is_vr_group(group_dir):
            continue
        if extract_condition(group_dir) != "VR":
            continue
        groups.append(group_dir)
    return sorted(set(groups))


def build_out_dir(group_dir: Path, data_dir: Path, out_root: Path, layout: str) -> Path:
    if layout == "group_id":
        return out_root / group_dir.name
    rel = group_dir.relative_to(data_dir)
    return out_root / rel


def add_optional_arg(cmd: list[str], name: str, value) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute mirage_sociogram.py sur tous les groupes VR.")
    parser.add_argument("--data-dir", type=Path, default=Path("D:/data_e2"), help="Racine des donnees.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualisation_sociale_all_vr",
        help="Dossier racine des sorties batch.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["relative", "group_id"],
        default="relative",
        help="Organisation des sorties: chemin relatif sous data-dir, ou seulement par group_id.",
    )
    parser.add_argument(
        "--script-path",
        type=Path,
        default=Path(__file__).resolve().with_name("mirage_sociogram.py"),
        help="Chemin du script mirage_sociogram.py.",
    )
    parser.add_argument("--group-name-filter", type=str, help="Ne garder que les groupes dont le nom contient ce texte.")
    parser.add_argument("--limit", type=int, help="Limiter le nombre de groupes traites.")
    parser.add_argument("--skip-existing", action="store_true", help="Sauter un groupe si run_metadata.json existe deja.")
    parser.add_argument("--dry-run", action="store_true", help="Afficher les commandes sans les lancer.")
    parser.add_argument("--stop-on-error", action="store_true", help="Arreter au premier echec.")

    parser.add_argument("--window-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--start-at", type=float)
    parser.add_argument("--end-at", type=float)
    parser.add_argument("--snapshot-at", type=float)
    parser.add_argument("--thr-db", type=float)
    parser.add_argument("--merge-gap", type=float)
    parser.add_argument("--min-speech-total-s", type=float)
    parser.add_argument("--min-speech-role-s", type=float)
    parser.add_argument("--audio-sr", type=int)
    parser.add_argument("--beep-scan-s", type=float)
    parser.add_argument("--frame-dpi", type=int, default=90)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-windows", type=int)

    parser.add_argument("--export-frames", action="store_true")
    parser.add_argument("--export-gif", action="store_true")
    parser.add_argument("--debug-timings", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    groups = discover_vr_groups(args.data_dir)
    if args.group_name_filter:
        needle = args.group_name_filter.lower()
        groups = [group_dir for group_dir in groups if needle in group_dir.name.lower()]
    if args.limit is not None:
        groups = groups[: max(0, int(args.limit))]

    if not groups:
        print("[WARN] Aucun groupe VR trouve.")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, int]] = []

    print(f"[INFO] {len(groups)} groupe(s) VR detecte(s).")
    for idx, group_dir in enumerate(groups, start=1):
        out_dir = build_out_dir(group_dir, args.data_dir, args.out_root, args.layout)
        metadata_path = out_dir / "run_metadata.json"

        scenario = extract_scenario(group_dir) or "UNK"
        timepoint = extract_timepoint(group_dir) or "UNK"
        print(f"[RUN {idx}/{len(groups)}] {group_dir.name} | VR | {scenario} | {timepoint}")

        if args.skip_existing and metadata_path.exists():
            print(f"  [SKIP] Sortie deja presente: {metadata_path}")
            continue

        cmd = [
            sys.executable,
            str(args.script_path),
            "--group-dir",
            str(group_dir),
            "--out-dir",
            str(out_dir),
            "--window-s",
            str(args.window_s),
            "--step-s",
            str(args.step_s),
            "--frame-dpi",
            str(args.frame_dpi),
            "--frame-stride",
            str(args.frame_stride),
        ]

        add_optional_arg(cmd, "--start-at", args.start_at)
        add_optional_arg(cmd, "--end-at", args.end_at)
        add_optional_arg(cmd, "--snapshot-at", args.snapshot_at)
        add_optional_arg(cmd, "--thr-db", args.thr_db)
        add_optional_arg(cmd, "--merge-gap", args.merge_gap)
        add_optional_arg(cmd, "--min-speech-total-s", args.min_speech_total_s)
        add_optional_arg(cmd, "--min-speech-role-s", args.min_speech_role_s)
        add_optional_arg(cmd, "--audio-sr", args.audio_sr)
        add_optional_arg(cmd, "--beep-scan-s", args.beep_scan_s)
        add_optional_arg(cmd, "--max-frames", args.max_frames)
        add_optional_arg(cmd, "--max-windows", args.max_windows)

        if args.export_frames:
            cmd.append("--export-frames")
        if args.export_gif:
            cmd.append("--export-gif")
        if args.debug_timings:
            cmd.append("--debug-timings")

        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures.append((group_dir, int(completed.returncode)))
            print(f"  [FAIL] code={completed.returncode}")
            if args.stop_on_error:
                break

    if failures:
        print("[WARN] Echecs detectes :")
        for group_dir, code in failures:
            print(f"  - {group_dir} -> code {code}")
        return 1

    print("[OK] Batch VR termine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
