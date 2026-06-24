#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_inv.py — Main entry point for INV (speech, gaze, face) analyses.

Runs selected INV analyses on a data directory, with filters on modality,
scenario, and which INV(s) to run.

Examples
--------
# Run all INV analyses on all data:
python run_inv.py --data-dir ../data_e2 --out-dir ../results

# Run only speech analysis:
python run_inv.py --data-dir ../data_e2 --inv speech --out-dir ../results

# Run speech + gaze:
python run_inv.py --data-dir ../data_e2 --inv speech gaze --out-dir ../results

# Compute HLF from existing INV outputs:
python run_inv.py --hlf-only \\
    --speech-csv ../results/INV/audio_features.csv \\
    --gaze-group-csv ../results/INV/gaze_directional/ALL_metrics_overall.csv \\
    --gaze-pair-csv ../results/INV/gaze_directional/ALL_metrics_pairs.csv \\
    --face-csv ../results/INV/face_emotion_metrics_all.csv \\
    --out-dir ../results/INV

# Run only INV structure analysis from an existing HLF audit CSV:
python run_inv.py --inv-structure-only \\
    --hlf-audit-csv ../results/INV/high_level_features_audit.csv \\
    --out-dir ../results/INV
"""

import sys
from pathlib import Path

# Ajoute le dossier parent (scripts/) au path pour importer common et les sous-modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import subprocess


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(p: str | None) -> str | None:
    """Resolve a user path to an absolute normalized string."""
    if p is None:
        return None
    return str(Path(p).expanduser().resolve())


def _default_inv_structure_out_dir(inv_out_dir: Path) -> Path:
    """
    Default location for analyze_inv_structure outputs.

    By convention, the default INV outputs live in results/INV while the
    structure analysis lives next to it in results/results_inv_structure.
    For custom output folders, keep the structure outputs nested under out_dir.
    """
    if inv_out_dir.name.lower() == "inv":
        return inv_out_dir.parent / "results_inv_structure"
    return inv_out_dir / "results_inv_structure"


def run_script(script_path: Path, args: list, label: str):
    """Run a Python script as a subprocess, printing its output."""
    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"  [{label}] {script_path.name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] {label} exited with code {result.returncode}")
    return result.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Run INV analyses (speech, gaze, face) and/or compute high-level features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory (e.g., ../data_e2). Required unless --hlf-only.",
    )
    ap.add_argument(
        "--inv",
        nargs="+",
        choices=["speech", "gaze", "face", "all"],
        default=["all"],
        help="Which INV analyses to run (default: all)",
    )
    ap.add_argument("--out-dir", type=str, default="results/INV", help="Output directory")

    # HLF-only mode
    ap.add_argument("--hlf-only", action="store_true", help="Skip INV analyses, only compute HLF from existing CSV outputs")
    ap.add_argument(
        "--speech-csv",
        type=str,
        default=None,
        help="Path to audio_features.csv for HLF.",
    )
    ap.add_argument("--gaze-group-csv", type=str, default=None, help="Path to gaze group metrics CSV (for HLF)")
    ap.add_argument("--gaze-pair-csv", type=str, default=None, help="Path to gaze pair metrics CSV (for HLF)")
    ap.add_argument("--face-csv", type=str, default=None, help="Path to face metrics CSV (for HLF)")
    ap.add_argument(
        "--inv-structure-only",
        action="store_true",
        help="Skip INV analyses and HLF, only run analyze_inv_structure.py from an existing HLF audit CSV",
    )
    ap.add_argument(
        "--hlf-audit-csv",
        type=str,
        default=None,
        help="Path to high_level_features_audit.csv for --inv-structure-only (default: <out-dir>/high_level_features_audit.csv)",
    )

    # Optional forwarded args
    ap.add_argument("--plot", action="store_true", help="Generate plots (forwarded to speech/face)")

    args = ap.parse_args()

    if args.hlf_only and args.inv_structure_only:
        ap.error("--hlf-only and --inv-structure-only cannot be used together")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_structure_out_dir = _default_inv_structure_out_dir(out_dir)

    invs = set(args.inv)
    if "all" in invs:
        invs = {"speech", "gaze", "face"}

    # Determine paths to scripts
    speech_script = SCRIPT_DIR / "analyse_inv" / "speech" / "analyze_audio.py"
    gaze_script = SCRIPT_DIR / "analyse_inv" / "gaze" / "analyze_gaze_directional.py"
    face_script = SCRIPT_DIR / "analyse_inv" / "face" / "analyze_aus_group.py"
    hlf_script = SCRIPT_DIR / "analyse_inv" / "hlf" / "compute_high_level_features.py"
    inv_structure_script = SCRIPT_DIR / "analyse_inv" / "analyze_inv_structure.py"

    speech_out = None
    gaze_group_out = None
    gaze_pair_out = None
    face_out = None
    hlf_audit_out = out_dir / "high_level_features_audit.csv"
    hlf_audit_input = None
    hlf_ran_successfully = False

    if not args.hlf_only and not args.inv_structure_only:
        if args.data_dir is None:
            ap.error("--data-dir is required unless --hlf-only or --inv-structure-only is set")

        data_dir = _resolve_path(args.data_dir)
        print(f"[INFO] data_dir resolved to: {data_dir}")
        print(f"[INFO] out_dir resolved to: {out_dir}")

        # ── Speech ──
        if "speech" in invs and speech_script.exists():
            speech_out = str((out_dir / "audio_features.csv").resolve())
            extra = ["--plot"] if args.plot else []
            run_script(speech_script, [data_dir, "--out", speech_out] + extra, "SPEECH")

        # ── Gaze (analyse directionnelle — sans maillage BIM) ──
        if "gaze" in invs and gaze_script.exists():
            gaze_out_dir = str((out_dir / "gaze_directional").resolve())
            corrected_dir = str((out_dir / ".." / ".." / "eyetracking_corrected").resolve())
            run_script(gaze_script, [
                "--corrected-dir", corrected_dir,
                "--data-dir", data_dir,
                "--out-dir", gaze_out_dir,
            ], "GAZE")
            gaze_group_out = str(Path(gaze_out_dir) / "ALL_metrics_overall.csv")
            gaze_pair_out = str(Path(gaze_out_dir) / "ALL_metrics_pairs.csv")

        # ── Face ──
        if "face" in invs and face_script.exists():
            face_out = str((out_dir / "face_emotion_metrics_all.csv").resolve())
            run_script(face_script, [data_dir, "--out", face_out], "FACE")

    elif args.hlf_only:
        # HLF-only mode: use provided paths
        speech_out = _resolve_path(args.speech_csv)
        gaze_group_out = _resolve_path(args.gaze_group_csv)
        gaze_pair_out = _resolve_path(args.gaze_pair_csv)
        face_out = _resolve_path(args.face_csv)
        print(f"[INFO] out_dir resolved to: {out_dir}")
    else:
        hlf_audit_input = _resolve_path(args.hlf_audit_csv)
        if hlf_audit_input is None:
            hlf_audit_input = str(hlf_audit_out.resolve())
        print(f"[INFO] out_dir resolved to: {out_dir}")

    # ── HLF (High-Level Features) ──
    if not args.inv_structure_only and hlf_script.exists():
        hlf_args = ["--out", str((out_dir / "high_level_features.csv").resolve())]
        _audio_feat_path = out_dir / "audio_features.csv"
        if _audio_feat_path.exists():
            hlf_args += ["--speech", str(_audio_feat_path.resolve())]
        elif speech_out and Path(speech_out).exists():
            hlf_args += ["--speech", speech_out]
        if gaze_group_out and Path(gaze_group_out).exists():
            hlf_args += ["--gaze-group", gaze_group_out]
        if gaze_pair_out and Path(gaze_pair_out).exists():
            hlf_args += ["--gaze-pair", gaze_pair_out]
        if face_out and Path(face_out).exists():
            hlf_args += ["--face", face_out]

        if len(hlf_args) > 2:  # at least --out + one input
            hlf_code = run_script(hlf_script, hlf_args, "HLF")
            hlf_ran_successfully = hlf_code == 0
        else:
            print("[INFO] Skipping HLF: no INV output CSV files found.")

    # ── INV structure analysis ──
    if inv_structure_script.exists():
        structure_data = None
        if args.inv_structure_only:
            structure_data = Path(hlf_audit_input)
        elif hlf_ran_successfully and hlf_audit_out.exists():
            structure_data = hlf_audit_out

        if structure_data is not None and structure_data.exists():
            inv_structure_out_dir.mkdir(parents=True, exist_ok=True)
            structure_args = [
                "--data",
                str(structure_data.resolve()),
                "--out",
                str(inv_structure_out_dir.resolve()),
            ]           
            structure_args_vr_only = [
                "--data",
                str(structure_data.resolve()),
                "--out",
                str(inv_structure_out_dir.resolve()),
                "--mode", "vr-only",
            ]
            run_script(inv_structure_script, structure_args, "INV STRUCTURE")
            run_script(inv_structure_script, structure_args_vr_only, "INV STRUCTURE (VR-only)")
        elif args.inv_structure_only:
            print(f"[INFO] Skipping INV STRUCTURE: HLF audit CSV not found at {hlf_audit_input}")

    print(f"\n[DONE] All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
