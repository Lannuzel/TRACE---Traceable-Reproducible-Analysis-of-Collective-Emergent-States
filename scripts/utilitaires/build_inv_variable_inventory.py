#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a consolidated inventory of variables found under results/INV.

Outputs:
- inv_variable_inventory.csv : one row per variable with metadata and sources
- inv_dataset_inventory.csv  : one row per dataset pattern scanned

The script scans all CSV files recursively, groups repeated gaze files by a
normalized dataset pattern (for example gaze/<group>/metrics_overall.csv), and
enriches variables with information from:
- scripts/config/inv_features_config.py
- Longitudinale/ci_multimodal_features_dimensions.csv (if available)

The implementation intentionally uses only the Python standard library so it
can run even when pandas is not installed on the machine.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
PROJECT_DIR = SCRIPT_PATH.parents[2]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config.inv_features_config import (
    get_canonical_feature_name,
    get_feature_metadata,
    infer_family_from_name,
)


DEFAULT_INV_DIR = PROJECT_DIR / "results" / "INV"
DEFAULT_VARIABLE_OUT = DEFAULT_INV_DIR / "inv_variable_inventory.csv"
DEFAULT_DATASET_OUT = DEFAULT_INV_DIR / "inv_dataset_inventory.csv"
DIMENSIONS_CSV = PROJECT_DIR / "ci_multimodal_features_dimensions.csv"
GENERATED_OUTPUT_NAMES = {
    "inv_variable_inventory.csv",
    "inv_dataset_inventory.csv",
    "inv_inventory_errors.csv",
}


KNOWN_DATASET_FLAGS = {
    "audio_features.csv": "present_in_audio_features",
    "face_emotion_metrics_all.csv": "present_in_face_metrics",
    "ALL_metrics_overall.csv": "present_in_gaze_overall",
    "ALL_metrics_pairs.csv": "present_in_gaze_pairs",
    "ALL_metrics_participants.csv": "present_in_gaze_participants",
    "high_level_features.csv": "present_in_high_level_features",
    "high_level_features_audit.csv": "present_in_high_level_audit",
    "high_level_features_missingness.csv": "present_in_high_level_missingness",
}


def normalize_dataset_pattern(path: Path, base_dir: Path) -> str:
    """Replace repeated group folders by a generic <group> token."""
    rel_parts = list(path.relative_to(base_dir).parts)
    norm_parts: list[str] = []
    for part in rel_parts:
        if re.fullmatch(r"bim\d+(?:_\d+)?", part, flags=re.IGNORECASE):
            norm_parts.append("<group>")
        else:
            norm_parts.append(part)
    return "/".join(norm_parts)


def classify_dataset(path: Path) -> str:
    """Return a coarse dataset category based on file name and location."""
    name = path.name.lower()
    rel = path.as_posix().lower()

    if "window" in name:
        return "window_detail"
    if "participant" in name:
        return "participant_metrics"
    if "pair" in name:
        return "pair_metrics"
    if "missingness" in name:
        return "missingness_summary"
    if "audit" in name:
        return "audit_dataset"
    if name == "high_level_features.csv":
        return "high_level_dataset"
    if name == "audio_features.csv":
        return "audio_dataset"
    if name == "face_emotion_metrics_all.csv":
        return "face_dataset"
    if name == "metrics_overall.csv":
        return "gaze_group_metrics"
    if name == "all_metrics_overall.csv":
        return "gaze_overall_dataset"
    if name == "metrics_pairs.csv":
        return "gaze_group_pair_metrics"
    if name == "all_metrics_pairs.csv":
        return "gaze_pair_dataset"
    if name == "metrics_participants.csv":
        return "gaze_group_participant_metrics"
    if name == "all_metrics_participants.csv":
        return "gaze_participant_dataset"
    if "/gaze/" in rel:
        return "gaze_other"
    return "other_csv"


def classify_variable_kind(column: str) -> str:
    """Assign a broad variable type for easier filtering."""
    metadata_cols = {
        "group",
        "group_id",
        "group_base_id",
        "condition",
        "scenario",
        "session",
        "timepoint",
        "participant",
        "pair",
        "source_face",
    }

    if column in metadata_cols:
        return "metadata"
    if column.endswith("_source"):
        return "provenance"
    if column.startswith("z_"):
        return "zscore"
    if (
        column.endswith("_thr_used")
        or column.endswith("_nwin")
        or column in {
            "thr_mode",
            "q",
            "z_k",
            "joy_thr_abs",
            "sad_thr_abs",
            "min_episode_s",
            "merge_gap_s",
            "dt_s",
            "min_overlap_s",
            "win_s",
            "step_s",
            "min_success",
            "dt_calc_s",
            "dt_model_s",
            "dt_lect_s",
        }
    ):
        return "parameter"
    return "metric"


def infer_family_extended(column: str) -> str:
    """Infer a family even for raw variables not yet documented in config."""
    family = infer_family_from_name(column)
    if family:
        return family

    name = column.lower()

    audio_prefixes = (
        "speech_",
        "silence_",
        "speak_sil_",
        "turns_",
        "turn_to_",
        "backchannels_",
        "speaking_turn_",
        "int_",
        "sync_",
        "mean_pause",
        "mean_turn",
        "pause_",
        "overlap_",
        "interrupt",
        "participation_",
        "total_speech",
        "total_turns",
        "max_speech",
        "tms_",
    )
    face_prefixes = (
        "joy_",
        "sad_",
        "affect_",
        "face_",
        "au",
        "pos_neg_",
    )
    gaze_prefixes = (
        "gaze_",
        "gaze_convergence_",
        "gaze_entropy_dir",
        "mutual_gaze_",
        "pair_convergence_",
        "pair_mutual_gaze",
        # Legacy (ancienne analyse par objet)
        "shared_obj",
        "pair_shared_obj",
        "transition_prob_gaze",
        "n_objects_fixated",
    )

    if name.startswith(audio_prefixes):
        return "audio"
    if name.startswith(face_prefixes):
        return "face"
    if name.startswith(gaze_prefixes):
        return "gaze"
    return ""


def detect_delimiter(path: Path) -> str:
    """Infer the CSV delimiter from a small text sample."""
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    candidates = [",", ";", "\t"]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        return str(dialect.delimiter)
    except Exception:
        counts = {sep: sample.count(sep) for sep in candidates}
        return max(counts, key=counts.get)


def read_columns(path: Path) -> list[str]:
    """Read a CSV robustly and return its header."""
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, [])
    return [str(col).strip() for col in header]


def load_dimensions_catalog(path: Path) -> dict[str, dict[str, str]]:
    """Load the existing feature-to-dimension catalog when available."""
    if not path.exists():
        return {}

    try:
        delimiter = detect_delimiter(path)
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = list(reader)
    except Exception:
        return {}

    if not rows:
        return {}

    first_row = rows[0]
    if "feature_pipeline" not in first_row:
        return {}

    catalog: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("feature_pipeline", "")).strip()
        if not key:
            continue
        catalog[key] = {
            "dimension_liee": str(row.get("dimension_liee", "")).strip(),
            "modalite": str(row.get("modalite", "")).strip(),
            "human_label": str(row.get("feature", "")).strip(),
            "sens_de_la_feature": str(row.get("sens_de_la_feature", "")).strip(),
            "calcul": str(row.get("calcul", "")).strip(),
            "source": str(row.get("source", "")).strip(),
        }
    return catalog


def join_sorted(values: set[str]) -> str:
    return " | ".join(sorted(v for v in values if v))


def stringify_list(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, (list, tuple, set)):
        return " | ".join(str(v) for v in values)
    return str(values)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to CSV with a stable header order."""
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_inventory(inv_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    csv_files = sorted(
        path for path in inv_dir.rglob("*.csv")
        if path.name not in GENERATED_OUTPUT_NAMES
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {inv_dir}")

    dimensions_catalog = load_dimensions_catalog(DIMENSIONS_CSV)

    variable_sources: dict[str, set[str]] = defaultdict(set)
    variable_files: dict[str, set[str]] = defaultdict(set)
    variable_basenames: dict[str, set[str]] = defaultdict(set)
    variable_categories: dict[str, set[str]] = defaultdict(set)
    variable_dataset_types: dict[str, set[str]] = defaultdict(set)
    variable_examples: dict[str, str] = {}
    dataset_columns: dict[str, set[str]] = defaultdict(set)
    dataset_files: dict[str, set[str]] = defaultdict(set)
    dataset_basenames: dict[str, set[str]] = defaultdict(set)
    dataset_types: dict[str, set[str]] = defaultdict(set)

    errors: list[dict[str, str]] = []

    for csv_path in csv_files:
        rel = csv_path.relative_to(inv_dir).as_posix()
        dataset_pattern = normalize_dataset_pattern(csv_path, inv_dir)
        dataset_type = classify_dataset(csv_path)

        try:
            columns = read_columns(csv_path)
        except Exception as exc:
            errors.append({"file": rel, "error": str(exc)})
            continue

        dataset_columns[dataset_pattern].update(columns)
        dataset_files[dataset_pattern].add(rel)
        dataset_basenames[dataset_pattern].add(csv_path.name)
        dataset_types[dataset_pattern].add(dataset_type)

        for column in columns:
            variable_sources[column].add(dataset_pattern)
            variable_files[column].add(rel)
            variable_basenames[column].add(csv_path.name)
            variable_categories[column].add(classify_variable_kind(column))
            variable_dataset_types[column].add(dataset_type)
            variable_examples.setdefault(column, rel)

    dataset_rows: list[dict[str, Any]] = []
    for dataset_pattern in sorted(dataset_columns):
        dataset_rows.append(
            {
                "dataset_pattern": dataset_pattern,
                "dataset_kind": join_sorted(dataset_types[dataset_pattern]),
                "n_distinct_columns": len(dataset_columns[dataset_pattern]),
                "n_concrete_files": len(dataset_files[dataset_pattern]),
                "basenames": join_sorted(dataset_basenames[dataset_pattern]),
                "example_files": join_sorted(dataset_files[dataset_pattern]),
                "columns": join_sorted(dataset_columns[dataset_pattern]),
            }
        )

    variable_rows: list[dict[str, Any]] = []
    for variable in sorted(variable_sources):
        canonical = get_canonical_feature_name(variable)
        config_meta = get_feature_metadata(variable) or {}
        family = config_meta.get("family") or infer_family_extended(variable)
        dim_meta = dimensions_catalog.get(variable) or dimensions_catalog.get(canonical) or {}

        row: dict[str, Any] = {
            "variable": variable,
            "variable_kind": join_sorted(variable_categories[variable]),
            "canonical_feature": canonical,
            "family": family,
            "documented_in_config": bool(config_meta),
            "documented_in_dimensions_catalog": bool(dim_meta),
            "description_short": config_meta.get("description", "") or dim_meta.get("human_label", ""),
            "theoretical_dimension": dim_meta.get("dimension_liee", ""),
            "human_label": dim_meta.get("human_label", ""),
            "interpretation": dim_meta.get("sens_de_la_feature", ""),
            "calculation": dim_meta.get("calcul", ""),
            "literature_or_source": dim_meta.get("source", ""),
            "priority": config_meta.get("priority", ""),
            "core": config_meta.get("core", ""),
            "core_hl": config_meta.get("core_hl", ""),
            "drop_if_redundant": config_meta.get("drop_if_redundant", ""),
            "redundant_with": stringify_list(config_meta.get("redundant_with", [])),
            "reason": config_meta.get("reason", ""),
            "found_in_n_dataset_patterns": len(variable_sources[variable]),
            "found_in_n_files": len(variable_files[variable]),
            "dataset_types": join_sorted(variable_dataset_types[variable]),
            "source_basenames": join_sorted(variable_basenames[variable]),
            "source_dataset_patterns": join_sorted(variable_sources[variable]),
            "example_file": variable_examples.get(variable, ""),
        }

        for basename, output_col in KNOWN_DATASET_FLAGS.items():
            row[output_col] = basename in variable_basenames[variable]

        variable_rows.append(row)

    return variable_rows, dataset_rows, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a consolidated inventory of variables under results/INV."
    )
    parser.add_argument(
        "--inv-dir",
        type=Path,
        default=DEFAULT_INV_DIR,
        help=f"Directory to scan (default: {DEFAULT_INV_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_VARIABLE_OUT,
        help=f"CSV output for variable inventory (default: {DEFAULT_VARIABLE_OUT})",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=DEFAULT_DATASET_OUT,
        help=f"CSV output for dataset inventory (default: {DEFAULT_DATASET_OUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inv_dir = args.inv_dir.resolve()
    out_path = args.out.resolve()
    dataset_out_path = args.dataset_out.resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_out_path.parent.mkdir(parents=True, exist_ok=True)

    variable_rows, dataset_rows, errors = build_inventory(inv_dir)
    write_csv(variable_rows, out_path)
    write_csv(dataset_rows, dataset_out_path)

    if errors:
        errors_path = inv_dir / "inv_inventory_errors.csv"
        write_csv(errors, errors_path)
        print(f"[WARN] Some files could not be scanned: {errors_path}")

    print(f"[OK] Variable inventory: {out_path}")
    print(f"[OK] Dataset inventory: {dataset_out_path}")
    print(f"[INFO] {len(variable_rows)} variables inventoried across {len(dataset_rows)} dataset patterns")


if __name__ == "__main__":
    main()
