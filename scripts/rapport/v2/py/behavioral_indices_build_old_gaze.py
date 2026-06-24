from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "rapport_v2" / "VR_only" / "behavioral_indices_vr"
SEED = 42

EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}
NAN_THRESHOLD = 0.20

INDEX_SPEC: dict[str, dict[str, Any]] = {
    "I_TMS_b": {
        "description": "Coordination / specialisation comportementale (TMS)",
        "features": [
            ("audio_participation_entropy", +1),
            ("audio_avg_speaking_turn_duration_s", +1),
            ("audio_successful_interruption_ratio", +1),
            ("skill_congruence_mean", +1),
        ],
        "min_features": 3,
    },
    "I_TAS_b": {
        "description": "Attention partagee comportementale (TAS)",
        "features": [
            ("shared_obj_ratio", +1),
            ("shared_obj_dur_mean_s", +1),
            ("mutual_gaze", +1),
            ("gaze_entropy_mean_participants", -1),
            ("face_facial_synchrony", +1),
        ],
        "min_features": 3,
    },
    "I_TRS_b": {
        "description": "Coordination cognitive comportementale (TRS)",
        "features": [
            ("audio_backchannel_rate_per_min", +1),
            ("audio_floor_exchange_pause_mean_s", -1),
            ("affect_alignment_idx", +1),
            ("gaze_attention_coordination_idx", +1),
        ],
        "min_features": 3,
    },
}


SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config.inv_features_config import INV_FEATURES  # noqa: E402


@dataclass
class RunLogger:
    lines: list[str]

    def log(self, level: str, message: str) -> None:
        self.lines.append(f"[{level}] {message}")

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def warn(self, message: str) -> None:
        self.log("WARN", message)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _normalize_group_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    return s or None


def _normalize_modality(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if "VR" in s:
        return "VR"
    if "PC" in s:
        return "PC"
    return s


def _normalize_scenario(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if "S1" in s:
        return "S1"
    if "S2" in s:
        return "S2"
    return s


def _normalize_timepoint(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if s in {"1", "T1"}:
        return "T1"
    if s in {"2", "T2"}:
        return "T2"
    return s


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_questionnaire_group_scores(log: RunLogger) -> pd.DataFrame:
    candidate_paths = [
        RESULTS_DIR / "questionnaire" / "global" / "pruned" / "scores_dimension_par_participant_pruned.csv",
        RESULTS_DIR / "questionnaire" / "analyse" / "scores_dimension_par_participant.csv",
    ]
    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Aucun fichier de scores questionnaire n'a ete trouve.")

    df = _read_csv(path)
    df["group_id"] = df["Groupe"].apply(_normalize_group_id)
    df["timepoint"] = df["Session"].apply(_normalize_timepoint)
    df["scenario"] = df["Scenario"].apply(_normalize_scenario)
    df["modalite"] = df["Modalite"].apply(_normalize_modality)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()

    out = (
        df.pivot_table(
            index=["group_id", "timepoint", "scenario", "modalite"],
            columns="dimension",
            values="score",
            aggfunc="mean",
        )
        .reset_index()
    )
    log.info(
        f"Questionnaire groupe charge depuis {path.name} : {len(out)} lignes "
        f"({(out['modalite'] == 'VR').sum()} VR / {(out['modalite'] == 'PC').sum()} PC)."
    )
    return out


def _load_performance(log: RunLogger) -> pd.DataFrame:
    path = RESULTS_DIR / "performance_task" / "recap_scores_all.csv"
    df = _read_csv(path)
    df["group_id"] = df["groupe"].apply(_normalize_group_id)
    df["modalite"] = df["modalite"].apply(_normalize_modality)
    df["scenario"] = df["scenario"].apply(_normalize_scenario)
    df["timepoint"] = df["group_id"].apply(lambda x: "T2" if str(x).endswith("_2") else "T1")
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = ["group_id", "timepoint", "scenario", "modalite", "Score_perf_tsk", "M1_consignes_%", "M2_nombre_%"]
    out = df[keep_cols].rename(columns={"M1_consignes_%": "M1", "M2_nombre_%": "M2"})
    log.info(
        f"Performance chargee : {len(out)} lignes "
        f"({(out['modalite'] == 'VR').sum()} VR / {(out['modalite'] == 'PC').sum()} PC)."
    )
    return out


def _load_high_level(log: RunLogger) -> pd.DataFrame:
    path = RESULTS_DIR / "INV" / "high_level_features.csv"
    df = _read_csv(path)
    df["group_id"] = df["group_id"].apply(_normalize_group_id)
    df["modalite"] = df["condition"].apply(_normalize_modality)
    df["scenario"] = df["scenario"].apply(_normalize_scenario)
    df["timepoint"] = df["timepoint"].apply(_normalize_timepoint)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = [
        "group_id",
        "timepoint",
        "scenario",
        "modalite",
        "audio_avg_speaking_turn_duration_s",
        "audio_successful_interruption_ratio",
        "audio_backchannel_rate_per_min",
        "audio_floor_exchange_pause_mean_s",
        "face_facial_synchrony",
        "affect_alignment_idx",
        "gaze_mutual_gaze_ratio",
        "gaze_attention_coordination_idx",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"High-level INV charge : {len(out)} lignes.")
    return out


def _load_audio_features(log: RunLogger) -> pd.DataFrame:
    path = RESULTS_DIR / "INV" / "audio_features.csv"
    df = _read_csv(path)
    df["group_id"] = df["group_id"].apply(_normalize_group_id)
    df["modalite"] = df["condition"].apply(_normalize_modality)
    df["scenario"] = df["scenario"].apply(_normalize_scenario)
    df["timepoint"] = df["timepoint"].apply(_normalize_timepoint)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = ["group_id", "timepoint", "scenario", "modalite", "audio_participation_entropy"]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"Features audio chargees : {len(out)} lignes.")
    return out


def _load_gaze_overall(log: RunLogger) -> pd.DataFrame:
    path = RESULTS_DIR / "INV" / "gaze" / "ALL_metrics_overall.csv"
    df = _read_csv(path)
    df["group_id"] = df["group_id"].apply(_normalize_group_id)
    df["modalite"] = df["condition"].apply(_normalize_modality)
    df["scenario"] = df["scenario"].apply(_normalize_scenario)
    df["timepoint"] = df["timepoint"].apply(_normalize_timepoint)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = [
        "group_id",
        "timepoint",
        "scenario",
        "modalite",
        "shared_obj_ratio",
        "shared_obj_dur_mean_s",
        "mutual_gaze_ratio_mean_pairs",
        "gaze_entropy_mean_participants",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"Features gaze overall chargees : {len(out)} lignes.")
    return out


def _load_riedl(log: RunLogger) -> pd.DataFrame:
    path = RESULTS_DIR / "indices_collab" / "riedl_group_summary.csv"
    df = _read_csv(path)
    df["group_id"] = df["GroupID"].apply(_normalize_group_id)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = ["group_id", "skill_congruence_mean", "strategy_norm", "effort_task_sum", "effort_task_norm"]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    log.info(f"Table Riedl chargee : {len(out)} groupes.")
    return out


def _load_tci(scope: str, log: RunLogger) -> pd.DataFrame:
    filename = "c_scores_allowed.csv" if scope == "allowed" else "c_scores_all.csv"
    path = RESULTS_DIR / "TCI" / filename
    df = _read_csv(path)
    df["group_id"] = df["group_id"].apply(_normalize_group_id)
    df = df[~df["group_id"].isin(EXCLUDED_GROUPS)].copy()
    keep_cols = ["group_id", "c_score", "rme_mean", "rme_max", "rme_min"]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out = out.rename(columns={"c_score": f"c_score_{scope}"})
    log.info(f"TCI {scope} charge : {len(out)} groupes depuis {filename}.")
    return out


def build_master_dataset(log: RunLogger | None = None) -> pd.DataFrame:
    log = log or RunLogger([])
    questionnaire = _load_questionnaire_group_scores(log)
    performance = _load_performance(log)
    high_level = _load_high_level(log)
    audio = _load_audio_features(log)
    gaze = _load_gaze_overall(log)
    riedl = _load_riedl(log)
    tci_allowed = _load_tci("allowed", log)
    tci_all = _load_tci("all", log)

    base = questionnaire[questionnaire["modalite"] == "VR"].copy()
    base = base.merge(
        performance,
        on=["group_id", "timepoint", "scenario", "modalite"],
        how="left",
    )
    base = base.merge(
        high_level,
        on=["group_id", "timepoint", "scenario", "modalite"],
        how="left",
    )
    base = base.merge(
        audio,
        on=["group_id", "timepoint", "scenario", "modalite"],
        how="left",
    )
    base = base.merge(
        gaze,
        on=["group_id", "timepoint", "scenario", "modalite"],
        how="left",
    )
    base = base.merge(riedl, on="group_id", how="left")
    base = base.merge(tci_allowed, on="group_id", how="left")
    base = base.merge(tci_all, on="group_id", how="left")

    base = base.sort_values(["timepoint", "scenario", "group_id"]).reset_index(drop=True)
    log.info(f"Dataset maitre VR construit : {len(base)} groupes.")

    n_tci_allowed = int(base["c_score_allowed"].notna().sum()) if "c_score_allowed" in base.columns else 0
    n_tci_all = int(base["c_score_all"].notna().sum()) if "c_score_all" in base.columns else 0
    n_perf = int(base["Score_perf_tsk"].notna().sum()) if "Score_perf_tsk" in base.columns else 0
    log.info(f"Sous-echantillon VR avec performance disponible : n={n_perf}.")
    log.info(f"Sous-echantillon VR avec TCI allowed disponible : n={n_tci_allowed}.")
    log.info(f"Sous-echantillon VR avec TCI all disponible : n={n_tci_all}.")

    allowed_set = set(base.loc[base["c_score_allowed"].notna(), "group_id"])
    all_set = set(base.loc[base["c_score_all"].notna(), "group_id"])
    if allowed_set == all_set:
        log.info(
            "Dans le perimetre VR valide courant, c_scores_allowed et c_scores_all "
            "retombent sur le meme sous-ensemble de groupes."
        )
    else:
        log.warn(
            f"Les sous-ensembles TCI allowed ({sorted(allowed_set)}) et all ({sorted(all_set)}) different."
        )

    return base


def _zscore(series: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=0))
    if not math.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=series.index, dtype=float), mean, std
    return (numeric - mean) / std, mean, std


def resolve_index_features(vr_df: pd.DataFrame, log: RunLogger | None = None) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    log = log or RunLogger([])
    resolved: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for index_name, spec in INDEX_SPEC.items():
        retained: list[dict[str, Any]] = []
        for feature_name, sign in spec["features"]:
            cfg = INV_FEATURES.get(feature_name)
            row: dict[str, Any] = {
                "index_name": index_name,
                "candidate_feature": feature_name,
                "sign": sign,
                "in_config": cfg is not None,
                "theoretical_core": bool(cfg.get("theoretical_core", False)) if cfg is not None else False,
                "status": "",
                "nan_rate_vr": np.nan,
            }

            if cfg is None:
                row["status"] = "excluded_absent_from_config"
                log.warn(
                    f"{index_name}: feature '{feature_name}' absente de inv_features_config.py -> retiree."
                )
                rows.append(row)
                continue

            if feature_name not in vr_df.columns:
                row["status"] = "excluded_absent_from_dataset"
                log.warn(
                    f"{index_name}: feature '{feature_name}' presente dans la config mais absente du dataset VR -> retiree."
                )
                rows.append(row)
                continue

            nan_rate = float(vr_df[feature_name].isna().mean())
            row["nan_rate_vr"] = nan_rate
            if nan_rate > NAN_THRESHOLD:
                row["status"] = "excluded_nan_rate_gt_20pct"
                log.warn(
                    f"{index_name}: feature '{feature_name}' retiree (NaN={nan_rate:.3f} > 0.20)."
                )
                rows.append(row)
                continue

            z_values, mean, std = _zscore(vr_df[feature_name])
            if z_values.notna().sum() == 0 or not math.isfinite(std) or std == 0:
                row["status"] = "excluded_zero_variance"
                log.warn(
                    f"{index_name}: feature '{feature_name}' retiree (variance nulle ou standardisation impossible)."
                )
                rows.append(row)
                continue

            row["status"] = "retained"
            row["mean_vr"] = mean
            row["sd_vr"] = std
            rows.append(row)
            retained.append(
                {
                    "feature_name": feature_name,
                    "sign": sign,
                    "theoretical_core": bool(cfg.get("theoretical_core", False)),
                    "mean_vr": mean,
                    "sd_vr": std,
                }
            )
            if not bool(cfg.get("theoretical_core", False)):
                log.warn(
                    f"{index_name}: '{feature_name}' retenue mais theoretical_core=False dans la config actuelle."
                )

        resolved[index_name] = {
            "description": spec["description"],
            "min_features": int(spec["min_features"]),
            "retained_features": retained,
        }
        log.info(
            f"{index_name}: {len(retained)}/{len(spec['features'])} features retenues "
            f"(min requis={spec['min_features']})."
        )

    return resolved, pd.DataFrame(rows)


def compute_behavioral_indices(
    vr_df: pd.DataFrame,
    resolved_specs: dict[str, dict[str, Any]],
    log: RunLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = log or RunLogger([])
    feature_z = vr_df[["group_id", "timepoint", "scenario", "modalite"]].copy()
    indices_df = feature_z.copy()

    for index_name, spec in resolved_specs.items():
        signed_cols: list[str] = []
        retained = spec["retained_features"]
        for feat in retained:
            feature_name = feat["feature_name"]
            sign = feat["sign"]
            z_values, _, _ = _zscore(vr_df[feature_name])
            signed_col = f"{feature_name}__signed_z"
            feature_z[signed_col] = z_values * sign
            signed_cols.append(signed_col)

        short_name = index_name.replace("I_", "").replace("_b", "")
        min_features = int(spec["min_features"])
        if signed_cols:
            available = feature_z[signed_cols].notna().sum(axis=1)
            mean_scores = feature_z[signed_cols].mean(axis=1, skipna=True)
            median_scores = feature_z[signed_cols].median(axis=1, skipna=True)
            indices_df[index_name] = np.where(available >= min_features, mean_scores, np.nan)
            indices_df[f"{index_name}_median"] = np.where(available >= min_features, median_scores, np.nan)
            indices_df[f"n_features_{short_name}"] = available
        else:
            indices_df[index_name] = np.nan
            indices_df[f"{index_name}_median"] = np.nan
            indices_df[f"n_features_{short_name}"] = 0
            log.warn(f"{index_name}: aucune feature retenue, indice non calculable.")

    return indices_df, feature_z


def save_behavioral_outputs(
    indices_df: pd.DataFrame,
    feature_z_df: pd.DataFrame,
    feature_selection_df: pd.DataFrame,
    log: RunLogger,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "indices": output_dir / "behavioral_indices_vr.csv",
        "feature_z": output_dir / "index_features_z.csv",
        "feature_selection": output_dir / "feature_selection_summary.csv",
        "log": output_dir / "log_indices.txt",
    }
    indices_df.to_csv(paths["indices"], index=False, encoding="utf-8")
    feature_z_df.to_csv(paths["feature_z"], index=False, encoding="utf-8")
    feature_selection_df.to_csv(paths["feature_selection"], index=False, encoding="utf-8")
    log.write(paths["log"])
    return paths


def build_behavioral_indices_pipeline(
    output_dir: Path = OUTPUT_DIR,
    write_outputs: bool = True,
) -> dict[str, Any]:
    log = RunLogger([])
    log.info("Debut du calcul des indices comportementaux VR.")
    vr_master = build_master_dataset(log)
    resolved_specs, feature_selection_df = resolve_index_features(vr_master, log)
    indices_df, feature_z_df = compute_behavioral_indices(vr_master, resolved_specs, log)

    analysis_df = vr_master.merge(
        indices_df,
        on=["group_id", "timepoint", "scenario", "modalite"],
        how="left",
    )

    paths: dict[str, Path] = {}
    if write_outputs:
        paths = save_behavioral_outputs(indices_df, feature_z_df, feature_selection_df, log, output_dir=output_dir)

    log.info("Calcul des indices comportementaux termine.")
    if write_outputs:
        log.write(output_dir / "log_indices.txt")

    return {
        "vr_master": vr_master,
        "analysis_df": analysis_df,
        "indices_df": indices_df,
        "feature_z_df": feature_z_df,
        "resolved_specs": resolved_specs,
        "feature_selection_df": feature_selection_df,
        "log": log,
        "paths": paths,
    }


def main() -> None:
    bundle = build_behavioral_indices_pipeline()
    print(f"[OK] Indices : {bundle['paths'].get('indices')}")
    print(f"[OK] Features z : {bundle['paths'].get('feature_z')}")
    print(f"[OK] Selection : {bundle['paths'].get('feature_selection')}")
    print(f"[OK] Log : {bundle['paths'].get('log')}")


if __name__ == "__main__":
    main()
