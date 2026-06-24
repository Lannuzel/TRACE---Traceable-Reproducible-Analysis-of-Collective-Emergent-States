from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
V2_DIR = Path(__file__).resolve().parents[1]
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from py.behavioral_indices_v2_common import (
    OUTPUT_DIR,
    PRIMARY_TARGETS,
    build_composite_score,
    correlation_metrics,
    ensure_output_dirs,
    load_base_bundle,
    perf_sample,
    save_v2_questionnaire_heatmap,
    sign_spec_to_text,
    write_csv,
    zscore_series,
)
from py.behavioral_indices_v2_signs import run_sign_inversion
from py.behavioral_indices_v2_silent import run_silent_division
from py.behavioral_indices_v2_tas import run_tas_variants
from py.behavioral_indices_v2_tms import run_tms_decomposition


TRS_V2_FEATURE_SIGNS = [
    ("audio_backchannel_rate_per_min", +1),
    ("audio_floor_exchange_pause_mean_s", +1),
    ("affect_alignment_idx", +1),
    ("gaze_attention_coordination_idx", +1),
]
TAS_V2_FEATURE_SIGNS = [("shared_obj_ratio", +1)]
TMS_V2_FEATURES = [
    "audio_participation_entropy",
    "audio_avg_speaking_turn_duration_s",
    "audio_successful_interruption_ratio",
]


def _tms_v2_feature_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    out = analysis_df[["group_id", "timepoint", "scenario", "modalite"]].copy()
    out["tms_participation_entropy_z"] = zscore_series(analysis_df["audio_participation_entropy"])
    out["tms_audio_avg_speaking_turn_duration_s_z"] = zscore_series(analysis_df["audio_avg_speaking_turn_duration_s"])
    out["tms_audio_successful_interruption_ratio_z"] = zscore_series(analysis_df["audio_successful_interruption_ratio"])
    return out


def _build_indices_v2_table(base_bundle: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis_df = base_bundle["analysis_df"].copy()
    indices_v2 = _tms_v2_feature_table(analysis_df)
    indices_v2["shared_obj_ratio_z"] = zscore_series(analysis_df["shared_obj_ratio"])
    indices_v2["I_TAS_b_v2"] = indices_v2["shared_obj_ratio_z"]

    trs_score, trs_matrix, trs_available = build_composite_score(
        analysis_df,
        TRS_V2_FEATURE_SIGNS,
        min_features=4,
        label="C_regulation_interactionnelle_v2",
    )
    trs_median = trs_matrix.median(axis=1, skipna=True)
    indices_v2["audio_backchannel_rate_per_min_z"] = zscore_series(analysis_df["audio_backchannel_rate_per_min"])
    indices_v2["audio_floor_exchange_pause_mean_s_z"] = zscore_series(analysis_df["audio_floor_exchange_pause_mean_s"])
    indices_v2["affect_alignment_idx_z"] = zscore_series(analysis_df["affect_alignment_idx"])
    indices_v2["gaze_attention_coordination_idx_z"] = zscore_series(analysis_df["gaze_attention_coordination_idx"])
    indices_v2["C_regulation_interactionnelle_v2"] = trs_score
    indices_v2["C_regulation_interactionnelle_v2_median"] = np.where(trs_available >= 4, trs_median, np.nan)
    indices_v2["I_TRS_b_v2"] = indices_v2["C_regulation_interactionnelle_v2"]
    indices_v2["n_features_C_regulation_interactionnelle_v2"] = trs_available

    metadata = {
        "tms_mode": "features_separees",
        "tms_features": TMS_V2_FEATURES,
        "tas_mode": "gaze_pure",
        "tas_feature_signs": TAS_V2_FEATURE_SIGNS,
        "trs_mode": "composite_additif_regulation_interactionnelle",
        "trs_feature_signs": TRS_V2_FEATURE_SIGNS,
    }
    return indices_v2, metadata


def _build_decision_table(
    base_bundle: dict[str, Any],
    signs_bundle: dict[str, Any],
    tas_bundle: dict[str, Any],
    silent_bundle: dict[str, Any],
) -> pd.DataFrame:
    internal = pd.read_csv(OUTPUT_DIR / "internal_consistency_indices.csv") if (OUTPUT_DIR / "internal_consistency_indices.csv").exists() else pd.DataFrame()
    if internal.empty and "paths" in base_bundle:
        internal = pd.DataFrame()
    grid_df = signs_bundle["grid_df"]
    targeted_df = signs_bundle["targeted_df"]
    silent_corr = silent_bundle["correlations_df"].set_index("target") if not silent_bundle["correlations_df"].empty else pd.DataFrame()
    tas_summary = tas_bundle["summary_df"].set_index("variant_name")

    tms_both = int(
        grid_df.loc[grid_df["index_name"] == "I_TMS_b", "joint_support_alpha_ge_050_and_perf_positive"].astype(bool).sum()
    ) if not grid_df.empty else 0
    trs_both = int(
        grid_df.loc[grid_df["index_name"] == "I_TRS_b", "joint_support_alpha_ge_050_and_perf_positive"].astype(bool).sum()
    ) if not grid_df.empty else 0
    tas_v1_alpha = float(tas_summary.loc["I_TAS_b_v1", "alpha_cronbach"]) if "I_TAS_b_v1" in tas_summary.index else np.nan
    tas_gaze_alpha = float(tas_summary.loc["I_TAS_b_gaze_pure", "alpha_cronbach"]) if "I_TAS_b_gaze_pure" in tas_summary.index else np.nan
    tas_plus_alpha = float(tas_summary.loc["I_TAS_b_gaze_plus_affect", "alpha_cronbach"]) if "I_TAS_b_gaze_plus_affect" in tas_summary.index else np.nan
    coord_rho_perf = float(silent_corr.loc["Score_perf_tsk", "rho"]) if "Score_perf_tsk" in silent_corr.index else np.nan
    coord_rho_cf = float(silent_corr.loc["c_factor", "rho"]) if "c_factor" in silent_corr.index else np.nan
    tms_inv = targeted_df[(targeted_df["index_name"] == "I_TMS_b") & (targeted_df["variant"] == "v1_inv")]
    tms_inv_q = float(tms_inv["rho_questionnaire"].iloc[0]) if not tms_inv.empty else np.nan
    trs_inv = targeted_df[(targeted_df["index_name"] == "I_TRS_b") & (targeted_df["variant"] == "v1_inv")]
    trs_inv_perf = float(trs_inv["rho_score_perf"].iloc[0]) if not trs_inv.empty else np.nan

    rows = [
        {
            "construct": "TMS",
            "decision": "Abandon de l'indice agrege ; maintien des trois features comme proxys separes",
            "retained_spec": "audio_participation_entropy, audio_avg_speaking_turn_duration_s, audio_successful_interruption_ratio",
            "reason": (
                "Alpha v1 negatif et structure instable ; une inversion partielle redresse certains liens, mais le signal reste porte par des features "
                "differenciees plutot que par un construit latent unique."
            ),
            "evidence_key": f"combinaisons alpha>=0.50 & rho_perf>0 = {tms_both}; rho_CRE v1_inv = {tms_inv_q:.3f}",
        },
        {
            "construct": "TAS",
            "decision": "Remplacement par la version gaze_pure (shared_obj_ratio z-scoree seule)",
            "retained_spec": "I_TAS_b_v2 = z(shared_obj_ratio)",
            "reason": (
                "La coherence de v1 est gonflee par la redondance interne shared_obj_ratio/shared_obj_dur_mean_s, tandis que face_facial_synchrony est "
                "quasi orthogonale aux composantes gaze."
            ),
            "evidence_key": f"alpha_v1 = {tas_v1_alpha:.3f}; alpha_gaze_pure = {tas_gaze_alpha}; alpha_gaze_plus_affect = {tas_plus_alpha:.3f}",
        },
        {
            "construct": "TRS",
            "decision": "Conservation sous forme de composite additif relabelle en score de regulation interactionnelle",
            "retained_spec": sign_spec_to_text(TRS_V2_FEATURE_SIGNS),
            "reason": (
                "Aucune combinaison de signes n'offre simultanement une coherence interne minimale et une relation positive a la performance ; le score est "
                "donc interprete comme une densite de regulation explicite, pas comme un indice latent de meilleur TRS."
            ),
            "evidence_key": f"combinaisons alpha>=0.50 & rho_perf>0 = {trs_both}; rho_perf v1_inv = {trs_inv_perf:.3f}",
        },
        {
            "construct": "H2",
            "decision": "Hypothese de division silencieuse retenue comme lecture principale",
            "retained_spec": "coordination_explicite_totale = I_TAS_b + I_TRS_b",
            "reason": (
                "Les groupes a coordination explicite faible ont une performance et un c-factor mediants plus eleves sur le sous-echantillon predictif."
            ),
            "evidence_key": f"rho(coord_total, perf) = {coord_rho_perf:.3f}; rho(coord_total, c_factor) = {coord_rho_cf:.3f}",
        },
    ]
    return pd.DataFrame(rows)


def _build_hypotheses_table(
    signs_bundle: dict[str, Any],
    silent_bundle: dict[str, Any],
    tas_bundle: dict[str, Any],
) -> pd.DataFrame:
    targeted_df = signs_bundle["targeted_df"]
    grid_df = signs_bundle["grid_df"]
    silent_corr = silent_bundle["correlations_df"].set_index("target")
    tas_summary = tas_bundle["summary_df"].set_index("variant_name")

    tms_inv = targeted_df[(targeted_df["index_name"] == "I_TMS_b") & (targeted_df["variant"] == "v1_inv")]
    tas_inv = targeted_df[(targeted_df["index_name"] == "I_TAS_b") & (targeted_df["variant"] == "v1_inv")]
    trs_inv = targeted_df[(targeted_df["index_name"] == "I_TRS_b") & (targeted_df["variant"] == "v1_inv")]

    h1_verdict = "partiel"
    h1_reason = (
        "L'inversion ciblee redresse partiellement TMS, mais ne corrige ni TAS ni TRS. "
        "Le grid search montre donc une dependance au signe pour certains sous-systemes, sans expliquer a lui seul le pattern inverse global."
    )
    if not tms_inv.empty and float(tms_inv["rho_questionnaire"].iloc[0]) > 0 and float(tms_inv["rho_score_perf"].iloc[0]) > 0:
        h1_reason += " Le cas TMS reste compatible avec une relecture specialisee du contexte BIM-VR."

    h2_verdict = "fort"
    h2_reason = (
        "La somme I_TAS_b + I_TRS_b est fortement et negativement associee a la performance et au c-factor, "
        "et les profils silencieux montrent des medianes superieures sur les deux variables."
    )
    coord_perf = float(silent_corr.loc["Score_perf_tsk", "rho"]) if "Score_perf_tsk" in silent_corr.index else np.nan
    coord_cf = float(silent_corr.loc["c_factor", "rho"]) if "c_factor" in silent_corr.index else np.nan

    h3_verdict = "fort"
    h3_reason = (
        "TMS v1 ne tient pas comme indice latent, TAS v1 est structurellement tire par une redondance interne, "
        "et TRS est plus defensible comme composite additif que comme construit unifie."
    )
    tas_v1_alpha = float(tas_summary.loc["I_TAS_b_v1", "alpha_cronbach"]) if "I_TAS_b_v1" in tas_summary.index else np.nan

    return pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "support_level": h1_verdict,
                "evidence": h1_reason,
                "key_metric": f"TMS v1_inv rho_CRE={float(tms_inv['rho_questionnaire'].iloc[0]) if not tms_inv.empty else np.nan:.3f}",
            },
            {
                "hypothesis": "H2",
                "support_level": h2_verdict,
                "evidence": h2_reason,
                "key_metric": f"rho coord_total/perf={coord_perf:.3f}; rho coord_total/c_factor={coord_cf:.3f}",
            },
            {
                "hypothesis": "H3",
                "support_level": h3_verdict,
                "evidence": h3_reason,
                "key_metric": f"alpha_TAS_v1={tas_v1_alpha:.3f}; combos TRS alpha>=0.50 & rho_perf>0 = {int(grid_df.loc[grid_df['index_name'] == 'I_TRS_b', 'joint_support_alpha_ge_050_and_perf_positive'].astype(bool).sum())}",
            },
        ]
    )


def _build_questionnaire_heatmap_rows(
    base_bundle: dict[str, Any],
    signs_bundle: dict[str, Any],
    indices_v2_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    analysis_df = base_bundle["analysis_df"].copy()
    targeted_df = signs_bundle["targeted_df"]
    rows: list[dict[str, Any]] = []

    label_map = {
        ("I_TMS_b", "v1"): "TMS v1",
        ("I_TMS_b", "v1_inv"): "TMS v1-inv",
        ("I_TAS_b", "v1"): "TAS v1",
        ("I_TAS_b", "v1_inv"): "TAS v1-inv",
        ("I_TRS_b", "v1"): "TRS v1",
        ("I_TRS_b", "v1_inv"): "TRS v1-inv",
    }
    for _, row in targeted_df.iterrows():
        key = (row["index_name"], row["variant"])
        if key not in label_map:
            continue
        rows.append(
            {
                "row_label": label_map[key],
                "questionnaire_target": row["questionnaire_target"],
                "rho_questionnaire": row["rho_questionnaire"],
            }
        )

    for feature_name, label in [
        ("tms_participation_entropy_z", "TMS v2: audio_participation_entropy"),
        ("tms_audio_avg_speaking_turn_duration_s_z", "TMS v2: mean_turn_duration"),
        ("tms_audio_successful_interruption_ratio_z", "TMS v2: successful_interruptions"),
    ]:
        metrics = correlation_metrics(indices_v2_df[feature_name], analysis_df["CRE"], use_bootstrap=False)
        rows.append(
            {
                "row_label": label,
                "questionnaire_target": "CRE",
                "rho_questionnaire": metrics["rho"],
            }
        )

    metrics_tas = correlation_metrics(indices_v2_df["I_TAS_b_v2"], analysis_df["TSK"], use_bootstrap=False)
    rows.append(
        {
            "row_label": "TAS v2: gaze_pure",
            "questionnaire_target": "TSK",
            "rho_questionnaire": metrics_tas["rho"],
        }
    )

    metrics_trs = correlation_metrics(indices_v2_df["C_regulation_interactionnelle_v2"], analysis_df["COM"], use_bootstrap=False)
    rows.append(
        {
            "row_label": "TRS v2: regulation_interactionnelle",
            "questionnaire_target": "COM",
            "rho_questionnaire": metrics_trs["rho"],
        }
    )
    return rows


def build_behavioral_indices_v2(
    *,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir, figures_dir = ensure_output_dirs(output_dir)
    base_bundle = load_base_bundle()
    log = base_bundle["log"]

    signs_bundle = run_sign_inversion(base_bundle, output_dir=output_dir, logger=log)
    tms_bundle = run_tms_decomposition(base_bundle, output_dir=output_dir, logger=log)
    tas_bundle = run_tas_variants(base_bundle, output_dir=output_dir, logger=log)
    silent_bundle = run_silent_division(base_bundle, output_dir=output_dir, logger=log)

    indices_v2_df, spec_meta = _build_indices_v2_table(base_bundle)
    write_csv(indices_v2_df, output_dir / "indices_v2.csv")

    decision_df = _build_decision_table(base_bundle, signs_bundle, tas_bundle, silent_bundle)
    write_csv(decision_df, output_dir / "specification_v2_decisions.csv")
    hypotheses_df = _build_hypotheses_table(signs_bundle, silent_bundle, tas_bundle)
    write_csv(hypotheses_df, output_dir / "hypotheses_support_v2.csv")

    heatmap_rows = _build_questionnaire_heatmap_rows(base_bundle, signs_bundle, indices_v2_df)
    heatmap_path = save_v2_questionnaire_heatmap(
        figures_dir / "heatmap_v1_v1inv_v2_questionnaires.png",
        heatmap_rows,
    )

    log.write(output_dir / "log_indices_v2.txt")

    return {
        "base_bundle": base_bundle,
        "signs": signs_bundle,
        "tms": tms_bundle,
        "tas": tas_bundle,
        "silent": silent_bundle,
        "indices_v2_df": indices_v2_df,
        "decision_df": decision_df,
        "hypotheses_df": hypotheses_df,
        "spec_meta": spec_meta,
        "questionnaire_heatmap_path": heatmap_path,
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "log_path": output_dir / "log_indices_v2.txt",
    }


def main() -> None:
    results = build_behavioral_indices_v2()
    print(f"[OK] Sorties v2 : {results['output_dir']}")
    print(f"[OK] Indices v2 : {results['output_dir'] / 'indices_v2.csv'}")


if __name__ == "__main__":
    main()
