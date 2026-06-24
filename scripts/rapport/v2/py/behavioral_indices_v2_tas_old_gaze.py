from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from py.behavioral_indices_v2_common import (
    OUTPUT_DIR,
    build_composite_score,
    correlation_metrics,
    ensure_output_dirs,
    load_base_bundle,
    metric_snapshot,
    perf_sample,
    write_csv,
    zscore_series,
)


TAS_VARIANTS: dict[str, dict[str, Any]] = {
    "I_TAS_b_v1": {
        "description": "Version v1 a 4 features retenues",
        "features": [
            ("shared_obj_ratio", +1),
            ("shared_obj_dur_mean_s", +1),
            ("gaze_entropy_mean_participants", -1),
            ("face_facial_synchrony", +1),
        ],
        "min_features": 3,
    },
    "I_TAS_b_gaze_pure": {
        "description": "Proxy mono-feature base sur shared_obj_ratio",
        "features": [("shared_obj_ratio", +1)],
        "min_features": 1,
    },
    "I_TAS_b_gaze_plus_affect": {
        "description": "Version reduite sans doublon shared_obj_dur_mean_s",
        "features": [
            ("shared_obj_ratio", +1),
            ("gaze_entropy_mean_participants", -1),
            ("face_facial_synchrony", +1),
        ],
        "min_features": 3,
    },
}


def run_tas_variants(
    base_bundle: dict[str, Any] | None = None,
    *,
    output_dir: Path = OUTPUT_DIR,
    logger=None,
) -> dict[str, Any]:
    bundle = base_bundle or load_base_bundle()
    log = logger or bundle["log"]
    output_dir, _ = ensure_output_dirs(output_dir)

    analysis_df = bundle["analysis_df"].copy()
    perf_df = perf_sample(analysis_df)
    summary_rows: list[dict[str, Any]] = []

    for variant_name, spec in TAS_VARIANTS.items():
        score, matrix, _ = build_composite_score(
            analysis_df,
            spec["features"],
            min_features=spec["min_features"],
            label=variant_name,
        )
        snapshot = metric_snapshot(
            index_name=variant_name,
            score=score,
            item_matrix=matrix,
            questionnaire_series=analysis_df["TSK"].rename("TSK"),
            perf_df=perf_df,
        )
        com_metrics = correlation_metrics(score, analysis_df["COM"], use_bootstrap=True)
        m2_metrics = correlation_metrics(
            score.loc[perf_df.index],
            perf_df["M2"],
            use_bootstrap=True,
            use_permutation=True,
            use_loo=True,
        )
        cf_metrics = correlation_metrics(
            score.loc[perf_df.index],
            perf_df["c_factor"],
            use_bootstrap=True,
            use_loo=True,
        )
        summary_rows.append(
            {
                "variant_name": variant_name,
                "description": spec["description"],
                "feature_signs": " | ".join(f"{name}:{'+' if sign > 0 else '-'}1" for name, sign in spec["features"]),
                "n_features": len(spec["features"]),
                "alpha_cronbach": snapshot["alpha_cronbach"],
                "omega_total": snapshot["omega_total"],
                "icc2k": snapshot["icc2k"],
                "icc2k_n_complete": snapshot["icc2k_n_complete"],
                "rho_tsk": snapshot["rho_questionnaire"],
                "p_tsk": snapshot["p_questionnaire"],
                "rho_com": com_metrics["rho"],
                "p_com": com_metrics["p_value"],
                "rho_score_perf": snapshot["rho_score_perf"],
                "p_score_perf": snapshot["p_score_perf"],
                "perm_p_score_perf": snapshot["perm_p_score_perf"],
                "loo_min_score_perf": snapshot["loo_min_score_perf"],
                "loo_max_score_perf": snapshot["loo_max_score_perf"],
                "rho_m2": m2_metrics["rho"],
                "p_m2": m2_metrics["p_value"],
                "perm_p_m2": m2_metrics["perm_p_value"],
                "loo_min_m2": m2_metrics["loo_min_rho"],
                "loo_max_m2": m2_metrics["loo_max_rho"],
                "rho_c_factor": cf_metrics["rho"],
                "p_c_factor": cf_metrics["p_value"],
                "loo_min_c_factor": cf_metrics["loo_min_rho"],
                "loo_max_c_factor": cf_metrics["loo_max_rho"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    write_csv(summary_df, output_dir / "tas_variants_comparison.csv")

    face_rows: list[dict[str, Any]] = []
    face_series = zscore_series(analysis_df["face_facial_synchrony"])
    for target, source_df in [
        ("SOC", analysis_df),
        ("COM", analysis_df),
        ("affect_alignment_idx", analysis_df),
        ("Score_perf_tsk", perf_df),
        ("c_factor", perf_df),
    ]:
        metrics = correlation_metrics(
            face_series.loc[source_df.index],
            source_df[target],
            use_bootstrap=True,
            use_permutation=target == "Score_perf_tsk",
            use_loo=target in {"Score_perf_tsk", "c_factor"},
        )
        face_rows.append(
            {
                "feature_name": "face_facial_synchrony",
                "target": target,
                **metrics,
            }
        )
    face_df = pd.DataFrame(face_rows)
    write_csv(face_df, output_dir / "tas_face_facial_synchrony.csv")

    log.info(
        f"Bloc C: variantes TAS exportees ({len(summary_df)} variantes, "
        f"{len(face_df)} correlations pour face_facial_synchrony)."
    )

    return {
        "summary_df": summary_df,
        "face_df": face_df,
    }
