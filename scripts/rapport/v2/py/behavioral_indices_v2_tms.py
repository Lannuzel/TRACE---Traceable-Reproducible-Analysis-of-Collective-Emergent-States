from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from py.behavioral_indices_v2_common import (
    OUTPUT_DIR,
    correlation_metrics,
    ensure_output_dirs,
    fit_ols_zscored,
    load_base_bundle,
    perf_sample,
    write_csv,
)


TMS_FEATURES = [
    "audio_participation_entropy",
    "audio_avg_speaking_turn_duration_s",
    "audio_successful_interruption_ratio",
]

TMS_QUESTIONNAIRE_TARGETS = ["COR", "CRE", "SPE"]
TMS_PERFORMANCE_TARGETS = ["Score_perf_tsk", "M1", "M2", "c_factor"]
TMS_EXTERNAL_TARGETS = ["effort_task_norm", "skill_congruence_mean", "strategy_norm"]


def run_tms_decomposition(
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

    rows: list[dict[str, Any]] = []
    for feature_name in TMS_FEATURES:
        for target in TMS_QUESTIONNAIRE_TARGETS:
            metrics = correlation_metrics(
                analysis_df[feature_name],
                analysis_df[target],
                use_bootstrap=True,
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "analysis_family": "questionnaire",
                    "target": target,
                    **metrics,
                }
            )
        for target in TMS_PERFORMANCE_TARGETS:
            metrics = correlation_metrics(
                perf_df[feature_name],
                perf_df[target],
                use_bootstrap=True,
                use_permutation=True,
                use_loo=True,
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "analysis_family": "performance",
                    "target": target,
                    **metrics,
                }
            )
        for target in TMS_EXTERNAL_TARGETS:
            metrics = correlation_metrics(
                perf_df[feature_name],
                perf_df[target],
                use_bootstrap=True,
            )
            rows.append(
                {
                    "feature_name": feature_name,
                    "analysis_family": "external",
                    "target": target,
                    **metrics,
                }
            )

    decomp_df = pd.DataFrame(rows)
    write_csv(decomp_df, output_dir / "tms_features_decomposition.csv")

    coef_table, model_summary, vif_df = fit_ols_zscored(
        perf_df,
        "Score_perf_tsk",
        TMS_FEATURES,
    )
    write_csv(coef_table, output_dir / "tms_features_ols.csv")
    write_csv(vif_df, output_dir / "tms_features_ols_vif.csv")
    pd.DataFrame([model_summary]).to_csv(output_dir / "tms_features_ols_summary.csv", index=False, encoding="utf-8")

    log.info(
        f"Bloc B: decomposition TMS exportee ({len(decomp_df)} lignes de correlation, "
        f"OLS n={model_summary.get('n', 0)})."
    )

    return {
        "decomposition_df": decomp_df,
        "ols_df": coef_table,
        "ols_summary_df": pd.DataFrame([model_summary]),
        "vif_df": vif_df,
    }
