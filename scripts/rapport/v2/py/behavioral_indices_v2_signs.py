from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from py.behavioral_indices_v2_common import (
    INDEX_ORDER,
    OUTPUT_DIR,
    PRIMARY_TARGETS,
    correlation_metrics,
    ensure_output_dirs,
    load_base_bundle,
    metric_snapshot,
    perf_sample,
    save_grid_histograms,
    sign_combo_id,
    sign_spec_to_text,
    write_csv,
)


TARGETED_SIGN_OVERRIDES: dict[str, dict[str, int]] = {
    "I_TMS_b": {
        "audio_successful_interruption_ratio": -1,
    },
    "I_TAS_b": {
        "face_facial_synchrony": -1,
        "gaze_entropy_dir_mean": +1,
        "gaze_entropy_mean_participants": +1,  # legacy alias
    },
    "I_TRS_b": {
        "audio_backchannel_rate_per_min": -1,
    },
}


def _retained_feature_signs(bundle: dict[str, Any], index_name: str) -> list[tuple[str, int]]:
    retained = bundle["resolved_specs"][index_name]["retained_features"]
    return [(feat["feature_name"], int(feat["sign"])) for feat in retained]


def _apply_targeted_overrides(
    feature_signs: list[tuple[str, int]],
    overrides: dict[str, int],
) -> list[tuple[str, int]]:
    return [(feature_name, int(overrides.get(feature_name, sign))) for feature_name, sign in feature_signs]


def _score_variant(
    df: pd.DataFrame,
    feature_signs: list[tuple[str, int]],
    *,
    min_features: int,
    label: str,
) -> tuple[pd.Series, pd.DataFrame]:
    from py.behavioral_indices_v2_common import build_composite_score

    score, matrix, _ = build_composite_score(df, feature_signs, min_features=min_features, label=label)
    return score, matrix


def run_sign_inversion(
    base_bundle: dict[str, Any] | None = None,
    *,
    output_dir: Path = OUTPUT_DIR,
    logger=None,
) -> dict[str, Any]:
    bundle = base_bundle or load_base_bundle()
    log = logger or bundle["log"]
    output_dir, figures_dir = ensure_output_dirs(output_dir)
    analysis_df = bundle["analysis_df"].copy()
    perf_df = perf_sample(analysis_df)

    targeted_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []

    for index_name in INDEX_ORDER:
        v1_feature_signs = _retained_feature_signs(bundle, index_name)
        inv_feature_signs = _apply_targeted_overrides(
            v1_feature_signs,
            TARGETED_SIGN_OVERRIDES.get(index_name, {}),
        )
        question_target = PRIMARY_TARGETS[index_name]
        min_features = int(bundle["resolved_specs"][index_name]["min_features"])

        for variant_name, feature_signs in [("v1", v1_feature_signs), ("v1_inv", inv_feature_signs)]:
            score, matrix = _score_variant(
                analysis_df,
                feature_signs,
                min_features=min_features,
                label=f"{index_name}_{variant_name}",
            )
            row = metric_snapshot(
                index_name=index_name,
                score=score,
                item_matrix=matrix,
                questionnaire_series=analysis_df[question_target].rename(question_target),
                perf_df=perf_df,
            )
            row.update(
                {
                    "variant": variant_name,
                    "feature_signs": sign_spec_to_text(feature_signs),
                }
            )
            targeted_rows.append(row)
            log.info(
                f"Bloc A1 {index_name}/{variant_name}: alpha={row['alpha_cronbach']:.3f} "
                f"rho_q={row['rho_questionnaire']:.3f} rho_perf={row['rho_score_perf']:.3f}."
            )

        features = [feature_name for feature_name, _ in v1_feature_signs]
        for signs in itertools.product([-1, 1], repeat=len(features)):
            combo_signs = list(zip(features, [int(s) for s in signs]))
            score, matrix = _score_variant(
                analysis_df,
                combo_signs,
                min_features=min_features,
                label=f"{index_name}_grid_{sign_combo_id(combo_signs)}",
            )
            q_metrics = correlation_metrics(score, analysis_df[question_target], use_bootstrap=False)
            p_metrics = correlation_metrics(
                score.loc[perf_df.index],
                perf_df["Score_perf_tsk"],
                use_bootstrap=False,
            )
            snapshot = metric_snapshot(
                index_name=index_name,
                score=score,
                item_matrix=matrix,
                questionnaire_series=analysis_df[question_target].rename(question_target),
                perf_df=perf_df,
            )
            grid_rows.append(
                {
                    "index_name": index_name,
                    "questionnaire_target": question_target,
                    "combo_id": sign_combo_id(combo_signs),
                    "feature_signs": sign_spec_to_text(combo_signs),
                    "alpha_cronbach": snapshot["alpha_cronbach"],
                    "omega_total": snapshot["omega_total"],
                    "rho_questionnaire": q_metrics["rho"],
                    "p_questionnaire": q_metrics["p_value"],
                    "n_questionnaire": q_metrics["n"],
                    "rho_score_perf": p_metrics["rho"],
                    "p_score_perf": p_metrics["p_value"],
                    "n_score_perf": p_metrics["n"],
                    "alpha_ge_050": bool(np.isfinite(snapshot["alpha_cronbach"]) and snapshot["alpha_cronbach"] >= 0.50),
                    "rho_score_perf_positive": bool(np.isfinite(p_metrics["rho"]) and p_metrics["rho"] > 0),
                }
            )

    targeted_df = pd.DataFrame(targeted_rows)
    grid_df = pd.DataFrame(grid_rows)
    if not grid_df.empty:
        grid_df["joint_support_alpha_ge_050_and_perf_positive"] = (
            grid_df["alpha_ge_050"].astype(bool) & grid_df["rho_score_perf_positive"].astype(bool)
        )
    write_csv(targeted_df, output_dir / "signes_inversion_cible.csv")
    write_csv(grid_df, output_dir / "signes_grid_search.csv")
    hist_path = save_grid_histograms(figures_dir / "grid_search_rho_performance_histograms.png", grid_df)

    return {
        "targeted_df": targeted_df,
        "grid_df": grid_df,
        "histogram_path": hist_path,
    }
