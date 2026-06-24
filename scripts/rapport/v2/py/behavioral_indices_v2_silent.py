from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from py.behavioral_indices_v2_common import (
    OUTPUT_DIR,
    correlation_metrics,
    ensure_output_dirs,
    load_base_bundle,
    mannwhitney_rank_biserial,
    perf_sample,
    ratio_m1_m2,
    save_boxplot,
    save_labeled_scatter,
    write_csv,
)


PROFILE_DIMENSIONS = ["COR", "CRE", "SPE", "SOC", "TSK", "COM"]


def run_silent_division(
    base_bundle: dict[str, Any] | None = None,
    *,
    output_dir: Path = OUTPUT_DIR,
    logger=None,
) -> dict[str, Any]:
    bundle = base_bundle or load_base_bundle()
    log = logger or bundle["log"]
    output_dir, figures_dir = ensure_output_dirs(output_dir)

    indices_df = bundle["indices_df"].copy()
    analysis_df = bundle["analysis_df"].copy()
    perf_df = perf_sample(analysis_df)

    perf_df = perf_df.merge(
        indices_df[["group_id", "I_TAS_b", "I_TRS_b"]],
        on="group_id",
        how="left",
        suffixes=("", "_idx"),
    )
    perf_df["coordination_explicite_totale"] = pd.to_numeric(perf_df["I_TAS_b"], errors="coerce") + pd.to_numeric(perf_df["I_TRS_b"], errors="coerce")
    perf_df["ratio_M1_M2"] = ratio_m1_m2(perf_df)
    median_threshold = float(perf_df["coordination_explicite_totale"].median(skipna=True))
    perf_df["profile_coordination"] = np.where(
        perf_df["coordination_explicite_totale"] <= median_threshold,
        "silencieux",
        "verbal",
    )
    write_csv(perf_df, output_dir / "silent_division_profiles.csv")

    corr_rows = []
    for target in ["Score_perf_tsk", "c_factor", "ratio_M1_M2"]:
        metrics = correlation_metrics(
            perf_df["coordination_explicite_totale"],
            perf_df[target],
            use_bootstrap=True,
            use_permutation=target == "Score_perf_tsk",
            use_loo=True,
        )
        corr_rows.append({"target": target, **metrics})
    corr_df = pd.DataFrame(corr_rows)
    write_csv(corr_df, output_dir / "silent_division_correlations.csv")

    comparison_rows = []
    for target in ["Score_perf_tsk", "c_factor"] + PROFILE_DIMENSIONS:
        stats = mannwhitney_rank_biserial(
            perf_df.loc[perf_df["profile_coordination"] == "silencieux", target],
            perf_df.loc[perf_df["profile_coordination"] == "verbal", target],
        )
        comparison_rows.append(
            {
                "target": target,
                "profile_a": "silencieux",
                "profile_b": "verbal",
                **stats,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    write_csv(comparison_df, output_dir / "silent_division_profile_comparisons.csv")

    strat_rows = []
    for stratum_name, sub in [
        ("c_factor_ge_0", perf_df.loc[pd.to_numeric(perf_df["c_factor"], errors="coerce") >= 0].copy()),
        ("c_factor_lt_0", perf_df.loc[pd.to_numeric(perf_df["c_factor"], errors="coerce") < 0].copy()),
    ]:
        for variable in ["I_TAS_b", "I_TRS_b", "coordination_explicite_totale"]:
            metrics = correlation_metrics(
                sub[variable],
                sub["Score_perf_tsk"],
                use_bootstrap=False,
                use_loo=False,
            )
            strat_rows.append(
                {
                    "stratum": stratum_name,
                    "variable": variable,
                    **metrics,
                }
            )
    strat_df = pd.DataFrame(strat_rows)
    write_csv(strat_df, output_dir / "silent_division_cfactor_strata.csv")

    scatter_path = save_labeled_scatter(
        figures_dir / "silent_coordination_vs_performance.png",
        perf_df,
        "coordination_explicite_totale",
        "Score_perf_tsk",
        title="Coordination explicite totale vs performance",
        xlabel="I_TAS_b + I_TRS_b",
        ylabel="Score_perf_tsk",
    )
    boxplot_path = save_boxplot(
        figures_dir / "performance_by_silent_profile.png",
        perf_df,
        "profile_coordination",
        "Score_perf_tsk",
        title="Performance par profil silencieux / verbal",
        ylabel="Score_perf_tsk",
    )

    log.info(
        f"Bloc D: analyse division silencieuse exportee (n={len(perf_df)}, seuil median={median_threshold:.3f})."
    )

    return {
        "profiles_df": perf_df,
        "correlations_df": corr_df,
        "comparisons_df": comparison_df,
        "strata_df": strat_df,
        "scatter_path": scatter_path,
        "boxplot_path": boxplot_path,
        "median_threshold": median_threshold,
    }
