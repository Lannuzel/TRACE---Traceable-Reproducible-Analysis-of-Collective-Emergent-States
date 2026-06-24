"""
Helpers de chargement, normalisation et fusion des donnees pour le rapport CI.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from pathlib import Path as _Path
from typing import Optional

import numpy as np
import pandas as pd

_scripts_dir = _Path(__file__).resolve().parents[2]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

_v2_dir = _Path(__file__).resolve().parents[1]
if str(_v2_dir) not in sys.path:
    sys.path.insert(0, str(_v2_dir))

from config import (
    AUDIO_ALIAS_PAIRS,
    CORE_FACE,
    CORE_GAZE,
    CORE_HL,
    CORE_RIEDL_COLS,
    CORE_SPEECH,
    CORE_FACE_V2 as V2_CORE_FACE,
    CORE_GAZE_V2 as V2_CORE_GAZE,
    CORE_HL_V2 as V2_CORE_HL,
    CORE_SPEECH_V2 as V2_CORE_SPEECH,
    PERFORMANCE_ANALYSIS_COLS,
    PERFORMANCE_EXCLUDED_COLS,
    filter_inv_dataframe,
    filter_inv_feature_names,
    is_excluded_inv_feature,
    infer_family_from_name,
)


UNIT_KEY_PRIORITY = ["group_id", "timepoint", "scenario", "modalite", "condition"]

ID_LIKE_COLS = {
    "group_id",
    "group_base_id",
    "timepoint",
    "scenario",
    "modalite",
    "condition",
    "session",
    "participant",
    "participant_id",
    "source_file",
    "_gaze_source",
    "_gaze_subdir",
    "dimension",
    "source",
}

QUESTIONNAIRE_DIMENSIONS = {"COM", "COR", "CRE", "SOC", "SPE", "TSK"}
PERFORMANCE_COLS = {
    *PERFORMANCE_ANALYSIS_COLS,
}

EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}

EXCLUSION_REASONS = {
    "bim002": "pas de fichier performance + donnees semblent corrompues",
    "bim065_2": "pas de fichier performance + pas de marker",
    "bim032": "pas de fichier performance, pas d'audio Mod/Calc, pas de marker Lecteur",
    "bim075": "piste audio absente + aucun marker + pas de dossier performance",
}


def add_audio_canonical_aliases(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Duplique les colonnes audio legacy vers leurs noms canoniques `audio_*`
    attendus par la v2 du rapport, sans supprimer les alias historiques.
    """
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    out = df.copy()
    for canonical, alias in AUDIO_ALIAS_PAIRS:
        if canonical not in out.columns and alias in out.columns:
            out[canonical] = out[alias]
    return out


def extract_timepoint_from_group_like(val) -> str | None:
    """Extrait T1/T2 depuis un identifiant de groupe type bim066 ou bim066_2."""
    if pd.isna(val):
        return None

    s = str(val).strip().lower()

    if re.fullmatch(r"bim\d{3}_2", s):
        return "T2"
    if re.fullmatch(r"bim\d{3}", s):
        return "T1"

    m = re.search(r"\bbim\d{3}(_2)?\b", s)
    if m:
        return "T2" if m.group(1) == "_2" else "T1"

    return None


def read_csv_auto(path: Path) -> pd.DataFrame | None:
    """
    Lecture CSV robuste :
    - essaie plusieurs couples (encodage, separateur, decimal)
    - choisit la lecture qui maximise d'abord le nombre de colonnes,
      puis le nombre de colonnes numeriques plausibles
    """
    if path is None or not Path(path).exists():
        return None

    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    seps = [";", ",", "\t"]
    decimals = {";": ",", ",": ".", "\t": "."}

    best_df = None
    best_score = (-1, -1)

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine="python", decimal=decimals[sep])
                if df is None or df.empty:
                    continue

                n_cols = df.shape[1]
                num_like = 0
                for c in df.columns:
                    if df[c].dtype == object:
                        s = (
                            df[c]
                            .astype(str)
                            .str.strip()
                            .str.replace(",", ".", regex=False)
                            .str.replace("%", "", regex=False)
                        )
                        num = pd.to_numeric(s, errors="coerce")
                        if num.notna().sum() >= max(3, int(0.1 * len(num))):
                            num_like += 1
                    elif pd.api.types.is_numeric_dtype(df[c]):
                        num_like += 1

                score = (n_cols, num_like)
                if score > best_score:
                    best_score = score
                    best_df = df
            except Exception:
                continue

    if best_df is not None:
        return best_df

    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return None


def add_group_base_id(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Ajoute group_base_id en retirant le suffixe final _N."""
    if df is None or df.empty:
        return df
    if "group_id" not in df.columns:
        return df

    out = df.copy()
    out["group_base_id"] = (
        out["group_id"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"_\d+$", "", regex=True)
    )
    return out


def exclude_bad_groups(df: pd.DataFrame | None, extra_excluded: list[str] | tuple[str, ...] | set[str] | None = None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    if "group_id" not in df.columns:
        return df

    out = df.copy()
    out["group_id"] = out["group_id"].astype(str).str.strip().str.lower()
    exclude_exact: set[str] = set(EXCLUDED_GROUPS)
    exclude_base: set[str] = set()
    if extra_excluded:
        for raw in extra_excluded:
            for token in re.split(r"[;,]", str(raw)):
                tok = token.strip().lower()
                if not tok:
                    continue
                if re.search(r"_\d+$", tok):
                    exclude_exact.add(tok)
                else:
                    exclude_base.add(tok)
    if exclude_base:
        base = out["group_id"].str.replace(r"_\d+$", "", regex=True)
        mask = out["group_id"].isin(exclude_exact) | base.isin(exclude_base)
    else:
        mask = out["group_id"].isin(exclude_exact)
    out = out[~mask].copy()
    return out.reset_index(drop=True)


def coerce_numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> pd.DataFrame:
    """Convertit en numerique les colonnes object qui ressemblent a des nombres."""
    if df is None or df.empty:
        return df

    exclude = exclude or set()
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip()
            s = s.str.replace(",", ".", regex=False).str.replace("%", "", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() >= max(3, int(0.2 * len(num))):
                out[c] = num
    return out


def available_unit_cols(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty:
        return []
    cols = [c for c in UNIT_KEY_PRIORITY if c in df.columns]
    return cols if "group_id" in cols else []


def common_unit_cols(*dfs: pd.DataFrame) -> list[str]:
    valid = [set(available_unit_cols(df)) for df in dfs if df is not None and not df.empty]
    if not valid:
        return ["group_id"]
    inter = set.intersection(*valid)
    out = [c for c in UNIT_KEY_PRIORITY if c in inter]
    return out if "group_id" in out else ["group_id"]


def has_real_timepoint(df: pd.DataFrame | None) -> bool:
    if df is None or df.empty or "timepoint" not in df.columns:
        return False
    tp = df["timepoint"].dropna().astype(str).str.upper().str.strip()
    return tp.isin(["T1", "T2"]).any()


def analysis_keys_for_df(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty or "group_id" not in df.columns:
        return []
    if has_real_timepoint(df):
        return ["group_id", "timepoint"]
    return ["group_id"]


def aggregate_numeric_by_unit(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty or "group_id" not in df.columns:
        return pd.DataFrame()

    group_cols = analysis_keys_for_df(df)
    if not group_cols:
        return pd.DataFrame()

    value_cols = [c for c in value_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    out = df.groupby(group_cols, dropna=False)[value_cols].mean().reset_index()
    if "timepoint" in out.columns:
        out = out.dropna(subset=["timepoint"])
    return out


def merge_on_unit(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy() if right is not None else pd.DataFrame()
    if right is None or right.empty:
        return left.copy()
    if "group_id" not in left.columns or "group_id" not in right.columns:
        return left.copy()

    left_has_tp = has_real_timepoint(left)
    right_has_tp = has_real_timepoint(right)

    if left_has_tp and right_has_tp and "timepoint" in left.columns and "timepoint" in right.columns:
        keys = ["group_id", "timepoint"]
    else:
        keys = ["group_id"]

    return left.merge(right, on=keys, how=how)


def normalize_timepoint(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "timepoint" not in df.columns:
        return df
    df["timepoint"] = df["timepoint"].astype(str).str.upper().str.strip()
    df.loc[~df["timepoint"].isin(["T1", "T2"]), "timepoint"] = np.nan
    return df


def normalize_group(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise l'identifiant de groupe sans supprimer le suffixe _2."""
    if df is None or df.empty:
        return df

    out = df.copy()
    candidate_cols = ["group_id", "GroupID", "groupe", "Groupe", "group"]
    col = next((c for c in candidate_cols if c in out.columns), None)
    if col is None:
        return out

    def _norm(val):
        if pd.isna(val):
            return val
        s = str(val).strip().lower()
        m = re.search(r"\bbim\s*0*(\d{1,3})(?:_(\d+))?\b", s)
        if not m:
            return None

        num = int(m.group(1))
        suffix = m.group(2)
        base = f"bim{num:03d}"
        if suffix is not None:
            return f"{base}_{suffix}"
        return base

    out["group_id"] = out[col].apply(_norm)
    out = out[out["group_id"].notna()].reset_index(drop=True)
    return out


def harmonize_timepoint(
    df: pd.DataFrame | None,
    *,
    session_col: str = "session",
    raw_group_candidates: list[str] | None = None,
) -> pd.DataFrame | None:
    """Harmonise la colonne timepoint a partir de plusieurs sources possibles."""
    if df is None or df.empty:
        return df

    out = df.copy()
    raw_group_candidates = raw_group_candidates or ["group_id", "GroupID", "groupe", "Groupe", "group"]

    if "timepoint" not in out.columns:
        out["timepoint"] = np.nan

    tp = out["timepoint"].astype(str).str.upper().str.strip()
    tp = tp.mask(tp.isin(["NAN", "NONE", ""]))

    if session_col in out.columns:
        sess = out[session_col].astype(str).str.upper().str.strip()
        sess = sess.replace({
            "1": "T1",
            "2": "T2",
            "S1": "T1",
            "S2": "T2",
            "T1": "T1",
            "T2": "T2",
            "SESSION1": "T1",
            "SESSION2": "T2",
        })
        tp = tp.where(tp.isin(["T1", "T2"]), sess)

    for c in raw_group_candidates:
        if c in out.columns:
            inferred = out[c].apply(extract_timepoint_from_group_like)
            tp = tp.where(tp.isin(["T1", "T2"]), inferred)

    out["timepoint"] = tp
    out.loc[~out["timepoint"].isin(["T1", "T2"]), "timepoint"] = np.nan
    return out


def _post_load_cleanup(df: pd.DataFrame, results_dir: Path, source_file: Optional[Path] = None) -> pd.DataFrame:
    """Normalise ids, timepoint, colonnes numeriques et memorise la source."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_group(df)
    df = add_group_base_id(df)
    df = harmonize_timepoint(
        df,
        session_col="session",
        raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
    )
    df = normalize_timepoint(df)
    df = coerce_numeric_columns(
        df,
        exclude={
            "group_id", "group_base_id", "timepoint", "modalite", "scenario",
            "session", "condition", "dimension", "participant",
        },
    )

    if source_file is not None:
        try:
            df["source_file"] = str(source_file.relative_to(results_dir))
        except Exception:
            df["source_file"] = str(source_file)

    df = exclude_bad_groups(df)
    return df.reset_index(drop=True)


def load_performance(results_dir: Path) -> pd.DataFrame:
    """Charge la performance depuis performance_task/recap_scores_all.csv ou les sous-dossiers par modalite."""
    all_path_candidates = [
        results_dir / "performance_task" / "recap_scores_all.csv",
        results_dir / "recap_scores_all.csv",
    ]
    all_path = next((p for p in all_path_candidates if p.exists()), all_path_candidates[0])

    if all_path.exists():
        df = read_csv_auto(all_path)
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "GroupID": "group_id",
            "groupe": "group_id",
            "Groupe": "group_id",
            "Modalite": "modalite",
            "Scenario": "scenario",
        })

        if "group_id" in df.columns:
            df["timepoint"] = df["group_id"].apply(extract_timepoint_from_group_like)

        df = _post_load_cleanup(df, results_dir, all_path)

        if "modalite" in df.columns:
            df["modalite"] = df["modalite"].astype(str).str.upper().str.strip()
        if "scenario" in df.columns:
            df["scenario"] = df["scenario"].astype(str).str.upper().str.strip()

        for c in ["group_id", "modalite", "scenario"]:
            if c in df.columns:
                df = df[df[c].notna()]

        return df.reset_index(drop=True)

    frames = []
    for scen in ["S1", "S2"]:
        for folder in [
            f"performance_task/performance_PC/{scen}",
            f"performance_PC/{scen}",
            f"PC_performance_revit/{scen}",
        ]:
            p = results_dir / folder
            if p.exists():
                for f in p.glob("recap_scores*.csv"):
                    df = read_csv_auto(f)
                    if df is None or df.empty:
                        continue
                    df = _post_load_cleanup(df, results_dir, f)
                    df["modalite"] = "PC"
                    df["scenario"] = scen
                    frames.append(df)
                break

    for scen in ["S1", "S2"]:
        for folder in [
            f"performance_task/performance_VR/{scen}",
            f"performance_VR/{scen}",
            f"VR_performance/{scen}",
        ]:
            p = results_dir / folder
            if p.exists():
                for f in p.glob("recap_scores*.csv"):
                    df = read_csv_auto(f)
                    if df is None or df.empty:
                        continue
                    df = _post_load_cleanup(df, results_dir, f)
                    df["modalite"] = "VR"
                    df["scenario"] = scen
                    frames.append(df)
                break

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out["modalite"] = out["modalite"].astype(str).str.upper().str.strip()
        out["scenario"] = out["scenario"].astype(str).str.upper().str.strip()
    return out


def load_riedl(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "indices_collab/riedl_group_summary.csv"
    df = read_csv_auto(p) if p.exists() else None
    return _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()


def resolve_tci_path(
    results_dir: Path,
    scope: str = "allowed",
    explicit_path: Path | None = None,
) -> Path | None:
    """
    Résout le fichier TCI à charger.

    Par défaut, le rapport principal privilégie la passe "allowed" afin
    d'éviter de dépendre implicitement du fichier legacy `c_scores.csv`.
    """
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    scope_norm = str(scope).strip().lower()
    if scope_norm == "all":
        candidates = [
            results_dir / "TCI" / "c_scores_all.csv",
            results_dir / "TCI" / "c_scores.csv",
        ]
    else:
        candidates = [
            results_dir / "TCI" / "c_scores_allowed.csv",
            results_dir / "TCI" / "c_scores.csv",
        ]

    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_tci(
    results_dir: Path,
    scope: str = "allowed",
    explicit_path: Path | None = None,
) -> pd.DataFrame:
    p = resolve_tci_path(results_dir, scope=scope, explicit_path=explicit_path)
    df = read_csv_auto(p) if p is not None and p.exists() else None
    out = _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()
    if not out.empty and p is not None:
        out.attrs["source_path"] = str(p)
        out.attrs["tci_scope"] = scope
    return out


def load_inv_face(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "INV" / "face_emotion_metrics_all.csv",
        results_dir / "INV" / "face" / "face_emotion_metrics_all.csv",
    ]
    p = next((c for c in candidates if c.exists()), None)
    df = read_csv_auto(p) if p is not None else None
    out = _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()
    return filter_inv_dataframe(out) if out is not None else pd.DataFrame()


def load_inv_speech(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "INV" / "audio_features.csv",
        results_dir / "INV" / "speech" / "audio_features.csv",
        results_dir / "INV" / "speech" / "metrics.csv",
    ]
    p = next((c for c in candidates if c.exists()), None)
    df = read_csv_auto(p) if p is not None else None
    out = _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()
    out = add_audio_canonical_aliases(out)
    return filter_inv_dataframe(out) if out is not None else pd.DataFrame()


def load_inv_gaze_all(results_dir: Path) -> pd.DataFrame:
    # Priorité : gaze_directional (nouvelle analyse sans maillage BIM)
    # Fallback : gaze (ancienne analyse par objet)
    gaze_dir_directional = results_dir / "INV" / "gaze_directional"
    gaze_dir_legacy = results_dir / "INV" / "gaze"
    gaze_dir = gaze_dir_directional if gaze_dir_directional.exists() else gaze_dir_legacy
    if not gaze_dir.exists():
        return pd.DataFrame()

    frames = []
    for f in gaze_dir.rglob("*.csv"):
        name = f.name.lower()
        if "overall" not in name:
            continue
        if not (name.startswith("all_") or "all_metrics" in name or "metrics_overall" in name):
            continue

        df = read_csv_auto(f)
        if df is None or df.empty:
            continue

        df = _post_load_cleanup(df, results_dir, f)
        df["_gaze_source"] = f.name
        try:
            df["_gaze_subdir"] = str(f.parent.relative_to(gaze_dir))
        except Exception:
            df["_gaze_subdir"] = str(f.parent)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    subset_cols = [c for c in ["group_id", "timepoint", "scenario", "modalite", "condition"] if c in out.columns]
    if subset_cols:
        out = out.drop_duplicates(
            subset=subset_cols + [c for c in out.columns if c not in {"_gaze_source", "_gaze_subdir"}]
        )
    return filter_inv_dataframe(out.reset_index(drop=True))


def load_high_level_features(results_dir: Path) -> tuple[pd.DataFrame, Optional[Path]]:
    candidates = [
        results_dir / "INV" / "high_level_features_audit.csv",
        results_dir / "high_level_features_audit.csv",
        results_dir / "high_level_features.csv",
        results_dir / "high_level_features_final.csv",
        results_dir / "INV" / "high_level_features.csv",
        results_dir / "INV" / "high_level_features_final.csv",
        results_dir / "INV" / "hlf" / "high_level_features.csv",
        results_dir / "INV" / "hlf" / "high_level_features_final.csv",
    ]
    hl_path = next((p for p in candidates if p.exists()), None)

    df = read_csv_auto(hl_path) if hl_path is not None else pd.DataFrame()
    if df is not None and not df.empty:
        df = normalize_group(df)
        df = add_group_base_id(df)
        df = harmonize_timepoint(
            df,
            session_col="session",
            raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
        )
        df = normalize_timepoint(df)
        df = coerce_numeric_columns(
            df,
            exclude={"group_id", "group_base_id", "timepoint", "condition", "modalite", "scenario", "session"},
        )
    else:
        df = pd.DataFrame()

    df = exclude_bad_groups(df)
    df = filter_inv_dataframe(df)
    return df, hl_path


def enrich_inv_face_with_high_level(inv_face: pd.DataFrame, hl: pd.DataFrame) -> pd.DataFrame:
    """Complete les donnees face avec les features core presentes dans high_level_features."""
    if hl is None or hl.empty or "group_id" not in hl.columns:
        return inv_face if inv_face is not None else pd.DataFrame()

    missing_core_face = [c for c in V2_CORE_FACE if c in hl.columns and (inv_face is None or c not in inv_face.columns)]
    if not missing_core_face:
        return inv_face if inv_face is not None else pd.DataFrame()

    hl_face_block = aggregate_numeric_by_unit(hl, missing_core_face)
    if hl_face_block.empty:
        return inv_face if inv_face is not None else pd.DataFrame()

    if inv_face is None or inv_face.empty:
        return hl_face_block

    return filter_inv_dataframe(merge_on_unit(inv_face, hl_face_block, how="left"))


def load_questionnaire_scores(results_dir: Path, use_pruned: bool = False) -> pd.DataFrame:
    """Charge les scores questionnaire par participant x dimension."""
    p_pruned = results_dir / "questionnaire/global/pruned/scores_dimension_par_participant_pruned.csv"
    p_orig = results_dir / "questionnaire/analyse/scores_dimension_par_participant.csv"
    p = p_pruned if use_pruned and p_pruned.exists() else p_orig

    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_group(df)
    df = add_group_base_id(df)
    df = harmonize_timepoint(
        df,
        session_col="session",
        raw_group_candidates=["group_id", "GroupID", "groupe", "Groupe", "group"],
    )
    df = normalize_timepoint(df)
    df = coerce_numeric_columns(
        df,
        exclude={
            "group_id", "group_base_id", "timepoint", "dimension", "participant",
            "modalite", "scenario", "session", "condition",
        },
    )

    try:
        df["source_file"] = str(p.relative_to(results_dir))
    except Exception:
        df["source_file"] = str(p)

    df = exclude_bad_groups(df)
    return df.reset_index(drop=True)


def load_questionnaire_descriptifs(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/analyse/descriptifs_scenario_modalite.csv"
    df = read_csv_auto(p) if p.exists() else None
    return _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()


def load_questionnaire_modeles(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/analyse/modeles_scenario_modalite_par_dimension.csv"
    df = read_csv_auto(p) if p.exists() else None
    return _post_load_cleanup(df, results_dir, p) if df is not None else pd.DataFrame()


def load_questionnaire_cronbach(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/cronbach_alpha_questionnaire.csv"
    if not p.exists():
        p = results_dir / "questionnaire/global/cronbach_alpha_tms.csv"
    df = read_csv_auto(p) if p.exists() else None

    if df is None or df.empty:
        return pd.DataFrame()

    ren = {}
    if "Dimension" in df.columns and "dimension" not in df.columns:
        ren["Dimension"] = "dimension"
    if "Alpha" in df.columns and "alpha" not in df.columns:
        ren["Alpha"] = "alpha"
    if "nItems" in df.columns and "n_items" not in df.columns:
        ren["nItems"] = "n_items"
    if ren:
        df = df.rename(columns=ren)

    if "dimension" in df.columns:
        df["dimension"] = df["dimension"].astype(str).str.upper().str.strip()
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    if "n_items" in df.columns:
        df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce")

    return df.reset_index(drop=True)


def load_questionnaire_cronbach_pruned(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/pruned/cronbach_alpha_pruned.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    if "dimension" in df.columns:
        df["dimension"] = df["dimension"].astype(str).str.upper().str.strip()
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    if "n_items" in df.columns:
        df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce")
    return df.reset_index(drop=True)


def load_alpha_comparison(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/pruned/alpha_comparison.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index(drop=True)


def load_exploratory_summary(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/exploratory_summary.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index(drop=True)


def load_desc_dim_pruned(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/pruned/desc_dim_pruned.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index(drop=True)


def load_questionnaire_participant_profile(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/participant_profile_responses.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    if "group_id" in df.columns:
        df["group_id"] = df["group_id"].astype(str).str.strip().str.lower()
    if "timepoint" in df.columns:
        df["timepoint"] = df["timepoint"].astype(str).str.strip().str.upper()
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.strip().str.upper()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def load_questionnaire_free_comments(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/free_comments_long.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    if "group_id" in df.columns:
        df["group_id"] = df["group_id"].astype(str).str.strip().str.lower()
    if "timepoint" in df.columns:
        df["timepoint"] = df["timepoint"].astype(str).str.strip().str.upper()
    if "modalite" in df.columns:
        df["modalite"] = df["modalite"].astype(str).str.strip().str.upper()
    if "scenario" in df.columns:
        df["scenario"] = df["scenario"].astype(str).str.strip().str.upper()
    return df.reset_index(drop=True)


def load_inv_pruned_features(results_dir: Path, inv_subdir: str = "results_inv_structure") -> list[str] | None:
    """Charge la liste des features INV conservées après hard pruning (PCA globale uniquement)."""
    inv_dir = results_dir / inv_subdir
    p = inv_dir / "inv_pruned_features.csv"
    if not p.exists():
        return None
    df = read_csv_auto(p)
    if df is None or df.empty:
        return None
    kept = df.loc[df["kept"] == 1, "feature"].tolist()
    kept = filter_inv_feature_names(list(dict.fromkeys(kept)))
    return kept if kept else None


def load_inv_pruned_features_full(results_dir: Path, inv_subdir: str = "results_inv_structure") -> pd.DataFrame | None:
    """Charge le détail complet du hard pruning pour affichage dans le rapport (PCA globale uniquement)."""
    inv_dir = results_dir / inv_subdir
    p = inv_dir / "inv_pruned_features.csv"
    if not p.exists():
        return None
    df = read_csv_auto(p)
    if df is None or df.empty:
        return None
    if "feature" in df.columns:
        df = df[~df["feature"].astype(str).apply(is_excluded_inv_feature)].copy()
    df["source"] = "inv_pruned_features"
    return df


def load_desc_dim_questionnaire(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "questionnaire/global/desc_dim_questionnaire.csv"
    df = read_csv_auto(p) if p.exists() else None
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index(drop=True)


def performance_score_col(perf: pd.DataFrame) -> Optional[str]:
    if perf is None or perf.empty:
        return None
    if "Score_perf_tsk" in perf.columns:
        return "Score_perf_tsk"
    return None


def perf_group_mean(perf: pd.DataFrame) -> pd.DataFrame:
    if perf is None or perf.empty or "group_id" not in perf.columns:
        return pd.DataFrame()

    sc = performance_score_col(perf)
    if sc is None or sc not in perf.columns:
        return pd.DataFrame()

    out = perf.copy()
    out[sc] = pd.to_numeric(out[sc], errors="coerce")
    g = out.groupby("group_id", dropna=False)[sc].mean().reset_index()
    g = g.rename(columns={sc: "Score_perf_tsk"})
    return g


def questionnaire_group_wide(q_scores: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if q_scores is None or q_scores.empty:
        return pd.DataFrame(), []
    if not {"group_id", "dimension", "score"}.issubset(q_scores.columns):
        return pd.DataFrame(), []

    qs = q_scores.copy()
    qs = harmonize_timepoint(qs, session_col="session", raw_group_candidates=["group_id"])
    qs = normalize_timepoint(qs)
    qs["dimension"] = qs["dimension"].astype(str)
    qs["score"] = pd.to_numeric(qs["score"], errors="coerce")
    qs = qs.dropna(subset=["group_id", "dimension", "score"])

    idx_cols = analysis_keys_for_df(qs)
    wide = (
        qs.pivot_table(index=idx_cols, columns="dimension", values="score", aggfunc="mean")
        .reset_index()
    )
    wide.columns.name = None
    cols = [c for c in wide.columns if c not in idx_cols]
    return wide, cols


def filter_df_by_group_ids(df: pd.DataFrame | None, keep_group_ids: set[str]) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()
    if "group_id" not in df.columns:
        return df.copy()

    out = df.copy()
    out["group_id"] = out["group_id"].astype(str).str.strip().str.lower()

    if not keep_group_ids:
        return out.iloc[0:0].copy()

    out = out[out["group_id"].isin(keep_group_ids)].copy()
    return out.reset_index(drop=True)


def apply_modality_filter(
    modality: str | None,
    perf: pd.DataFrame,
    riedl: pd.DataFrame,
    tci: pd.DataFrame,
    inv_face: pd.DataFrame,
    inv_speech: pd.DataFrame,
    inv_gaze_all: pd.DataFrame,
    hl: pd.DataFrame,
    q_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filtre les tables pour une modalite donnee (PC ou VR)."""
    if modality is None:
        return perf, riedl, tci, inv_face, inv_speech, inv_gaze_all, hl, q_scores

    modality = str(modality).upper().strip()

    def _filter_on_modalite(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None:
            return pd.DataFrame()
        if df.empty:
            return df.copy()

        out = df.copy()
        rename_map = {}
        for col in out.columns:
            c = str(col).strip().lower()
            if c in ["modalite", "modalité"]:
                rename_map[col] = "modalite"
            elif c == "condition":
                rename_map[col] = "condition"

        if rename_map:
            out = out.rename(columns=rename_map)

        if "modalite" not in out.columns and "condition" in out.columns:
            cond = out["condition"].astype(str).str.strip().str.upper()
            if cond.isin(["PC", "VR"]).any():
                out["modalite"] = cond

        if "modalite" not in out.columns:
            return out.copy()

        out["modalite"] = out["modalite"].astype(str).str.strip().str.upper()
        out = out[out["modalite"] == modality].copy()
        return out.reset_index(drop=True)

    perf_f = _filter_on_modalite(perf)
    q_scores_f = _filter_on_modalite(q_scores)

    keep_ids: set[str] = set()
    for df in [perf_f, q_scores_f]:
        if df is not None and not df.empty and "group_id" in df.columns:
            keep_ids |= set(df["group_id"].astype(str).str.strip().str.lower().dropna().unique())

    riedl_f = filter_df_by_group_ids(riedl, keep_ids)
    tci_f = filter_df_by_group_ids(tci, keep_ids)

    inv_face_f = _filter_on_modalite(inv_face)
    inv_speech_f = _filter_on_modalite(inv_speech)
    inv_gaze_f = _filter_on_modalite(inv_gaze_all)
    hl_f = _filter_on_modalite(hl)

    inv_face_f = filter_df_by_group_ids(inv_face_f, keep_ids)
    inv_speech_f = filter_df_by_group_ids(inv_speech_f, keep_ids)
    inv_gaze_f = filter_df_by_group_ids(inv_gaze_f, keep_ids)
    hl_f = filter_df_by_group_ids(hl_f, keep_ids)

    if inv_speech is not None and not inv_speech.empty:
        n_before = len(inv_speech)
        n_after = len(inv_speech_f)
        if n_after == 0 and n_before > 0:
            has_modalite = "modalite" in inv_speech.columns or "condition" in inv_speech.columns
            has_groupid = "group_id" in inv_speech.columns
            speech_groups = set(inv_speech["group_id"].astype(str).str.lower().unique()) if has_groupid else set()
            matching_groups = speech_groups & keep_ids if has_groupid else set()
            print(f"  [DIAG] inv_speech: {n_before} lignes avant filtrage -> {n_after} apres")
            print(f"         - Colonne modalite/condition: {'oui' if has_modalite else 'non'}")
            print(f"         - keep_ids (n={len(keep_ids)}): {sorted(list(keep_ids)[:5])}{'...' if len(keep_ids) > 5 else ''}")
            print(f"         - speech_groups (n={len(speech_groups)}): {sorted(list(speech_groups)[:5])}{'...' if len(speech_groups) > 5 else ''}")
            print(f"         - Intersection (n={len(matching_groups)}): {sorted(list(matching_groups)[:5]) if matching_groups else 'vide'}")

    return perf_f, riedl_f, tci_f, inv_face_f, inv_speech_f, inv_gaze_f, hl_f, q_scores_f


def build_group_master_csv(
    perf: pd.DataFrame,
    riedl: pd.DataFrame,
    tci: pd.DataFrame,
    inv_face: pd.DataFrame,
    inv_speech: pd.DataFrame,
    inv_gaze_all: pd.DataFrame,
    hl: pd.DataFrame,
    q_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Construit une table fusionnee finale au niveau unite d'analyse."""
    pieces = []
    merge_log = []

    perf_block = pd.DataFrame()
    if perf is not None and not perf.empty and "group_id" in perf.columns:
        sc = performance_score_col(perf)
        if sc is not None and sc in perf.columns:
            perf_block = perf.copy()
            perf_block[sc] = pd.to_numeric(perf_block[sc], errors="coerce")

            group_cols = analysis_keys_for_df(perf_block)
            keep_cols = group_cols + [sc]

            extra_id_cols = [c for c in ["modalite", "scenario", "condition", "session", "group_base_id"] if c in perf_block.columns]
            keep_cols += [c for c in extra_id_cols if c not in keep_cols]

            perf_block = perf_block[keep_cols].copy()

            agg_map = {sc: "mean"}
            for c in extra_id_cols:
                agg_map[c] = "first"

            perf_block = perf_block.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
            perf_block = perf_block.rename(columns={sc: "Score_perf_tsk"})
            merge_log.append(f"perf_v2: {len(perf_block)} lignes, cols={list(perf_block.columns)}")
            pieces.append(perf_block)

    if riedl is not None and not riedl.empty and "group_id" in riedl.columns:
        r_num = [c for c in riedl.columns if pd.api.types.is_numeric_dtype(riedl[c]) and c not in ID_LIKE_COLS]
        r_block = aggregate_numeric_by_unit(riedl, r_num)
        if not r_block.empty:
            pieces.append(r_block)

    if tci is not None and not tci.empty and "group_id" in tci.columns:
        t_num = [c for c in tci.columns if pd.api.types.is_numeric_dtype(tci[c]) and c not in ID_LIKE_COLS]
        t_block = aggregate_numeric_by_unit(tci, t_num)
        if not t_block.empty:
            pieces.append(t_block)

    if q_scores is not None and not q_scores.empty:
        q = q_scores.copy()
        q = normalize_group(q)
        q = harmonize_timepoint(q, session_col="session", raw_group_candidates=["group_id"])
        q = normalize_timepoint(q)

        if {"group_id", "dimension", "score"}.issubset(q.columns):
            q["score"] = pd.to_numeric(q["score"], errors="coerce")
            idx_cols = analysis_keys_for_df(q)
            if "dimension" in q.columns and idx_cols:
                q_block = (
                    q.pivot_table(index=idx_cols, columns="dimension", values="score", aggfunc="mean")
                    .reset_index()
                )
                q_block.columns.name = None
                if not q_block.empty:
                    pieces.append(q_block)

    if inv_face is not None and not inv_face.empty and "group_id" in inv_face.columns:
        f_cols = [c for c in V2_CORE_FACE if c in inv_face.columns]
        if not f_cols:
            f_cols = [c for c in inv_face.columns if pd.api.types.is_numeric_dtype(inv_face[c]) and c not in ID_LIKE_COLS and not is_excluded_inv_feature(c)]
        face_block = aggregate_numeric_by_unit(inv_face, f_cols)
        if not face_block.empty:
            pieces.append(face_block)

    if inv_speech is not None and not inv_speech.empty and "group_id" in inv_speech.columns:
        s_cols = [c for c in V2_CORE_SPEECH if c in inv_speech.columns]
        if not s_cols:
            s_cols = [c for c in inv_speech.columns if pd.api.types.is_numeric_dtype(inv_speech[c]) and c not in ID_LIKE_COLS and not is_excluded_inv_feature(c)]
        speech_block = aggregate_numeric_by_unit(inv_speech, s_cols)
        if not speech_block.empty:
            pieces.append(speech_block)

    if inv_gaze_all is not None and not inv_gaze_all.empty and "group_id" in inv_gaze_all.columns:
        g_cols = [c for c in V2_CORE_GAZE if c in inv_gaze_all.columns]
        if not g_cols:
            g_cols = [c for c in inv_gaze_all.columns if pd.api.types.is_numeric_dtype(inv_gaze_all[c]) and c not in ID_LIKE_COLS and not is_excluded_inv_feature(c)]
        gaze_block = aggregate_numeric_by_unit(inv_gaze_all, g_cols)
        if not gaze_block.empty:
            pieces.append(gaze_block)

    if hl is not None and not hl.empty and "group_id" in hl.columns:
        hl_cols = [c for c in V2_CORE_HL if c in hl.columns]
        if not hl_cols:
            hl_cols = [c for c in hl.columns if pd.api.types.is_numeric_dtype(hl[c]) and c not in ID_LIKE_COLS and not is_excluded_inv_feature(c)]
        hl_block = aggregate_numeric_by_unit(hl, hl_cols)
        if not hl_block.empty:
            pieces.append(hl_block)

    if not pieces:
        return pd.DataFrame()

    master = pieces[0].copy()
    for block in pieces[1:]:
        master = merge_on_unit(master, block, how="outer")

    for base_col in ["modalite", "scenario", "condition", "session", "group_base_id"]:
        x_col = f"{base_col}_x"
        y_col = f"{base_col}_y"
        if x_col in master.columns:
            master[base_col] = master[x_col].fillna(master.get(y_col, master[x_col]))
            master = master.drop(columns=[c for c in [x_col, y_col] if c in master.columns])
        elif y_col in master.columns:
            master[base_col] = master[y_col]
            master = master.drop(columns=[y_col])

    # Résoudre tous les doublons _x/_y issus du merge inter-blocs (audio, face, etc.)
    # Pour chaque paire (col_x, col_y), conserver col = coalesce(_x, _y) et supprimer les doublons.
    for cx in [c for c in master.columns if c.endswith("_x")]:
        base = cx[:-2]
        cy = base + "_y"
        if cy not in master.columns:
            continue
        if base in master.columns:
            master = master.drop(columns=[cx, cy])
        else:
            sx = pd.to_numeric(master[cx], errors="coerce")
            sy = pd.to_numeric(master[cy], errors="coerce")
            master[base] = sx.fillna(sy)
            master = master.drop(columns=[cx, cy])

    master = normalize_group(master)
    master = add_group_base_id(master)
    master = harmonize_timepoint(master, session_col="session", raw_group_candidates=["group_id"])
    master = normalize_timepoint(master)
    master = coerce_numeric_columns(master, exclude=ID_LIKE_COLS | {"group_base_id"})

    first_cols = [c for c in [
        "group_id", "group_base_id", "timepoint", "scenario", "modalite",
        "condition", "session",
        "Score_perf_tsk",
    ] if c in master.columns]
    other_cols = [c for c in master.columns if c not in first_cols]
    master = master[first_cols + other_cols]

    sort_cols = [c for c in ["group_id", "scenario", "modalite", "timepoint"] if c in master.columns]
    if sort_cols:
        master = master.sort_values(sort_cols).reset_index(drop=True)

    key_cols = [c for c in ["group_id", "scenario", "modalite", "timepoint"] if c in master.columns]
    if key_cols:
        dups = master.duplicated(subset=key_cols, keep=False)
        n_dups = dups.sum()
        if n_dups > 0:
            print(f"[WARN] {n_dups} lignes dupliquees detectees sur {key_cols}")
            master = master.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
            print(f"[INFO] Doublons supprimes, {len(master)} lignes restantes")

    for msg in merge_log:
        print(f"  [MERGE] {msg}")
    print(f"  [MERGE] master final: {master.shape[0]} lignes x {master.shape[1]} colonnes")

    return master


def classify_merged_dataset_column(col: str) -> str:
    """Retourne le bloc fonctionnel principal d'une colonne du merged dataset."""
    if col in ID_LIKE_COLS or col in {"n_missing_speech_core", "n_missing_gaze_core", "n_missing_face_core"}:
        return "Identifiants / contexte"

    if col in PERFORMANCE_COLS or col.startswith("Score_perf"):
        return "Performance"

    if col in CORE_RIEDL_COLS or col.startswith(("effort_", "strategy_", "skill_", "contribution_", "n_tasks")):
        return "Riedl / Hackman"

    if col in QUESTIONNAIRE_DIMENSIONS:
        return "Questionnaire"

    if col in {"c_score", "rme_mean", "rme_max", "rme_min", "rme_total", "RME", "RME_mean"} or col.startswith("rme"):
        return "TCI"

    if col in V2_CORE_SPEECH:
        return "INV Speech"
    if col in V2_CORE_FACE:
        return "INV Face"
    if col in V2_CORE_GAZE:
        return "INV Gaze"
    if col in V2_CORE_HL:
        return "INV High-level"

    if col.startswith("n_missing_"):
        return "Diagnostics"
    if col.endswith("_source"):
        return "Provenance"

    family = infer_family_from_name(col)
    if family is not None:
        return f"INV {family.capitalize()}"

    return "Autre"


def build_merged_dataset_column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un inventaire riche des colonnes du merged dataset."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "column", "block", "dtype", "n_non_na", "n_missing",
            "pct_missing", "n_unique", "sample_values",
        ])

    n_rows = len(df)
    rows: list[dict] = []
    for col in df.columns:
        series = df[col]
        n_non_na = int(series.notna().sum())
        n_missing = int(n_rows - n_non_na)
        pct_missing = round((100.0 * n_missing / n_rows), 2) if n_rows > 0 else np.nan
        n_unique = int(series.nunique(dropna=True))
        sample_values = " | ".join(series.dropna().astype(str).head(3).tolist())
        rows.append({
            "column": col,
            "block": classify_merged_dataset_column(col),
            "dtype": str(series.dtype),
            "n_non_na": n_non_na,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "sample_values": sample_values,
        })

    return pd.DataFrame(rows).sort_values(["block", "pct_missing", "column"]).reset_index(drop=True)


def build_merged_dataset_block_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    """Résumé par bloc conceptuel du merged dataset."""
    if inventory is None or inventory.empty:
        return pd.DataFrame(columns=[
            "block", "n_columns", "all_missing_columns",
            "avg_missing_pct", "median_missing_pct",
        ])

    summary = (
        inventory.groupby("block", dropna=False)
        .agg(
            n_columns=("column", "size"),
            all_missing_columns=("n_non_na", lambda s: int((s == 0).sum())),
            avg_missing_pct=("pct_missing", "mean"),
            median_missing_pct=("pct_missing", "median"),
        )
        .reset_index()
    )
    for col in ["avg_missing_pct", "median_missing_pct"]:
        summary[col] = summary[col].round(2)
    return summary.sort_values(["block"]).reset_index(drop=True)


def build_merged_dataset_unit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compte les lignes par combinaison de contexte disponible."""
    if df is None or df.empty:
        return pd.DataFrame()

    keys = [c for c in ["condition", "scenario", "modalite", "timepoint"] if c in df.columns]
    if not keys:
        return pd.DataFrame()

    return (
        df.groupby(keys, dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def build_source_table_summary(source_tables: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """Résumé des tables sources réellement chargées avant fusion."""
    if not source_tables:
        return pd.DataFrame(columns=["source", "n_rows", "n_cols", "has_group_id", "analysis_keys"])

    rows: list[dict] = []
    for name, table in source_tables.items():
        if table is None:
            rows.append({
                "source": name,
                "n_rows": 0,
                "n_cols": 0,
                "has_group_id": False,
                "analysis_keys": "",
            })
            continue

        rows.append({
            "source": name,
            "n_rows": int(table.shape[0]),
            "n_cols": int(table.shape[1]),
            "has_group_id": bool("group_id" in table.columns),
            "analysis_keys": ", ".join(analysis_keys_for_df(table)),
        })

    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


def export_merged_dataset_bundle(
    df: pd.DataFrame,
    out_dir: Path,
    suffix: str,
    *,
    inv_analysis_mode: str,
    modality_filter: str | None,
    inv_subdir: str,
    source_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """
    Exporte le merged dataset et plusieurs artefacts de documentation.

    Fichiers générés :
    - merged_dataset_complete_<suffix>.csv
    - merged_dataset_columns_<suffix>.csv
    - merged_dataset_blocks_<suffix>.csv
    - merged_dataset_unit_counts_<suffix>.csv
    - merged_dataset_sources_<suffix>.csv
    - merged_dataset_report_<suffix>.md
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"merged_dataset_complete_{suffix}.csv"
    inventory_path = out_dir / f"merged_dataset_columns_{suffix}.csv"
    block_path = out_dir / f"merged_dataset_blocks_{suffix}.csv"
    counts_path = out_dir / f"merged_dataset_unit_counts_{suffix}.csv"
    sources_path = out_dir / f"merged_dataset_sources_{suffix}.csv"
    report_path = out_dir / f"merged_dataset_report_{suffix}.md"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    inventory = build_merged_dataset_column_inventory(df)
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    block_summary = build_merged_dataset_block_summary(inventory)
    block_summary.to_csv(block_path, index=False, encoding="utf-8-sig")

    unit_counts = build_merged_dataset_unit_counts(df)
    unit_counts.to_csv(counts_path, index=False, encoding="utf-8-sig")

    source_summary = build_source_table_summary(source_tables)
    source_summary.to_csv(sources_path, index=False, encoding="utf-8-sig")

    key_cols = [c for c in ["group_id", "scenario", "modalite", "timepoint"] if c in df.columns]
    n_dups = int(df.duplicated(subset=key_cols, keep=False).sum()) if key_cols else 0
    full_missing_cols = inventory.loc[inventory["n_non_na"] == 0, "column"].tolist()
    top_missing = inventory.sort_values(["pct_missing", "column"], ascending=[False, True]).head(15)

    lines = [
        "# Merged Dataset Export",
        "",
        f"- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- INV analysis mode: {inv_analysis_mode}",
        f"- Modality filter: {'all' if modality_filter is None else str(modality_filter).lower()}",
        f"- INV source directory used by the report: {inv_subdir}",
        f"- Rows: {len(df)}",
        f"- Columns: {len(df.columns)}",
        f"- Duplicate rows on keys {key_cols if key_cols else 'N/A'}: {n_dups}",
        "",
        "## Output Files",
        "",
        f"- Main CSV: `{csv_path.name}`",
        f"- Column inventory: `{inventory_path.name}`",
        f"- Block summary: `{block_path.name}`",
        f"- Unit counts: `{counts_path.name}`",
        f"- Source summary: `{sources_path.name}`",
        "",
    ]

    if not source_summary.empty:
        lines.extend([
            "## Source Tables Loaded",
            "",
            "```text",
            source_summary.to_string(index=False),
            "```",
            "",
        ])

    if not block_summary.empty:
        lines.extend([
            "## Block Summary",
            "",
            "```text",
            block_summary.to_string(index=False),
            "```",
            "",
        ])

    if not unit_counts.empty:
        lines.extend([
            "## Counts by Condition / Scenario / Timepoint",
            "",
            "```text",
            unit_counts.to_string(index=False),
            "```",
            "",
        ])

    if full_missing_cols:
        lines.extend([
            "## Fully Missing Columns",
            "",
            *[f"- `{col}`" for col in full_missing_cols],
            "",
        ])

    if not top_missing.empty:
        lines.extend([
            "## Top Missingness (Top 15 columns)",
            "",
            "```text",
            top_missing[["column", "block", "pct_missing", "n_non_na", "n_unique"]].to_string(index=False),
            "```",
            "",
        ])

    lines.extend([
        "## Notes",
        "",
        "- This export is report-driven: it reflects the tables actually loaded and filtered by the current report mode.",
        "- The merged dataset depends on the selected INV analysis mode because the report can load pruning-specific questionnaire and INV diagnostics.",
        "- The detailed schema is stored in the companion CSV files generated alongside this report.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "csv": csv_path,
        "inventory": inventory_path,
        "blocks": block_path,
        "unit_counts": counts_path,
        "sources": sources_path,
        "report": report_path,
    }
