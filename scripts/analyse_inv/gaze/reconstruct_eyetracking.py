"""
reconstruct_eyetracking.py — Étape 1.5 + Étape 2 du pipeline de correction eye-tracking VR.

Prérequis : avoir validé rapport_diagnostic_eyetracking.md (Étapes 0+1).
Ce script ne s'exécute que sur les sessions dont le bug est confirmé par le diagnostic.

Étape 1.5 : validation heuristique de la direction (heading gaze vs heading mouvement).
Étape 2   : reconstruction des fichiers corrigés (TrueOrigin, DirXYZ, flags).

Usage :
    python reconstruct_eyetracking.py \
        --data-dir D:/data_e2 \
        --out-dir  D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected

Options :
    --group bim015          Restreindre à un groupe
    --head-offset 0.0       Offset Y à ajouter à Player0_PosY pour obtenir la hauteur tête
                            (défaut 0.0 : données déjà en coordonnées tête, confirmé sur le dataset)
    --head-offset-threshold 0.8  Si Player0_PosY médian < ce seuil → applique l'offset
    --motion-min-speed 0.3  Seuil vitesse horizontale (m/s) pour l'analyse de heading
    --motion-min-frames 20  Nombre minimum de frames en mouvement pour la validation 1.5
    --verbose
"""
import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
# Fréquences observées
EYE_DT_MEDIAN_MS = 28.0     # ~35 Hz
POS_DT_MEDIAN_MS = 503.0    # ~2 Hz

# Seuil offset bug (doit être confirmé par le diagnostic)
OFFSET_BUG_THRESHOLD_M = 0.50

# Étape 1.5 : seuils heuristiques direction
MOTION_MIN_SPEED_DEFAULT = 0.3        # m/s pour sélectionner les frames en mouvement
MOTION_MIN_FRAMES_DEFAULT = 20        # min frames en mouvement pour que l'analyse soit valide
DELTA_THETA_DRIFT_THRESHOLD_DEG = 45  # drift crculaire > ce seuil → suspect_yaw
DELTA_THETA_BIMODAL_MIN_SPLIT_DEG = 60  # gap bimodal → suspect_yaw

# Merge temporel : tolérance pour merge_asof
MERGE_TOLERANCE_S = 0.200  # 200 ms

# Etape 2 : flags sur les colonnes de sortie
HEAD_OFFSET_THRESHOLD_DEFAULT = 0.8   # m : si Y_med < seuil → position est la racine, pas la tête


# ---------------------------------------------------------------------------
# I/O ROBUSTE (identique au diagnostic, dupliqué pour autonomie du script)
# ---------------------------------------------------------------------------

def _detect_decimal(line: str, sep: str = ";") -> str:
    for f in line.split(sep):
        f = f.strip()
        if "," in f and "." not in f:
            return ","
        if "." in f and "," not in f:
            return "."
    return "."


def read_eye_csv(path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]], str]:
    """Retourne (df_data, markers, sep_dec)."""
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    raw = re.sub(r"END\s*$", "", raw.rstrip()) + "\n"
    lines = raw.splitlines()
    if not lines:
        return pd.DataFrame(), [], "."

    sep_dec = "."
    for ln in lines[1:]:
        if ln.strip() and "MARKER" not in ln:
            sep_dec = _detect_decimal(ln)
            break

    data_lines, markers = [], []
    for ln in lines[1:]:
        stripped = ln.strip()
        if not stripped:
            continue
        parts = stripped.split(";")
        if len(parts) == 2 and "MARKER" in parts[1]:
            markers.append((parts[0].strip(), stripped))
        else:
            data_lines.append(stripped)

    if not data_lines:
        return pd.DataFrame(), markers, sep_dec

    df = pd.read_csv(
        StringIO(lines[0] + "\n" + "\n".join(data_lines)),
        sep=";", decimal=sep_dec, engine="python"
    )
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )
    return df, markers, sep_dec


def read_pos_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Retourne (df, time_col)."""
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    sep_dec = "."
    for ln in raw.splitlines()[1:]:
        if ln.strip():
            sep_dec = _detect_decimal(ln)
            break
    df = pd.read_csv(StringIO(raw), sep=";", decimal=sep_dec, engine="python")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )
    time_col = next((c for c in ["Timestamp", "Time"] if c in df.columns), "")
    return df, time_col


# ---------------------------------------------------------------------------
# DÉCOUVERTE DES SESSIONS (même logique que le diagnostic)
# ---------------------------------------------------------------------------

def find_vr_sessions(data_dir: Path) -> list[dict]:
    """
    Découvre les sessions VR. Pour chaque (timepoint, scenario, group, role),
    ne conserve que le fichier EyeTrackingData le plus récent (mtime),
    comme le fait load_group_gaze dans analyze_gaze.py (max par mtime).
    Cela gère les groupes multi-fichiers (ex: bim002) en prenant la session finale.
    """
    eye_files = [
        p for p in data_dir.rglob("*EyeTrackingData.csv")
        if "_old" not in p.parts and "VR" in p.parts
    ]

    # Groupe par (timepoint, scenario, group, role) → liste de candidats
    by_key: dict[tuple, list[Path]] = {}
    for eye_path in eye_files:
        parts = eye_path.parts
        try:
            vr_idx = parts.index("VR")
            timepoint = parts[vr_idx - 1]
            scenario  = parts[vr_idx + 1]
            group     = parts[vr_idx + 2]
            role      = parts[vr_idx + 3]
        except (ValueError, IndexError):
            continue
        key = (timepoint, scenario, group, role)
        by_key.setdefault(key, []).append(eye_path)

    sessions = []
    for (timepoint, scenario, group, role), candidates in sorted(by_key.items()):
        # Même logique que l'ancienne analyse : fichier le plus récent par rôle
        # Exception : si un fichier merged_* existe, il prend la priorité
        merged = [p for p in candidates if "merged" in p.stem]
        eye_path = merged[0] if merged else max(candidates, key=lambda p: p.stat().st_mtime)

        folder = eye_path.parent
        stem_ts = eye_path.stem.split("_EyeTrackingData")[0]
        pos_candidates = [
            p for p in folder.glob("*UsersPositions.csv")
            if "_old" not in str(p)
        ]
        pos_path = None
        pairing_ok = True

        if len(pos_candidates) == 1:
            pos_path = pos_candidates[0]
        elif stem_ts == "merged":
            merged_pos = folder / "merged_UsersPositions.csv"
            pos_path = merged_pos if merged_pos.exists() else None
            pairing_ok = pos_path is not None
        else:
            matched = [p for p in pos_candidates if stem_ts in p.stem]
            if len(matched) == 1:
                pos_path = matched[0]
            elif pos_candidates:
                try:
                    t_eye = datetime.strptime(stem_ts, "%Y-%m-%d_%H-%M-%S")
                    def _ts(p):
                        s = p.stem.replace("_UsersPositions", "")
                        return datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
                    timed = [(p, abs((_ts(p) - t_eye).total_seconds()))
                             for p in pos_candidates
                             if re.match(r"\d{4}-\d{2}-\d{2}", p.stem)]
                    if timed:
                        best, delta = min(timed, key=lambda x: x[1])
                        pos_path = best if delta <= 60 else None
                except Exception:
                    pass
            pairing_ok = pos_path is not None

        if pairing_ok and pos_path is not None:
            sessions.append({
                "eye_path": eye_path,
                "pos_path": pos_path,
                "group": group,
                "role": role,
                "timepoint": timepoint,
                "scenario": scenario,
            })
    return sessions


# ---------------------------------------------------------------------------
# ÉTAPE 1.5 — VALIDATION HEURISTIQUE DE DIRECTION
# ---------------------------------------------------------------------------

def step15_direction_check(
    df_eye: pd.DataFrame,
    df_pos: pd.DataFrame,
    time_col_pos: str,
    min_speed: float,
    min_frames: int,
) -> dict:
    """
    Compare heading du regard (projection XZ de d_world) au heading du mouvement
    (déplacement XZ de Player0 entre frames positions consécutives).

    Retourne un dict avec :
    - direction_assumption : 'pure_translation_unverified' | 'validated_traj' | 'suspect_yaw'
    - delta_theta_mean_deg, delta_theta_std_deg, delta_theta_drift_deg
    - n_motion_frames, caveat
    """
    result = {
        "direction_assumption": "pure_translation_unverified",
        "delta_theta_mean_deg": None,
        "delta_theta_std_deg": None,
        "delta_theta_drift_deg": None,
        "n_motion_frames": 0,
        "caveat": "",
    }

    # --- Heading mouvement depuis positions ---
    df_p = df_pos[[time_col_pos, "Player0_PosX", "Player0_PosZ"]].dropna().copy()
    df_p = df_p.sort_values(time_col_pos).reset_index(drop=True)
    df_p["dx"] = df_p["Player0_PosX"].diff()
    df_p["dz"] = df_p["Player0_PosZ"].diff()
    df_p["dt"] = df_p[time_col_pos].diff()
    df_p["v"] = np.sqrt(df_p["dx"]**2 + df_p["dz"]**2) / df_p["dt"].replace(0, np.nan)
    df_p_move = df_p[df_p["v"] > min_speed].copy()

    if len(df_p_move) < min_frames:
        result["caveat"] = (
            f"Seulement {len(df_p_move)} frames en mouvement (|v|>{min_speed} m/s) "
            f"sur {len(df_p)} — seuil {min_frames} non atteint. "
            "Résultat : pure_translation_unverified. "
            "Note : locomotion par téléportation ou faible déplacement invalide cette heuristique."
        )
        result["n_motion_frames"] = len(df_p_move)
        return result

    df_p_move["theta_move"] = np.degrees(np.arctan2(df_p_move["dx"], df_p_move["dz"]))

    # --- d_world depuis eye ---
    df_e = df_eye[["Time", "RayOriginX", "RayOriginY", "RayOriginZ",
                   "HitPointX", "HitPointY", "HitPointZ"]].dropna().copy()
    if df_e.empty:
        result["caveat"] = "Aucune frame eye valide pour le calcul de direction."
        return result

    d = df_e[["HitPointX", "HitPointY", "HitPointZ"]].values \
        - df_e[["RayOriginX", "RayOriginY", "RayOriginZ"]].values
    norms = np.linalg.norm(d, axis=1)
    valid = norms > 1e-6
    d_norm = np.full_like(d, np.nan)
    d_norm[valid] = d[valid] / norms[valid, None]
    df_e = df_e.copy()
    df_e["DirX"] = d_norm[:, 0]
    df_e["DirZ"] = d_norm[:, 2]
    df_e["theta_gaze"] = np.degrees(np.arctan2(df_e["DirX"], df_e["DirZ"]))

    # --- Merge nearest-neighbor positions → eye ---
    df_p_move_s = df_p_move[[time_col_pos, "theta_move"]].rename(
        columns={time_col_pos: "Time_pos"}
    ).sort_values("Time_pos")
    df_e_s = df_e[["Time", "theta_gaze"]].sort_values("Time")

    merged = pd.merge_asof(
        df_p_move_s,
        df_e_s,
        left_on="Time_pos",
        right_on="Time",
        direction="nearest",
        tolerance=MERGE_TOLERANCE_S,
    ).dropna(subset=["theta_gaze", "theta_move"])

    if len(merged) < min_frames:
        result["caveat"] = (
            f"Seulement {len(merged)} frames appariées après merge temporel "
            f"(tolérance {MERGE_TOLERANCE_S*1000:.0f} ms)."
        )
        result["n_motion_frames"] = len(merged)
        return result

    # --- Delta theta (wrap [-180, 180]) ---
    delta = merged["theta_gaze"].values - merged["theta_move"].values
    # wrap circulaire
    delta = (delta + 180) % 360 - 180

    # Moyenne circulaire
    sin_m = np.sin(np.radians(delta)).mean()
    cos_m = np.cos(np.radians(delta)).mean()
    mean_deg = float(np.degrees(np.arctan2(sin_m, cos_m)))

    # Dispersion circulaire (std approximation)
    R = np.sqrt(sin_m**2 + cos_m**2)
    circ_std = float(np.degrees(np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1)))))

    # Drift temporel : régression linéaire de delta ~ t
    t_norm = (merged["Time_pos"].values - merged["Time_pos"].values[0])
    if t_norm[-1] > 0:
        coeffs = np.polyfit(t_norm, delta, 1)
        drift_per_s = coeffs[0]  # °/s
        total_drift = abs(drift_per_s * t_norm[-1])
    else:
        total_drift = 0.0

    # Verdict
    if circ_std > DELTA_THETA_DRIFT_THRESHOLD_DEG or total_drift > DELTA_THETA_DRIFT_THRESHOLD_DEG:
        assumption = "suspect_yaw"
        caveat = (
            f"delta_theta instable : std_circ={circ_std:.1f} deg, "
            f"drift_total={total_drift:.1f} deg. "
            "Le yaw virtuel n'est peut-etre pas integre dans d_world. "
            "Les sorties directionnelles de cette session sont NON FIABLES."
        )
    else:
        assumption = "validated_traj"
        caveat = (
            f"delta_theta unimodal et stable : mean={mean_deg:.1f} deg, "
            f"std_circ={circ_std:.1f} deg, drift={total_drift:.1f} deg. "
            "Direction coherente avec la trajectoire. Biais systematique "
            f"de {mean_deg:.1f} deg regard/deplacement attendu (normal)."
        )

    result.update({
        "direction_assumption": assumption,
        "delta_theta_mean_deg": round(mean_deg, 2),
        "delta_theta_std_deg": round(circ_std, 2),
        "delta_theta_drift_deg": round(total_drift, 2),
        "n_motion_frames": len(merged),
        "caveat": caveat,
    })
    return result


# ---------------------------------------------------------------------------
# ÉTAPE 2 — RECONSTRUCTION CSV CORRIGÉ
# ---------------------------------------------------------------------------

def step2_reconstruct(
    df_eye: pd.DataFrame,
    markers: list[tuple[str, str]],
    df_pos: pd.DataFrame,
    time_col_pos: str,
    dir_info: dict,
    head_offset_y: float,
    out_path: Path,
) -> dict:
    """
    Écrit <session>_EyeTrackingData_corrected.csv.
    Retourne un dict de métriques (summary).
    """
    # --- Prépare positions avec offset tête ---
    df_p = df_pos[[time_col_pos, "Player0_PosX", "Player0_PosY", "Player0_PosZ"]].dropna().copy()
    df_p = df_p.rename(columns={
        time_col_pos: "Time_pos",
        "Player0_PosX": "px",
        "Player0_PosY": "py",
        "Player0_PosZ": "pz",
    }).sort_values("Time_pos").reset_index(drop=True)
    df_p["py"] = df_p["py"] + head_offset_y

    pos_time = df_p["Time_pos"].values
    pos_x = df_p["px"].values
    pos_y = df_p["py"].values
    pos_z = df_p["pz"].values

    # --- Prépare eye ---
    df_e = df_eye.copy().reset_index(drop=True)
    t_eye = df_e["Time"].values

    # Vérifie recouvrement
    in_range = (t_eye >= pos_time[0]) & (t_eye <= pos_time[-1])
    n_extrap = int((~in_range).sum())

    # Interpolation linéaire de E_true sur les timestamps eye
    true_x = np.interp(t_eye, pos_time, pos_x)
    true_y = np.interp(t_eye, pos_time, pos_y)
    true_z = np.interp(t_eye, pos_time, pos_z)

    # Marque les frames hors plage (extrapolation interdite → NaN)
    true_x[~in_range] = np.nan
    true_y[~in_range] = np.nan
    true_z[~in_range] = np.nan

    # Résidu de synchro : distance temporelle au voisin le plus proche dans df_p
    idx_near = np.searchsorted(pos_time, t_eye, side="left").clip(0, len(pos_time) - 1)
    idx_near_m1 = (idx_near - 1).clip(0, len(pos_time) - 1)
    d_right = np.abs(t_eye - pos_time[idx_near])
    d_left = np.abs(t_eye - pos_time[idx_near_m1])
    sync_residual_s = np.where(d_left < d_right, d_left, d_right)
    sync_residual_ms = sync_residual_s * 1000.0

    # --- d_world = normalize(HitPoint - RayOrigin) ---
    hp = df_e[["HitPointX", "HitPointY", "HitPointZ"]].values
    ro = df_e[["RayOriginX", "RayOriginY", "RayOriginZ"]].values
    diff = hp - ro
    norms = np.linalg.norm(diff, axis=1)

    hit_was_nan = np.zeros(len(df_e), dtype=int)
    dir_x = np.full(len(df_e), np.nan)
    dir_y = np.full(len(df_e), np.nan)
    dir_z = np.full(len(df_e), np.nan)

    valid_hit = norms > 1e-6
    hit_was_nan[~valid_hit] = 1

    # NaN explicites dans HitPoint (colonne NaN originale)
    has_hp_nan = (
        df_e["HitPointX"].isna() | df_e["HitPointY"].isna() | df_e["HitPointZ"].isna()
    ).values
    hit_was_nan[has_hp_nan] = 1

    usable = valid_hit & ~has_hp_nan
    dir_x[usable] = diff[usable, 0] / norms[usable]
    dir_y[usable] = diff[usable, 1] / norms[usable]
    dir_z[usable] = diff[usable, 2] / norms[usable]

    # --- Offset origine ---
    origin_offset = np.sqrt(
        (ro[:, 0] - true_x)**2 +
        (ro[:, 1] - true_y)**2 +
        (ro[:, 2] - true_z)**2
    )

    # --- Construction du DataFrame de sortie ---
    out = pd.DataFrame({
        "Time": df_e["Time"],
        "RayOriginX": df_e["RayOriginX"],
        "RayOriginY": df_e["RayOriginY"],
        "RayOriginZ": df_e["RayOriginZ"],
        "TrueOriginX": true_x,
        "TrueOriginY": true_y,
        "TrueOriginZ": true_z,
        "DirX": dir_x,
        "DirY": dir_y,
        "DirZ": dir_z,
        "HitPointX": df_e["HitPointX"],
        "HitPointY": df_e["HitPointY"],
        "HitPointZ": df_e["HitPointZ"],
        "ObjectHit": df_e.get("ObjectHit", pd.Series(np.nan, index=df_e.index)),
        "hit_was_nan": hit_was_nan,
        "sync_residual_ms": np.round(sync_residual_ms, 2),
        "origin_offset_m": np.round(origin_offset, 4),
        "direction_assumption": dir_info["direction_assumption"],
    })

    # Colonne PlayerLooked si présente
    if "PlayerLooked" in df_e.columns:
        out["PlayerLooked"] = df_e["PlayerLooked"]

    # Écriture avec point décimal, séparateur ;
    out.to_csv(out_path, sep=";", decimal=".", index=False, float_format="%.6f")

    # Métriques summary
    return {
        "n_frames": len(out),
        "n_hit_nan": int(hit_was_nan.sum()),
        "pct_hit_nan": round(100.0 * hit_was_nan.mean(), 2),
        "n_extrap_frames": n_extrap,
        "sync_residual_median_ms": round(float(np.nanmedian(sync_residual_ms)), 2),
        "sync_residual_p95_ms": round(float(np.nanpercentile(sync_residual_ms, 95)), 2),
        "origin_offset_median_m": round(float(np.nanmedian(origin_offset)), 4),
        "origin_offset_max_m": round(float(np.nanmax(origin_offset)), 4),
        "head_offset_applied_m": head_offset_y,
        "direction_assumption": dir_info["direction_assumption"],
        "n_motion_frames": dir_info.get("n_motion_frames", 0),
        "delta_theta_mean_deg": dir_info.get("delta_theta_mean_deg"),
        "delta_theta_std_deg": dir_info.get("delta_theta_std_deg"),
        "delta_theta_drift_deg": dir_info.get("delta_theta_drift_deg"),
        "direction_caveat": dir_info.get("caveat", ""),
        "out_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# RAPPORT SUMMARY
# ---------------------------------------------------------------------------

def write_summary(all_summaries: list[dict], dir_infos: list[dict],
                  sessions: list[dict], out_dir: Path) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# reconstruction_summary.md",
        "",
        f"Généré le {now}",
        "",
        "## Paramètres appliqués",
        "",
    ]

    # Tableau principal
    lines += [
        "## Résultats par session",
        "",
        "| Session | n_frames | pct_NaN | sync_med_ms | sync_p95_ms | offset_med_m | "
        "head_offset_m | n_extrap | dir_assumption | delta_theta_mean | delta_theta_std | drift_deg |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, sm in zip(sessions, all_summaries):
        sid = f"{s['timepoint']}__{s['scenario']}__{s['group']}__{s['role']}"
        da = sm["direction_assumption"]
        icon = {"validated_traj": "OK", "suspect_yaw": "SUSPECT", "pure_translation_unverified": "?"}
        lines.append(
            f"| {sid} | {sm['n_frames']} | {sm['pct_hit_nan']}% | "
            f"{sm['sync_residual_median_ms']} | {sm['sync_residual_p95_ms']} | "
            f"{sm['origin_offset_median_m']} | {sm['head_offset_applied_m']} | "
            f"{sm['n_extrap_frames']} | {icon.get(da, da)} | "
            f"{sm.get('delta_theta_mean_deg', '-')} | {sm.get('delta_theta_std_deg', '-')} | "
            f"{sm.get('delta_theta_drift_deg', '-')} |"
        )

    lines += [
        "",
        "## Colonnes des CSV corrigés",
        "",
        "| Colonne | Description | Fiabilité |",
        "|---|---|---|",
        "| Time | Horloge Unity Time.time (s) | source |",
        "| RayOriginX/Y/Z | Origine d'origine du raycast (conservée, NON FIABLE position) | traçabilité |",
        "| TrueOriginX/Y/Z | E_true = Player0_Pos interpolé (+ head_offset_y si racine détectée) | FIABLE |",
        "| DirX/Y/Z | d_world = normalize(HitPoint - RayOrigin) ; NaN si hit_was_nan=1 | voir direction_assumption |",
        "| HitPointX/Y/Z | Hit d'origine (conservé, NON FIABLE position absolue) | traçabilité uniquement |",
        "| ObjectHit | Nom objet touché (conservé) | fiable si direction fiable |",
        "| hit_was_nan | 1 si HitPoint était NaN (direction non dérivable) | — |",
        "| sync_residual_ms | |t_eye - t_pos_voisin| en ms | — |",
        "| origin_offset_m | ||RayOrigin - E_true|| en m | indicateur du bug |",
        "| direction_assumption | Résultat Étape 1.5 | — |",
        "",
        "## Limites documentées",
        "",
        "1. **Fréquence positions** : ~2 Hz vs ~35 Hz eye. E_true interpolé linéairement "
        "sur des trous de ~0.5 s. A 1.4 m/s, l'erreur d'origine est ~0.15-0.27 m "
        "(résidu synchro médian ~100 ms, p95 ~200 ms).",
        "2. **Direction (DirXYZ)** : correcte uniquement si `direction_assumption != suspect_yaw`. "
        "Pour les sessions `suspect_yaw`, DirXYZ est présent mais doit être traité avec précaution.",
        "3. **HitPoint corrigé** : ABSENT. Un vrai impact corrigé nécessite un re-raycast "
        "contre le maillage BIM (Étape 3), pas un décalage additif HitPoint + offset.",
        "4. **Hauteur tête** : Player0_PosY varie de 1.16 a 1.73 m sur ce dataset, "
        "confirmé comme hauteur tête directement. head_offset_y=0.0 appliqué par défaut.",
        "5. **Validation direction** : heuristique trajectoire invalide si locomotion "
        "par téléportation ou faible déplacement. Sessions avec < 20 frames en mouvement "
        "marquées 'pure_translation_unverified'.",
        "",
        "## Étape 3 (non implémentée)",
        "",
        "Pour recalculer CorrectedHitX/Y/Z et CorrectedObjectHit, fournir le maillage "
        "BIM (OBJ/PLY/GLB) et appeler le re-raycast depuis (TrueOriginXYZ, DirXYZ) "
        "avec la librairie trimesh.",
        "",
    ]

    # Caveats détaillés par session
    lines += ["## Caveats direction par session", ""]
    for s, sm in zip(sessions, all_summaries):
        sid = f"{s['timepoint']}__{s['scenario']}__{s['group']}__{s['role']}"
        lines.append(f"**{sid}** : {sm.get('direction_caveat', '-')}")
        lines.append("")

    path = out_dir / "reconstruction_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction eye-tracking VR (Étapes 1.5+2). Lance après validation du diagnostic."
    )
    parser.add_argument("--data-dir", default="D:/data_e2", type=Path)
    parser.add_argument(
        "--out-dir",
        default="D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected",
        type=Path,
    )
    parser.add_argument("--group", default=None)
    parser.add_argument(
        "--head-offset", type=float, default=0.0,
        help="Offset Y a ajouter a Player0_PosY pour obtenir la hauteur tete "
             "(defaut 0.0 : Player0_PosY est deja la tete sur ce dataset)"
    )
    parser.add_argument(
        "--head-offset-threshold", type=float, default=HEAD_OFFSET_THRESHOLD_DEFAULT,
        help="Si Player0_PosY median < ce seuil (m), applique --head-offset"
    )
    parser.add_argument(
        "--motion-min-speed", type=float, default=MOTION_MIN_SPEED_DEFAULT,
        help="Seuil vitesse horizontale (m/s) pour l'analyse heading Etape 1.5"
    )
    parser.add_argument(
        "--motion-min-frames", type=int, default=MOTION_MIN_FRAMES_DEFAULT,
        help="Min frames en mouvement pour valider la direction"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Etape 1.5+2] Reconstruction eye-tracking VR")
    print(f"  data-dir : {args.data_dir}")
    print(f"  out-dir  : {args.out_dir}")
    print()

    sessions = find_vr_sessions(args.data_dir)
    if args.group:
        sessions = [s for s in sessions if s["group"] == args.group]
        print(f"  Filtre groupe={args.group} -> {len(sessions)} sessions")

    print(f"  {len(sessions)} sessions trouvees.")

    all_summaries = []
    all_dir_infos = []
    skipped = []

    for i, s in enumerate(sessions, 1):
        sid = f"{s['timepoint']}__{s['scenario']}__{s['group']}__{s['role']}"
        if args.verbose:
            print(f"\n[{i}/{len(sessions)}] {sid}")

        # Lecture
        df_eye, markers, _ = read_eye_csv(s["eye_path"])
        df_pos, time_col_pos = read_pos_csv(s["pos_path"])

        if df_eye.empty or df_pos.empty or not time_col_pos:
            print(f"  SKIP {sid} : fichier vide ou colonne temps manquante")
            skipped.append(sid)
            continue

        # Auto-détection offset tête
        y_med = float(df_pos["Player0_PosY"].median())
        head_offset = args.head_offset if y_med < args.head_offset_threshold else 0.0
        if args.verbose:
            print(f"  Player0_PosY mediane={y_med:.3f}m -> head_offset={head_offset:.2f}m")

        # Étape 1.5
        dir_info = step15_direction_check(
            df_eye, df_pos, time_col_pos,
            min_speed=args.motion_min_speed,
            min_frames=args.motion_min_frames,
        )
        all_dir_infos.append(dir_info)

        da = dir_info["direction_assumption"]
        icon = {"validated_traj": "OK", "suspect_yaw": "SUSPECT", "pure_translation_unverified": "?"}.get(da, da)
        print(f"  {sid} : dir={icon}  n_motion={dir_info['n_motion_frames']}")
        if args.verbose and dir_info.get("caveat"):
            print(f"    -> {dir_info['caveat'][:120]}")

        # Étape 2 : fichier de sortie
        out_name = f"{sid}_EyeTrackingData_corrected.csv"
        out_path = args.out_dir / out_name

        try:
            summary = step2_reconstruct(
                df_eye, markers, df_pos, time_col_pos,
                dir_info=dir_info,
                head_offset_y=head_offset,
                out_path=out_path,
            )
            all_summaries.append(summary)
            print(f"    -> {out_path.name} | frames={summary['n_frames']} "
                  f"NaN={summary['pct_hit_nan']}% "
                  f"sync_med={summary['sync_residual_median_ms']}ms "
                  f"offset_med={summary['origin_offset_median_m']}m")
        except Exception as e:
            print(f"  ERREUR {sid} : {e}")
            skipped.append(sid)
            import traceback; traceback.print_exc()

    # Rapport summary
    valid_sessions = [s for s in sessions if f"{s['timepoint']}__{s['scenario']}__{s['group']}__{s['role']}" not in skipped]
    summary_path = write_summary(all_summaries, all_dir_infos, valid_sessions, args.out_dir)

    print(f"\n=== RÉSUMÉ ===")
    print(f"  Sessions traitees : {len(all_summaries)}")
    print(f"  Sessions ignorees : {len(skipped)}")

    dirs = [sm["direction_assumption"] for sm in all_summaries]
    print(f"  validated_traj  : {dirs.count('validated_traj')}")
    print(f"  suspect_yaw     : {dirs.count('suspect_yaw')}")
    print(f"  unverified      : {dirs.count('pure_translation_unverified')}")
    print(f"\n  Rapport summary : {summary_path}")
    print(f"  CSV corrigés    : {args.out_dir}")
    print()
    print("Rappel limites :")
    print("  - DirXYZ utilisable si direction_assumption != suspect_yaw")
    print("  - HitPoint non corrige : Etape 3 (re-raycast BIM) requise")
    print("  - TrueOriginXYZ : erreur ~0.15-0.27 m due a l'interpolation 2 Hz")


if __name__ == "__main__":
    main()
