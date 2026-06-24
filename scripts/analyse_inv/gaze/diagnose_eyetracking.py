"""
diagnose_eyetracking.py — Étapes 0 + 1 du pipeline de diagnostic eye-tracking VR.

Étape 0 : inventaire, appariement eye ↔ positions, inspection des fichiers.
Étape 1 : diagnostic du bug de position (RayOrigin figé), validation direction,
           alignement temporel, production de rapport_diagnostic_eyetracking.md.

Usage :
    python diagnose_eyetracking.py --data-dir D:/data_e2 --out-dir D:/Analyse_donnee/Longitudinale/results/eyetracking_diagnostic

Après validation du rapport, la reconstruction (Étape 2) sera dans reconstruct_eyetracking.py.
"""
import argparse
import sys
import re
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
EXCLUDED_GROUPS = {"bim002", "bim032", "bim065_2", "bim075"}

EYE_COLS_REQUIRED = [
    "Time", "RayOriginX", "RayOriginY", "RayOriginZ",
    "HitPointX", "HitPointY", "HitPointZ", "ObjectHit",
]
POS_COLS_TIME = {"Timestamp"}
POS_PATTERN_XYZ = re.compile(r"Player(\d+)_Pos([XYZ])")

# Seuil : si l'étendue de RayOrigin (diag bbox) < ce seuil en mètres → probablement figé
FROZEN_DIAG_THRESHOLD_M = 2.0
# Seuil : offset médian RayOrigin vs tête vraie > ce seuil → bug confirmé
OFFSET_BUG_THRESHOLD_M = 0.50
# Tolérance de fusion temporelle (ms) pour le merge nearest-neighbor
MERGE_TOLERANCE_MS = 200.0
# Tolérance angle (°) pour valider que d_world pointe vers le joueur regardé
ANGLE_VALID_DEG = 30.0


# ---------------------------------------------------------------------------
# LECTURE ROBUSTE
# ---------------------------------------------------------------------------

def _detect_decimal(raw_line: str, sep: str) -> str:
    """Détecte le séparateur décimal dans une ligne de données (virgule ou point)."""
    fields = raw_line.strip().split(sep)
    for f in fields:
        f = f.strip()
        if "," in f and "." not in f:
            return ","
        if "." in f and "," not in f:
            return "."
    return "."  # défaut


def read_eye_csv(path: Path) -> tuple[pd.DataFrame, list[str], str, str]:
    """
    Lit un fichier EyeTrackingData CSV avec gestion des lignes MARKER et du token END.
    Renvoie (df_data, markers_list, sep_dec, warnings_str).
    - df_data : lignes données uniquement (MARKER et END exclus)
    - markers_list : [(time_str, raw_line), ...]
    """
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    # Supprime le token END collé à la dernière ligne
    raw = re.sub(r"END\s*$", "", raw.rstrip()) + "\n"

    lines = raw.splitlines()
    if not lines:
        return pd.DataFrame(), [], ".", "fichier vide"

    header_line = lines[0]
    sep = ";"
    data_lines = []
    marker_lines = []
    warnings = []

    # Détection décimale sur la première ligne de données
    sep_dec = "."
    for ln in lines[1:]:
        if ln.strip() == "" or "MARKER" in ln:
            continue
        sep_dec = _detect_decimal(ln, sep)
        break

    for ln in lines[1:]:
        stripped = ln.strip()
        if stripped == "":
            continue
        parts = stripped.split(sep)
        if len(parts) == 2 and "MARKER" in parts[1]:
            marker_lines.append((parts[0].strip(), stripped))
            continue
        data_lines.append(stripped)

    if not data_lines:
        return pd.DataFrame(), marker_lines, sep_dec, "aucune ligne de données"

    # Parse
    from io import StringIO
    content = header_line + "\n" + "\n".join(data_lines)
    try:
        df = pd.read_csv(
            StringIO(content),
            sep=sep,
            decimal=sep_dec,
            engine="python",
            encoding="utf-8",
        )
    except Exception as e:
        return pd.DataFrame(), marker_lines, sep_dec, f"erreur parse: {e}"

    # Normalise les noms de colonnes
    df.columns = [c.strip() for c in df.columns]

    # Convertit Time en float si besoin
    if "Time" in df.columns and df["Time"].dtype == object:
        df["Time"] = pd.to_numeric(
            df["Time"].str.replace(",", "."), errors="coerce"
        )

    for col in ["RayOriginX", "RayOriginY", "RayOriginZ",
                "HitPointX", "HitPointY", "HitPointZ"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    return df, marker_lines, sep_dec, "; ".join(warnings) if warnings else "OK"


def read_pos_csv(path: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Lit un fichier UsersPositions CSV.
    Renvoie (df, time_col, warnings_str).
    """
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    lines = raw.splitlines()
    if not lines:
        return pd.DataFrame(), "", "fichier vide"

    sep = ";"
    sep_dec = "."
    for ln in lines[1:]:
        if ln.strip():
            sep_dec = _detect_decimal(ln, sep)
            break

    from io import StringIO
    try:
        df = pd.read_csv(
            StringIO(raw),
            sep=sep,
            decimal=sep_dec,
            engine="python",
            encoding="utf-8",
        )
    except Exception as e:
        return pd.DataFrame(), "", f"erreur parse: {e}"

    df.columns = [c.strip() for c in df.columns]

    # Identifie colonne temps
    time_col = ""
    for cand in ["Timestamp", "Time"]:
        if cand in df.columns:
            time_col = cand
            break
    if not time_col:
        return df, "", "colonne temps non identifiée"

    if df[time_col].dtype == object:
        df[time_col] = pd.to_numeric(
            df[time_col].astype(str).str.replace(",", "."), errors="coerce"
        )

    for col in df.columns:
        if col == time_col:
            continue
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    return df, time_col, "OK"


# ---------------------------------------------------------------------------
# DÉCOUVERTE DES SESSIONS
# ---------------------------------------------------------------------------

def find_vr_sessions(data_dir: Path) -> list[dict]:
    """
    Retourne une liste de dicts par fichier eye-tracking VR (hors _old/).
    Chaque dict : {eye_path, pos_path, group, role, timepoint, scenario,
                   eye_stem, pos_stem, pairing_status}
    """
    eye_files = [
        p for p in data_dir.rglob("*EyeTrackingData.csv")
        if "_old" not in p.parts and "VR" in p.parts
    ]

    sessions = []
    for eye_path in sorted(eye_files):
        parts = eye_path.parts
        # Extrait timepoint, scenario, group, role depuis le chemin
        try:
            vr_idx = parts.index("VR")
            timepoint = parts[vr_idx - 1]   # ex: T1_FISA_A3
            scenario = parts[vr_idx + 1]    # S1 ou S2
            group = parts[vr_idx + 2]
            role = parts[vr_idx + 3]
        except (ValueError, IndexError):
            continue

        # Cherche le UsersPositions correspondant dans le même dossier
        folder = eye_path.parent
        stem_ts = eye_path.stem.split("_EyeTrackingData")[0]  # ex: 2025-10-24_13-49-40 ou merged
        pos_candidates = list(folder.glob("*UsersPositions.csv"))
        pos_candidates = [p for p in pos_candidates if "_old" not in str(p)]

        pos_path = None
        pairing_status = "OK"

        if len(pos_candidates) == 1:
            pos_path = pos_candidates[0]
        elif len(pos_candidates) == 0:
            pairing_status = "STOP:aucun_UsersPositions"
        else:
            # Plusieurs candidats : essaie de matcher par horodatage
            if stem_ts == "merged":
                # Cherche merged_UsersPositions.csv
                merged_pos = folder / "merged_UsersPositions.csv"
                if merged_pos.exists():
                    pos_path = merged_pos
                else:
                    pairing_status = f"STOP:appariement_ambigu ({len(pos_candidates)} fichiers, pas de merged)"
            else:
                matched = [p for p in pos_candidates if stem_ts in p.stem]
                if len(matched) == 1:
                    pos_path = matched[0]
                elif len(matched) == 0:
                    # Essai par proximité temporelle du nom
                    pairing_status = f"AMBIGUOUS:{len(pos_candidates)}_candidats"
                    # On prend le plus proche temporellement si possible
                    try:
                        t_eye = datetime.strptime(stem_ts, "%Y-%m-%d_%H-%M-%S")
                        def ts_of(p):
                            s = p.stem.replace("_UsersPositions", "")
                            return datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
                        pos_candidates_ts = [(p, abs((ts_of(p) - t_eye).total_seconds()))
                                             for p in pos_candidates if re.match(r"\d{4}-\d{2}-\d{2}", p.stem)]
                        if pos_candidates_ts:
                            best, delta = min(pos_candidates_ts, key=lambda x: x[1])
                            if delta <= 60:
                                pos_path = best
                                pairing_status = f"APPROX:{delta:.0f}s"
                            else:
                                pairing_status = f"STOP:appariement_ambigu_delta>{delta:.0f}s"
                    except Exception:
                        pairing_status = "STOP:appariement_ambigu"
                else:
                    pairing_status = f"STOP:appariement_ambigu_{len(matched)}_matches"

        sessions.append({
            "eye_path": eye_path,
            "pos_path": pos_path,
            "group": group,
            "role": role,
            "timepoint": timepoint,
            "scenario": scenario,
            "pairing_status": pairing_status,
        })

    return sessions


# ---------------------------------------------------------------------------
# CHECK 1 — ORIGINE FIGÉE ?
# ---------------------------------------------------------------------------

def check_origin_frozen(df: pd.DataFrame) -> dict:
    """Retourne les stats spatiales de RayOrigin et HitPoint."""
    result = {}
    for prefix in ["RayOrigin", "HitPoint"]:
        cols = [f"{prefix}X", f"{prefix}Y", f"{prefix}Z"]
        if not all(c in df.columns for c in cols):
            result[prefix] = {"available": False}
            continue
        sub = df[cols].dropna()
        if sub.empty:
            result[prefix] = {"available": False, "n": 0}
            continue
        mins = sub.min()
        maxs = sub.max()
        extents = maxs - mins
        diag = float(np.sqrt((extents**2).sum()))
        result[prefix] = {
            "available": True,
            "n": len(sub),
            "extent_x": float(extents.iloc[0]),
            "extent_y": float(extents.iloc[1]),
            "extent_z": float(extents.iloc[2]),
            "diag_bbox_m": diag,
            "std_x": float(sub.iloc[:, 0].std()),
            "std_y": float(sub.iloc[:, 1].std()),
            "std_z": float(sub.iloc[:, 2].std()),
        }
    frozen = (
        result.get("RayOrigin", {}).get("diag_bbox_m", 999) < FROZEN_DIAG_THRESHOLD_M
        and result.get("HitPoint", {}).get("diag_bbox_m", 0) > FROZEN_DIAG_THRESHOLD_M
    )
    result["verdict_frozen"] = frozen
    return result


# ---------------------------------------------------------------------------
# ALIGNEMENT TEMPOREL + CHECK 2 — OFFSET ORIGINE vs VRAIE TÊTE
# ---------------------------------------------------------------------------

def align_and_check_offset(df_eye: pd.DataFrame, df_pos: pd.DataFrame,
                            time_col_pos: str) -> dict:
    """
    Aligne df_eye (col Time) et df_pos (col time_col_pos) par plus proche voisin.
    Calcule la distance ||RayOrigin - PlayerPos|| frame par frame.
    """
    t_eye = df_eye["Time"].dropna().values
    t_pos = df_pos[time_col_pos].dropna().values

    if len(t_eye) == 0 or len(t_pos) == 0:
        return {"status": "STOP:temps_vide"}

    # Vérifie recouvrement
    overlap_start = max(t_eye.min(), t_pos.min())
    overlap_end = min(t_eye.max(), t_pos.max())
    if overlap_start >= overlap_end:
        return {
            "status": "STOP:aucun_recouvrement_temporel",
            "eye_range": (float(t_eye.min()), float(t_eye.max())),
            "pos_range": (float(t_pos.min()), float(t_pos.max())),
        }

    # Merge nearest-neighbor
    df_e = df_eye[["Time", "RayOriginX", "RayOriginY", "RayOriginZ"]].dropna(subset=["Time"])
    df_p = df_pos[[time_col_pos] + [c for c in df_pos.columns if c != time_col_pos]].dropna(subset=[time_col_pos])
    df_p = df_p.rename(columns={time_col_pos: "Time_pos"})

    df_e_sorted = df_e.sort_values("Time").reset_index(drop=True)
    df_p_sorted = df_p.sort_values("Time_pos").reset_index(drop=True)

    # Utilise merge_asof
    merged = pd.merge_asof(
        df_e_sorted,
        df_p_sorted,
        left_on="Time",
        right_on="Time_pos",
        direction="nearest",
        tolerance=MERGE_TOLERANCE_MS / 1000.0,
    )
    merged = merged.dropna(subset=["Time_pos"])

    residuals_ms = (merged["Time"] - merged["Time_pos"]).abs() * 1000.0

    # Identifie colonnes Player0
    pos_x_col = next((c for c in df_p.columns if "Player0_PosX" in c), None)
    pos_y_col = next((c for c in df_p.columns if "Player0_PosY" in c), None)
    pos_z_col = next((c for c in df_p.columns if "Player0_PosZ" in c), None)

    if not all([pos_x_col, pos_y_col, pos_z_col]):
        return {
            "status": "STOP:colonnes_position_non_identifiées",
            "pos_cols": list(df_p.columns),
        }

    offsets = np.sqrt(
        (merged["RayOriginX"] - merged[pos_x_col])**2 +
        (merged["RayOriginY"] - merged[pos_y_col])**2 +
        (merged["RayOriginZ"] - merged[pos_z_col])**2
    ).dropna()

    n_merged = len(merged)
    pct_merged = 100.0 * n_merged / max(len(df_e), 1)

    verdict = "bug_confirmé" if (offsets.median() > OFFSET_BUG_THRESHOLD_M) else "origine_saine"

    return {
        "status": "OK",
        "n_merged": n_merged,
        "pct_merged": pct_merged,
        "residual_median_ms": float(residuals_ms.median()),
        "residual_p95_ms": float(residuals_ms.quantile(0.95)),
        "offset_median_m": float(offsets.median()),
        "offset_p25_m": float(offsets.quantile(0.25)),
        "offset_p75_m": float(offsets.quantile(0.75)),
        "offset_max_m": float(offsets.max()),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# CHECK 3 — DIRECTION FIABLE ?
# ---------------------------------------------------------------------------

def check_direction_reliability(df_eye: pd.DataFrame, df_pos: pd.DataFrame,
                                 time_col_pos: str) -> dict:
    """
    Pour les frames où PlayerLooked est renseigné, calcule l'angle entre
    d_world = normalize(HitPoint - RayOrigin) et (pos_PlayerLooked - E_true).
    Si la colonne PlayerLooked n'existe pas, retourne un warning non bloquant.
    """
    if "PlayerLooked" not in df_eye.columns:
        return {"status": "SKIP:colonne_PlayerLooked_absente"}

    df_looked = df_eye[df_eye["PlayerLooked"].notna() & (df_eye["PlayerLooked"] != "")].copy()
    if df_looked.empty:
        return {"status": "SKIP:aucune_frame_PlayerLooked"}

    # Identifie les colonnes Player dans df_pos
    player_cols = {}
    for col in df_pos.columns:
        m = POS_PATTERN_XYZ.match(col)
        if m:
            pid, axis = int(m.group(1)), m.group(2)
            player_cols.setdefault(pid, {})[axis] = col

    if not player_cols:
        return {"status": "SKIP:aucune_colonne_player_pos"}

    return {"status": "SKIP:PlayerLooked_absent_de_ce_dataset",
            "note": "La colonne PlayerLooked n'est pas présente dans les fichiers de cette étude."}


# ---------------------------------------------------------------------------
# INSPECTION COMPLÈTE D'UNE SESSION
# ---------------------------------------------------------------------------

def inspect_session(session: dict) -> dict:
    """Inspecte une session et retourne toutes les métriques."""
    result = {**session}
    eye_path: Path = session["eye_path"]
    pos_path: Path = session.get("pos_path")

    # --- Eye tracking ---
    df_eye, markers, sep_dec, warn_eye = read_eye_csv(eye_path)
    result["eye_warn"] = warn_eye
    result["eye_sep_dec"] = sep_dec
    result["eye_markers"] = markers
    result["eye_n_markers"] = len(markers)

    if df_eye.empty:
        result["eye_status"] = "STOP:vide"
        return result

    result["eye_cols"] = list(df_eye.columns)
    result["eye_n_rows"] = len(df_eye)
    result["eye_dtypes"] = {c: str(df_eye[c].dtype) for c in df_eye.columns}
    result["eye_nan_pct"] = {
        c: round(100.0 * df_eye[c].isna().mean(), 2) for c in df_eye.columns
    }

    if "Time" in df_eye.columns:
        t = df_eye["Time"].dropna()
        dt = t.diff().dropna()
        result["eye_time_range"] = (float(t.min()), float(t.max()))
        result["eye_dt_median_ms"] = float(dt.median() * 1000) if len(dt) else None
    else:
        result["eye_time_range"] = None
        result["eye_dt_median_ms"] = None

    # Check colonnes requises manquantes
    missing_cols = [c for c in EYE_COLS_REQUIRED if c not in df_eye.columns]
    result["eye_missing_cols"] = missing_cols

    # --- Position ---
    if pos_path is None:
        result["pos_status"] = "absent"
        result["check1"] = check_origin_frozen(df_eye)
        result["check2"] = {"status": "STOP:pas_de_fichier_positions"}
        result["check3"] = {"status": "SKIP:pas_de_fichier_positions"}
        return result

    df_pos, time_col_pos, warn_pos = read_pos_csv(pos_path)
    result["pos_warn"] = warn_pos
    result["pos_time_col"] = time_col_pos

    if df_pos.empty or not time_col_pos:
        result["pos_status"] = f"STOP:{warn_pos}"
        result["check1"] = check_origin_frozen(df_eye)
        result["check2"] = {"status": f"STOP:{warn_pos}"}
        result["check3"] = {"status": "SKIP"}
        return result

    result["pos_cols"] = list(df_pos.columns)
    result["pos_n_rows"] = len(df_pos)
    t_pos = df_pos[time_col_pos].dropna()
    result["pos_time_range"] = (float(t_pos.min()), float(t_pos.max()))

    # --- Checks ---
    result["check1"] = check_origin_frozen(df_eye)
    result["check2"] = align_and_check_offset(df_eye, df_pos, time_col_pos)
    result["check3"] = check_direction_reliability(df_eye, df_pos, time_col_pos)

    return result


# ---------------------------------------------------------------------------
# GÉNÉRATION DU RAPPORT MARKDOWN
# ---------------------------------------------------------------------------

def format_check1(c1: dict, session_id: str) -> str:
    lines = ["#### Check 1 — Origine figée ?"]
    for prefix in ["RayOrigin", "HitPoint"]:
        d = c1.get(prefix, {})
        if not d.get("available"):
            lines.append(f"- **{prefix}** : données non disponibles")
            continue
        lines.append(
            f"- **{prefix}** : diag_bbox={d['diag_bbox_m']:.3f} m | "
            f"extent (X,Y,Z)=({d['extent_x']:.3f},{d['extent_y']:.3f},{d['extent_z']:.3f}) m | "
            f"std (X,Y,Z)=({d['std_x']:.4f},{d['std_y']:.4f},{d['std_z']:.4f}) m | n={d['n']}"
        )
    frozen = c1.get("verdict_frozen", False)
    icon = "🔴" if frozen else "🟢"
    verdict = "FIGÉE (diag_bbox RayOrigin < 2 m, HitPoint large)" if frozen else "Saine (étendue normale)"
    lines.append(f"\n**Verdict** {icon} : RayOrigin {verdict}")
    return "\n".join(lines)


def format_check2(c2: dict) -> str:
    lines = ["#### Check 2 — Offset RayOrigin vs vraie tête (UsersPositions)"]
    status = c2.get("status", "")
    if status != "OK":
        lines.append(f"⛔ **{status}**")
        for k, v in c2.items():
            if k != "status":
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    lines.append(
        f"- Frames mergées : {c2['n_merged']} ({c2['pct_merged']:.1f}%)"
    )
    lines.append(
        f"- Résidu synchro : médiane={c2['residual_median_ms']:.1f} ms | p95={c2['residual_p95_ms']:.1f} ms"
    )
    lines.append(
        f"- Offset ||RayOrigin – E_true|| : médiane={c2['offset_median_m']:.3f} m | "
        f"p25={c2['offset_p25_m']:.3f} m | p75={c2['offset_p75_m']:.3f} m | max={c2['offset_max_m']:.3f} m"
    )
    verdict = c2.get("verdict", "")
    icon = "🔴" if verdict == "bug_confirmé" else "🟢"
    labels = {
        "bug_confirmé": "BUG CONFIRMÉ — offset > 0.50 m, E_true = UsersPositions",
        "origine_saine": "Origine saine — offset faible, hypothèse invalidée, NE PAS CORRIGER",
    }
    lines.append(f"\n**Verdict** {icon} : {labels.get(verdict, verdict)}")
    return "\n".join(lines)


def format_check3(c3: dict) -> str:
    lines = ["#### Check 3 — Direction d_world fiable ?"]
    status = c3.get("status", "")
    if status.startswith("SKIP"):
        lines.append(f"ℹ️ {status} — {c3.get('note', '')}")
        lines.append(
            "→ Colonne `PlayerLooked` absente de ce dataset (absente du schéma CSV observé). "
            "La validation de direction par cible nommée est impossible. "
            "La direction est considérée correcte par hypothèse (voir Check 1 + 2)."
        )
    elif status.startswith("STOP"):
        lines.append(f"⛔ {status}")
    else:
        for k, v in c3.items():
            if k != "status":
                lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def make_session_id(s: dict) -> str:
    return f"{s['timepoint']}__{s['scenario']}__{s['group']}__{s['role']}"


def generate_report(sessions_results: list[dict], out_dir: Path) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Rapport diagnostic eye-tracking VR",
        f"",
        f"Généré le {now}  ",
        f"Données : `D:/data_e2`  ",
        f"Script : `diagnose_eyetracking.py`",
        f"",
        "---",
        "",
        "## Résumé global",
        "",
    ]

    # Résumé
    total = len(sessions_results)
    stop_pair = [s for s in sessions_results if "STOP" in s.get("pairing_status", "")]
    stop_eye = [s for s in sessions_results if "STOP" in s.get("eye_status", "")]
    frozen_sessions = [
        s for s in sessions_results
        if s.get("check1", {}).get("verdict_frozen") is True
    ]
    bug_confirmed = [
        s for s in sessions_results
        if s.get("check2", {}).get("verdict") == "bug_confirmé"
    ]
    origine_saine = [
        s for s in sessions_results
        if s.get("check2", {}).get("verdict") == "origine_saine"
    ]

    lines += [
        f"| Métrique | N |",
        f"|---|---|",
        f"| Sessions eye-tracking VR trouvées | {total} |",
        f"| Arrêts (appariement ambigu) | {len(stop_pair)} |",
        f"| Arrêts (fichier eye vide) | {len(stop_eye)} |",
        f"| RayOrigin figée (Check 1) | {len(frozen_sessions)} |",
        f"| Bug confirmé (Check 2 offset > 0.50 m) | {len(bug_confirmed)} |",
        f"| Origine saine (Check 2) | {len(origine_saine)} |",
        f"",
    ]

    # Tableau récapitulatif
    lines += [
        "## Tableau récapitulatif",
        "",
        "| Session | Pairing | n_eye | n_pos | dec | Origin_diag_m | HitPoint_diag_m | Offset_médian_m | Résidu_ms | Check1 | Check2 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for s in sessions_results:
        sid = make_session_id(s)
        pair = s.get("pairing_status", "?")
        n_eye = s.get("eye_n_rows", "—")
        n_pos = s.get("pos_n_rows", "—")
        dec = s.get("eye_sep_dec", "?")

        c1 = s.get("check1", {})
        orig_diag = f"{c1.get('RayOrigin', {}).get('diag_bbox_m', float('nan')):.3f}" if c1.get("RayOrigin", {}).get("available") else "—"
        hit_diag = f"{c1.get('HitPoint', {}).get('diag_bbox_m', float('nan')):.3f}" if c1.get("HitPoint", {}).get("available") else "—"
        frozen_v = "🔴FIGÉ" if c1.get("verdict_frozen") else ("🟢OK" if c1.get("RayOrigin", {}).get("available") else "—")

        c2 = s.get("check2", {})
        offset_med = f"{c2.get('offset_median_m', float('nan')):.3f}" if c2.get("status") == "OK" else "—"
        residu = f"{c2.get('residual_median_ms', float('nan')):.1f}" if c2.get("status") == "OK" else "—"
        v2 = c2.get("verdict", c2.get("status", "—"))
        v2_icon = "🔴BUG" if v2 == "bug_confirmé" else ("🟢OK" if v2 == "origine_saine" else f"⛔{v2[:20]}")

        lines.append(f"| {sid} | {pair} | {n_eye} | {n_pos} | {dec} | {orig_diag} | {hit_diag} | {offset_med} | {residu} | {frozen_v} | {v2_icon} |")

    lines.append("")

    # Sections détaillées
    lines += ["---", "", "## Détail par session", ""]

    for s in sessions_results:
        sid = make_session_id(s)
        lines += [f"### {sid}", ""]
        lines.append(f"- **Fichier eye** : `{s['eye_path']}`")
        lines.append(f"- **Fichier pos** : `{s.get('pos_path', 'ABSENT')}`")
        lines.append(f"- **Appariement** : {s.get('pairing_status', '?')}")

        if "STOP" in s.get("pairing_status", ""):
            lines += [
                "",
                f"> ⛔ **ARRÊT** — {s['pairing_status']}  ",
                "> Corriger l'appariement manuellement avant de relancer.",
                "",
            ]
            continue

        if s.get("eye_status", "").startswith("STOP"):
            lines += ["", f"> ⛔ **ARRÊT** — {s['eye_status']}", ""]
            continue

        # Infos eye
        lines += [
            "",
            "**Fichier eye-tracking**",
            f"- Colonnes : `{s.get('eye_cols', [])}`",
            f"- Colonnes manquantes : `{s.get('eye_missing_cols', [])}`",
            f"- N lignes données : {s.get('eye_n_rows', '?')} | N marqueurs : {s.get('eye_n_markers', 0)}",
            f"- Séparateur décimal : `{s.get('eye_sep_dec', '?')}`",
            f"- Plage temps : {s.get('eye_time_range')} s | pas médian : {s.get('eye_dt_median_ms')} ms",
            f"- NaN% par colonne : {s.get('eye_nan_pct', {})}",
            "",
        ]

        if s.get("pos_n_rows"):
            lines += [
                "**Fichier positions**",
                f"- Colonnes : `{s.get('pos_cols', [])}`",
                f"- N lignes : {s.get('pos_n_rows', '?')}",
                f"- Plage temps : {s.get('pos_time_range')} s",
                "",
            ]

        c1 = s.get("check1", {})
        if c1:
            lines.append(format_check1(c1, sid))
            lines.append("")

        c2 = s.get("check2", {})
        if c2:
            lines.append(format_check2(c2))
            lines.append("")

        c3 = s.get("check3", {})
        if c3:
            lines.append(format_check3(c3))
            lines.append("")

        # BARRIÈRE si bug non confirmé
        v2 = s.get("check2", {}).get("verdict", "")
        if v2 == "origine_saine":
            lines += [
                "> 🟢 **Origine saine** — l'hypothèse de freeze est invalidée pour cette session.  ",
                "> **NE PAS CORRIGER**. Vérifier si un autre problème est présent.",
                "",
            ]
        elif v2 == "bug_confirmé":
            lines += [
                "> 🔴 **Bug confirmé** — RayOrigin figée, offset > 0.50 m.  ",
                "> Session éligible à la reconstruction (Étape 2) après validation du rapport.",
                "",
            ]

        lines.append("---")
        lines.append("")

    lines += [
        "## ⛔ BARRIÈRE D'ARRÊT",
        "",
        "Ce rapport documente l'Étape 0 + Étape 1 **uniquement**.  ",
        "**Aucune correction n'a été appliquée.**  ",
        "",
        "Pour procéder à l'Étape 2 (reconstruction), valider ce rapport puis lancer :  ",
        "```",
        "python reconstruct_eyetracking.py --data-dir D:/data_e2 --out-dir <SORTIE>",
        "```",
        "",
    ]

    report_path = out_dir / "rapport_diagnostic_eyetracking.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic eye-tracking VR (Étapes 0+1). Rapport seulement, aucune correction."
    )
    parser.add_argument("--data-dir", default="D:/data_e2", type=Path)
    parser.add_argument(
        "--out-dir",
        default="D:/Analyse_donnee/Longitudinale/results/eyetracking_diagnostic",
        type=Path,
    )
    parser.add_argument("--group", default=None, help="Restreindre à un groupe (ex: bim015)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Étape 0] Découverte des sessions VR dans {args.data_dir} …")
    sessions = find_vr_sessions(args.data_dir)

    if args.group:
        sessions = [s for s in sessions if s["group"] == args.group]
        print(f"  Filtre groupe={args.group} -> {len(sessions)} sessions")

    print(f"  {len(sessions)} sessions eye-tracking VR trouvées.")

    # Signale les arrêts d'appariement immédiatement
    stops = [s for s in sessions if "STOP" in s.get("pairing_status", "")]
    if stops:
        print(f"\n⚠️  {len(stops)} session(s) avec appariement ambigu (STOP) :")
        for s in stops:
            print(f"   - {make_session_id(s)} : {s['pairing_status']}")

    print(f"\n[Étape 1] Inspection et diagnostic par session …")
    results = []
    for i, s in enumerate(sessions, 1):
        sid = make_session_id(s)
        if args.verbose:
            print(f"  [{i}/{len(sessions)}] {sid}")
        r = inspect_session(s)
        results.append(r)

        # Affiche verdicts en temps réel
        c1v = r.get("check1", {}).get("verdict_frozen")
        c2v = r.get("check2", {}).get("verdict", r.get("check2", {}).get("status", "?"))
        icon1 = "🔴FIGÉ" if c1v else ("🟢" if c1v is False else "—")
        icon2 = {"bug_confirmé": "🔴BUG", "origine_saine": "🟢OK"}.get(c2v, f"⛔{c2v[:15]}")
        print(f"  {sid} : Check1={icon1}  Check2={icon2}")

    print(f"\n[Rapport] Génération du rapport markdown …")
    report_path = generate_report(results, args.out_dir)
    print(f"  ✅ Rapport écrit : {report_path}")

    # Résumé console
    bug_sessions = [r for r in results if r.get("check2", {}).get("verdict") == "bug_confirmé"]
    saine_sessions = [r for r in results if r.get("check2", {}).get("verdict") == "origine_saine"]
    print(f"\n=== RÉSUMÉ ===")
    print(f"  Sessions analysées : {len(results)}")
    print(f"  Bug confirmé (origine figée) : {len(bug_sessions)}")
    print(f"  Origine saine (hypothèse invalide) : {len(saine_sessions)}")
    print(f"  Arrêts (appariement/données) : {len(stops)}")
    print(f"\n>>> BARRIÈRE — Consulte le rapport avant toute correction :")
    print(f"    {report_path}")
    print(f"\n>>> Si les Checks 1+2 confirment le bug, lancer :")
    print(f"    python reconstruct_eyetracking.py --data-dir {args.data_dir} --out-dir {args.out_dir}")


if __name__ == "__main__":
    main()
