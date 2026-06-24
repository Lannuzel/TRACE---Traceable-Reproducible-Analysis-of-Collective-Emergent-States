"""
refine_yaw_eyetracking.py — Étape 1.6 : estimation du yaw playspace→monde par Kabsch.

Principe géométrique (à lire avant toute modification) :
    Unity écrit RayOrigin = transform.position = position de la tête dans le repère
    PHYSIQUE (playspace). Le logger World écrit Player0_Pos = position de la tête dans le
    repère VIRTUEL (monde). Ces deux logs mesurent le MÊME objet physique (la tête) dans
    deux repères liés par une transformation rigide :
        p_monde(t) = R_yaw(t) · p_play(t) + T(t)
    où R_yaw est la rotation horizontale entre le sol physique et le sol virtuel, et T(t)
    est la translation. Ces deux paramètres CHANGENT à chaque téléportation avec
    orientation (le joueur peut pivoter en se téléportant).

    Conséquence : sur un PLATEAU entre deux téléportations, R_yaw et T sont constants.
    En démoyennant p_monde et p_play sur ce plateau, T disparaît ; Kabsch sur les résidus
    centrés estime R_yaw pour ce plateau, depuis le BALLANT PHYSIQUE DE TÊTE.
    Il n'y a PAS de yaw global cohérent entre plateaux.

Limitations critiques documentées ici :
    L1. Ambiguïté de signe R vs R+180° : un ballant purement linéaire peut donner theta
        ou theta+180°. Elle n'est PAS résolue par continuité inter-plateaux (le yaw
        change à chaque téléportation). Elle est détectée par la stabilité INTRA-plateau
        (sous-fenêtres de 20 s) : si les sous-fenêtres convergent vers la même valeur,
        theta est fiable ; si elles oscillent ±90°, l'ambiguïté n'est pas levée.
    L2. Ballant insuffisant (<SWAY_MIN_STD_M en XZ) → fenêtre rejetée ; pas de signal.
    L3. Résidu Kabsch élevé (>KABSCH_RESID_MAX_M) → mauvais fit ; fenêtre rejetée.
    L4. UsersPositions ~2 Hz : RayOrigin interpolé aux timestamps de Pos.
    L5. Stabilité intra-plateau : std des sous-fenêtres > INTRA_STD_MAX_DEG → plateau
        avec ambiguïté non résolue, DirCorr marqué NaN pour ce plateau.
    L6. TELEPORT_GUARD_S : exclusion des N secondes après chaque téléportation.

Statuts possibles (direction_assumption) :
    "translation_pure_confirmee"  : |theta| < NEAR_ZERO_DEG sur tous les plateaux valides
    "yaw_recupere"                : theta estimé et stable intra-plateau pour ≥1 plateau
    "indetermine"                 : ballant insuffisant ou ambiguïté non résolue
    (les statuts heuristiques de l'étape 1.5 sont écrasés si on obtient mieux ici)

Usage :
    python refine_yaw_eyetracking.py \
        --corrected-dir D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected \
        --data-dir      D:/data_e2 \
        --out-dir       D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected \
        --group bim015
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
# PARAMÈTRES (tous modifiables en CLI)
# ---------------------------------------------------------------------------
TELEPORT_THRESH_M    = 0.5    # saut monde > ce seuil → nouvelle position (téléportation)
TELEPORT_GUARD_S     = 1.0    # exclure N secondes après chaque téléportation
PLATEAU_MIN_DUR_S    = 3.0    # durée minimale d'un plateau pour être utilisé
PLATEAU_MIN_POS_PTS  = 4      # nombre minimal de points Pos dans un plateau
SWAY_MIN_STD_M       = 0.008  # ballant minimal (std XZ) pour contraindre Kabsch (8 mm)
KABSCH_RESID_MAX_M   = 0.040  # résidu Kabsch max acceptable (4 cm)
NEAR_ZERO_DEG        = 5.0    # |theta| < ce seuil → translation pure
# Stabilité intra-plateau (sous-fenêtres de 20 s)
INTRA_WINDOW_S       = 20.0   # durée des sous-fenêtres pour validation intra-plateau
INTRA_STEP_S         = 15.0   # pas entre sous-fenêtres
INTRA_MIN_WINS       = 2      # nombre minimal de sous-fenêtres valides
INTRA_STD_MAX_DEG    = 15.0   # std intra-plateau max → ambiguïté levée
# Fraction minimale de plateaux valides (avec theta stable) pour "yaw_recupere"
MIN_VALID_PLATEAU_FRAC = 0.30


# ---------------------------------------------------------------------------
# I/O (minimal, autonome)
# ---------------------------------------------------------------------------

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )
    return df


def read_pos(path: Path) -> tuple[pd.DataFrame, str]:
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    sep_dec = "."
    for ln in raw.splitlines()[1:]:
        if ln.strip():
            if "," in ln.split(";")[0].split(",")[0] if ";" in ln else ",":
                sep_dec = ","
            break
    df = pd.read_csv(StringIO(raw), sep=";", decimal=",", engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = _to_numeric(df)
    time_col = next((c for c in ["Timestamp", "Time"] if c in df.columns), "")
    return df.dropna(subset=[time_col]) if time_col else df, time_col


def read_eye_raw(path: Path) -> pd.DataFrame:
    raw = path.read_bytes().decode("utf-8-sig", errors="replace")
    raw = re.sub(r"END\s*$", "", raw.rstrip()) + "\n"
    lines = raw.splitlines()
    data_lines = [l for l in lines[1:] if l.strip() and "MARKER" not in l]
    df = pd.read_csv(
        StringIO(lines[0] + "\n" + "\n".join(data_lines)),
        sep=";", decimal=",", engine="python"
    )
    df.columns = [c.strip() for c in df.columns]
    return _to_numeric(df).dropna(subset=["Time"])


def read_corrected(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=".", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return _to_numeric(df)


# ---------------------------------------------------------------------------
# KABSCH 2D (yaw seulement, plan XZ)
# ---------------------------------------------------------------------------

def kabsch_yaw(A_xz: np.ndarray, B_xz: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Aligne A (playspace XZ) sur B (monde XZ) par rotation pure (pas de réflexion).
    Retourne (theta_deg, resid_m, R_2x2).
    A et B doivent déjà être centrés (démoyen és).
    """
    H = A_xz.T @ B_xz          # 2×2
    U, S, Vt = np.linalg.svd(H)
    # Assure det=+1 (rotation pure, pas de réflexion)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T         # 2×2
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    A_rot = (R @ A_xz.T).T
    resid = float(np.sqrt(((A_rot - B_xz) ** 2).sum(axis=1)).mean())
    return theta, resid, R


def wrap180(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


# ---------------------------------------------------------------------------
# IDENTIFICATION DES PLATEAUX
# ---------------------------------------------------------------------------

def find_plateaux(df_pos: pd.DataFrame, time_col: str) -> list[dict]:
    """
    Segmente df_pos en plateaux entre téléportations.
    Exclut TELEPORT_GUARD_S secondes après chaque saut.
    Retourne liste de dicts {t_start, t_end, mean_x, mean_z, idx_start, idx_end}.
    """
    ts = df_pos[time_col].values
    px = df_pos["Player0_PosX"].values
    pz = df_pos["Player0_PosZ"].values

    dxz = np.sqrt(np.diff(px) ** 2 + np.diff(pz) ** 2)
    teleport_at = np.where(dxz > TELEPORT_THRESH_M)[0] + 1  # indices dans ts

    # Construire les intervalles entre téléportations
    boundaries = [0] + list(teleport_at) + [len(ts)]
    plateaux = []
    for i in range(len(boundaries) - 1):
        i0 = boundaries[i]
        i1 = boundaries[i + 1]
        if i1 - i0 < PLATEAU_MIN_POS_PTS:
            continue
        # Exclure TELEPORT_GUARD_S après le début
        t0_raw = ts[i0]
        t0 = t0_raw + TELEPORT_GUARD_S if i > 0 else t0_raw
        t1 = ts[i1 - 1]
        if t1 - t0 < PLATEAU_MIN_DUR_S:
            continue
        # Points effectifs dans la fenêtre gardée
        mask = (ts[i0:i1] >= t0)
        idx_eff = np.where(mask)[0] + i0
        if len(idx_eff) < PLATEAU_MIN_POS_PTS:
            continue
        plateaux.append({
            "t_start": float(ts[idx_eff[0]]),
            "t_end": float(ts[idx_eff[-1]]),
            "mean_x": float(px[idx_eff].mean()),
            "mean_z": float(pz[idx_eff].mean()),
            "idx_pos": idx_eff,
        })
    return plateaux


# ---------------------------------------------------------------------------
# KABSCH PAR PLATEAU — avec validation intra-plateau (sous-fenêtres)
# ---------------------------------------------------------------------------

def _kabsch_window(ray_x_at_pos: np.ndarray, ray_z_at_pos: np.ndarray,
                   pos_x: np.ndarray, pos_z: np.ndarray) -> tuple[float, float] | None:
    """
    Kabsch 2D sur un ensemble de points déjà appariés (interp faite par l'appelant).
    Retourne (theta_deg, resid_m) ou None si données insuffisantes / SVD échoue.
    """
    A = np.column_stack([ray_x_at_pos, ray_z_at_pos])
    B = np.column_stack([pos_x, pos_z])
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    std_A = float(np.sqrt(np.var(Ac)))
    std_B = float(np.sqrt(np.var(Bc)))
    if std_A < SWAY_MIN_STD_M or std_B < SWAY_MIN_STD_M:
        return None
    H = Ac.T @ Bc
    if not np.all(np.isfinite(H)):
        return None
    try:
        U, _S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    theta = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    Ar = (R @ Ac.T).T
    resid = float(np.sqrt(((Ar - Bc) ** 2).sum(axis=1)).mean())
    return theta, resid


def _intra_plateau_stability(
    t_eye: np.ndarray, ray_x: np.ndarray, ray_z: np.ndarray,
    t_pos: np.ndarray, pos_x: np.ndarray, pos_z: np.ndarray,
) -> dict:
    """
    Découpe un plateau en sous-fenêtres glissantes (INTRA_WINDOW_S, pas INTRA_STEP_S).
    Dans chaque sous-fenêtre, interpole RayOrigin aux instants Pos puis lance Kabsch.
    Retourne un dict avec la liste des thetas valides, leur std, et le verdict d'ambiguïté.

    Résolution de l'ambiguïté R/R+180° INTRA-plateau :
      - Si toutes les sous-fenêtres donnent des thetas à moins de INTRA_STD_MAX_DEG l'un
        de l'autre (après test des deux candidats theta / theta±180°), l'ambiguïté est levée.
      - Sinon : "ambiguity_unresolved".
    """
    t_total_start = t_pos[0]
    t_total_end = t_pos[-1]
    duration = t_total_end - t_total_start

    win_thetas = []
    win_residuals = []

    ws = 0.0
    while ws + INTRA_WINDOW_S <= duration + 1e-3:
        s0 = t_total_start + ws
        s1 = s0 + INTRA_WINDOW_S

        mask_p = (t_pos >= s0) & (t_pos <= s1)
        if mask_p.sum() < PLATEAU_MIN_POS_PTS:
            ws += INTRA_STEP_S
            continue

        tp_w = t_pos[mask_p]
        mask_e = (t_eye >= s0) & (t_eye <= s1)
        te_w = t_eye[mask_e]
        rx_w = ray_x[mask_e]
        rz_w = ray_z[mask_e]
        if len(te_w) < 5:
            ws += INTRA_STEP_S
            continue

        rx_at_p = np.interp(tp_w, te_w, rx_w)
        rz_at_p = np.interp(tp_w, te_w, rz_w)

        res = _kabsch_window(rx_at_p, rz_at_p, pos_x[mask_p], pos_z[mask_p])
        if res is not None:
            win_thetas.append(res[0])
            win_residuals.append(res[1])

        ws += INTRA_STEP_S

    if len(win_thetas) < INTRA_MIN_WINS:
        return {
            "status": "too_few_subwindows",
            "n_subwins": len(win_thetas),
            "theta_deg": None, "intra_std_deg": None, "resid_m": None,
        }

    thetas = np.array(win_thetas)
    residuals = np.array(win_residuals)

    # Résolution ambiguïté : tester si theta ou theta±180° donne un ensemble plus compact.
    # On cherche le flip de signe binaire (vecteur {0,1}^N) minimisant la variance circulaire.
    # Pour N petit (<20), exhaustif. Pour N grand, greedy depuis la médiane.
    best_thetas = _resolve_ambiguity_intra(thetas, residuals)
    intra_std = float(np.std(best_thetas))
    theta_circ = _circ_mean(best_thetas)

    if intra_std > INTRA_STD_MAX_DEG:
        status = "ambiguity_unresolved"
    elif residuals.mean() > KABSCH_RESID_MAX_M:
        status = "resid_eleve"
    else:
        status = "ok"

    return {
        "status": status,
        "n_subwins": len(win_thetas),
        "theta_deg": float(theta_circ),
        "intra_std_deg": intra_std,
        "resid_m": float(residuals.mean()),
        "thetas_subwins": best_thetas.tolist(),
    }


def _resolve_ambiguity_intra(thetas: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """
    Lève l'ambiguïté theta / theta+180° INTRA-plateau par greedy depuis la fenêtre
    la mieux contrainte (résidu minimal), en propageant le candidat le plus proche.
    Le yaw ne dérive pas au sein d'un plateau → contrainte de continuité valide ici.
    """
    if len(thetas) <= 1:
        return thetas.copy()
    out = thetas.copy().astype(float)
    anchor = int(np.argmin(residuals))
    # Propage vers l'avant
    for i in range(anchor + 1, len(thetas)):
        prev = out[i - 1]
        c0 = thetas[i]
        c1 = wrap180(thetas[i] + 180.0)
        out[i] = c0 if abs(wrap180(c0 - prev)) <= abs(wrap180(c1 - prev)) else c1
    # Propage vers l'arrière
    for i in range(anchor - 1, -1, -1):
        prev = out[i + 1]
        c0 = thetas[i]
        c1 = wrap180(thetas[i] + 180.0)
        out[i] = c0 if abs(wrap180(c0 - prev)) <= abs(wrap180(c1 - prev)) else c1
    return out


def _circ_mean(thetas: np.ndarray) -> float:
    """Moyenne circulaire d'un tableau d'angles en degrés."""
    r = np.radians(thetas)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def estimate_theta_plateau(
    plateau: dict,
    df_pos: pd.DataFrame,
    time_col: str,
    df_eye: pd.DataFrame,
) -> dict | None:
    """
    Estime le yaw d'un plateau par Kabsch global + validation intra-plateau.
    Retourne None seulement si le plateau est trop court pour être analysé du tout.
    Sinon retourne toujours un dict avec un champ "status".
    """
    t0, t1 = plateau["t_start"], plateau["t_end"]

    ew = df_eye[(df_eye["Time"] >= t0) & (df_eye["Time"] <= t1)]
    if len(ew) < 10:
        return None

    idx = plateau["idx_pos"]
    pg = df_pos.iloc[idx][[time_col, "Player0_PosX", "Player0_PosZ"]].dropna()
    if len(pg) < PLATEAU_MIN_POS_PTS:
        return None

    t_pos = pg[time_col].values
    t_eye = ew["Time"].values
    ray_x = ew["RayOriginX"].values
    ray_z = ew["RayOriginZ"].values
    pos_x = pg["Player0_PosX"].values
    pos_z = pg["Player0_PosZ"].values

    # Kabsch global (sur tout le plateau)
    ray_x_at_pos = np.interp(t_pos, t_eye, ray_x)
    ray_z_at_pos = np.interp(t_pos, t_eye, ray_z)
    A = np.column_stack([ray_x_at_pos, ray_z_at_pos])
    B = np.column_stack([pos_x, pos_z])
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    std_A = float(np.sqrt(np.var(Ac)))
    std_B = float(np.sqrt(np.var(Bc)))

    base = {
        "t_start": t0, "t_end": t1, "duration": t1 - t0,
        "n_pos": len(pg), "std_A_m": std_A, "std_B_m": std_B,
    }

    if std_A < SWAY_MIN_STD_M or std_B < SWAY_MIN_STD_M:
        return {**base, "theta_deg": None, "resid_m": None, "R": None,
                "intra_std_deg": None, "n_subwins": 0, "status": "ballant_insuffisant"}

    global_res = _kabsch_window(ray_x_at_pos, ray_z_at_pos, pos_x, pos_z)
    if global_res is None:
        return {**base, "theta_deg": None, "resid_m": None, "R": None,
                "intra_std_deg": None, "n_subwins": 0, "status": "svd_failed"}

    theta_global, resid_global = global_res

    # Validation intra-plateau (lève l'ambiguïté R/R+180° et mesure la stabilité)
    duration = t1 - t0
    if duration >= INTRA_WINDOW_S:
        intra = _intra_plateau_stability(t_eye, ray_x, ray_z, t_pos, pos_x, pos_z)
    else:
        # Plateau trop court pour les sous-fenêtres : on utilise le global seul,
        # l'ambiguïté reste non résolue (statut = too_few_subwindows)
        intra = {
            "status": "too_few_subwindows", "n_subwins": 0,
            "theta_deg": theta_global, "intra_std_deg": None, "resid_m": resid_global,
        }

    # Statut final du plateau
    intra_status = intra["status"]
    if intra_status == "ok":
        # Utilise le theta intra (ambiguïté résolue, plus robuste)
        theta_final = intra["theta_deg"]
        status = "ok"
    elif intra_status == "too_few_subwindows":
        # Plateau court : theta global, ambiguïté non résolue mais résidu peut être bon
        theta_final = theta_global
        status = "ambiguity_unresolved" if resid_global <= KABSCH_RESID_MAX_M else "resid_eleve"
    elif intra_status == "ambiguity_unresolved":
        theta_final = intra["theta_deg"]  # theta intra-moyen (non fiable)
        status = "ambiguity_unresolved"
    else:  # resid_eleve
        theta_final = intra["theta_deg"]
        status = "resid_eleve"

    # Reconstruit R à partir de theta_final
    th = np.radians(theta_final) if theta_final is not None else 0.0
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    return {
        **base,
        "theta_deg": theta_final,
        "theta_global_deg": theta_global,
        "resid_m": intra.get("resid_m", resid_global),
        "resid_global_m": resid_global,
        "R": R,
        "intra_std_deg": intra.get("intra_std_deg"),
        "n_subwins": intra.get("n_subwins", 0),
        "status": status,
    }


# ---------------------------------------------------------------------------
# VERDICT PAR SESSION
# ---------------------------------------------------------------------------

def session_verdict(plateau_results: list[dict]) -> dict:
    """
    Agrège les résultats de Kabsch par plateau en un verdict de session.
    Chaque plateau a son propre theta (le yaw change à chaque téléportation).
    Le verdict porte sur la FRACTION de plateaux bien contraints.
    """
    n_total = len(plateau_results)
    n_ballant_insuf = sum(1 for p in plateau_results if p.get("status") == "ballant_insuffisant")
    n_resid_eleve   = sum(1 for p in plateau_results if p.get("status") == "resid_eleve")
    n_ambig         = sum(1 for p in plateau_results if p.get("status") == "ambiguity_unresolved")
    n_svd_fail      = sum(1 for p in plateau_results if p.get("status") in ("svd_failed", "too_few_subwindows"))

    # Plateaux "ok" : theta estimé et ambiguïté levée
    valid = [p for p in plateau_results if p.get("status") == "ok" and p.get("theta_deg") is not None]

    base = {
        "n_plateaux_total": n_total,
        "n_plateaux_valid": len(valid),
        "n_ballant_insuf": n_ballant_insuf,
        "n_resid_eleve": n_resid_eleve,
        "n_ambiguity_unresolved": n_ambig,
        "n_svd_or_short": n_svd_fail,
    }

    if not valid:
        return {
            **base,
            "direction_assumption": "indetermine",
            "theta_median_deg": None,
            "theta_std_deg": None,
            "resid_median_m": None,
            "note": (
                f"Aucun plateau avec ambiguïté levée ({n_ambig} ambigus, "
                f"{n_ballant_insuf} ballant insuf., {n_resid_eleve} résidu élevé). "
                "Re-raycast maillage requis."
            ),
        }

    thetas   = np.array([p["theta_deg"] for p in valid])
    residuals = np.array([p["resid_m"]   for p in valid])
    frac_valid = len(valid) / max(n_total, 1)

    theta_med  = float(np.median(thetas))
    theta_std  = float(np.std(thetas))
    resid_med  = float(np.median(residuals))
    theta_circ = _circ_mean(thetas)

    if frac_valid < MIN_VALID_PLATEAU_FRAC:
        assumption = "indetermine"
        note = (
            f"Seulement {len(valid)}/{n_total} plateaux valides "
            f"({100*frac_valid:.0f}% < {100*MIN_VALID_PLATEAU_FRAC:.0f}%). "
            "Re-raycast maillage requis."
        )
    elif abs(theta_circ) < NEAR_ZERO_DEG and theta_std < NEAR_ZERO_DEG:
        assumption = "translation_pure_confirmee"
        note = (
            f"Yaw ~ 0° sur tous les plateaux valides "
            f"(moy.circ.={theta_circ:.2f}°, std={theta_std:.2f}°). "
            "d_world valide sans correction."
        )
    else:
        assumption = "yaw_recupere"
        note = (
            f"{len(valid)}/{n_total} plateaux valides. "
            f"Theta par plateau : moy.circ.={theta_circ:.2f}°, "
            f"std_inter={theta_std:.2f}°, résidu médian={resid_med*100:.2f} cm. "
            "DirCorrXYZ appliqué plateau par plateau dans le CSV."
        )

    return {
        **base,
        "direction_assumption": assumption,
        "theta_median_deg": round(theta_med, 3),
        "theta_circ_mean_deg": round(theta_circ, 3),
        "theta_std_deg": round(theta_std, 3),
        "resid_median_m": round(resid_med, 5),
        "note": note,
        "thetas_per_plateau": thetas.tolist(),
        "t_starts": [p["t_start"] for p in valid],
        "t_ends":   [p["t_end"]   for p in valid],
    }


# ---------------------------------------------------------------------------
# APPLICATION DE LA CORRECTION DIRECTIONNELLE
# ---------------------------------------------------------------------------

def apply_dir_correction(
    df_corr: pd.DataFrame,
    verdict: dict,
    plateau_results: list[dict],
) -> pd.DataFrame:
    """
    Ajoute DirCorrX/Y/Z (correction yaw) et met à jour direction_assumption.
    - Si "translation_pure_confirmee" : DirCorr = Dir (copie).
    - Si "yaw_recupere" : applique R(t) interpolé frame-par-frame depuis les plateaux.
    - Sinon : DirCorr = NaN.
    Conserve toujours DirX/Y/Z original.
    """
    df = df_corr.copy()
    da = verdict["direction_assumption"]
    df["direction_assumption"] = da

    n = len(df)
    dir_corr_x = np.full(n, np.nan)
    dir_corr_y = np.full(n, np.nan)
    dir_corr_z = np.full(n, np.nan)

    if da == "translation_pure_confirmee":
        dir_corr_x = df["DirX"].values.copy()
        dir_corr_y = df["DirY"].values.copy()
        dir_corr_z = df["DirZ"].values.copy()

    elif da == "yaw_recupere":
        valid = [p for p in plateau_results if p.get("status") == "ok" and p["theta_deg"] is not None]
        if valid:
            t_eye = df["Time"].values
            dir_x = df["DirX"].values
            dir_y = df["DirY"].values
            dir_z = df["DirZ"].values
            theta_interp = np.full(len(t_eye), np.nan)

            # Applique chaque theta dans la fenêtre de son plateau uniquement.
            # Entre plateaux (téléportations), DirCorr reste NaN : le yaw est inconnu.
            for p in valid:
                mask = (t_eye >= p["t_start"]) & (t_eye <= p["t_end"])
                theta_interp[mask] = p["theta_deg"]

            # Frames avec theta connu → applique la rotation XZ
            has_theta = ~np.isnan(theta_interp)
            valid_dir = ~np.isnan(dir_x)
            apply_mask = has_theta & valid_dir

            cos_t = np.where(apply_mask, np.cos(np.radians(theta_interp)), np.nan)
            sin_t = np.where(apply_mask, np.sin(np.radians(theta_interp)), np.nan)

            new_x = np.where(apply_mask, cos_t * dir_x - sin_t * dir_z, np.nan)
            new_z = np.where(apply_mask, sin_t * dir_x + cos_t * dir_z, np.nan)
            new_y = np.where(apply_mask, dir_y, np.nan)

            # Re-normalise (rotation 2D ne préserve pas la norme 3D à cause de DirY)
            norms = np.sqrt(new_x ** 2 + new_y ** 2 + new_z ** 2)
            safe = norms > 1e-9
            dir_corr_x = np.where(safe, new_x / norms, np.nan)
            dir_corr_y = np.where(safe, new_y / norms, np.nan)
            dir_corr_z = np.where(safe, new_z / norms, np.nan)

            df["yaw_theta_applied_deg"] = theta_interp

    df["DirCorrX"] = dir_corr_x
    df["DirCorrY"] = dir_corr_y
    df["DirCorrZ"] = dir_corr_z
    return df


# ---------------------------------------------------------------------------
# DÉCOUVERTE DES SESSIONS (lecture depuis corrected-dir)
# ---------------------------------------------------------------------------

def find_sessions(corrected_dir: Path, data_dir: Path, group_filter: str | None) -> list[dict]:
    """
    Mappe les CSV corrigés → fichiers sources (RayOrigin brut + UsersPositions).
    """
    # Pattern : {timepoint}__{scenario}__{group}__{role}_EyeTrackingData_corrected.csv
    # Champs séparés par __ (double underscore). group peut contenir _ simple (ex: bim073_2).
    # On split sur __ puis on isole le suffixe fixe _EyeTrackingData_corrected.csv.
    sessions = []
    for f in sorted(corrected_dir.glob("*_EyeTrackingData_corrected.csv")):
        stem = f.name.replace("_EyeTrackingData_corrected.csv", "")
        parts = stem.split("__")
        if len(parts) != 4:
            continue
        tp, sc, grp, role = parts
        if group_filter and grp != group_filter:
            continue

        # Trouver la source dans data_dir
        role_dir = data_dir / tp / "VR" / sc / grp / role
        if not role_dir.exists():
            continue

        eye_files = [p for p in role_dir.glob("*EyeTrackingData.csv") if "_old" not in str(p)]
        pos_files = [p for p in role_dir.glob("*UsersPositions.csv") if "_old" not in str(p)]

        if not eye_files or not pos_files:
            continue

        # Même logique que analyze_gaze.py : merged en priorité, sinon le plus récent (mtime)
        eye_path = next((p for p in eye_files if "merged" in p.name),
                        max(eye_files, key=lambda p: p.stat().st_mtime))
        pos_path = next((p for p in pos_files if "merged" in p.name),
                        max(pos_files, key=lambda p: p.stat().st_mtime))

        sessions.append({
            "corrected_path": f,
            "eye_path": eye_path,
            "pos_path": pos_path,
            "timepoint": tp, "scenario": sc, "group": grp, "role": role,
            "session_id": f"{tp}__{sc}__{grp}__{role}",
        })
    return sessions


# ---------------------------------------------------------------------------
# RAPPORT MARKDOWN
# ---------------------------------------------------------------------------

def write_report(all_verdicts: list[tuple[dict, dict]], out_dir: Path) -> Path:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Rapport Étape 1.6 — Estimation yaw playspace→monde (Kabsch)",
        "",
        f"Généré le {now}",
        "",
        "## Principe",
        "",
        "RayOrigin (playspace) et Player0_Pos (monde) mesurent le même ballant physique de tête "
        "dans deux repères liés par `p_monde = R_yaw · p_play + T(t)`. "
        "Sur chaque plateau entre téléportations, T est constant ; Kabsch sur les résidus centrés "
        "estime R_yaw (1 DOF, plan XZ).",
        "",
        "**Limitations** : (L1) ambiguïté R/R+180° résolue INTRA-plateau par sous-fenêtres "
        "(le yaw change à chaque téléportation → pas de continuité inter-plateaux) ; "
        "(L2) ballant < 8 mm → rejeté ; (L3) résidu > 4 cm → rejeté ; "
        "(L4) UsersPositions ~2 Hz, RayOrigin interpolé aux instants Pos ; "
        "(L5) DirCorr = NaN hors fenêtres de plateau valides (transitions entre téléportations).",
        "",
        "---",
        "",
        "## Tableau récapitulatif",
        "",
        "| Session | dir_assumption | theta_med° | theta_std° | "
        "resid_med_cm | n_valid | n_insuf | n_resid_elev | n_ambig |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for s, v in all_verdicts:
        sid = s["session_id"]
        da = v["direction_assumption"]
        icon = {"translation_pure_confirmee": "OK", "yaw_recupere": "YAW", "indetermine": "??"}.get(da, "?")
        th = f"{v['theta_median_deg']:.2f}" if v.get("theta_median_deg") is not None else "—"
        ts = f"{v['theta_std_deg']:.2f}" if v.get("theta_std_deg") is not None else "—"
        rm = f"{v['resid_median_m']*100:.2f}" if v.get("resid_median_m") is not None else "—"
        lines.append(
            f"| {sid} | {icon} {da} | {th} | {ts} | {rm} | "
            f"{v['n_plateaux_valid']} | {v['n_ballant_insuf']} | {v['n_resid_eleve']} | "
            f"{v.get('n_ambiguity_unresolved', 0)} |"
        )

    lines += ["", "---", "", "## Détail par session", ""]
    for s, v in all_verdicts:
        sid = s["session_id"]
        lines += [
            f"### {sid}",
            "",
            f"**Statut** : `{v['direction_assumption']}`  ",
            f"**Note** : {v['note']}",
            "",
            f"- Plateaux : {v['n_plateaux_total']} total | "
            f"{v['n_plateaux_valid']} valides (ambiguïté levée) | "
            f"{v['n_ballant_insuf']} ballant insuf. | "
            f"{v['n_resid_eleve']} résidu élevé | "
            f"{v.get('n_ambiguity_unresolved', 0)} ambigus | "
            f"{v.get('n_svd_or_short', 0)} SVD/courts",
        ]
        if v.get("theta_median_deg") is not None:
            lines += [
                f"- theta médian inter-plateaux : {v['theta_median_deg']:.3f}°  "
                f"(moy. circ. {v.get('theta_circ_mean_deg', 0):.3f}°, "
                f"std_inter={v['theta_std_deg']:.3f}°)",
                f"- résidu Kabsch médian : {v['resid_median_m']*100:.3f} cm",
            ]
            if v.get("thetas_per_plateau"):
                thetas_str = ", ".join(f"{t:.1f}" for t in v["thetas_per_plateau"])
                lines.append(f"- thetas par plateau valide : [{thetas_str}]")
        lines += ["", "---", ""]

    lines += [
        "## Colonnes ajoutées aux CSV",
        "",
        "| Colonne | Description |",
        "|---|---|",
        "| DirCorrX/Y/Z | Direction corrigée du yaw (R_median·d_world) ; NaN si indéterminé ou hit_was_nan |",
        "| yaw_theta_applied_deg | Theta interpolé appliqué à chaque frame (yaw_récupéré seulement) |",
        "| direction_assumption | Statut mis à jour (remplace l'heuristique Étape 1.5) |",
        "",
        "**Rappel** : DirX/Y/Z (brut, sans correction yaw) est conservé intact.",
        "",
        "**Étape 3** : pour les sessions `indetermine`, seul un re-raycast contre le maillage BIM "
        "depuis (TrueOriginXYZ, DirCorrXYZ ou DirXYZ) permettra de récupérer un CorrectedHitX/Y/Z.",
        "",
    ]

    path = out_dir / "rapport_etape16_yaw.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Étape 1.6 : estimation yaw playspace→monde par Kabsch + correction DirXYZ."
    )
    parser.add_argument(
        "--corrected-dir",
        default="D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected",
        type=Path,
    )
    parser.add_argument("--data-dir", default="D:/data_e2", type=Path)
    parser.add_argument(
        "--out-dir",
        default="D:/Analyse_donnee/Longitudinale/results/eyetracking_corrected",
        type=Path,
    )
    parser.add_argument("--group", default=None, help="Restreindre à un groupe")
    parser.add_argument(
        "--teleport-thresh", type=float, default=TELEPORT_THRESH_M,
        help="Seuil de saut pour détecter une téléportation (m)"
    )
    parser.add_argument(
        "--sway-min-std", type=float, default=SWAY_MIN_STD_M,
        help="Ballant minimal (std XZ, m) pour que Kabsch soit contraint"
    )
    parser.add_argument(
        "--kabsch-resid-max", type=float, default=KABSCH_RESID_MAX_M,
        help="Résidu Kabsch max acceptable (m)"
    )
    parser.add_argument(
        "--intra-std-max", type=float, default=INTRA_STD_MAX_DEG,
        help="Std max intra-plateau (deg) pour considerer l'ambiguite levee"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    g = globals()
    g["TELEPORT_THRESH_M"] = args.teleport_thresh
    g["SWAY_MIN_STD_M"]    = args.sway_min_std
    g["KABSCH_RESID_MAX_M"] = args.kabsch_resid_max
    g["INTRA_STD_MAX_DEG"] = args.intra_std_max

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(args.corrected_dir, args.data_dir, args.group)
    print(f"[Etape 1.6] {len(sessions)} sessions trouvees.")

    all_verdicts = []

    for i, s in enumerate(sessions, 1):
        sid = s["session_id"]
        if args.verbose:
            print(f"\n[{i}/{len(sessions)}] {sid}")

        # Lecture
        df_eye = read_eye_raw(s["eye_path"])
        df_pos, time_col = read_pos(s["pos_path"])
        df_corr = read_corrected(s["corrected_path"])

        if df_eye.empty or df_pos.empty or not time_col:
            print(f"  SKIP {sid}: fichier source manquant")
            continue

        # Identification des plateaux
        plateaux = find_plateaux(df_pos, time_col)
        if args.verbose:
            print(f"  {len(plateaux)} plateaux identifies")

        # Kabsch par plateau
        plateau_results = []
        for pl in plateaux:
            res = estimate_theta_plateau(pl, df_pos, time_col, df_eye)
            if res is not None:
                plateau_results.append(res)

        # Verdict de session
        verdict = session_verdict(plateau_results)

        da = verdict["direction_assumption"]
        icon = {"translation_pure_confirmee": "OK", "yaw_recupere": "YAW", "indetermine": "??"}.get(da, "?")
        th = f"{verdict['theta_median_deg']:.2f}deg" if verdict["theta_median_deg"] is not None else "N/A"
        st = f"std={verdict['theta_std_deg']:.2f}" if verdict["theta_std_deg"] is not None else ""
        print(f"  {sid}: {icon} theta={th} {st} valid={verdict['n_plateaux_valid']}/{len(plateau_results)}")

        # Mise à jour CSV corrigé
        df_updated = apply_dir_correction(df_corr, verdict, plateau_results)
        df_updated.to_csv(s["corrected_path"], sep=";", decimal=".", index=False, float_format="%.6f")

        all_verdicts.append((s, verdict))

    # Rapport
    report_path = write_report(all_verdicts, args.out_dir)
    print(f"\n[Rapport] {report_path}")

    # Résumé console
    dists = [v["direction_assumption"] for _, v in all_verdicts]
    print(f"\n=== RESUME ===")
    print(f"  Sessions traitees : {len(all_verdicts)}")
    print(f"  translation_pure_confirmee : {dists.count('translation_pure_confirmee')}")
    print(f"  yaw_recupere               : {dists.count('yaw_recupere')}")
    print(f"  indetermine                : {dists.count('indetermine')}")
    print(f"\nColonnes ajoutees : DirCorrX/Y/Z, yaw_theta_applied_deg (mis a jour dans CSV existants)")
    print(f"Rappel : DirX/Y/Z brut conserve intact.")


if __name__ == "__main__":
    main()
