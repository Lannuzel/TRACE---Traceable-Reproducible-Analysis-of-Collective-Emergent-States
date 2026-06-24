from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}

# ---------------------------
# CONFIG: OpenFace executable
# ---------------------------
def which_openface_feature_extraction(explicit_path: Optional[str] = None) -> str:
    """
    Path vers FeatureExtraction (OpenFace 2.0).
    Sous Windows: FeatureExtraction.exe
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
        raise RuntimeError(f"OpenFace FeatureExtraction introuvable: {explicit_path}")

    # Essai via PATH
    exe = shutil.which("FeatureExtraction")
    if exe:
        return exe
    exe = shutil.which("FeatureExtraction.exe")
    if exe:
        return exe

    raise RuntimeError(
        "FeatureExtraction (OpenFace) introuvable. "
        "Passe --openface_exe C:\\path\\to\\FeatureExtraction.exe"
    )

def is_inside_raw(path: Path) -> bool:
    return any(part.lower() == "raw" for part in path.parts)

def iter_videos(root: Path) -> List[Path]:
    """
    Sélectionne toutes les vidéos hors raw/.
    Si tu veux ne traiter QUE celles déjà normalisées, tu peux filtrer sur 'processed_openface'.
    """
    vids = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        if is_inside_raw(f):
            continue
        vids.append(f)
    return sorted(vids)

def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Commande échouée:\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )

# ------------------------------------
# Mapping OpenFace -> noms "VR-like"
# ------------------------------------
# OpenFace: AU01_r, AU02_r, ... intensités
# Ton format: "InnerBrowRaiserL/R", "BrowLowererL/R", etc.
#
# IMPORTANT: OpenFace ne fournit pas "L/R" distinct pour la plupart des AUs (souvent une seule intensité).
# Pour produire L/R, on peut:
#   - dupliquer la valeur (même intensité pour L et R) si tu veux une structure identique
#   - ou laisser vide si tu veux être strict
#
# Ci-dessous: mapping minimal, à étendre si tu veux.
AU_TO_LABEL = {
    "AU01_r": "InnerBrowRaiser",   # AU1
    "AU02_r": "OuterBrowRaiser",   # AU2
    "AU04_r": "BrowLowerer",       # AU4
    "AU05_r": "UpperLidRaiser",    # AU5
    "AU06_r": "CheekRaiser",       # AU6
    "AU07_r": "LidTightener",      # AU7
    "AU09_r": "NoseWrinkler",      # AU9
    "AU10_r": "UpperLipRaiser",    # AU10
    "AU12_r": "LipCornerPuller",   # AU12
    "AU14_r": "Dimpler",           # AU14
    "AU15_r": "LipCornerDepressor",# AU15
    "AU17_r": "ChinRaiser",        # AU17
    "AU20_r": "LipStretcher",      # AU20
    "AU23_r": "LipTightener",      # AU23
    "AU24_r": "LipPressor",        # AU24
    "AU25_r": "LipsPart",          # AU25 (souvent "JawDrop/LipsPart" selon taxo)
    "AU26_r": "JawDrop",           # AU26
    "AU28_r": "LipSuck",           # AU28
    "AU45_r": "EyesClosed",        # AU45
}

def expand_lr_columns(base_label: str) -> List[str]:
    """
    Ton exemple attend L/R pour beaucoup de variables.
    Ici on génère <label>L et <label>R.
    """
    return [f"{base_label}L", f"{base_label}R"]

def build_target_columns() -> List[str]:
    """
    Construit la liste finale des colonnes au format demandé.
    On met Timestamp en premier, puis toutes les colonnes dérivées du mapping.
    """
    cols = ["Timestamp"]
    for _, base in AU_TO_LABEL.items():
        # Si le label a déjà une structure non latéralisée, on duplique L/R
        # (si tu veux être strict, remplace par [base] et adapte ton downstream)
        cols.extend(expand_lr_columns(base))
    return cols

TARGET_COLUMNS = build_target_columns()

def openface_extract(video_path: Path, openface_exe: str, out_dir: Path) -> Path:
    """
    Lance OpenFace FeatureExtraction.
    Retourne le chemin du CSV généré par OpenFace.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    # Nom attendu
    csv_path = out_dir / f"{video_path.stem}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 10_000:
        return csv_path
    # OpenFace écrit <video_name>.csv dans out_dir, sauf si on précise -of
    cmd = [
        openface_exe,
        "-f", str(video_path),
        "-out_dir", str(out_dir),
        "-aus",          # Action Units
        "-pose",         # head pose (optionnel)
        "-gaze",         # gaze (optionnel)
    ]
    run_cmd(cmd)


    if not csv_path.exists():
        # certains builds ajoutent .avi.csv etc. On cherche le CSV le plus récent
        candidates = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Aucun CSV OpenFace trouvé dans {out_dir}")
        csv_path = candidates[0]

    return csv_path

def convert_openface_to_target(openface_csv: Path, target_csv: Path) -> None:
    """
    Convertit le CSV OpenFace (virgule) en CSV format demandé:
    - séparateur ';'
    - décimales ','
    - colonnes: Timestamp + labels VR-like (L/R)
    """
    df = pd.read_csv(openface_csv, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    # OpenFace peut fournir 'timestamp' (minuscule) ou 'Timestamp' selon versions
    ts_col = "timestamp" if "timestamp" in df.columns else ("Timestamp" if "Timestamp" in df.columns else None)
    if ts_col is None:
        raise RuntimeError(f"Colonne timestamp introuvable dans {openface_csv.name}")

    out = pd.DataFrame()
    out["Timestamp"] = df[ts_col].astype(float)

    # Mapping AU -> label (duplication L/R)
    for au_col, base_label in AU_TO_LABEL.items():
        if au_col not in df.columns:
            # si l'AU n'existe pas, on crée des colonnes à 0 (ou NaN)
            out[f"{base_label}L"] = 0.0
            out[f"{base_label}R"] = 0.0
            continue

        vals = df[au_col].astype(float)
        out[f"{base_label}L"] = vals
        out[f"{base_label}R"] = vals

    # Réordonner
    out = out[TARGET_COLUMNS]

    # Écriture avec ; et virgule décimale
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    # On écrit en texte en remplaçant le point décimal par virgule
    # (sauf pour le séparateur ;)
    with target_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(out.columns)

        for row in out.itertuples(index=False):
            formatted = []
            for x in row:
                if isinstance(x, (float, int)):
                    # format stable, 6 décimales (ajuste si besoin)
                    s = f"{float(x):.6f}".replace(".", ",")
                    formatted.append(s)
                else:
                    formatted.append(str(x))
            writer.writerow(formatted)

def main():
    parser = argparse.ArgumentParser(
        description="Batch OpenFace (FeatureExtraction) + export FACS CSV au format VR-like."
    )
    parser.add_argument("root", type=str, help="Chemin racine (ex: D:\\data_e2)")
    parser.add_argument("--openface_exe", type=str, default=None,
                        help="Chemin vers FeatureExtraction.exe (OpenFace)")
    parser.add_argument("--only_processed_openface", action="store_true",
                        help="Ne traiter que les vidéos situées dans 'processed_openface/'")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Chemin racine introuvable: {root}")

    openface_exe = which_openface_feature_extraction(args.openface_exe)

    videos = iter_videos(root)
    if args.only_processed_openface:
        videos = [v for v in videos if "processed_openface" in (p.lower() for p in v.parts)]

    if not videos:
        print("Aucune vidéo à traiter (hors raw/).")
        return

    print(f"Vidéos à traiter: {len(videos)}")

    ok, failed = 0, 0

    for v in tqdm(videos, desc="OpenFace + FACS export", unit="vidéo"):
        try:
            # Dossiers de sortie
            # - openface_raw: sortie brute OpenFace
            # - facs: format final demandé
            out_root = v.parent  # à côté de la vidéo
            openface_dir = out_root / "openface_raw"
            facs_dir = out_root / "facs"

            # 1) OpenFace extraction
            openface_csv = openface_extract(v, openface_exe=openface_exe, out_dir=openface_dir)

            # 2) Conversion vers format cible
            target_csv = facs_dir / f"{v.stem}__FACS.csv"
            convert_openface_to_target(openface_csv, target_csv)

            ok += 1

        except Exception as e:
            failed += 1
            tqdm.write(f"[ERREUR] {v} -> {e}")

    print(f"\nTerminé. OK={ok}, ERREURS={failed}")
    print("Chaque vidéo a un CSV final dans un dossier 'facs/' à côté de la vidéo.")

if __name__ == "__main__":
    main()
