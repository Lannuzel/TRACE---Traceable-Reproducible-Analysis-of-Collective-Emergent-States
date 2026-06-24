# revit_to_pc_like_inplace.py
# ------------------------------------------------------------
# Objectif
# - Parcourt un dossier racine (ex: D:\data_e2) et cherche, dans chaque dossier bimXXX,
#   les exports Revit (CSV avec entête: ElementId;Type;X_m;Y_m;Z_m;ElementName).
# - Pour chaque fichier détecté:
#     1) Renomme l’original en "<nom>_old.csv"
#     2) Écrit un nouveau fichier avec le nom initial (même chemin),
#        au format "PC-like" compatible avec tes scripts:
#           ElementId;Type;PositionX;PositionY;PositionZ;ElementName;Duration(s)
#        avec:
#           PositionX <- X_m
#           PositionZ <- Y_m   (plan Revit XY mappé sur Unity XZ)
#           PositionY <- Z_m   (altitude conservée pour info)
#        et filtre par défaut: ElementName == "Access" (chaises)
# - Injecte la durée (Duration(s)) si (Scenario, Groupe) est présent dans DURATIONS.
#
# Hypothèses
# - L’arborescence contient ...\PC\S1\bim067\... ou ...\PC\S2\bim074\...
# - Les fichiers Revit à transformer ont tous le même nom par dossier bimXXX
#   (mais on n’en a pas besoin: on détecte par l’en-tête).
#
# Usage (PowerShell):
#   python .\revit_to_pc_like_inplace.py --root "D:\data_e2" --dry-run
#   python .\revit_to_pc_like_inplace.py --root "D:\data_e2"
#
# Options utiles:
#   --pattern "*.csv"               (défaut)
#   --chair-name "Access"           (défaut)
#   --keep-all                      (ne filtre pas ElementName)
#   --non-strict                    (contains au lieu de ==)
#   --overwrite-old                 (si <name>_old.csv existe déjà, on le remplace)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ---- Durées connues (HH:MM:SS) -> injectées en secondes dans "Duration(s)" ----
DURATIONS = {
    ("S1", "bim067"): "00:25:00",
    ("S1", "bim007"): "00:24:11",
    ("S1", "bim009"): "00:15:00",
    ("S1", "bim025"): "00:15:37",
    ("S2", "bim057"): "00:25:03",
    ("S2", "bim074"): "00:25:01",
    ("S2", "bim077"): "00:25:01",
}

REQUIRED_REVIT_COLS = ["ElementId", "Type", "X_m", "Y_m", "Z_m", "ElementName"]


def hms_to_seconds(hms: str) -> int:
    m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", hms.strip())
    if not m:
        raise ValueError(f"Format durée invalide: {hms} (attendu HH:MM:SS)")
    hh, mm, ss = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss


DURATIONS_S = {(s, g): hms_to_seconds(v) for (s, g), v in DURATIONS.items()}


def infer_scenario_and_group(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Infère (S1/S2/...) et (bimXXX) depuis le chemin du fichier.
    Ex: ...\\PC\\S2\\bim074\\raw\\xxx.csv -> ("S2","bim074")
    """
    p = str(path).replace("\\", "/")
    scen = None
    grp = None

    ms = re.search(r"/(S\d)\b", p, flags=re.IGNORECASE)
    if ms:
        scen = ms.group(1).upper()

    mg = re.search(r"\b(bim\d{3})\b", p, flags=re.IGNORECASE)
    if mg:
        grp = mg.group(1).lower()

    return scen, grp


def looks_like_revit_export(csv_path: Path) -> bool:
    """
    Détecte un export Revit par son en-tête (sans lire tout le fichier).
    """
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
            header = f.readline().strip()
    except Exception:
        return False

    # Tolère espaces
    cols = [c.strip() for c in header.split(";")]
    return cols[: len(REQUIRED_REVIT_COLS)] == REQUIRED_REVIT_COLS


def read_revit(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_REVIT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")
    return df


def filter_elements(df: pd.DataFrame, chair_name: str, keep_all: bool, strict: bool) -> pd.DataFrame:
    if keep_all:
        return df.copy()
    s = df["ElementName"].astype(str).str.strip()
    keep = s.eq(chair_name) if strict else s.str.contains(chair_name, case=False, na=False)
    return df.loc[keep].copy()


def convert_to_pc_like(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ElementId": df["ElementId"].astype("Int64", errors="ignore"),
            "Type": df["Type"].astype(str),
            "PositionX": pd.to_numeric(df["X_m"], errors="coerce"),
            "PositionY": pd.to_numeric(df["Z_m"], errors="coerce"),
            "PositionZ": pd.to_numeric(df["Y_m"], errors="coerce"),
            "ElementName": df["ElementName"].astype(str).str.strip(),
        }
    )
    out = out.dropna(subset=["PositionX", "PositionZ"]).reset_index(drop=True)
    return out


def make_old_name(path: Path) -> Path:
    # bim074.csv -> bim074_old.csv
    return path.with_name(f"{path.stem}_old{path.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Transforme exports Revit en CSV PC-like in-place (rename *_old).")
    ap.add_argument("--root", required=True, help="Dossier racine (ex: D:\\data_e2).")
    ap.add_argument("--pattern", default="*.csv", help="Pattern des fichiers à scanner (défaut: *.csv).")
    ap.add_argument("--chair-name", default="Access", help='Nom d’élément à conserver (défaut: "Access").')
    ap.add_argument("--keep-all", action="store_true", help="Ne filtre pas ElementName (convertit tout).")
    ap.add_argument("--non-strict", action="store_true", help="Filtre ElementName par contains au lieu de ==.")
    ap.add_argument("--dry-run", action="store_true", help="N’écrit rien, affiche seulement les actions.")
    ap.add_argument(
        "--overwrite-old",
        action="store_true",
        help="Si un fichier *_old existe déjà, le remplace (sinon on skip).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERREUR] root introuvable: {root}")
        return 2

    strict = not args.non_strict

    candidates = list(root.rglob(args.pattern))
    if not candidates:
        print(f"[WARN] Aucun fichier trouvé sous {root} avec pattern {args.pattern}")
        return 0

    n_found = 0
    n_done = 0
    n_skipped = 0
    n_errors = 0

    for csv_path in candidates:
        # On évite de retraiter les *_old
        if csv_path.stem.endswith("_old"):
            continue

        if not looks_like_revit_export(csv_path):
            continue

        n_found += 1
        scen, grp = infer_scenario_and_group(csv_path)
        dur = DURATIONS_S.get((scen, grp)) if scen and grp else None

        old_path = make_old_name(csv_path)

        # Gestion old existant
        if old_path.exists() and not args.overwrite_old:
            print(f"[SKIP] {csv_path} -> old existe déjà: {old_path} (utilise --overwrite-old pour forcer)")
            n_skipped += 1
            continue

        try:
            df = read_revit(csv_path)
            df = filter_elements(df, chair_name=args.chair_name, keep_all=args.keep_all, strict=strict)

            if df.empty:
                print(f"[SKIP] {csv_path} -> aucun élément après filtre (chair-name={args.chair_name})")
                n_skipped += 1
                continue

            out = convert_to_pc_like(df)
            if dur is not None:
                out["Duration(s)"] = dur
            else:
                out["Duration(s)"] = pd.NA  # explicite: non disponible

            if args.dry_run:
                print(f"[DRY] Revit détecté: {csv_path}")
                print(f"      rename -> {old_path.name}")
                print(f"      write  -> {csv_path.name} (rows={len(out)}, scen={scen}, group={grp}, Duration(s)={dur})")
            else:
                # Renommer original -> *_old (en remplaçant si demandé)
                if old_path.exists() and args.overwrite_old:
                    old_path.unlink()
                shutil.move(str(csv_path), str(old_path))

                # Écrire nouveau fichier au nom original
                out.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")

                print(f"[OK] {csv_path} (rows={len(out)}) | old={old_path.name} | Duration(s)={dur}")
            n_done += 1

        except Exception as e:
            print(f"[ERREUR] {csv_path}: {type(e).__name__}: {e}")
            n_errors += 1

    print("\n---- Résumé ----")
    print(f"Revit détectés : {n_found}")
    print(f"Transformés    : {n_done}")
    print(f"Skippés        : {n_skipped}")
    print(f"Erreurs        : {n_errors}")

    return 1 if n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
