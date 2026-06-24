#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_group_subfolders.py

Tu donnes un dossier "groupe" (ex: .../bim066) contenant plusieurs sous-dossiers.
Dans CHAQUE sous-dossier, le script fusionne les CSV ayant la même fin de nom (kind),
ex: 2025-10-24_15-58-42_UsersPositions.csv + 2025-10-24_16-16-31_UsersPositions.csv

Règles:
- Ordre: plus vieux -> plus récent
- Continuité temporelle: on décale le temps du fichier i pour qu'il commence juste après fin(i-1)
- Colonne temps: Time OU Timestamp (casse acceptée)
- MARKER: supprimé sur tous sauf le plus récent (on conserve un seul MARKER final)
- END: supprimé sur tous sauf le plus récent (si présent)

Sortie:
- par défaut: merged_<kind>.csv dans le même dossier
- --inplace: écrase le fichier le plus récent (par kind) dans ce dossier
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


FNAME_RE = re.compile(
    r"^(?P<dt>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<kind>.+\.csv)$"
)

TIME_COL_CANDIDATES = ["Timestamp", "Time", "timestamp", "time"]


def parse_dt_from_name(name: str) -> Optional[datetime]:
    m = FNAME_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("dt"), "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


def kind_from_name(name: str) -> Optional[str]:
    m = FNAME_RE.match(name)
    return m.group("kind") if m else None


def find_time_col(df: pd.DataFrame) -> str:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Aucune colonne temps trouvée parmi {TIME_COL_CANDIDATES}.")


def to_float_timestamp(series: pd.Series) -> pd.Series:
    # "490,4797" -> 490.4797 ; "END" -> NaN
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def estimate_dt(ts_num: pd.Series) -> float:
    x = ts_num.dropna()
    if len(x) < 3:
        return 0.0
    diffs = x.diff().dropna()
    diffs = diffs[diffs > 0]
    return float(diffs.median()) if len(diffs) else 0.0


def fmt_decimal_comma(x: float, ndigits: int = 6) -> str:
    s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


def is_marker_row(df: pd.DataFrame) -> pd.Series:
    return df.astype(str).apply(lambda col: col.str.contains(r"\bMARKER\b", na=False)).any(axis=1)


def is_end_row(df: pd.DataFrame, time_col: str) -> pd.Series:
    return df[time_col].astype(str).str.fullmatch(r"END", na=False)


@dataclass
class CsvChunk:
    path: Path
    dt: datetime
    kind: str
    df: pd.DataFrame
    time_col: str


def read_semicolon_csv(path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, sep=";", engine="python")
    time_col = find_time_col(df)
    return df, time_col


def merge_chunks(chunks: List[CsvChunk]) -> pd.DataFrame:
    """
    chunks triés vieux -> récent, même kind, même dossier.
    - supprime MARKER et END sur tous sauf dernier
    - aligne le temps pour continuité
    """
    if len(chunks) < 2:
        return chunks[0].df

    merged_parts: List[pd.DataFrame] = []
    prev_end: Optional[float] = None

    for i, ch in enumerate(chunks):
        df = ch.df.copy()
        tcol = ch.time_col

        # On garde MARKER uniquement pour le plus vieux (i == 0)
        if i != 0:
            df = df.loc[~is_marker_row(df)].copy()
            df = df.loc[~is_end_row(df, tcol)].copy()

        ts_num = to_float_timestamp(df[tcol])

        if i == 0:
            # chunk initial inchangé
            if ts_num.dropna().shape[0]:
                prev_end = float(ts_num.dropna().iloc[-1])
            merged_parts.append(df)
            continue

        if prev_end is None or ts_num.dropna().shape[0] == 0:
            merged_parts.append(df)
            continue

        start = float(ts_num.dropna().iloc[0])
        dt_est = estimate_dt(ts_num)
        shift = (prev_end + dt_est) - start

        ts_shifted = ts_num.where(ts_num.isna(), ts_num + shift)

        # Réécriture de la colonne temps (on laisse les valeurs non-num (ex END) intactes)
        out_t = df[tcol].astype(str).copy()
        mask = ts_num.notna()
        out_t.loc[mask] = ts_shifted.loc[mask].map(lambda v: fmt_decimal_comma(float(v), 6))
        df[tcol] = out_t

        prev_end = float(ts_shifted.dropna().iloc[-1]) if ts_shifted.dropna().shape[0] else prev_end
        merged_parts.append(df)

    return pd.concat(merged_parts, ignore_index=True)


def merge_one_folder(folder: Path, inplace: bool, out_dir: Optional[Path]) -> List[Path]:
    """
    Dans un dossier donné, détecte les groupes de fichiers par kind et les fusionne.
    Retourne la liste des fichiers écrits.
    """
    csvs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]

    # Regrouper par kind dans CE dossier
    by_kind: Dict[str, List[Tuple[datetime, Path]]] = {}
    for p in csvs:
        k = kind_from_name(p.name)
        dt = parse_dt_from_name(p.name)
        if k and dt:
            by_kind.setdefault(k, []).append((dt, p))

    written: List[Path] = []

    for k, lst in by_kind.items():
        if len(lst) < 2:
            continue

        lst.sort(key=lambda t: t[0])  # vieux -> récent

        chunks: List[CsvChunk] = []
        for dt, p in lst:
            df, tcol = read_semicolon_csv(p)
            chunks.append(CsvChunk(path=p, dt=dt, kind=k, df=df, time_col=tcol))

        merged = merge_chunks(chunks)

        newest_path = chunks[-1].path
        if inplace:
            out_path = newest_path
        else:
            target_dir = out_dir if out_dir else folder
            out_path = target_dir / f"merged_{k}"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # on prend la colonne temps du plus récent (celle que tu veux conserver)
        tcol_out = chunks[-1].time_col

        # si certains chunks avaient "Time" et d'autres "Timestamp", on harmonise sur tcol_out
        for c in ["Time", "Timestamp", "time", "timestamp"]:
            if c in merged.columns and c != tcol_out and tcol_out not in merged.columns:
                merged = merged.rename(columns={c: tcol_out})

        write_csv_with_marker_compact(merged, out_path, time_col=tcol_out)
        written.append(out_path)

    return written

def marker_second_field(df: pd.DataFrame, time_col: str) -> str:
    # colonne où se trouve "MARKER" (autre que time_col)
    for c in df.columns:
        if c == time_col:
            continue
        if df[c].astype(str).str.contains(r"\bMARKER\b", na=False).any():
            return c
    # fallback: première colonne après time_col
    cols = [c for c in df.columns if c != time_col]
    return cols[0] if cols else time_col


def write_csv_with_marker_compact(df: pd.DataFrame, out_path: Path, time_col: str) -> None:
    """
    Écrit un CSV ';' où :
    - lignes MARKER => "time;MARKER" (2 champs)
    - lignes END (si time_col == END) => "END" (1 champ) ou conserve la ligne telle quelle si tu préfères
    - autres lignes => toutes les colonnes (nb champs complet)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Détecter la colonne qui porte MARKER (souvent la 2e)
    marker_col = marker_second_field(df, time_col)

    # Si header présent (pandas a lu des colonnes), on le garde
    header = ";".join(df.columns)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")

        for _, row in df.iterrows():
            # END ?
            if str(row.get(time_col, "")).strip() == "END":
                # si tu veux exactement "END" seul :
                f.write("END\n")
                continue

            # MARKER ?
            is_marker = False
            for c in df.columns:
                v = row.get(c, "")
                if isinstance(v, str) and "MARKER" in v:
                    is_marker = True
                    break

            if is_marker:
                t = "" if pd.isna(row.get(time_col)) else str(row.get(time_col))
                f.write(f"{t};MARKER\n")
                continue

            # Ligne normale: écrire toutes colonnes, vides si NaN
            vals = []
            for c in df.columns:
                v = row.get(c, "")
                if pd.isna(v):
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(";".join(vals) + "\n")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-root", required=True, help="Dossier du groupe (ex: .../bim066)")
    ap.add_argument("--inplace", action="store_true", help="Écrase le plus récent au lieu d'écrire merged_<kind>.csv")
    ap.add_argument("--out", default=None, help="Dossier de sortie (même arborescence). Si absent: écrit dans le dossier source.")
    ap.add_argument("--min-files", type=int, default=2, help="Fusionne seulement si >= min-files pour un kind.")
    args = ap.parse_args()

    group_root = Path(args.group_root).resolve()
    if not group_root.exists():
        raise SystemExit(f"[ERREUR] group-root introuvable: {group_root}")

    out_root = Path(args.out).resolve() if args.out else None

    # Parcours de tous les dossiers (y compris root)
    folders = [group_root] + [p for p in group_root.rglob("*") if p.is_dir()]

    total_written = 0
    for folder in folders:
        # Si --out est fourni, préserver l'arborescence relative
        out_dir = None
        if out_root:
            rel = folder.relative_to(group_root)
            out_dir = out_root / rel

        written = merge_one_folder(folder, inplace=args.inplace, out_dir=out_dir)

        # min-files: on filtre après coup si besoin (simple)
        # (si tu veux strict, on peut le faire avant, mais là on fusionne uniquement si >=2 déjà)
        if written:
            total_written += len(written)
            for w in written:
                print(f"[OK] {folder} -> {w}")

    print(f"[DONE] fichiers écrits: {total_written}")


if __name__ == "__main__":
    main()
