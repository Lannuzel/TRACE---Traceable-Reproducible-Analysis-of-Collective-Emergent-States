from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple

from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}

def which_ffmpeg(explicit_path: Optional[str] = None) -> str:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return str(p)
        raise RuntimeError(f"ffmpeg introuvable à l'emplacement fourni: {explicit_path}")
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    raise RuntimeError(
        "ffmpeg introuvable dans le PATH. "
        "Installe ffmpeg ou passe --ffmpeg_path C:\\path\\to\\ffmpeg.exe"
    )

def is_inside_raw(path: Path) -> bool:
    """
    True si le fichier est dans un dossier nommé exactement 'raw' (insensible à la casse),
    à n'importe quel niveau de l'arborescence.
    """
    return any(part.lower() == "raw" for part in path.parts)

def iter_videos_excluding_raw(root: Path) -> Iterable[Path]:
    """
    Parcourt root et renvoie tous les fichiers vidéo hors dossiers raw/.
    """
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        if is_inside_raw(f):
            continue
        yield f

def stable_id(path: Path, root: Path) -> str:
    """
    ID stable basé sur le chemin relatif, pour éviter les collisions de noms.
    """
    rel = str(path.relative_to(root)).replace("\\", "/")
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]

def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Commande ffmpeg échouée:\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )

def normalize_video_for_openface(
    ffmpeg: str,
    in_video: Path,
    out_video: Path,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    crf: int = 20,
    preset: str = "veryfast",
    overwrite: bool = False,
) -> None:
    """
    Normalisation OpenFace-friendly:
    - CFR fps
    - scale+pad en 1280x720 (sans déformer) + letterbox
    - H.264 + yuv420p
    - audio AAC conservé (mais on extrait aussi un WAV à part)
    """
    # Note: force_original_aspect_ratio=decrease + pad centre = robuste
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        ffmpeg, "-hide_banner",
        "-y" if overwrite else "-n",
        "-i", str(in_video),

        # CFR 30 fps
        "-r", str(fps),

        # Video filters
        "-vf", vf,

        # Video codec settings
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", preset,
        "-crf", str(crf),

        # Audio: garder une piste standard (AAC) pour compat
        "-c:a", "aac",
        "-b:a", "192k",

        str(out_video)
    ]
    run_cmd(cmd)

def extract_audio_wav(
    ffmpeg: str,
    in_video: Path,
    out_wav: Path,
    sr: int = 44100,
    mono: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Extrait l'audio en PCM 16-bit, utile clap / analyses.
    """
    cmd = [
        ffmpeg, "-hide_banner",
        "-y" if overwrite else "-n",
        "-i", str(in_video),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_wav)
    ]
    run_cmd(cmd)

def process_one_video(
    ffmpeg: str,
    root: Path,
    video_path: Path,
    fps: int,
    width: int,
    height: int,
    crf: int,
    overwrite: bool,
) -> Tuple[Path, Path]:
    """
    Crée:
    - <stem>__<id>__openface.mp4
    - <stem>__<id>__audio.wav
    dans processed_openface/
    """
    out_dir = video_path.parent / "processed_openface"
    out_dir.mkdir(exist_ok=True)

    sid = stable_id(video_path, root)

    out_video = out_dir / f"{video_path.stem}__{sid}__openface.mp4"
    out_wav = out_dir / f"{video_path.stem}__{sid}__audio.wav"

    # Skip si déjà fait (sauf overwrite)
    if (not overwrite) and out_video.exists() and out_wav.exists():
        return out_video, out_wav

    normalize_video_for_openface(
        ffmpeg=ffmpeg,
        in_video=video_path,
        out_video=out_video,
        fps=fps,
        width=width,
        height=height,
        crf=crf,
        overwrite=overwrite,
    )

    extract_audio_wav(
        ffmpeg=ffmpeg,
        in_video=video_path,
        out_wav=out_wav,
        overwrite=overwrite,
    )

    return out_video, out_wav

def main():
    parser = argparse.ArgumentParser(
        description="Prépare les vidéos (hors dossiers raw/) pour OpenFace + extraction audio WAV."
    )
    parser.add_argument("root", type=str, help="Chemin racine (ex: D:\\data_e2)")
    parser.add_argument("--ffmpeg_path", type=str, default=None, help="Chemin vers ffmpeg.exe si non dans PATH")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--crf", type=int, default=20, help="Qualité x264 (18-22 typiquement)")
    parser.add_argument("--overwrite", action="store_true", help="Écrase les sorties existantes")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Chemin racine introuvable: {root}")

    ffmpeg = which_ffmpeg(args.ffmpeg_path)

    videos = sorted(iter_videos_excluding_raw(root))
    if not videos:
        print(f"Aucune vidéo trouvée hors dossiers raw/ sous: {root}")
        return

    print(f"Vidéos détectées (hors raw/): {len(videos)}")

    ok = 0
    failed = 0

    for v in tqdm(videos, desc="Préparation OpenFace", unit="vidéo"):
        try:
            out_v, out_a = process_one_video(
                ffmpeg=ffmpeg,
                root=root,
                video_path=v,
                fps=args.fps,
                width=args.width,
                height=args.height,
                crf=args.crf,
                overwrite=args.overwrite,
            )
            ok += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"[ERREUR] {v} -> {e}")

    print(f"\nTerminé. OK={ok}, ERREURS={failed}")
    print("Les sorties sont dans des dossiers 'processed_openface' à côté des vidéos sources.")

if __name__ == "__main__":
    main()
