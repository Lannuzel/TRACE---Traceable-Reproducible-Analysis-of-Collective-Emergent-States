#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparatif avant/apres des variables modifiees par MOD-1 a MOD-9.

Utilise les colonnes `_old` (= ancien calcul) et les colonnes canoniques
(= nouveau calcul) presentes dans le meme CSV pour produire un rapport
lisible par groupe et en agregat.

Usage :
    python generate_comparison_report.py \
        --audio  results/INV/audio_features.csv \
        --audit  results/INV/high_level_features_audit.csv \
        --out    scripts/analyse_inv/comparison_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paires (nouveau, ancien) a comparer, avec label et MOD associe
# ---------------------------------------------------------------------------
AUDIO_PAIRS = [
    # (nouveau,                               ancien,                                     label,                                    mod)
    ("audio_avg_speaking_turn_duration_s",    "audio_avg_speaking_turn_duration_s_old",   "Duree moy. tour (s)",                    "MOD-1"),
    ("audio_total_speaking_turns",            "audio_total_speaking_turns_old",           "Nombre total de tours",                  "MOD-1"),
    ("audio_floor_exchange_pause_mean_s",     "audio_floor_exchange_pause_mean_s_old",    "Pause inter-tours moy. (s)",             "MOD-1"),
    ("n_floor_exchanges",                     "n_floor_exchanges_old",                    "N echanges de plancher",                 "MOD-1"),
    ("mean_turn_calculateur_s",               "mean_turn_calculateur_s_old",              "Duree moy. tour calculateur (s)",        "MOD-1"),
    ("mean_turn_modelisateur_s",              "mean_turn_modelisateur_s_old",             "Duree moy. tour modelisateur (s)",       "MOD-1"),
    ("mean_turn_lecteur_s",                   "mean_turn_lecteur_s_old",                  "Duree moy. tour lecteur (s)",            "MOD-1"),
    ("turns_calculateur_n",                   "turns_calculateur_n_old",                  "N tours calculateur",                    "MOD-1"),
    ("turns_modelisateur_n",                  "turns_modelisateur_n_old",                 "N tours modelisateur",                   "MOD-1"),
    ("turns_lecteur_n",                       "turns_lecteur_n_old",                      "N tours lecteur",                        "MOD-1"),
    ("overlap_ratio",                         "overlap_ratio_old",                        "Ratio overlap (occupancy vs direct)",    "MOD-2"),
    ("backchannels_calculateur_n",            "backchannels_calculateur_n_old",           "Backchannels calculateur (n)",           "MOD-3"),
    ("backchannels_modelisateur_n",           "backchannels_modelisateur_n_old",          "Backchannels modelisateur (n)",          "MOD-3"),
    ("backchannels_lecteur_n",                "backchannels_lecteur_n_old",               "Backchannels lecteur (n)",               "MOD-3"),
    ("backchannel_rate_per_min",              "audio_backchannel_rate_per_min_old",       "Taux backchannels (/min)",               "MOD-3"),
    ("audio_overlap_takeover_ratio",          "audio_successful_interruption_ratio_old",  "Overlap takeover ratio",                 "MOD-4"),
]

HLF_PAIRS = [
    ("gaze_attention_coordination_idx",       "gaze_attention_coordination_idx_old",      "Gaze coord. idx (poids entropy)",        "MOD-9"),
    ("face_sadness_marker_ratio",             "face_negative_affect_ratio_old",           "Sadness marker ratio",                   "MOD-6"),
]

NEW_COLS_MOD2 = [
    "audio_pct_time_0_speakers",
    "audio_pct_time_1_speaker",
    "audio_pct_time_2_speakers",
    "audio_pct_time_3_speakers",
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(int(v)) if float(v) == int(float(v)) else f"{v:.3f}"


def _stats(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    return float(s.mean()), float(s.std(ddof=1)), float(s.min()), float(s.max()), int(len(s))


def _sign(delta: float) -> str:
    if abs(delta) < 1e-6:
        return "="
    return "+" if delta > 0 else "-"


def build_report(audio_path: str, audit_path: str) -> str:
    lines: list[str] = []

    audio = pd.read_csv(audio_path) if audio_path and Path(audio_path).exists() else pd.DataFrame()
    audit = pd.read_csv(audit_path) if audit_path and Path(audit_path).exists() else pd.DataFrame()

    lines.append("# Comparatif avant/apres — variables modifiees MOD-1 a MOD-9\n")
    lines.append(f"Date : 2026-05-07  |  N groupes audio = {len(audio)}  |  N groupes HLF = {len(audit)}\n")
    lines.append(
        "> **Lecture** : colonne `nouveau` = calcul post-modification (valeur canonique actuelle). "
        "Colonne `ancien` = ancien calcul conserve sous suffixe `_old`. "
        "`delta_moy` = nouveau - ancien. `signe_attendu` = direction theorique attendue "
        "selon la spec de chaque MOD.\n"
    )

    # =========================================================================
    # SECTION 1 — Agregats (moyenne sur les 23 groupes)
    # =========================================================================
    lines.append("\n---\n\n## 1. Agregats sur les 23 groupes\n")
    lines.append("| MOD | Variable | moy_nouveau | sd_nouveau | moy_ancien | sd_ancien | delta_moy | signe_attendu | OK |")
    lines.append("|-----|----------|------------|-----------|-----------|----------|----------|--------------|-----|")

    SIGN_EXPECTED = {
        "MOD-1/duree_tour":     "+",   # tours CA plus longs (fusion micro-pauses)
        "MOD-1/n_tours":        "?",   # peut augmenter (micro-segs fusionnes) ou diminuer
        "MOD-1/pause_inter":    "-",   # pauses CA excluent les pauses intra-locuteur
        "MOD-1/n_echanges":     "?",   # depend du ratio intra/inter
        "MOD-2/overlap":        "~",   # recalcul depuis occupancy, ecart faible attendu
        "MOD-3/backchannels":   "-",   # filtres stricts reduisent le compte
        "MOD-4/takeover":       "=",   # renommage pur, valeur identique
        "MOD-9/gaze_coord":     "?",   # poids entropy change, sens depend des donnees
    }

    def _expected(mod, label):
        if mod == "MOD-1":
            if "tour" in label.lower() and "duree" in label.lower():
                return "+"
            if "n tours" in label.lower() or "nombre" in label.lower():
                return "?"
            if "pause" in label.lower():
                return "-"
            return "?"
        if mod == "MOD-2":
            return "~"
        if mod == "MOD-3":
            return "-"
        if mod == "MOD-4":
            return "="
        if mod == "MOD-6":
            return "="
        if mod == "MOD-9":
            return "?"
        return "?"

    for new_col, old_col, label, mod in AUDIO_PAIRS:
        if new_col not in audio.columns or old_col not in audio.columns:
            lines.append(f"| {mod} | {label} | N/A | N/A | N/A | N/A | N/A | N/A | ⚠ col absente |")
            continue
        mn, sn, _, _, nn = _stats(audio[new_col])
        mo, so, _, _, no = _stats(audio[old_col])
        delta = (mn - mo) if (not np.isnan(mn) and not np.isnan(mo)) else np.nan
        exp = _expected(mod, label)
        obs = _sign(delta) if not np.isnan(delta) else "?"
        ok = "OK" if (exp == "?" or exp == "~" or exp == obs or (exp == "=" and abs(delta) < 1e-4)) else "!!"
        lines.append(
            f"| {mod} | {label} | {_fmt(mn)} | {_fmt(sn)} | {_fmt(mo)} | {_fmt(so)} "
            f"| {_fmt(delta)} | {exp} | {ok} |"
        )

    for new_col, old_col, label, mod in HLF_PAIRS:
        if new_col not in audit.columns or old_col not in audit.columns:
            lines.append(f"| {mod} | {label} | N/A | N/A | N/A | N/A | N/A | N/A | ⚠ col absente |")
            continue
        mn, sn, _, _, _ = _stats(audit[new_col])
        mo, so, _, _, _ = _stats(audit[old_col])
        delta = (mn - mo) if (not np.isnan(mn) and not np.isnan(mo)) else np.nan
        exp = _expected(mod, label)
        obs = _sign(delta) if not np.isnan(delta) else "?"
        ok = "OK" if (exp == "?" or exp == "~" or exp == obs) else "!!"
        lines.append(
            f"| {mod} | {label} | {_fmt(mn)} | {_fmt(sn)} | {_fmt(mo)} | {_fmt(so)} "
            f"| {_fmt(delta)} | {exp} | {ok} |"
        )

    # Nouvelles colonnes MOD-2 (occupancy — pas de _old)
    lines.append("\n### Nouvelles colonnes MOD-2 (occupancy — pas d'equivalent avant)\n")
    lines.append("| variable | moy | sd | min | max | N |")
    lines.append("|----------|-----|----|-----|-----|---|")
    for col in NEW_COLS_MOD2:
        if col in audio.columns:
            mn, sn, mi, ma, nn = _stats(audio[col])
            lines.append(f"| {col} | {_fmt(mn)} | {_fmt(sn)} | {_fmt(mi)} | {_fmt(ma)} | {nn} |")

    # =========================================================================
    # SECTION 2 — Tableau par groupe (variables clés)
    # =========================================================================
    lines.append("\n---\n\n## 2. Tableau par groupe — variables cles\n")

    key_pairs = [
        ("audio_avg_speaking_turn_duration_s", "audio_avg_speaking_turn_duration_s_old", "tour_s",   "MOD-1"),
        ("audio_total_speaking_turns",         "audio_total_speaking_turns_old",         "n_tours",  "MOD-1"),
        ("backchannel_rate_per_min",           "audio_backchannel_rate_per_min_old",     "bc/min",   "MOD-3"),
        ("overlap_ratio",                      "overlap_ratio_old",                      "overlap",  "MOD-2"),
    ]

    # Header
    header_parts = ["group_id"]
    for _, _, lbl, mod in key_pairs:
        header_parts += [f"{lbl}_new ({mod})", f"{lbl}_old", f"delta"]
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

    if not audio.empty and "group_id" in audio.columns:
        for _, row in audio.sort_values("group_id").iterrows():
            cells = [str(row["group_id"])]
            for new_col, old_col, lbl, mod in key_pairs:
                nv = pd.to_numeric(row.get(new_col, np.nan), errors="coerce")
                ov = pd.to_numeric(row.get(old_col, np.nan), errors="coerce")
                delta = nv - ov if (pd.notna(nv) and pd.notna(ov)) else np.nan
                cells += [_fmt(nv), _fmt(ov), _fmt(delta)]
            lines.append("| " + " | ".join(cells) + " |")

    # =========================================================================
    # SECTION 3 — Backchannels avant/apres par groupe (MOD-3)
    # =========================================================================
    lines.append("\n---\n\n## 3. Backchannels par groupe (MOD-3 : filtre strict vs laxiste)\n")
    lines.append("| group_id | bc_new/min | bc_old/min | delta | dans_[2,25]_new | dans_[2,25]_old |")
    lines.append("|----------|-----------|-----------|-------|-----------------|-----------------|")

    if not audio.empty and "backchannel_rate_per_min" in audio.columns:
        bc_new = audio[["group_id", "backchannel_rate_per_min"]].copy()
        bc_new["bc_new"] = pd.to_numeric(bc_new["backchannel_rate_per_min"], errors="coerce")
        if "audio_backchannel_rate_per_min_old" in audio.columns:
            bc_new["bc_old"] = pd.to_numeric(audio["audio_backchannel_rate_per_min_old"], errors="coerce")
        else:
            bc_new["bc_old"] = np.nan
        for _, r in bc_new.sort_values("group_id").iterrows():
            nv, ov = r["bc_new"], r["bc_old"]
            delta = nv - ov if (pd.notna(nv) and pd.notna(ov)) else np.nan
            rng_n = ("OK" if 2 <= nv <= 25 else ("!! > 25" if nv > 25 else "!! < 2")) if pd.notna(nv) else "N/A"
            rng_o = ("OK" if 2 <= ov <= 25 else ("!! > 25" if ov > 25 else "!! < 2")) if pd.notna(ov) else "N/A"
            lines.append(
                f"| {r['group_id']} | {_fmt(nv)} | {_fmt(ov)} | {_fmt(delta)} | {rng_n} | {rng_o} |"
            )

    # =========================================================================
    # SECTION 4 — Assertions de coherence
    # =========================================================================
    lines.append("\n---\n\n## 4. Assertions de coherence\n")
    lines.append("| ID | Description | Resultat | Detail |")
    lines.append("|----|-------------|---------|--------|")

    assertions = []

    # A1 : mean_turn_s_new >= mean_turn_s_old pour chaque groupe
    if "audio_avg_speaking_turn_duration_s" in audio.columns and "audio_avg_speaking_turn_duration_s_old" in audio.columns:
        nv = pd.to_numeric(audio["audio_avg_speaking_turn_duration_s"], errors="coerce")
        ov = pd.to_numeric(audio["audio_avg_speaking_turn_duration_s_old"], errors="coerce")
        mask = nv.notna() & ov.notna()
        violations = int((nv[mask] < ov[mask] - 1e-4).sum())
        status = "PASS" if violations == 0 else f"FAIL ({violations} groupes)"
        assertions.append(("A1", "mean_turn_s_CA >= mean_turn_s_IPU (MOD-1)", status,
                            f"moy_CA={nv.mean():.3f}s vs moy_IPU={ov.mean():.3f}s"))

    # A2 : floor_exchange_pause_NEW <= floor_exchange_pause_OLD en moyenne
    if "audio_floor_exchange_pause_mean_s" in audio.columns and "audio_floor_exchange_pause_mean_s_old" in audio.columns:
        nv = pd.to_numeric(audio["audio_floor_exchange_pause_mean_s"], errors="coerce")
        ov = pd.to_numeric(audio["audio_floor_exchange_pause_mean_s_old"], errors="coerce")
        violations = int((nv.dropna() > ov.dropna() + 1e-4).sum())
        status = "PASS" if violations == 0 else f"FAIL ({violations}/{len(nv.dropna())} groupes)"
        assertions.append(("A2", "floor_exchange_pause_NEW <= OLD (MOD-1)", status,
                            f"moy_NEW={nv.mean():.3f}s vs moy_OLD={ov.mean():.3f}s"))

    # A3 : sum(pct_time_*) = 1.0 pour chaque groupe (MOD-2)
    pct_cols = [c for c in audio.columns if c.startswith("audio_pct_time_") and "_speakers" in c or c == "audio_pct_time_1_speaker"]
    pct_cols = [c for c in audio.columns if "audio_pct_time_" in c and ("_speakers" in c or "_speaker" in c)]
    if len(pct_cols) == 4:
        sums = audio[pct_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        violations = int(((sums - 1.0).abs() >= 1e-3).sum())
        status = "PASS" if violations == 0 else f"FAIL ({violations} groupes)"
        assertions.append(("A3", "sum(pct_time_*) = 1.0 +/- 1e-3 (MOD-2)", status,
                            f"max ecart = {(sums - 1.0).abs().max():.6f}"))
    else:
        assertions.append(("A3", "sum(pct_time_*) = 1.0 (MOD-2)", "N/A",
                            f"Colonnes trouvees: {pct_cols}"))

    # A4 : bc_rate_new <= bc_rate_old pour chaque groupe (MOD-3)
    if "backchannel_rate_per_min" in audio.columns and "audio_backchannel_rate_per_min_old" in audio.columns:
        nv = pd.to_numeric(audio["backchannel_rate_per_min"], errors="coerce")
        ov = pd.to_numeric(audio["audio_backchannel_rate_per_min_old"], errors="coerce")
        mask = nv.notna() & ov.notna()
        violations = int((nv[mask] > ov[mask] + 1e-4).sum())
        pct_ok = ((nv >= 2) & (nv <= 25)).mean() * 100
        status = "PASS" if violations == 0 else f"FAIL ({violations} groupes)"
        assertions.append(("A4", "bc_rate_NEW <= bc_rate_OLD (MOD-3 filtre strict)", status,
                            f"moy_NEW={nv.mean():.1f}/min vs moy_OLD={ov.mean():.1f}/min | {pct_ok:.0f}% dans [2,25]"))

    # A5 : overlap_takeover_ratio = successful_interruption_ratio_old (MOD-4, renommage pur)
    if "audio_overlap_takeover_ratio" in audio.columns and "audio_successful_interruption_ratio_old" in audio.columns:
        nv = pd.to_numeric(audio["audio_overlap_takeover_ratio"], errors="coerce")
        ov = pd.to_numeric(audio["audio_successful_interruption_ratio_old"], errors="coerce")
        max_diff = (nv - ov).abs().max()
        status = "PASS" if max_diff < 1e-6 else f"FAIL (max_diff={max_diff:.2e})"
        assertions.append(("A5", "audio_overlap_takeover_ratio == old (MOD-4 renommage pur)", status,
                            f"max diff = {max_diff:.2e}"))

    for aid, desc, res, det in assertions:
        icon = "OK" if "PASS" in res else ("N/A" if "N/A" in res else "!!")
        lines.append(f"| {aid} | {desc} | {icon} {res} | {det} |")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio",  default="results/INV/audio_features.csv")
    ap.add_argument("--audit",  default="results/INV/high_level_features_audit.csv")
    ap.add_argument("--out",    default="scripts/analyse_inv/comparison_report.md")
    args = ap.parse_args()

    report = build_report(args.audio, args.audit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[OK] Rapport ecrit : {out}")


if __name__ == "__main__":
    main()
