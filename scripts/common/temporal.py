"""
temporal.py — Time-series and temporal analysis utilities.

Sliding windows, episode detection, segment overlap computations.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def sliding_windows(t0: float, t1: float, win: float, step: float):
    """Yield (start, end) tuples for sliding windows over [t0, t1)."""
    s = t0
    while s < t1:
        e = min(s + win, t1)
        yield s, e
        s += step


def episodes_from_bool(mask: np.ndarray, fs: float) -> Tuple[int, float, float]:
    """
    Detect contiguous True episodes in a boolean array sampled at *fs* Hz.

    Returns:
        (n_episodes, total_duration_s, mean_duration_s)
    """
    if mask.size == 0 or not np.any(mask):
        return 0, 0.0, np.nan

    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    on_ix = np.where(edges == 1)[0]
    off_ix = np.where(edges == -1)[0]
    durs = (off_ix - on_ix) / fs
    n = int(len(durs))
    tot = float(np.sum(durs)) if n else 0.0
    mean = float(np.mean(durs)) if n else np.nan
    return n, tot, mean


def segment_duration(segs: List[Tuple[float, float]]) -> float:
    """Total duration of a list of (start, end) segments."""
    return sum(e - s for s, e in segs)


def segment_overlap_len(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Overlap duration between two segments."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)


def pairwise_overlap(
    a: List[Tuple[float, float]], b: List[Tuple[float, float]]
) -> float:
    """Total overlap between two sorted segment lists (sweep-line)."""
    i = j = 0
    ovl = 0.0
    while i < len(a) and j < len(b):
        ovl += segment_overlap_len(a[i], b[j])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return ovl


def union_duration(list_of_lists: List[List[Tuple[float, float]]]) -> float:
    """Duration of the union of multiple segment lists."""
    pts = sorted(
        [seg for lst in list_of_lists for seg in lst], key=lambda x: x[0]
    )
    if not pts:
        return 0.0
    merged = [list(pts[0])]
    for s, e in pts[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged)


def silence_intervals(
    segs_by_role: Dict[str, List[Tuple[float, float]]], total: float
) -> List[Tuple[float, float]]:
    """Intervals where nobody speaks (group pauses)."""
    all_segs = sorted(
        [seg for lst in segs_by_role.values() for seg in lst],
        key=lambda x: x[0],
    )
    if not all_segs:
        return [(0.0, total)]
    merged = [list(all_segs[0])]
    for s, e in all_segs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    sil = []
    prev = 0.0
    for s, e in merged:
        if s > prev:
            sil.append((prev, s))
        prev = e
    if prev < total:
        sil.append((prev, total))
    return sil


def compute_floor_exchanges(
    segs_by_role: Dict[str, List[Tuple[float, float]]],
    max_gap: float = 2.0,
    min_turn_sec: float = 1.0,
) -> List[float]:
    """
    Compute transition pauses between speakers.

    Returns a list of gap durations (s) for valid floor exchanges.
    """
    events = []
    for role, segs in segs_by_role.items():
        for s, e in segs:
            if (e - s) >= min_turn_sec:
                events.append((s, e, role))

    events.sort(key=lambda x: x[0])

    gaps = []
    for i in range(len(events) - 1):
        s1, e1, r1 = events[i]
        s2, e2, r2 = events[i + 1]

        if r1 == r2:
            continue

        gap = s2 - e1
        if 0 <= gap <= max_gap:
            gaps.append(gap)

    return gaps


def count_interruptions(
    segs_by_role: Dict[str, List[Tuple[float, float]]],
    roles: List[str],
    min_overlap: float = 0.1,
    min_post_takeover: float = 0.5,
    min_turn_sec: float = 1.0,
) -> Tuple[int, int]:
    """
    Count interruption attempts and successes across roles.

    Returns:
        (attempts, successes)
    """
    attempts = 0
    successes = 0

    filtered = {
        role: [(s, e) for s, e in segs if (e - s) >= min_turn_sec]
        for role, segs in segs_by_role.items()
    }

    for a_role in roles:
        for b_role in roles:
            if a_role == b_role:
                continue
            for a_s, a_e in filtered[a_role]:
                for b_s, b_e in filtered[b_role]:
                    if a_s < b_s < a_e:
                        ovl = min(a_e, b_e) - b_s
                        if ovl >= min_overlap:
                            attempts += 1
                            if b_e - a_e >= min_post_takeover:
                                successes += 1

    return attempts, successes
