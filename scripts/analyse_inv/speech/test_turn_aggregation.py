#!/usr/bin/env python3
"""Test unitaire pour aggregate_ipus_to_ca_turns (MOD-1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def test_ca_turn_aggregation():
    from analyse_inv.speech.analyze_audio import aggregate_ipus_to_ca_turns

    # Cas 1 : 3 segments A consecutifs avec pauses internes, 1 B, 1 A final
    segs_by_role = {
        "A": [(0.0, 2.0), (3.0, 4.0), (5.5, 7.0), (10.0, 11.0)],
        "B": [(8.0, 9.0)],
    }
    turns = aggregate_ipus_to_ca_turns(segs_by_role, turn_min_s=1.0)
    assert len(turns) == 3, f"Expected 3 turns, got {len(turns)}: {turns}"
    assert turns[0]["role"] == "A" and turns[0]["start"] == 0.0 and turns[0]["end"] == 7.0, f"Turn 0 wrong: {turns[0]}"
    assert turns[1]["role"] == "B", f"Turn 1 wrong: {turns[1]}"
    assert turns[2]["role"] == "A" and turns[2]["start"] == 10.0, f"Turn 2 wrong: {turns[2]}"
    print("  Cas 1 PASS")

    # Cas 2 : filtre duree (turn_min=1.0)
    # A(0.0,2.0) et A(3.0,3.5) sont consecutifs -> fusionnes en A(0.0,3.5) par la logique CA
    # B(5.0,7.0) = tour separé
    # Le tour A(0.0,3.5) dure 3.5s >= 1.0 => conserve
    # Resultat attendu : 2 tours [A(0.0,3.5), B(5.0,7.0)]
    segs_by_role2 = {
        "A": [(0.0, 2.0), (3.0, 3.5)],
        "B": [(5.0, 7.0)],
    }
    turns2 = aggregate_ipus_to_ca_turns(segs_by_role2, turn_min_s=1.0)
    assert len(turns2) == 2, f"Expected 2 turns, got {len(turns2)}: {turns2}"
    assert turns2[0]["role"] == "A" and turns2[0]["start"] == 0.0 and turns2[0]["end"] == 3.5, (
        f"Expected A(0.0,3.5), got {turns2[0]}"
    )
    assert turns2[1]["role"] == "B", f"Expected B, got {turns2[1]}"
    print("  Cas 2 PASS")

    # Cas 3 : un seul role sans alternance
    segs_by_role3 = {
        "A": [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0)],
        "B": [],
    }
    turns3 = aggregate_ipus_to_ca_turns(segs_by_role3, turn_min_s=1.0)
    assert len(turns3) == 1, f"Expected 1 turn, got {len(turns3)}: {turns3}"
    assert turns3[0]["end"] == 8.0, f"Expected end=8.0, got {turns3[0]}"
    print("  Cas 3 PASS")

    # Cas 4 : bug passe 3 — micro-tour B s'intercale entre deux tours A longs,
    # B est filtre (< 1s) mais sans passe 3, A resterait en deux tours distincts.
    # Exemple reel : lect(21-22), calc(22.67-22.72, 0.05s), lect(22.73-24.95)
    segs_by_role4 = {
        "A": [(0.0, 5.0), (5.5, 10.0)],
        "B": [(2.0, 2.4)],  # micro-backchannel 0.4s, filtre
    }
    turns4 = aggregate_ipus_to_ca_turns(segs_by_role4, turn_min_s=1.0)
    # Sans passe 3 : [A(0-5), A(5.5-10)] car B filtre mais A deja divise
    # Avec passe 3 : [A(0-10)] fusionne
    assert len(turns4) == 1, f"Expected 1 turn after pass-3 re-merge, got {len(turns4)}: {turns4}"
    assert turns4[0]["role"] == "A"
    assert turns4[0]["start"] == 0.0 and turns4[0]["end"] == 10.0, f"Expected A(0,10), got {turns4[0]}"
    print("  Cas 4 PASS (passe 3 re-fusion apres filtrage)")

    # Cas 5 : micro-tour B interpose entre deux A, mais un VRAI tour C entre les deux A
    # => les deux A ne doivent PAS fusionner
    segs_by_role5 = {
        "A": [(0.0, 3.0), (6.0, 9.0)],
        "B": [(2.0, 2.3)],   # micro-tour filtre
        "C": [(3.5, 5.5)],   # vrai tour >= 1s qui separe les deux A
    }
    turns5 = aggregate_ipus_to_ca_turns(segs_by_role5, turn_min_s=1.0)
    assert len(turns5) == 3, f"Expected 3 turns, got {len(turns5)}: {turns5}"
    assert turns5[0]["role"] == "A" and turns5[1]["role"] == "C" and turns5[2]["role"] == "A", (
        f"Expected A/C/A, got {[t['role'] for t in turns5]}"
    )
    print("  Cas 5 PASS (vrai tour C empeche la re-fusion des deux A)")

    print("[OK] test_ca_turn_aggregation PASSED")

if __name__ == "__main__":
    test_ca_turn_aggregation()
