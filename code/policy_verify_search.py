"""Verification-only agent: SEARCH-enabled wrapper for the last_stats fix.

Pack CONF (13 keys) + m3s2r1 weights. Used by the war-room-ordered
20-game verification field (bigeval.py policy_verify_search.py
exp/echo2.py 993000 20 verify_search): confirms summaries carry
search_games>0 and a_stats populated after the fix. Not a submission
agent, not in any training pool.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CONF = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True,
        'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2,
        'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95,
        'START_DL': None}


def make_agent():
    import cnet2_search as S
    w = "/home/arm/junyang/s26stage/data/weights/cnet2_m3s2r1L.npz"
    return S.make_search_agent(CONF, weights=w)
