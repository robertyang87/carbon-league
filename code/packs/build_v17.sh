#!/bin/bash
# Rebuilds the v17 family of single-file packs (generated ~5.7 MB files, not in git) from this exact config.
# Policy b128v9 int8, rollout s64v8 f16. *_nc = uncapped clock twins for 60-process cloud exams (never shipped).
# Option defaults when absent: MASTER_NO_PLANTER off, LATE_STEP_MASTER off, MAX_KEY_LATE off, EH_* off.
cd "$(dirname "$0")/.."
W=../data/weights; export CNET2_QMODE=int8 CNET2_QMODE_ROLL=f16
BASE="'MODE': 'mcts', 'CLOCK': 'wall', 'ROOT_S': 10, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7), 'ID_PRIOR': 0.0"
CAP="'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None"          # shipped (ladder) clock caps
NC="'BUDGET': 99.0, 'HARD': 99.0, 'ABORT_DL': None, 'START_DL': None"          # exam twins
THREAT="'THREAT_FIRST': True, 'THREAT_MIN': 100.0"
EH="'EH_STEP': 270, 'EH_HEAVY': 300.0, 'EH_HEAVY_STEP': 285"                   # edge rule from 270; heavy (>=300) cars go home from 285
P() { python3 cnet2_search_pack.py exp/submission_cnet2_$1.py "{$2}" $W/cnet2_b128v9.npz $W/cnet2_s64v8.npz; }
# P v17      "$BASE, $CAP"                                   # on the ladder (md5 fc36b4ab... was built from older source)
P v17c_nc  "$BASE, $NC"                                    # control
P v17t_nc  "$BASE, $NC, $THREAT, $EH, 'EH_SHADOW': True"  # threat-first; go-home rule recorded only (same-yardstick ledger)
P v17te_nc "$BASE, $NC, $THREAT, $EH"                      # + go-home rule
P v17tb_nc "$BASE, $NC, $THREAT, $EH, 'EH_BASE_PICK': True" # + ambush-aware base choice
# v17 search with the dg2 DAgger policy (pure-net exam vs Master2s stand-in: +205 +- 80 over v9)
python3 cnet2_search_pack.py exp/submission_cnet2_v17d2_nc.py "{$BASE, $NC}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# v18 candidate (exam twin): dg2 policy + threat-first + go-home rule
python3 cnet2_search_pack.py exp/submission_cnet2_v18c_nc.py "{$BASE, $NC, $THREAT, $EH}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
