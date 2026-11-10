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
# shipped (ladder) build of v17d2: same CONF as v17, dg2 policy
python3 cnet2_search_pack.py exp/submission_cnet2_v17d2.py "{$BASE, $CAP}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# v18e (exam twin): dg2 policy + go-home rule only (no THREAT_FIRST: v18c showed it costs the hunting slot)
python3 cnet2_search_pack.py exp/submission_cnet2_v18e_nc.py "{$BASE, $NC, $EH}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# v18cc (exam twin): dg2 policy + car-curve base rule (Master2s collector curve: 3rd..8th car at steps 13/22/32/39/48/55)
python3 cnet2_search_pack.py exp/submission_cnet2_v18cc_nc.py "{$BASE, $NC, 'CAR_CURVE': (13, 22, 32, 39, 48, 55)}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# endgame search leaf (exam twins on the v17d2 policy). Threat = enemy collector with cargo <= ours at torus distance <= 1.
# v18lr: road threat only, last 30 steps; v18lb: + base-mouth ambush (car <= 5 from its base); v18lb100: v18lb from step 200;
# v18lk: v18lb + key slot shared (even steps) with loaded cars whose way home is tight. One discount (0.5), never multiplied.
LTR="'LEAF_THREAT': 0.5, 'LT_LEFT': 30"
python3 cnet2_search_pack.py exp/submission_cnet2_v18lr_nc.py "{$BASE, $NC, $LTR}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v18lb_nc.py "{$BASE, $NC, $LTR, 'LT_BASE': True}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v18lb100_nc.py "{$BASE, $NC, 'LEAF_THREAT': 0.5, 'LT_LEFT': 100, 'LT_BASE': True}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v18lk_nc.py "{$BASE, $NC, $LTR, 'LT_BASE': True, 'KEY_LATE_HOME': True, 'KLH_STEP': 270, 'KLH_MIN': 100.0, 'KLH_SLACK': 4, 'KLH_SHARE': 1}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# v19 opening search (2026-11-09): planters as key units in steps <= 100 + carbon-based tree leaf value
PK="'PLANTER_KEY': True, 'PK_STEP': 100, 'PK_MAX': 1, 'PK_R': 2"; TC="'TREE_C': 0.0375, 'TC_ONCE': True"
python3 cnet2_search_pack.py exp/submission_cnet2_v19br_nc.py "{$BASE, $NC, 'MAX_BR': 9}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz            # control: more branches only
python3 cnet2_search_pack.py exp/submission_cnet2_v19tc_nc.py "{$BASE, $NC, $TC}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz                     # leaf only
python3 cnet2_search_pack.py exp/submission_cnet2_v19pk_nc.py "{$BASE, $NC, 'MAX_BR': 9, $PK, $TC}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz   # planters searched
python3 cnet2_search_pack.py exp/submission_cnet2_v19pd_nc.py "{$BASE, $NC, 'MAX_BR': 9, $PK, $TC, 'PD_W': 3.0, 'PG_MIN': 100.0}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz   # + distance to good cell
# v19 round 2 (after review): saturating tree value TREE_C 1.0 * nb4 * (1 - 0.9625^min(50-age, left, 30)), NO TC_ONCE
TS="'TREE_C': 1.0, 'TREE_SAT': 30"
python3 cnet2_search_pack.py exp/submission_cnet2_v19tc2_nc.py "{$BASE, $NC, $TS}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz                    # leaf only (new base)
python3 cnet2_search_pack.py exp/submission_cnet2_v19pk2_nc.py "{$BASE, $NC, 'MAX_BR': 9, $PK, $TS}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz  # + planters searched
python3 cnet2_search_pack.py exp/submission_cnet2_v19pd2_nc.py "{$BASE, $NC, 'MAX_BR': 9, $PK, $TS, 'PD_W': 3.0, 'PG_MIN': 100.0}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# v19 round 3 (2nd review): 2 planters per step (MAX_BR 13); PD_POT = best potential tree within 3 steps, same scale as trees
python3 cnet2_search_pack.py exp/submission_cnet2_v19p2_nc.py "{$BASE, $NC, 'MAX_BR': 13, $PK, 'PK_MAX': 2, $TS}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v19pp_nc.py "{$BASE, $NC, 'MAX_BR': 13, $PK, 'PK_MAX': 2, $TS, 'PD_POT': True, 'PD_K': 3}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v19pp.py "{$BASE, $CAP, 'MAX_BR': 13, $PK, 'PK_MAX': 2, $TS, 'PD_POT': True, 'PD_K': 3}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz   # capped (ship-candidate) twin
# v19pq (3rd review): pp + PK_CASH (planters searched only when cash covers planting) + PK_H2 (depth 2 on planter-only steps); _nc and capped
PQ="'MAX_BR': 13, $PK, 'PK_MAX': 2, $TS, 'PD_POT': True, 'PD_K': 3, 'PK_CASH': True, 'PK_H2': True"
python3 cnet2_search_pack.py exp/submission_cnet2_v19pq_nc.py "{$BASE, $NC, $PQ}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v19pq.py "{$BASE, $CAP, $PQ}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_v19pq7_nc.py "{$BASE, $NC, $PQ, 'PD_DISC': 0.7}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz   # potential tree discounted 0.7/step (plant-now favoured)
# v19sd: v17d2 (capped) + stay-dodge in steps <= 100: a planter standing still with an enemy planter 1 step away moves (also when about to plant / seize)
python3 cnet2_search_pack.py exp/submission_cnet2_v19sd.py "{$BASE, $CAP, 'PLANTER_AVOID': True, 'PA_ALL': True, 'PA_MANH': True, 'PA_STEP': 100}" $W/cnet2_b128dg2.npz $W/cnet2_s64v8.npz
# Grove1 candidate: v17d2 search (capped) + AWR round-1 opening policy (slot 2 for steps 0-100, then 5, 7 from 240)
python3 cnet2_search_pack.py exp/submission_cnet2_awr1.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}" $W/cnet2_b128awr_r1.npz $W/cnet2_s64v8.npz
# Master2s as teacher (slot 2, all steps; dg5 base set with old slot-2 rows moved to slot 1), capped search:
# m2tA = slot 2 all game, m2tB = slot 2 until 240 then slot 7
python3 cnet2_search_pack.py exp/submission_cnet2_m2tA.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 999}" $W/cnet2_b128m2t.npz $W/cnet2_s64v8.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2tB.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t.npz $W/cnet2_s64v8.npz
# new rollout net s64mm (Master2s slot 2 + M family slot 4, from s64v8): helps only when the policy's style matches
python3 cnet2_search_pack.py exp/submission_cnet2_m2tBr.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3Br.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3Bq.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64m3.npz   # matched rollout (s64 on the Master2s x3 set)
# m2t3BL: late recruits/planters searched (bases from 200, planters from 240, typed worker leaf values with <= 100 left)
# m2t3BW: same but bases searched and workers valued by type for the WHOLE game (Master2s: over-investment in trees all game)
LATE="'MAX_BR': 9, 'BASE_KEY': True, 'BK_STEP': 200, 'BK_MAX': 1, 'PLANTER_KEY': True, 'PK_STEP': -1, 'PK_LATE': 240, 'PK_MAX': 1, 'PK_R': 2, 'TREE_C': 1.0, 'TREE_SAT': 30, 'PK_CASH': True, 'PK_H2': True, 'WT_LEFT': 100, 'WC_RATE': 4.4, 'WC_UTIL': 0.5"
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BL.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $LATE}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
WHOLE="'MAX_BR': 9, 'BASE_KEY': True, 'BK_STEP': 0, 'BK_MAX': 1, 'PLANTER_KEY': True, 'PK_STEP': -1, 'PK_LATE': 240, 'PK_MAX': 1, 'PK_R': 2, 'TREE_C': 1.0, 'TREE_SAT': 30, 'PK_CASH': True, 'PK_H2': True, 'WT_LEFT': 300, 'WC_RATE': 4.4, 'WC_UTIL': 0.5"
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BW.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $WHOLE}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# after review of 784dc9a: car value minus a 10-step round trip, late tree survival 0.6, MAX_BR 11, base keys only with
# < 10 workers and never for rule/guard-edited bases, bk_in/bk_cut counters; BLN = typed values only for new workers
L2="'MAX_BR': 11, 'BASE_KEY': True, 'BK_STEP': 200, 'BK_MAX': 1, 'PLANTER_KEY': True, 'PK_STEP': -1, 'PK_LATE': 240, 'PK_MAX': 1, 'PK_R': 2, 'TREE_C': 1.0, 'TREE_SAT': 30, 'PK_CASH': True, 'PK_H2': True, 'WT_LEFT': 100, 'WC_RATE': 4.4, 'WC_UTIL': 0.5, 'WC_TRIP': 10, 'TSURV': 0.6"
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BL2.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $L2}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BLN.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $L2, 'WT_NEW_ONLY': True}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
ST="'KEY_LATE_CARGO': True, 'KEY_LATE_STEP': 270, 'MAX_KEY_LATE': 5, 'KLC_MIN': 10.0, 'MAX_BR_LATE': 21"   # endgame short trips: loaded cars (>= 10) searched from 270, up to 5, late branch cap 21
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BL2S.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $L2, $ST}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# measurement: m2t3Br + hard recruit cuts (no planters from 256, nothing from 276) -- upper bound of a late recruiting discipline
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BC.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'REC_CUT_PLANTER': 256, 'REC_CUT_ALL': 276}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# Grove1 + endgame short trips only (no late base / planter search)
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrS.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, $ST}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# Grove1 late-teacher tests: m2t3BrA = slot 2 the whole game; m2t3BrL = slot 2 until 270 then slot 7
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrA.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 999}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrL.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 270}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# switch-point sweep around m2t3BrL: slot 2 until N then slot 7
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrL260.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 260}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrL265.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 265}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrL275.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 275}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrL280.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 280}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# exam-only fixed-budget twins (CLOCK sims; FIX_SIMS = host-4 capped medians per 50-step phase, 2026-11-10); never shipped
python3 cnet2_search_pack.py exp/fx_m2t3L240.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3L260.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 260}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3L265.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 265}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3L270.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 270}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3L275.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 275}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3L280.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 280}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_awr1.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}" $W/cnet2_b128awr_r1.npz $W/cnet2_s64v8.npz   # Scout1 fixed-budget twin
# camper-bust rule (exam twin) and search-budget probes (fixed 20 / 40 sims per step)
python3 cnet2_search_pack.py exp/fx_m2t3CB.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'CB_STEP': 100}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3S20.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (20,), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3S40.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (40,), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
# safe-entry rule exam twins: SEv = veto only unsafe planned moves of loaded (>=150) cars within 3 of base; SEc = cars >=400 within 3 head for the nearest safe cell
python3 cnet2_search_pack.py exp/fx_m2t3SEv.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'SE_STEP': 100, 'SE_MODE': 'veto', 'SE_MINC': 150.0}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3SEc.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'SE_STEP': 100, 'SE_MINC': 400.0}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz
python3 cnet2_search_pack.py exp/fx_m2t3SEw.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'SE_STEP': 100, 'SE_MODE': 'wait', 'SE_MINC': 300.0, 'SE_R': 6, 'SE_WAIT': 8}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz   # SEw: only cars loaded >=300 sitting 1-6 from base 8+ steps
python3 cnet2_search_pack.py exp/fx_m2t3SEt.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'SE_STEP': 100, 'SE_MODE': 'wait', 'SE_MINC': 300.0, 'SE_R': 6, 'SE_WAIT': 8, 'SE_TEAM': 1500.0}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz   # SEt: team stall (>=1500 on board, 8 steps no delivery)
# merged: Scout1's net (awr r1, slot 2) plays steps 0-99 with no search, then Grove1 (m2t3 + search, slot 2 -> 7 at 240)
python3 cnet2_search_pack.py exp/submission_cnet2_m2t3BrO.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'OPEN_NET_STEP': 100, 'OPEN_NET_T': 2}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz $W/cnet2_b128awr_r1.npz
python3 cnet2_search_pack.py exp/fx_m2t3O.py "{$BASE, $CAP, 'CLOCK': 'sims', 'FIX_SIMS': (10, 10, 10, 10, 10, 11), 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'OPEN_NET_STEP': 100, 'OPEN_NET_T': 2}" $W/cnet2_b128m2t3.npz $W/cnet2_s64mm.npz $W/cnet2_b128awr_r1.npz
