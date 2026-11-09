#!/bin/bash
# new rollout net s64mm: three capped search packs vs capped v17d2 (200 head-to-head games each) on host 4
cd /Users/baggioustb/dev/carbon/carbon-league/code; K=$HOME/.ssh/carbon_aliyun_ed25519; H=root@47.117.170.99
until ssh -o ConnectTimeout=60 dgx-via-spot3 'grep -q TRAIN_DONE ~/junyang/carbon/data/cnet2_train_s64mm.log' </dev/null 2>/dev/null; do sleep 60; done
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_s64mm.npz ../data/weights/ || exit 1
W=../data/weights; export CNET2_QMODE=int8 CNET2_QMODE_ROLL=f16
BASE="'MODE': 'mcts', 'CLOCK': 'wall', 'ROOT_S': 10, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7), 'ID_PRIOR': 0.0"
CAP="'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None"; OP="'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240"
../.venv/bin/python cnet2_search_pack.py exp/submission_cnet2_v17d2r.py "{$BASE, $CAP}" $W/cnet2_b128dg2.npz $W/cnet2_s64mm.npz > /dev/null 2>&1
../.venv/bin/python cnet2_search_pack.py exp/submission_cnet2_m2tBr.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t.npz $W/cnet2_s64mm.npz > /dev/null 2>&1
../.venv/bin/python cnet2_search_pack.py exp/submission_cnet2_awr1r.py "{$BASE, $CAP, $OP}" $W/cnet2_b128awr_r1.npz $W/cnet2_s64mm.npz > /dev/null 2>&1
ls exp/submission_cnet2_v17d2r.py exp/submission_cnet2_m2tBr.py exp/submission_cnet2_awr1r.py >> ../data/s64_eval.log 2>&1 || exit 1
scp -q -i $K exp/submission_cnet2_v17d2r.py exp/submission_cnet2_m2tBr.py exp/submission_cnet2_awr1r.py ${H}:/root/carbon/code/exp/
ssh -i $K $H 'while pgrep -f "[g]ate_awr1_old.sh" >/dev/null; do sleep 20; done; cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
for v in v17d2r m2tBr awr1r; do EVO_WORKERS=20 ../.venv/bin/python bigeval.py exp/submission_cnet2_$v.py exp/submission_cnet2_v17d2.py 998000 200 h2h_${v}_v17d2 >> ../data/gate.log 2>&1 < /dev/null & done; wait' </dev/null
cd ../data; for t in h2h_v17d2r_v17d2 h2h_m2tBr_v17d2 h2h_awr1r_v17d2; do scp -q -i $K ${H}:/root/carbon/data/bigeval_live/$t.jsonl bigeval_live/; done
python3 h2h_report.py h2h_v17d2r_v17d2 h2h_m2tBr_v17d2 h2h_awr1r_v17d2 >> s64_eval.log; echo S64_DONE >> s64_eval.log
