#!/bin/bash
cd /Users/baggioustb/dev/carbon/carbon-league/code; K=$HOME/.ssh/carbon_aliyun_ed25519; H=root@47.117.170.99
until ssh -o ConnectTimeout=60 dgx-via-spot3 'grep -q TRAIN_DONE ~/junyang/carbon/data/cnet2_train_s64m3.log' </dev/null 2>/dev/null; do sleep 60; done
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_s64m3.npz ../data/weights/ || exit 1
W=../data/weights; export CNET2_QMODE=int8 CNET2_QMODE_ROLL=f16
BASE="'MODE': 'mcts', 'CLOCK': 'wall', 'ROOT_S': 10, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7), 'ID_PRIOR': 0.0"
CAP="'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None"
../.venv/bin/python cnet2_search_pack.py exp/submission_cnet2_m2t3Bq.py "{$BASE, $CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}" $W/cnet2_b128m2t3.npz $W/cnet2_s64m3.npz > /dev/null 2>&1 || exit 1
echo "python3 cnet2_search_pack.py exp/submission_cnet2_m2t3Bq.py \"{\$BASE, \$CAP, 'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}\" \$W/cnet2_b128m2t3.npz \$W/cnet2_s64m3.npz   # matched rollout (s64 on the Master2s x3 set)" >> packs/build_v17.sh
scp -q -i $K exp/submission_cnet2_m2t3Bq.py ${H}:/root/carbon/code/exp/
ssh -i $K $H 'while pgrep -f "[g]ate_r2.sh" >/dev/null; do sleep 20; done; cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=30
../.venv/bin/python bigeval.py exp/submission_cnet2_m2t3Bq.py exp/submission_cnet2_v17d2.py 998000 200 h2h_m2t3Bq_v17d2 >> ../data/gate.log 2>&1 < /dev/null &
../.venv/bin/python bigeval.py exp/submission_cnet2_m2t3Bq.py exp/submission_cnet2_v17d2.py 999000 200 h2h_m2t3Bq_v17d2b >> ../data/gate.log 2>&1 < /dev/null &
wait' </dev/null
cd ../data; for t in h2h_m2t3Bq_v17d2 h2h_m2t3Bq_v17d2b; do scp -q -i $K ${H}:/root/carbon/data/bigeval_live/$t.jsonl bigeval_live/; done
cat bigeval_live/h2h_m2t3Bq_v17d2.jsonl bigeval_live/h2h_m2t3Bq_v17d2b.jsonl > bigeval_live/h2h_m2t3Bq_v17d2all.jsonl
python3 h2h_report.py h2h_m2t3Bq_v17d2all > s64m3_eval.out; echo Q_DONE >> s64m3_eval.out
