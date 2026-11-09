#!/bin/bash
# Master2s-teacher net: when trained, pure-net whole-game head-to-heads on the Mac (slot 2 all game vs dg5 slot 5->7),
# then the same vs the fitted M (+50) and vs 1C-v9.
cd /Users/baggioustb/dev/carbon/carbon-league/data
until ssh -o ConnectTimeout=60 dgx-via-spot3 'grep -q "^best" ~/junyang/carbon/data/cnet2_train_b128m2t.log' </dev/null 2>/dev/null; do sleep 60; done
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128m2t.npz weights/ || exit 1
cd ../code; python3 - <<'PY'
s = open("exp/cnet2_dg5e5s57_240.py").read()
for name, conf in (("m2t_all2", "'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 999}"), ("m2t_27", "'TEACHER': 2, 'TEACHER_LATE': 7, 'LATE_STEP': 240}")):
    t = s.replace("'cnet2_b128dg5_ep5.npz'", "'cnet2_b128m2t.npz'").replace("'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}", conf)
    open(f"exp/cnet2_{name}.py", "w").write(t)
PY
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
for v in m2t_all2 m2t_27; do ../.venv/bin/python bigeval.py exp/cnet2_$v.py exp/cnet2_dg5e5s57.py 998000 200 h2h_${v}_dg5 >> ../data/m2t_eval.log 2>&1 < /dev/null; done
EVAL_OPP_CASH=50 ../.venv/bin/python bigeval.py exp/cnet2_m2t_all2.py exp/mfit_dg5.py 998000 200 h2h_m2t_all2_mfit >> ../data/m2t_eval.log 2>&1 < /dev/null
../.venv/bin/python bigeval.py exp/cnet2_dg5e5s57.py exp/mfit_dg5.py 998000 0 noop > /dev/null 2>&1
EVAL_OPP_CASH=50 ../.venv/bin/python bigeval.py exp/cnet2_dg5e5s57.py exp/mfit_dg5.py 998000 200 h2h_dg5_mfit >> ../data/m2t_eval.log 2>&1 < /dev/null
cd ../data; python3 h2h_report.py h2h_m2t_all2_dg5 h2h_m2t_27_dg5 h2h_m2t_all2_mfit h2h_dg5_mfit > m2t_eval.out; echo M2T_DONE >> m2t_eval.out
