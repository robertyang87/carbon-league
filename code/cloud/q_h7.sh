#!/bin/bash
# host 7 (2026-11-10 evening): the ladder-like Grove2 candidate exam (cap 6, uniform, 25 sims) vs Grove1 at 20 sims,
# plus its paired 1C-v9 check (same seeds, same machine)
cd /root/carbon/code; export OMP_NUM_THREADS=1
EVO_WORKERS=46 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3U6B25.py exp/fx_m2t3S20.py 988000 400 p3_U6B25_vs_S20_h7 > ../data/p3_u6_h7.out 2>&1 < /dev/null &
EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3U6B25.py exp/opp_1cz_v9.py 987000 100 c9_U6B25_h7 > ../data/c9_u6_h7.out 2>&1 < /dev/null &
EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S20.py exp/opp_1cz_v9.py 987000 100 c9_S20_h7 > ../data/c9_s20_h7.out 2>&1 < /dev/null &
wait
