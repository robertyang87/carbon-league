#!/bin/bash
pkill -f "python dagger_gen.py"; sleep 3
ps -eo pid,ppid,args | grep "[m]ultiprocessing\|[c]oncurrent" | awk '$2==1 {print $1}' | xargs -r kill 2>/dev/null
cd ~/carbon/code; rm -rf ../data/dagger1; mkdir -p ../data/dagger1
nohup env OMP_NUM_THREADS=1 BC_W=64 ../.venv/bin/python dagger_gen.py ../data/dagger1 exp/cnet2_v9s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 7000 400 24 > ../data/dagger1.out 2>&1 </dev/null &
