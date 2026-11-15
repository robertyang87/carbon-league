#!/bin/bash
echo "STUDIO_PROBE_BEGIN"
hostname
date "+%Y-%m-%d %H:%M:%S %Z"
echo "--- train procs ---"
pgrep -fl "cnet2_train" || echo "NO_CNET2_TRAIN_PROC"
ps -Ao pid,etime,command | grep "[p]ython" | head -20
echo "--- data dir ---"
ls -la ~/junyang/carbon/data | head -40
echo "--- ds_m3s2 top ---"
ls -la ~/junyang/carbon/data/ds_m3s2 2>/dev/null | head -30
du -sh ~/junyang/carbon/data/ds_m3s2 2>/dev/null
echo "--- meta.json ---"
head -c 1500 ~/junyang/carbon/data/ds_m3s2/meta.json 2>/dev/null; echo
echo "--- init weights locate ---"
find ~/junyang/carbon -name "cnet2_arch_n1_last.npz" 2>/dev/null
echo "--- weights dir ---"
ls -la ~/junyang/carbon/data/weights 2>/dev/null | head -30
echo "--- disk ---"
df -h ~ | tail -2
echo "--- torch ---"
~/junyang/carbon/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
echo "STUDIO_PROBE_END"
