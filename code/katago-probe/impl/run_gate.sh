#!/bin/bash
# KataGo probe-2 construction gate on rootset32 (6,336 registered roots).
# Frozen DEFAULTS (PREREG rev.1); arms value / const / heur in one driver run,
# per-root paired seeds. Diagnostic only -- never enters any training pool.
# Launch (only when the DGX window is verified open):
#   cd /home/arm/junyang/katago_probe_scratch/probe2 && setsid nohup ./run_gate.sh > gate.log 2>&1 &
set -u
cd /home/arm/junyang/katago_probe_scratch/probe2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /home/arm/junyang/carbon/.venv/bin/python gate_driver.py \
  --moddir modules \
  --replays /home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/roots \
  --rootset /home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/rootset32.json \
  --weights /home/arm/junyang/s26stage/data/weights/cnet2_m3s2r1L.npz \
  --rweights /home/arm/junyang/carbon/data/weights/cnet2_s64.npz \
  --out out_gate \
  --workers 16
