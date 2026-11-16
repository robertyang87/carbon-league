#!/bin/bash
# adj1 tier-1 signature measurement driver: 10 pool packs x diagnostic set
# (20291101-20292100), chunks of 200 seeds, 10 pinned workers, nice 10.
# Resume-safe: existing chunk files are skipped. Progress: ls /root/carbon/sig_adj1/
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /root/carbon/probe_m2/code
mkdir -p /root/carbon/sig_adj1
PACKS="m3s2r1_base pool_m3s2g1L pool_m7bsr1L pool_m9xj260wr1_fromstudio pool_m9xj260r1 pool_m6r1L pool_m7bsv2r1 pool_m9xj260g1 pool_m7bsv2g1 pool_caragor1_ep19"
for p in $PACKS; do
  for i in 0 1 2 3 4; do
    lo=$((20291101 + i * 200)); hi=$((lo + 200))
    out=/root/carbon/sig_adj1/${p}_s${i}.jsonl
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo "run $p chunk $i seeds $lo-$hi $(date +%H:%M:%S)"
    nice -n 10 /root/carbon/.venv/bin/python /root/carbon/signature_measure.py exp/${p}.py $lo $hi $out 10 || echo "CHUNK_FAIL $p $i"
  done
done
echo ALL_CHUNKS_DONE
