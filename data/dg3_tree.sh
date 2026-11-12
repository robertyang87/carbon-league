cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
until grep -q "games 40" ../data/dagger_probe_dg3.out 2>/dev/null; do sleep 30; done
for v in dg2 dg3; do ../.venv/bin/python bigeval.py exp/cnet2_${v}s57_240.py exp/cnet2_m2s2stand.py 996000 200 m2_${v}n_m2s2 >> ../data/dg3_tree.log 2>&1; done
python3 ../data/planter_check.py m2_v9n_m2s2 m2_dg1n_m2s2 m2_dg2n_m2s2 m2_dg3n_m2s2
python3 ../data/paired.py m2_dg3n_m2s2 m2_dg2n_m2s2 2>&1 | tail -3
