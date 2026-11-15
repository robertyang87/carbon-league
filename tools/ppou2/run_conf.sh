#!/bin/bash
# ppou2 early confirmation exam (strategy-group ruling 2026-11-15 11:19).
# Reserve blocks: M2 984000-984499, E2 984500-984999 (500 games per field).
# Baselines freshly produced from the frozen start pack (ah 11d80059),
# M2 two batches; candidate = ppou2_m_it20 pack (ah f4fff163) as-is.
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
PY=/root/carbon/.venv/bin/python
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 984000 500 ppou2_conf_base_m2_a > /root/carbon/probe_m2/conf_base_m2_a.log 2>&1
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 984000 500 ppou2_conf_base_m2_b > /root/carbon/probe_m2/conf_base_m2_b.log 2>&1
$PY bigeval.py exp/m3s2r1_base.py exp/echo2.py 984500 500 ppou2_conf_base_e2 > /root/carbon/probe_m2/conf_base_e2.log 2>&1
$PY bigeval.py exp/ppou2_m_it20.py exp/opp_master2r.py 984000 500 ppou2_conf_it20_m2 > /root/carbon/probe_m2/conf_it20_m2.log 2>&1
$PY bigeval.py exp/ppou2_m_it20.py exp/echo2.py 984500 500 ppou2_conf_it20_e2 > /root/carbon/probe_m2/conf_it20_e2.log 2>&1
echo done > /root/carbon/probe_m2/conf_done.txt
