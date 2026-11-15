#!/bin/bash
# DGX status probe trip 12:58 - follow-up B. Marker DGXSTATUS1258B. Read-only.
echo "DGXSTATUS1258B $(date '+%H:%M:%S %Z')"
echo "== GATE STOP EVIDENCE =="
cat /home/arm/junyang/gate_fields.log 2>&1; echo
tail -n 6 /home/arm/junyang/carbon/data/gate_dgx.log 2>&1 | cut -c1-300
tail -n 6 /home/arm/junyang/carbon/data/gate_dgx_b.log 2>&1 | cut -c1-300
ls -la /home/arm/junyang/carbon/data/gate_guided_master2/ 2>&1
echo "== GEN1 ROUND2 LOG =="
tail -n 12 /home/arm/junyang/katago_chain_gen1b.log 2>&1 | cut -c1-300
echo "== VERIFY FIELD PROGRESS =="
ls -lt /home/arm/junyang/carbon/data/bigeval_live/ 2>&1 | head -8
ls -l /home/arm/junyang/verify_search.log 2>&1
ps -eo pid,etime,cmd | grep -E 'policy_verify|bigeval' | grep -v grep || echo NO_VERIFY_PROC
echo "== HEAD BEHIND =="
cd /home/arm/junyang/carbon-repo && git rev-list --count HEAD..fed5adc0768ca01505a5045b93bba4fb5616a44f 2>&1
git log --oneline fed5adc0768ca01505a5045b93bba4fb5616a44f -5 2>&1
echo "== DEPLOY EVIDENCE (files changed since 12:46) =="
find /home/arm/junyang -maxdepth 2 -newermt '2026-11-13 12:46' -type f 2>/dev/null | grep -v -E 'bigeval_live|__pycache__|\.venv' | head -15
ps -eo pid,etime,cmd | grep -E 'ssh.*100.98.67.20|deploy' | grep -v grep || echo NO_DEPLOY_PROC
echo "== GPU NOW =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader
echo "DGXSTATUS1258B-END $(date '+%H:%M:%S %Z')"
