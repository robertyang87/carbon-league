#!/bin/bash
# DGX status probe 0128C (2026-11-15 ~01:33 CST). Marker DGXSTATUS0128C. Read-only. Minimal: clone index lengths + recent pull logs.
MARK=DGXSTATUS0128C
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
M=DGXSTATUS0128C
VPY=/home/arm/junyang/carbon/.venv/bin/python
echo '--CLONE-INDEX--'
for d in m3s2_corpus carago_corpus fable_corpus krypton-csplant_corpus; do echo -n "$d clone_index="; $VPY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d))' /home/arm/junyang/carbon-repo/data/$d/index.json 2>&1 | head -1; done
echo '--PULL-LOGS--'
ls -lat /home/arm/junyang/pull*.log 2>/dev/null | head -8
echo '--FABLE-INDEX-MTIME--'
ls -la /home/arm/junyang/carbon-repo/data/fable_corpus/index.json 2>/dev/null
echo '--DUEL-STILL--'
ps -eo pid,etime,args | grep -E 'k0_duel.py' | grep -v grep | wc -l
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"
