#!/bin/bash
# DGX status probe 0128B (2026-11-15 ~01:32 CST). Marker DGXSTATUS0128B. Read-only. Follow-up: k0_calib outputs, fable growth, carbon work-dir counts, HEAD behind-count.
MARK=DGXSTATUS0128B
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
M=DGXSTATUS0128B
VPY=/home/arm/junyang/carbon/.venv/bin/python
echo '--K0CALIB-TREE--'
ls -la /home/arm/junyang/k0_calib/ 2>/dev/null | head -25
echo '--CELLS-GRID1--'
ls -la /home/arm/junyang/k0_calib/cells/lr0.0002_s100_b1_seed7.* 2>/dev/null
echo '--DUELS-OUT--'
ls -la /home/arm/junyang/k0_calib/duels/ 2>/dev/null
ls -la /home/arm/junyang/k0_calib/duels/lr0.0002_s100_b1_seed7.json 2>/dev/null || echo "duel out json NOT YET PRESENT"
echo '--DUEL-JSON-COUNT--'
$VPY -c 'import json,sys; d=json.load(open("/home/arm/junyang/k0_calib/duels/lr0.0002_s100_b1_seed7.json")); print("type:", type(d).__name__, "len:", len(d) if hasattr(d,"__len__") else "NA")' 2>&1 | head -3
echo '--DRIVER-LOG--'
ls -lat /home/arm/junyang/k0_calib/*.log 2>/dev/null | head -5
tail -5 /home/arm/junyang/k0_calib/driver.log 2>/dev/null; echo "(end driver.log tail)"
tail -3 /home/arm/junyang/k0_calib/cells/lr0.0002_s100_b1_seed7.log.jsonl 2>/dev/null; echo "(end grid1 log tail)"
echo '--FABLE-NEWEST--'
ls -lat /home/arm/junyang/carbon-repo/data/fable_corpus/ 2>/dev/null | head -25
echo '--CARBON-WORKDIR-CORPUS--'
for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d total="; ls /home/arm/junyang/carbon/data/$d/ 2>/dev/null | wc -l; echo -n "$d gz="; ls /home/arm/junyang/carbon/data/$d/*.json.gz 2>/dev/null | wc -l; echo -n "$d index="; $VPY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d))' /home/arm/junyang/carbon/data/$d/index.json 2>&1 | head -1; done
for d in ds_carago ds_fable; do echo -n "$d meta="; $VPY -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m.get("total", m))' /home/arm/junyang/carbon/data/$d/meta.json 2>&1 | head -1; done
echo '--HEAD-BEHIND--'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse origin/main 2>&1 | head -1; timeout 60 git rev-list --count HEAD..origin/main 2>&1 | head -1; timeout 60 git cat-file -t 102dc97046009bcbd4e29e4f3981905f566670f1 2>&1 | head -1; timeout 60 git rev-list --count HEAD..102dc97046009bcbd4e29e4f3981905f566670f1 2>&1 | head -1
echo '--GPU-MEM-RECHECK--'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"
