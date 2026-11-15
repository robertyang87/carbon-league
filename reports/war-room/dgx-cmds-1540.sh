#!/bin/bash
# Read-only probe: exact DGX-local commands used for m3s2/m7bs index build, dataset build, dual-arm train launch.
# Requested by war room 15:37 for Carago (183) + Fable (182). READ-ONLY: cat/grep/ls/head only, no writes.
# Marker: DGXCMDS1540X7Q2 (claim only output containing this marker)
MARK=DGXCMDS1540X7Q2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXCMDS1540X7Q2 DGX-BEGIN $(date '+%F %T %Z')"
hostname
C=/home/arm/junyang/carbon
echo '--- TREE-TOP ---'
ls -la $C | head -40
echo '--- CODE-DIR ---'
ls $C/code 2>/dev/null | head -60
echo '--- SH-SCRIPTS ---'
find $C -maxdepth 3 -name '*.sh' -newermt '2026-11-10' 2>/dev/null | head -30
echo '--- M7BS-SH-FULL ---'
for f in $(find $C -maxdepth 3 -name 'm7bs*.sh' 2>/dev/null | head -3); do echo "== $f"; cat "$f"; done
echo '--- BUILD-LOGS-LIST ---'
find $C -maxdepth 4 -name 'build*.log' -o -maxdepth 4 -name '*build*.log' 2>/dev/null | head -10
ls -lt $C/data/dq_logs/ 2>/dev/null | head -25
echo '--- BUILD-M7BS-LOG-HEAD ---'
for f in $(find $C -maxdepth 4 -iname '*build*m7bs*.log' 2>/dev/null | head -2); do echo "== $f"; head -40 "$f"; done
echo '--- BUILD-M3S2-LOG-HEAD ---'
for f in $(find $C -maxdepth 4 -iname '*build*m3s2*.log' 2>/dev/null | head -2); do echo "== $f"; head -40 "$f"; done
echo '--- FEAT-TEACHERS ---'
for f in $(find $C -maxdepth 3 -name 'cnet2_feat.py' 2>/dev/null | head -2); do echo "== $f"; grep -n -A15 'TEACHERS' "$f" | head -40; done
echo '--- BUILD-PY-HEAD ---'
for f in $(find $C -maxdepth 3 \( -name 'cnet2_build.py' -o -name 'build_top.py' \) 2>/dev/null | head -3); do echo "== $f"; head -60 "$f"; done
echo '--- SET-UW ---'
find $C -maxdepth 3 -iname '*set_uw*' 2>/dev/null | head -5
grep -rn 'set_uw' $C/code/*.py 2>/dev/null | head -10
echo '--- INDEX-SCRIPTS ---'
grep -rln 'index.json' $C/code/*.py $C/*.py $C/*.sh 2>/dev/null | head -10
find $C -maxdepth 3 -iname '*index*' \( -name '*.py' -o -name '*.sh' \) 2>/dev/null | head -10
echo '--- M3S2-CORPUS-DIR ---'
ls $C/data/m3s2_corpus/ 2>/dev/null | head; ls $C/data/m7bs_corpus/ 2>/dev/null | head
echo '--- INDEX-JSON-HEAD ---'
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/index.json'));e=(d if isinstance(d,list) else d.get('entries') or d);print(type(d));print(json.dumps(e[0] if isinstance(e,list) else list(e.items())[0], ensure_ascii=False)[:600])" 2>&1
echo '--- BASH-HISTORY-RELEVANT ---'
grep -aE 'cnet2_build|cnet2_train|build_top|set_uw|index|OUT_DIR|REPLAY_DIR|CNET2_' ~/.bash_history 2>/dev/null | tail -80
echo "DGXCMDS1540X7Q2 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
