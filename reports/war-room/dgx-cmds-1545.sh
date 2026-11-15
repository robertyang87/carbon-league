#!/bin/bash
# Read-only probe 2: index build provenance - cat m3s2.sh, ladder_corpus.py index logic, matches_fresh.json sample.
# Marker: DGXCMDS1545Y3R8
MARK=DGXCMDS1545Y3R8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXCMDS1545Y3R8 DGX-BEGIN $(date '+%F %T %Z')"
C=/home/arm/junyang/carbon
echo '--- M3S2-SH-FULL ---'
cat $C/code/m3s2.sh 2>/dev/null
echo '--- CODE-GREP-FIX-INDEX ---'
ls $C/code | grep -iE 'fix|index|corpus|ladder'
echo '--- LADDER-CORPUS-GREP ---'
grep -n 'index\|steps\|winner\|strategies' $C/code/ladder_corpus.py 2>/dev/null | head -30
echo '--- LADDER-CORPUS-HEAD ---'
head -50 $C/code/ladder_corpus.py 2>/dev/null
echo '--- MATCHES-FRESH-SAMPLE ---'
python3 -c "
import json
d=json.load(open('/home/arm/junyang/carbon/data/m3s2_corpus/matches_fresh.json'))
print(type(d), len(d) if hasattr(d,'__len__') else '')
e = d[0] if isinstance(d,list) else list(d.values())[0] if isinstance(d,dict) else d
print(json.dumps(e, ensure_ascii=False)[:800])
" 2>&1
echo '--- ANY-FIX4-FILES ---'
find $C -maxdepth 3 -iname '*fix*' -newermt '2026-11-11' 2>/dev/null | head -10
grep -rln 'bak88\|FIX4\|fix4' $C/code/*.py $C/code/*.sh 2>/dev/null | head -5
echo "DGXCMDS1545Y3R8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
