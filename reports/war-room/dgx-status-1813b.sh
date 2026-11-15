#!/bin/bash
# DGX status probe 1813b (correction): replay counts under replays/ + branch-tip HEAD check.
# Marker: DGXSTATUS1813F2L9
MARK=DGXSTATUS1813F2L9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1813F2L9 DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon/data 2>/dev/null
for d in m3s2_corpus carago_corpus fable_corpus m6_corpus k4v2_corpus; do echo -n "$d/replays="; ls "$d/replays" 2>/dev/null | wc -l; done
echo -n "m6 index: "; python3 -c "import json;print(len(json.load(open('m6_corpus/index.json'))))" 2>/dev/null || echo MISSING
echo -n "k4v2 dir: "; ls k4v2_corpus 2>/dev/null | head -3; ls k4v2_corpus 2>/dev/null | wc -l
cd /home/arm/junyang/carbon-repo 2>/dev/null
echo "local HEAD: $(git rev-parse --short HEAD) $(git log -1 --format=%ci_%s | head -c 100)"
echo "remote branch tip:"; timeout 60 git ls-remote origin cnet2-search-v12 2>&1
echo "DGXSTATUS1813F2L9 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
