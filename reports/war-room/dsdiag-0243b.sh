#!/bin/bash
# ds meta values probe 0243b. Read-only. Marker: DGXDS0243B
MARK=DGXDS0243B
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXDS0243B
cd /home/arm/junyang/carbon
for d in ds_carago ds_fable; do python3 -c "import json,os;m=json.load(open('data/$d/meta.json'));print('$d', 'mtime', __import__('datetime').datetime.fromtimestamp(os.path.getmtime('data/$d/meta.json')).strftime('%m%d_%H:%M'), 'games=', m.get('games'), 'total=', m.get('total'), 'teachers=', m.get('teachers'))" 2>&1; done
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"
