#!/bin/bash
# ds meta diagnostic probe 0243. Read-only. Marker: DGXDS0243A
MARK=DGXDS0243A
echo "$MARK BEGIN $(date -u +%H:%M:%S)"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
M=DGXDS0243A
cd /home/arm/junyang/carbon
echo "== ls data/ds_carago data/ds_fable:"
ls -la data/ds_carago data/ds_fable 2>&1 | head -20
echo "== find meta.json under data/ds_*:"
timeout 60 find data -maxdepth 3 -name 'meta.json' -path '*ds_*' 2>/dev/null
echo "== python read attempt (verbatim error):"
python3 -c "import json;m=json.load(open('data/ds_carago/meta.json'));print(m.get('num_samples') or m.get('n') or list(m.keys())[:10])" 2>&1 | head -5
echo "== ds dirs anywhere under carbon:"
timeout 60 find /home/arm/junyang/carbon -maxdepth 2 -iname 'ds_*' 2>/dev/null | head
echo "$M END"
DGXEOF
echo "$MARK END rc=$?"
