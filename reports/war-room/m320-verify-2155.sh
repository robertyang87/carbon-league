#!/bin/bash
# it320 chain verify probe 2: poller disarm check + training trio + _w dataset audit.
# Marker: M320VERIFY2155
MARK=M320VERIFY2155
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M320VERIFY2155 DGX-BEGIN $(date '+%F %T %Z')"
hostname
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo '--- POLLER-264410 ---'
if [ -d /proc/264410 ]; then ps -p 264410 -o pid,etime,args; tr '\0' ' ' < /proc/264410/cmdline; echo; else echo POLLER-264410-GONE; fi
echo '--- ANY-POLLER-OR-CHAIN-PROC ---'
ps -eo pid,etime,args | grep -E 'wait_freeze|WAIT|poll|replica_chain|build_top|m320n' | grep -v grep || echo NO-POLLER-CHAIN-PROC
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- CHAIN-STATE ---'
cat /home/arm/junyang/carbon/data/m9xj320_chain.state; echo
echo '--- READY ---'
cat /home/arm/junyang/carbon/data/M9XJ320_READY.txt
echo '--- MANIFEST ---'
cat /home/arm/junyang/carbon/data/m9xj320_manifest.json; echo
echo '--- W-DATASET-AUDIT ---'
python3 - <<'PYEOF'
import json, hashlib, os
base='/home/arm/junyang/carbon/data/'
for d in ['ds_m9xj320','ds_m9xj320_w']:
    p=base+d+'/meta.json'
    try:
        m=json.load(open(p))
        print(d, {k:(len(v) if isinstance(v,list) else v) for k,v in m.items()})
    except Exception as e:
        print(d, 'META-ERR', e)
# matches: Echo2 count + structure
mj=json.load(open(base+'m9xj320_matches.json'))
print('matches_type', type(mj).__name__, 'n=', len(mj))
items = mj if isinstance(mj,list) else mj.get('matches') or mj.get('entries') or []
if items:
    print('first_entry_keys', list(items[0].keys()))
    import collections
    s=json.dumps(items)
    print('echo2_mentions_in_json', s.count('Echo2'))
PYEOF
echo '--- TRAIN-LOG-TAILS ---'
for f in m9xj320r1 m9xj320g1; do echo "== $f"; tail -2 /home/arm/junyang/carbon/data/dq_logs/$f.log; grep -E '"ep": 19|best' /home/arm/junyang/carbon/data/dq_logs/$f.log | tail -2; done
echo '--- WEIGHTS ---'
ls -lt --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/ | grep m9xj320 | head -10
echo "M320VERIFY2155 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
