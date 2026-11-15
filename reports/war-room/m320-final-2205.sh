#!/bin/bash
# it320 chain FINAL probe: training completion, best vals, weight md5s, val-shard identity.
# Marker: M320FINAL2205
MARK=M320FINAL2205
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M320FINAL2205 DGX-BEGIN $(date '+%F %T %Z')"
hostname
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS (expect none if done) ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo TRAIN-DONE-NO-PROC
echo '--- R1-LOG-FULL-TAIL ---'
tail -4 /home/arm/junyang/carbon/data/dq_logs/m9xj320r1.log
echo '--- G1-LOG-FULL-TAIL ---'
tail -4 /home/arm/junyang/carbon/data/dq_logs/m9xj320g1.log
echo '--- BEST-VALS ---'
python3 - <<'PYEOF'
import json
for arm in ['r1','g1']:
    p=f'/home/arm/junyang/carbon/data/weights/cnet2_m9xj320{arm}_log.jsonl'
    best=None; last=None; n=0
    try:
        for line in open(p):
            d=json.loads(line)
            if 'val' in d:
                n+=1; last=d
                if best is None or d['val']['all']>best['val']['all']: best=d
        print(arm, 'epochs_with_val=', n, 'BEST ep=', best['ep'], 'val_all=', best['val']['all'], 'val=', best['val'], 'LAST ep=', last['ep'], 'val_all=', last['val']['all'])
    except Exception as e:
        print(arm, 'ERR', e)
PYEOF
echo '--- WEIGHT-MD5 ---'
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320g1.npz
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320g1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1_ep19.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320g1_ep19.npz 2>&1
echo '--- VAL-SHARD-IDENTITY ---'
python3 - <<'PYEOF'
import os, hashlib
def shardmap(d):
    out={}
    sd=os.path.join(d,'shards')
    if not os.path.isdir(sd): return out
    for f in sorted(os.listdir(sd)):
        h=hashlib.md5()
        with open(os.path.join(sd,f),'rb') as fh:
            for chunk in iter(lambda: fh.read(1<<20), b''): h.update(chunk)
        out[f]=h.hexdigest()
    return out
a=shardmap('/home/arm/junyang/carbon/data/ds_m9xj320')
b=shardmap('/home/arm/junyang/carbon/data/ds_m9xj320_w')
common=set(a)&set(b)
diff=[f for f in common if a[f]!=b[f]]
print('ds_shards=',len(a),' ds_w_shards=',len(b),' common_names=',len(common),' common_with_diff_md5=',len(diff), diff[:5])
PYEOF
echo '--- GPU-PROCS-NOW ---'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true
echo "M320FINAL2205 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
