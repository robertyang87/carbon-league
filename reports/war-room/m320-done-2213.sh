#!/bin/bash
# it320 chain completion probe: both arms done, best vals, md5s, val-row byte identity.
# Marker: M320DONE2213
MARK=M320DONE2213
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M320DONE2213 DGX-BEGIN $(date '+%F %T %Z')"
hostname
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- TRAIN-PROCS (expect none) ---'
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo TRAIN-DONE-NO-PROC
echo '--- LOG-TAILS ---'
tail -2 /home/arm/junyang/carbon/data/dq_logs/m9xj320r1.log
tail -1 /home/arm/junyang/carbon/data/dq_logs/m9xj320g1.log
echo '--- BEST-VALS-AND-MD5 ---'
python3 - <<'PYEOF'
import json, hashlib, os
for arm in ['r1','g1']:
    p=f'/home/arm/junyang/carbon/data/weights/cnet2_m9xj320{arm}_log.jsonl'
    best=None; last=None; n=0
    for line in open(p):
        d=json.loads(line)
        if 'val' in d:
            n+=1; last=d
            if best is None or d['val']['all']>best['val']['all']: best=d
    w=f'/home/arm/junyang/carbon/data/weights/cnet2_m9xj320{arm}.npz'
    h=hashlib.md5(open(w,'rb').read()).hexdigest()
    st=os.stat(w)
    import time
    print(arm,'epochs=',n,'BEST ep=',best['ep'],'val_all=',best['val']['all'],'LAST ep=',last['ep'],'val_all=',last['val']['all'],'weight_md5=',h,'size=',st.st_size,'mtime=',time.strftime('%F %T',time.localtime(st.st_mtime)))
PYEOF
echo '--- VAL-ROW-IDENTITY ---'
python3 - <<'PYEOF'
import numpy as np, collections
base='/home/arm/junyang/carbon/data/'
def load(d):
    out={}
    for name in ['gid','step','carbon']:
        try: out[name]=np.load(base+d+'/'+name+'.npy')
        except Exception as e: print(d,name,'ERR',e)
    return out
a=load('ds_m9xj320'); b=load('ds_m9xj320_w')
print('rows ds=',len(a['gid']),' ds_w=',len(b['gid']))
ca=collections.Counter(a['gid'].tolist()); cb=collections.Counter(b['gid'].tolist())
unchanged=[g for g in ca if cb.get(g)==ca[g]]
print('games_total=',len(ca),' games_rowcount_unchanged=',len(unchanged))
ok=True; vrows=0
for g in unchanged:
    ia=np.where(a['gid']==g)[0]; ib=np.where(b['gid']==g)[0]
    vrows+=len(ia)
    if not (np.array_equal(a['step'][ia],b['step'][ib]) and np.array_equal(a['carbon'][ia],b['carbon'][ib])):
        ok=False; print('VAL-ROW-DIFF gid',g)
print('val_rows_checked=',vrows,'val_rows_identical=',ok)
PYEOF
echo '--- CHAIN-STATE ---'
cat /home/arm/junyang/carbon/data/m9xj320_chain.state
echo "M320DONE2213 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
