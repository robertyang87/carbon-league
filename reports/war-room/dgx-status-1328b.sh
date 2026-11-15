#!/bin/bash
# DGX status probe trip 13:28 follow-up - read-only. Marker DGXSTATUS1328B.
echo "DGXSTATUS1328B $(date '+%H:%M:%S %Z')"
python3 -c "
import json
rows=[json.loads(l) for l in open('/home/arm/junyang/carbon-repo/data/bigeval_live/verify_search.jsonl')]
done=[r for r in rows if r.get('err') is None and r.get('margin') is not None]
errs=[r for r in rows if r.get('err') is not None]
print('total_rows=',len(rows),'completed=',len(done),'err_rows=',len(errs))
print('wins=',sum(1 for r in done if r['margin']>0),'losses=',sum(1 for r in done if r['margin']<0))
print('mean_margin=',round(sum(r['margin'] for r in done)/len(done),1) if done else None)
print('seeds_done=',sorted(r['seed'] for r in done))
print('last_t=',max(r.get('t','') for r in done))
"
echo "== DEPLOY CHILDREN =="
ps -eo pid,etime,cmd | grep -E 'deploy|rsync|scp|ssh.*100.98' | grep -v grep || echo NO_DEPLOY_CHILD
ls -lt /home/arm/junyang/*.log 2>/dev/null | head -5
echo "DGXSTATUS1328B-END $(date '+%H:%M:%S %Z')"
