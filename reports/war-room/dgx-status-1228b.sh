#!/bin/bash
# DGX status probe trip 12:28 follow-up - read-only. Marker DGXSTATUS1228B.
echo "DGXSTATUS1228B $(date '+%H:%M:%S %Z')"
echo "== GEN1 DONE PROOF =="
cat /home/arm/junyang/carbon/data/weights/cnet2_gen1g1_log.jsonl 2>&1
cat /home/arm/junyang/carbon/data/weights/cnet2_gen1r1_log.jsonl 2>&1
ls -l /home/arm/junyang/carbon/data/weights/ | grep -E 'gen1|heads_v2'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -name '*gen1*.log*' 2>/dev/null | head
echo "== HEADS V2 PROGRESS =="
ps -o pid,etime,cmd -p 566235 2>&1
ls -lt /home/arm/junyang/*.log 2>/dev/null | head -5
tail -n 8 /home/arm/junyang/train_heads.log 2>&1
echo "== HEAD BEHIND =="
cd /home/arm/junyang/carbon-repo && git rev-list --count HEAD..origin/main 2>&1; git log -1 --format='%H %ci' origin/main 2>&1; git log --oneline -4 origin/main 2>&1
echo "== LAST_STATS PUBLISHER =="
grep -rn 'last_stats *=' /home/arm/junyang/carbon-repo/code/ 2>/dev/null | head -8
grep -rn 'last_stats' /home/arm/junyang/carbon-repo/code/cnet2_search.py | grep -v getattr | head -8
echo "== DS COUNTS =="
python3 -c "
import json
for d in ['ds_carago','ds_fable','ds_m3s2','ds_gen1']:
    m=json.load(open(f'/home/arm/junyang/carbon/data/{d}/meta.json'))
    g=m.get('games',[])
    print(d,'games=',len(g),'samples=',m.get('samples',m.get('n_samples','?')))
" 2>&1
echo "== UNINDEXED REPLAYS =="
python3 -c "
import json,os
for c in ['carago_corpus','fable_corpus']:
    idx=json.load(open(f'/home/arm/junyang/carbon/data/{c}/index.json'))
    keys=set(idx.keys()) if isinstance(idx,dict) else set(x.get('key',x) for x in idx)
    files=set(os.listdir(f'/home/arm/junyang/carbon/data/{c}/replays'))
    print(c,'extra_files_not_in_index=',sorted(files-keys))
" 2>&1
echo "== GOLDEN README MD5 =="
md5sum /home/arm/junyang/golden/README.md
echo "== IT460 SEARCH =="
ls /home/arm/junyang/carbon-repo/data/ | grep -i 460; ls /home/arm/junyang/carbon/data/ | grep -i 460; echo IT460-SEARCH-DONE
echo "DGXSTATUS1228B-END $(date '+%H:%M:%S %Z')"
