#!/bin/bash
# DGX status probe trip 12:28 follow-up C - read-only. Marker DGXSTATUS1228C.
echo "DGXSTATUS1228C $(date '+%H:%M:%S %Z')"
echo "== HEADS V2 REAL LOG =="
tail -n 4 /home/arm/junyang/train_heads_v2.log 2>&1
ps -o pid,etime,cmd -p 566235 2>&1
ls -l /home/arm/junyang/carbon/data/weights/ | grep heads_v2 || echo NO_HEADS_V2_WEIGHT_YET
echo "== GATE PILOT =="
cat /home/arm/junyang/gate_pilot_greedy.log 2>&1
ls -l /home/arm/junyang/gate_pilot_*.log 2>&1
ps -eo pid,etime,cmd | grep -i 'gate_pilot\|994000' | grep -v grep || echo NO_GATE_PILOT_PROC
echo "== GEN1 TRAIN-DONE LINES =="
grep -E 'TRAIN-DONE|DONE|Error|Traceback' /home/arm/junyang/carbon/data/dq_logs/gen1g1.log /home/arm/junyang/carbon/data/dq_logs/gen1r1.log 2>&1 | tail -6
tail -n 2 /home/arm/junyang/carbon/data/dq_logs/gen1g1.log 2>&1
tail -n 2 /home/arm/junyang/carbon/data/dq_logs/gen1r1.log 2>&1
echo "== LAST_STATS FIX COMMIT? =="
cd /home/arm/junyang/carbon-repo && git log --oneline --grep=last_stats -5 2>&1; git log --oneline -3 -- code/awr_us_env.py code/m_rules.py 2>&1
echo "== M9XJ460 =="
ls /home/arm/junyang/carbon-repo/data/m9xj460_corpus 2>/dev/null | wc -l
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon-repo/data/m9xj460_matches.json'));print('m9xj460_matches_entries=',len(d))" 2>&1
ls /home/arm/junyang/carbon/data/ | grep -i m9xj460 || echo NO_M9XJ460_WORKDIR
echo "== UNINDEXED EXTRAS =="
python3 -c "
import json,os
for c in ['carago_corpus','fable_corpus']:
    idx=json.load(open(f'/home/arm/junyang/carbon/data/{c}/index.json'))
    names=set()
    for e in (idx if isinstance(idx,list) else idx.values()):
        if isinstance(e,dict):
            v=e.get('key') or e.get('file') or e.get('id') or ''
            names.add(os.path.basename(str(v)))
        else:
            names.add(os.path.basename(str(e)))
    files=set(os.listdir(f'/home/arm/junyang/carbon/data/{c}/replays'))
    print(c,'extra=',sorted(files-names),'missing_from_disk=',sorted(names-files)[:5])
" 2>&1
echo "DGXSTATUS1228C-END $(date '+%H:%M:%S %Z')"
