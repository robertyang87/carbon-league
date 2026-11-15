#!/bin/bash
# DGX status probe B - read-only. Marker: DGXSTATUS1143B
echo "MARK DGXSTATUS1143B"
echo "--- GEN1 LOGS ---"
cat /home/arm/junyang/katago_chain_gen1.log 2>/dev/null || echo NO_GEN1_LOG
cat /home/arm/junyang/katago_chain_gen1b.log 2>/dev/null || echo NO_GEN1B_LOG
echo "--- GEN1 OUTPUT ---"
ls /home/arm/junyang/carbon/data/gen1_shard0 2>/dev/null | wc -l
ls -lt --time-style=full-iso /home/arm/junyang/carbon/data/gen1_shard0 2>/dev/null | head -5
python3 -c "import json,glob;fs=glob.glob('/home/arm/junyang/carbon/data/gen1_shard0/*.json');print('json_files',len(fs))" 2>/dev/null || echo NO_GEN1_JSON
echo "--- SNAPLOOP PRECISE ---"
ps -eo pid,etime,args | grep -i snaploop | grep -v grep || echo NO_SNAPLOOP_PROC
ps -eo pid,etime,args | grep -E 's29-carago' | grep -v grep || echo NO_S29_CARAGO
echo "--- REPO-SIDE NEW BATCHES ---"
for d in data/carago1_corpus data/godslayer_cvi_corpus data/m9xj320_corpus data/lite_v7_corpus; do
  echo "repo/$d files=$(ls /home/arm/junyang/carbon-repo/$d 2>/dev/null | wc -l) index=$(python3 -c "import json;print(len(json.load(open('/home/arm/junyang/carbon-repo/$d/index.json'))))" 2>/dev/null || echo NO_INDEX)"
done
echo "--- END DGXSTATUS1143B ---"
