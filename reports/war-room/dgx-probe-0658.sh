#!/bin/bash
# DGX direct-read probe, trip marker DGXSTATUS0658. Read-only.
echo "MARKER DGXSTATUS0658 START $(date '+%F %T %Z')"
hostname
uptime
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo "--- procs (train/pack/pull/git/fetch) ---"
ps -eo pid,etime,args | grep -Ei 'python|train|pack|pull_gap|git |rsync|scp' | grep -v grep || echo "NO_MATCHING_PROCS"
echo "--- git fetch/pull/checkout procs ---"
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo "NO_GIT_NET_PROCS"
cd /home/arm/junyang/carbon-repo || { echo "REPO_CD_FAIL"; exit 1; }
echo "--- repo HEAD ---"
git rev-parse HEAD
git log -1 --format='%H %ci %s'
echo "--- remote HEAD (timeout 60) ---"
timeout 60 git ls-remote origin HEAD || echo "LS_REMOTE_FAIL"
echo "--- it85 pack ---"
ls -la data/weights/ckpt_85_frozen.npz code/exp/it85.py 2>&1
md5sum code/exp/it85.py 2>&1
echo "--- corpora counts ---"
for d in data/m3s2_corpus data/carago_corpus data/fable_corpus data/hakimi3_corpus data/fable6_corpus data/hakimi3pro_corpus data/k4v3_corpus data/krypton_corpus; do
  if [ -d "$d" ]; then
    echo "== $d"
    echo -n "  replay files: "; find "$d" -type f | wc -l
    if [ -f "$d/index.json" ]; then python3 -c "import json;d=json.load(open('$d/index.json'));print('  index entries:', len(d) if isinstance(d,list) else d.get('count', list(d.keys())[:3]))" 2>&1; fi
  fi
done
echo "--- datasets meta ---"
for m in data/ds_carago/meta.json data/ds_fable/meta.json; do
  if [ -f "$m" ]; then echo "== $m"; python3 -c "import json;print(json.load(open('$m')))" 2>&1 | head -c 600; echo; fi
done
echo "--- gap manifests ---"
ls -la /home/arm/junyang/pull0030_hakimi3_gap.manifest /home/arm/junyang/pull0030_fable6_gap.manifest /home/arm/junyang/pull0030_hakimi3pro_gap.manifest /home/arm/junyang/pull0030_k4v3_gap.manifest /home/arm/junyang/pull0030_krypton_gap.manifest 2>&1
echo "--- recent changes (60 min, data + code/exp) ---"
find data code/exp -newermt '60 minutes ago' -type f 2>/dev/null | head -30
echo "--- newest npz ---"
ls -lt data/weights/*.npz 2>/dev/null | head -5
echo "MARKER DGXSTATUS0658 END $(date '+%F %T %Z')"
