#!/bin/bash
# M9XJ three-arm pre-stage probe (2026-11-12 ~19:05): READ-ONLY except one repo pull on DGX.
# Purpose: count m9xj (it260) / m9xj320 (it320) games+corpus, verify teacher names
# character-exact, verify replica_chain.sh currency, check Studio readiness.
# Do NOT build any dataset, do NOT launch training, do NOT touch running processes.
# NOTE: DGX is busy with the M6 build (agent 264eab30, launched ~18:49); Studio has a
# NEW owner-ordered job (19:04): weighted control arm cnet2_m3s2wg1 -- this probe only
# READS whether wg1 has started; it does not dispatch or interfere with it.
# Marker: M9XJPRE1905A7B (claim only output containing this marker)
MARK=M9XJPRE1905A7B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- DGX (via ssh, timeout-wrapped) ---'
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M9XJPRE1905A7B DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- GPU + TRAIN PROCS (read-only; M6 build may be running -- DO NOT TOUCH) ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
ps -eo pid,etime,args | grep -E 'cnet2_train|replica_chain|build_top|ladder_corpus' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GIT PROCS (if any git fetch/pull alive, repo pull below is SKIPPED) ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- REPO PULL (carbon-repo, branch cnet2-search-v12) ---'
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && git log -1 --format='%h %ci %s' && git branch --show-current
echo -n 'remote tip: '; timeout 60 git ls-remote origin cnet2-search-v12 2>&1 || echo LSREMOTE-FAILED
if pgrep -f 'git (fetch|pull|checkout)' >/dev/null 2>&1; then
  echo 'PULL-SKIPPED git-busy-on-DGX (M6 agent may be pulling/building)'
else
  timeout 60 git pull --ff-only 2>&1 | tail -3 || echo PULL-FAILED
  echo -n 'HEAD after pull: '; git rev-parse HEAD; git log -1 --format='%h %ci %s'
fi
echo '--- REPLICA_CHAIN.SH currency ---'
git log -1 --format='%H %ci %s' -- code/replica_chain.sh
md5sum code/replica_chain.sh
ls -l code/replica_chain.sh code/build_top.py code/set_uw.py code/ladder_corpus.py 2>&1
echo '--- M9XJ COUNTS (repo) ---'
python3 - <<'PYEOF'
import json, glob, os
for tag in ('m9xj', 'm9xj320'):
    m = json.load(open(f'data/{tag}_matches.json'))
    ms = m if isinstance(m, list) else m.get('matches', m.get('games', []))
    print(tag, 'matches=', len(ms))
    names = {}
    for e in ms:
        st = e.get('strategies', {}) or {}
        for s in ('a', 'b'):
            v = st.get(s)
            nm = v.get('name') if isinstance(v, dict) else None
            if nm:
                names[nm] = names.get(nm, 0) + 1
    print(' names=', names)
    teach = [n for n in names if 'M9XJ' in n]
    print(' teacher_candidates=', teach)
    side = {'a': 0, 'b': 0}
    if teach:
        for e in ms:
            st = e.get('strategies', {}) or {}
            for s in ('a', 'b'):
                v = st.get(s)
                nm = v.get('name') if isinstance(v, dict) else None
                if nm == teach[0]:
                    side[s] += 1
    print(' teacher_side_as_a_b=', side)
    print(' sample_keys=', sorted(ms[0].keys()) if ms else None)
    print(' corpus_gz_files=', len(glob.glob(f'data/{tag}_corpus/*.json.gz')),
          ' corpus_index_exists=', os.path.exists(f'data/{tag}_corpus/index.json'))
PYEOF
echo '--- WORKDIR CORPUS (build input for replica_chain.sh) ---'
for d in m9xj_corpus m9xj320_corpus; do
  echo "== $d"
  echo -n 'replays='; ls /home/arm/junyang/carbon/data/$d/replays/*.json.gz 2>/dev/null | wc -l
  echo -n 'index_exists='; [ -f /home/arm/junyang/carbon/data/$d/index.json ] && echo yes || echo no
  ls -l /home/arm/junyang/carbon/data/$d/index.json 2>/dev/null || true
done
echo '--- DS_M3S2_W PRECEDENT (weighted-build method) ---'
ls -l /home/arm/junyang/carbon/data/ds_m3s2_w/meta.json 2>/dev/null || echo NO-DS-M3S2-W-META
python3 -c "import json;m=json.load(open('/home/arm/junyang/carbon/data/ds_m3s2_w/meta.json'));print('meta:',{k:m.get(k) for k in ('games','samples','total','rows','train','val') if k in m})" 2>/dev/null || true
find /home/arm/junyang/carbon -maxdepth 3 \( -iname '*weight*' -o -iname '*m3s2w*' -o -iname '*_w.py' \) 2>/dev/null | head -20
ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | grep -i 'm3s2w\|weight' || echo NO-WEIGHTED-BUILD-LOG
echo '--- DGX DISK ---'
df -h /home/arm/junyang/carbon /home/arm/junyang/carbon-repo 2>/dev/null
du -sh /home/arm/junyang/carbon/data/ds_* 2>/dev/null | sort -h | tail -12
echo "M9XJPRE1905A7B DGX-END"
DGXEOF
echo "DGX-SSH-RC=$?"
echo '--- Studio (via ssh, timeout-wrapped) ---'
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "M9XJPRE1905A7B STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- VENV ---'
/Users/arm/junyang/carbon/.venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())" 2>&1
ls -ld /Users/arm/junyang/carbon/.venv 2>&1
echo '--- TRAIN PROCS (read-only; wg1 job may have started -- DO NOT TOUCH) ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2_train|train' | grep -v grep || echo NO-TRAIN-PROC
ps -ax -o pid,etime,command | grep -E 'ship|scp|rsync' | grep -v grep | head -5 || echo NO-SHIP-PROC
echo '--- DISK ---'
df -h / | tail -1
echo '--- DS_K4V2 (prune-or-keep input) ---'
du -sh /Users/arm/junyang/carbon/data/ds_k4v2 2>/dev/null || echo NO-DS-K4V2
find /Users/arm/junyang/carbon/data/ds_k4v2 -type f 2>/dev/null | wc -l
echo '--- DATASETS + WEIGHTS ---'
du -sh /Users/arm/junyang/carbon/data/* 2>/dev/null | sort -h | tail -15
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -10
echo '--- LAUNCH SCRIPTS ON STUDIO ---'
ls -l /Users/arm/junyang/carbon/*.sh 2>/dev/null | head -10 || echo NO-LAUNCH-SCRIPTS
echo '--- DS_M3S2_W PRESENT? (wg1 job input) ---'
du -sh /Users/arm/junyang/carbon/data/ds_m3s2_w 2>/dev/null || echo NO-DS-M3S2-W-ON-STUDIO
ls -l /Users/arm/junyang/carbon/data/m3s2wg1.log /Users/arm/junyang/carbon/data/*wg1*.log 2>/dev/null || echo NO-WG1-LOG-YET
echo "M9XJPRE1905A7B STUDIO-END"
STEOF
echo "STUDIO-SSH-RC=$?"
echo "$MARK L20-END $(date '+%F %T %Z')"
