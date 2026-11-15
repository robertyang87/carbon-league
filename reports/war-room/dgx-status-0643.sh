#!/bin/bash
# DGX status probe 0643 — read-only. Tailscale SSH from this VM to DGX.
# Cloud-assistant channel is RETIRED — never use it; only this direct read.
echo "DGXSTATUS0643 L20-BEGIN $(date -u +%FT%TZ)"
timeout -k 30 420 ssh -o ConnectTimeout=30 -o BatchMode=yes -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null arm@100.84.148.73 'bash -s' <<'DGXEOF'
echo "DGXSTATUS0643 DGX-BEGIN"
hostname
uptime
date
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "--- load ---"
cat /proc/loadavg
echo "--- training procs ---"
ps -eo pid,etime,cmd | grep -E "rl2|ppo|train|rollout|build_top|cnet2|selfplay|bigeval|fastsim" | grep -v grep || echo NOPROC
echo "--- git residue ---"
ps -eo pid,etime,cmd | grep -E "git (fetch|pull|checkout)" | grep -v grep || echo NOGIT
echo "--- pull driver procs ---"
ps -eo pid,etime,cmd | grep -E "pull_gap|ladder|gap" | grep -v grep || echo NOPULL
echo "--- all python procs ---"
ps -eo pid,etime,cmd | grep python3 | grep -v grep | head -20 || echo NOPY
cd /home/arm/junyang/carbon 2>/dev/null || echo "NOWORKDIR"
pwd
echo "--- repo corpus gz counts ---"
for c in k4v3 krypton hakimi3 fable6 hakimi3pro lite-v8 m3s2 carago fable m3s3z; do
  d="/home/arm/junyang/carbon-repo/data/${c}_corpus"
  if [ -d "$d" ]; then echo "$c repo_gz=$(ls $d/*.gz 2>/dev/null | wc -l)"; else echo "$c repo_MISSING"; fi
done
echo "--- gap-pull progress (hakimi3/fable6/hakimi3pro) ---"
ls -la /home/arm/junyang/pull_gap* /home/arm/junyang/pull0031* /home/arm/junyang/pull0030* 2>/dev/null | tail -30
echo "--- gap-pull logs tail ---"
for f in /home/arm/junyang/pull_gap*.log /home/arm/junyang/pull_gap_hakimi*.log /home/arm/junyang/pull0031*.log; do
  [ -f "$f" ] && { echo "== $f =="; tail -5 "$f"; }
done
echo "--- manifests ---"
ls -la /home/arm/junyang/*.manifest 2>/dev/null | tail -10
for m in /home/arm/junyang/pull0031_*.manifest /home/arm/junyang/pull_gap_*.manifest; do
  [ -f "$m" ] && echo "$m lines=$(wc -l < "$m") tail=$(tail -1 "$m")"
done
echo "--- workdir corpus counts ---"
for c in m3s2_corpus carago_corpus fable_corpus; do
  if [ -d "data/$c" ]; then
    idx=$(python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo ERR)
    rep=$(ls data/$c/replays 2>/dev/null | wc -l)
    echo "$c idx=$idx replays=$rep"
  else
    echo "$c MISSING"
  fi
done
echo "--- ds meta ---"
for d in ds_carago ds_fable; do
  if [ -f "data/$d/meta.json" ]; then
    echo "$d $(stat -c '%y' data/$d/meta.json | cut -c1-16) $(python3 -c "import json;m=json.load(open('data/$d/meta.json'));print('games=',len(m.get('games',[])),'total=',m.get('total'))" 2>/dev/null || echo ERR)"
  else
    echo "$d MISSING"
  fi
done
echo "--- newest weights ---"
find /home/arm/junyang -name "*.npz" -type f -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -5
echo "--- files changed last 60m ---"
find /home/arm/junyang/carbon /home/arm/junyang/s26stage -type f -mmin -60 2>/dev/null | head -30
echo "--- git HEAD ---"
cd /home/arm/junyang/carbon-repo 2>/dev/null && git rev-parse HEAD && git branch --show-current && timeout 60 git ls-remote origin HEAD || echo "NOGITREPO"
echo "DGXSTATUS0643 DGX-END"
DGXEOF
echo "DGXSTATUS0643 L20-END $(date -u +%FT%TZ)"
