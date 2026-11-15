#!/bin/bash
# DGX status direct read, marker DGXSTATUS0713
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o ConnectTimeout=30)
timeout -k 30 420 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 <<'REMOTE_EOF'
echo MARKER=DGXSTATUS0713
hostname; uptime; date
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv
echo ---PS---
ps aux | grep -E 'python|rsync|git' | grep -v grep | head -40
echo ---IT85---
ls -l /home/arm/junyang/carbon-repo/code/exp/it85.py /home/arm/junyang/carbon-repo/data/weights/ckpt_85_frozen.npz 2>&1
md5sum /home/arm/junyang/carbon-repo/code/exp/it85.py 2>&1
echo ---CORPUS---
cd /home/arm/junyang/carbon-repo/data || exit 1
for d in hakimi3_corpus fable6_corpus hakimi3pro_corpus carago_corpus fable_corpus m3s2_corpus; do
  echo -n "$d replays="; ls $d/replays 2>/dev/null | wc -l
  echo -n "$d index="; python3 -c "import json;print(len(json.load(open('$d/index.json'))))" 2>/dev/null || echo NA
done
echo ---DS---
for f in ds_carago ds_fable; do echo -n "$f: "; python3 -c "import json;m=json.load(open('$f/meta.json'));print(m)" 2>&1 | head -2; done
echo ---GIT---
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD
echo ---END---
REMOTE_EOF
