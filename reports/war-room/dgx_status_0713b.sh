#!/bin/bash
# DGX follow-up: corpus layout check, marker DGXSTATUS0713B
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o ConnectTimeout=30)
timeout -k 30 240 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 <<'REMOTE_EOF'
echo MARKER=DGXSTATUS0713B
cd /home/arm/junyang/carbon-repo/data && pwd && ls
echo ---CORPUS_LAYOUT---
for d in hakimi3_corpus fable6_corpus hakimi3pro_corpus carago_corpus fable_corpus m3s2_corpus; do echo "== $d"; ls $d 2>&1 | head -5; echo "  files=$(find $d -type f 2>/dev/null | wc -l)"; done
echo ---DS_LAYOUT---
ls -d ds_carago ds_fable 2>&1; find ds_carago ds_fable -maxdepth 2 -type f 2>/dev/null | head -10
echo ---REMOTE_HEAD---
cd /home/arm/junyang/carbon-repo && timeout 60 git ls-remote origin main | head -2
echo ---END---
REMOTE_EOF
