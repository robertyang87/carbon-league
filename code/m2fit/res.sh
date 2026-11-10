#!/bin/bash
# res.sh TAG... : summary lines of finished runs on host 4 + live counts
K=$HOME/.ssh/carbon_aliyun_ed25519
for t in "$@"; do ssh -i $K root@47.117.170.99 "cd /root/carbon/data; echo -n \"$t live=\"; wc -l < bigeval_live/$t.jsonl 2>/dev/null; grep -h '\"tag\": \"$t\"' $t.out 2>/dev/null | cut -c1-400"; done
