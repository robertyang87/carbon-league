#!/bin/bash
# DGX status probe trip 13:58 follow-up F - read-only. Marker DGXSTATUS1358F.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "DGXSTATUS1358F $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358F; cd /home/arm/junyang/carbon-repo && echo ---LOG-ORIGIN-MAIN---; git log --oneline -5 origin/main; echo ---LOG-LOCAL---; git log --oneline -3 HEAD; echo ---SHOW-NEW---; git show -s --format="%ci %s" 86cbbc3 478fd7e; echo ---LITE-V7-WORKDIR---; ls -la /home/arm/junyang/carbon/data/lite_v7_corpus/; stat -c "%y %n" /home/arm/junyang/carbon/data/lite_v7_corpus/index.json; echo ---INDEX-VS-FILES---; python3 -c "
import json,os,glob
idx=json.load(open(\"/home/arm/junyang/carbon/data/lite_v7_corpus/index.json\"))
ids=[e.get(\"id\",e) if isinstance(e,dict) else e for e in (idx if isinstance(idx,list) else idx.get(\"entries\",idx))]
files={os.path.basename(f).replace(\"-1.json.gz\",\"\") for f in glob.glob(\"/home/arm/junyang/carbon-repo/data/lite_v7_corpus/*.gz\")}
print(\"index entries:\",len(ids),\"repo files:\",len(files))
print(\"in-index-not-in-files:\",sorted(set(ids)-files))
print(\"in-files-not-in-index:\",sorted(files-set(ids)))
m=json.load(open(\"/home/arm/junyang/carbon-repo/data/lite-v7_matches.json\"))
print(\"matches.json:\",len(m))
"; echo ---DQ-LOGS---; ls -lt /home/arm/junyang/carbon/data/dq_logs/ 2>/dev/null | head -5; echo ---END-F---' 2>&1
echo "rcF=$?"
echo "DGXSTATUS1358F-END $(date '+%H:%M:%S %Z')"
