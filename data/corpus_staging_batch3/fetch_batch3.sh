#!/bin/bash
# Batch-3: remote md5 of the 48 resolved files + scp into staging.
set -u
STAGE=/home/hatch/workspace/carbon-live/corpus_staging_batch3
SSH_OPTS=(-i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")

cd "$STAGE" || exit 1
mapfile -t FILES < fetch_list.txt
echo "files to fetch: ${#FILES[@]}"

# 1) md5 table straight from DGX (source of truth)
timeout -k 30 240 ssh "${SSH_OPTS[@]}" arm@100.84.148.73 "cd /home/arm/junyang/carbon-repo/data && md5sum ${FILES[*]}" > dgx_md5.txt 2>dgx_md5.err
echo "dgx md5 rc=$? lines=$(wc -l < dgx_md5.txt)"
cat dgx_md5.err

# 2) scp the 48 files (only these)
fail=0
while read -r _ path; do
  lab="${path%%_corpus/*}_corpus"
  file="${path##*/}"
  timeout -k 30 120 scp "${SSH_OPTS[@]}" "arm@100.84.148.73:/home/arm/junyang/carbon-repo/data/$path" "$STAGE/$lab/$file" || { echo "SCP FAIL: $path"; fail=1; }
done < dgx_md5.txt
echo "scp done, fail=$fail"
find . -name '*.json.gz' | sort | wc -l
du -sb . | cut -f1
