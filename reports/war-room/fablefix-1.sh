echo "== FABLEFIXQ7A BEGIN =="
date '+%F %T %Z'
timeout 480 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
DGXD=/home/arm/junyang/carbon/data
fail(){ echo "FABLEFIXQ7A FAIL: $1"; exit 1; }
echo "STUDIO $(hostname) $(date '+%F %T %Z')"
[ -d "$D" ] || fail "data dir missing: $D"
echo "-- DGX preflight --"
DGX_META=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "python3 -c \"import json;d=json.load(open('$DGXD/ds_fable/meta.json'));print(len(d['games']), d['total'])\"") || fail "DGX meta read failed"
echo "DGX_META games_total: $DGX_META"
DGX_COUNT=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "find $DGXD/ds_fable -type f | wc -l") || fail "DGX count failed"
DGX_COUNT=$(echo $DGX_COUNT | tr -d ' ')
echo "DGX_FILE_COUNT: $DGX_COUNT"
DGX_AGG=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "cd $DGXD/ds_fable && find . -type f | LC_ALL=C sort | xargs md5sum | awk '{print \$1, \$2}' | md5sum | awk '{print \$1}'") || fail "DGX agg md5 failed"
echo "DGX_AGG: $DGX_AGG"
echo "-- existing local ds_fable --"
if [ -e "$D/ds_fable" ]; then mv "$D/ds_fable" "$D/ds_fable.partial-$(date +%H%M%S)" && echo "moved partial aside"; else echo "none present"; fi
echo "-- tar transfer DGX-LAN -> Studio --"
set -o pipefail
ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 $DGX "tar -C $DGXD -cf - ds_fable" | tar -C "$D" -xf - || fail "tar transfer failed"
echo "TRANSFER_OK $(date '+%F %T %Z')"
echo "-- verify local --"
du -sh "$D/ds_fable"
[ -f "$D/ds_fable/meta.json" ] || fail "meta.json missing"
[ -f "$D/ds_fable/maps.npy" ] || fail "maps.npy missing"
LOCAL_META=$($HOME/junyang/carbon/.venv/bin/python -c "import json;d=json.load(open('$D/ds_fable/meta.json'));print(len(d['games']), d['total'], d['teachers'])" 2>/dev/null || python3 -c "import json;d=json.load(open('$D/ds_fable/meta.json'));print(len(d['games']), d['total'], d['teachers'])")
echo "LOCAL_META: $LOCAL_META"
echo "$LOCAL_META" | grep -q "^181 54119" || fail "meta mismatch: $LOCAL_META"
LOCAL_COUNT=$(find "$D/ds_fable" -type f | wc -l | tr -d ' ')
echo "LOCAL_FILE_COUNT: $LOCAL_COUNT"
[ "$LOCAL_COUNT" = "$DGX_COUNT" ] || fail "file count mismatch local=$LOCAL_COUNT dgx=$DGX_COUNT"
ST_AGG=$(cd "$D/ds_fable" && find . -type f | LC_ALL=C sort | xargs md5 -r | awk '{print $1, $2}' | md5 | awk '{print $NF}')
echo "STUDIO_AGG: $ST_AGG"
[ "$ST_AGG" = "$DGX_AGG" ] || fail "aggregate md5 mismatch studio=$ST_AGG dgx=$DGX_AGG"
echo "FABLEFIXQ7A PASS"
STEOF
echo "SSH_RC=$?"
echo "== FABLEFIXQ7A END =="
