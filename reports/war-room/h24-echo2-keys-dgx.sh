echo H24E2KEY1952_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGX-BEGIN $(date '+%F %T %Z')"
ls -la ~/.ssh/
echo "--- PUBKEY FINGERPRINTS ---"
for p in ~/.ssh/*.pub; do [ -e "$p" ] && ssh-keygen -lf "$p"; done
echo "--- TRY KEYS TO H24 ---"
for k in ~/.ssh/id_* ~/.ssh/*_access ~/.ssh/carbon*; do
  case "$k" in *.pub) continue;; esac
  [ -f "$k" ] || continue
  if timeout 20 ssh -i "$k" -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new root@47.117.16.90 'echo KEY-OK' 2>/dev/null; then echo "WORKS: $k"; fi
done
echo "DGX-END"
DGXEOF
echo "L20-END rc=$?"
echo H24E2KEY1952_END
