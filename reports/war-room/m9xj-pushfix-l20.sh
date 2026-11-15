#!/bin/bash
# Push ds_m9xj_w DGX -> Studio. 1) diagnose DGX->Studio direct ssh failure.
# 2) relay via L20 (established channels both hops). Marker: M9XJPUSH2050B
MARK=M9XJPUSH2050B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
DGXKEY=/root/.ssh/dgx_access
echo '--- L20 ssh keys ---'; ls -la /root/.ssh/ 2>/dev/null | head
echo '--- DGX-side direct-to-Studio diagnosis ---'
timeout -k 30 300 ssh -i $DGXKEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "dgx host $(hostname)"; ls ~/.ssh/ 2>/dev/null
echo '--- route/arp for Studio ZT 10.211.43.68 ---'
ip route get 10.211.43.68 2>&1 | head -2; arp -a 2>/dev/null | head -8
echo '--- ssh DGX->Studio verbose (first error lines) ---'
timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'echo DIRECT-OK $(hostname)' 2>&1 | tail -4
echo '--- try mDNS name ---'
timeout 45 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@mac-studio.local 'echo MDNS-OK $(hostname)' 2>&1 | tail -2
DGXEOF
echo '--- relay via L20 ---'
AGG='import hashlib,os,sys
root=sys.argv[1]; h=hashlib.md5()
for dp,dn,fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p=os.path.join(dp,f); h.update(os.path.relpath(p,root).encode())
        with open(p,"rb") as fh:
            for chunk in iter(lambda: fh.read(1<<20), b""): h.update(chunk)
print(h.hexdigest())'
rm -rf /root/m9xj_stage; mkdir -p /root/m9xj_stage
timeout -k 30 420 ssh -i $DGXKEY -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'tar cf - -C /home/arm/junyang/carbon/data ds_m9xj_w' | tar xf - -C /root/m9xj_stage && echo RELAY-HOP1-OK
L20AGG=$(python3 -c "$AGG" /root/m9xj_stage/ds_m9xj_w); echo "DSW-L20-AGGMD5 $L20AGG"
timeout -k 30 420 ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'echo STUDIO-REACHABLE $(hostname)' || { echo "$MARK STUDIO-UNREACHABLE-FROM-L20"; echo "$MARK L20-END"; exit 0; }
tar cf - -C /root/m9xj_stage ds_m9xj_w | timeout -k 30 600 ssh -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.68 'mkdir -p ~/junyang/carbon/data && tar xf - -C ~/junyang/carbon/data' && echo RELAY-HOP2-OK
STAGG=$(timeout -k 10 180 ssh -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.68 "~/junyang/carbon/.venv/bin/python -c \"$AGG\" ~/junyang/carbon/data/ds_m9xj_w" 2>&1 | tail -1)
echo "DSW-STUDIO-AGGMD5 $STAGG"
echo "DSW-DGX-AGGMD5 expected 9920540853a148f3e0252477bb5d9d51"
[ "$STAGG" = "9920540853a148f3e0252477bb5d9d51" ] && echo DSW-MD5-MATCH || echo DSW-MD5-MISMATCH
echo "$MARK L20-END rc=$?"
