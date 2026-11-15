#!/bin/bash
# L20 wrapper: install m9xj chain script on DGX and launch it detached.
# Marker: M9XJCHN2026A
MARK=M9XJCHN2026A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'OUTER'
cat > /home/arm/junyang/carbon/m9xj_chain.sh <<'CHAINSCRIPT'
__CHAIN_BODY__
CHAINSCRIPT
chmod +x /home/arm/junyang/carbon/m9xj_chain.sh
mkdir -p /home/arm/junyang/carbon/data/dq_logs
if pgrep -f 'm9xj_chain[.]sh' >/dev/null; then echo "M9XJCHN2026A CHAIN-ALREADY-RUNNING"; pgrep -af 'm9xj_chain[.]sh'; exit 0; fi
nohup bash /home/arm/junyang/carbon/m9xj_chain.sh >> /home/arm/junyang/carbon/data/dq_logs/m9xj_chain.log 2>&1 < /dev/null &
echo "M9XJCHN2026A CHAIN-LAUNCHED pid $!"
sleep 6
tail -8 /home/arm/junyang/carbon/data/dq_logs/m9xj_chain.log
OUTER
echo "$MARK L20-END rc=$?"
