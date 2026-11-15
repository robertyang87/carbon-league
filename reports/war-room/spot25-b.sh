#!/bin/bash
# spot25 step B probe: L20 -> DGX -> read h24 wrappers + sizes, check DGX->spot25 ssh.
MARK="SPOT25B8Q3"
echo "MARKER $MARK START"
date '+%F %T %Z'
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
set -u
H24=47.117.16.90
S25=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "--- H24 REACH ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@"$H24" 'hostname; uptime'
echo "--- S25 REACH FROM DGX ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new root@"$S25" 'hostname; uptime'
echo "--- ECHO2 WRAPPER (full) ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'cat /root/carbon/code/exp/echo2.py'
echo "--- E0 + M3S2R1 WRAPPER WEIGHT LINES ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'grep -n "npz\|WEIGHT\|weights" /root/carbon/code/exp/echo_e0.py /root/carbon/code/exp/m3s2r1.py'
echo "--- SIZES ON H24 ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'du -sh /root/carbon/code /root/carbon/.venv /root/carbon/data/weights 2>/dev/null; ls -la /root/carbon/data/weights/ | head -30'
echo "--- DGX WEIGHTS (for cross-check) ---"
ls -la /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -30
md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null || echo "NO-M3S2R1-ON-DGX"
DGXEOF
echo "DGX-SSH-RC=$?"
echo "MARKER $MARK END"
