echo "M320DONE2240A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_done_studio.sh <<'STEOF'
echo "M320DONE2240A STUDIO $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
pgrep -fl 'cnet2_train.py' || echo NO-TRAINER-RUNNING
echo '--- all val lines ---'
grep '"val"' $D/m9xj320wr1.log | tail -25
echo '--- weights ---'
ls -lt $D/weights/ | grep -i m9xj320wr1 | head -8
echo '--- ship log ---'
cat $D/m9xj320wr1_ship.log 2>/dev/null || echo NO-SHIP-LOG-YET
echo '--- local md5s ---'
for f in $D/weights/cnet2_m9xj320wr1.npz $D/weights/cnet2_m9xj320wr1_ep19.npz; do [ -f "$f" ] && md5 -q "$f" && echo "  ^ $f"; done
STEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m320_done_studio.sh
echo "studio_rc=$?"
cat > /tmp/m320_done_dgx.sh <<'DGEOF'
echo "M320DONE2240A DGX $(date '+%F %T %Z')"
W=/home/arm/junyang/carbon/data/weights
ls -la $W/cnet2_m9xj320wr1_fromstudio.npz 2>/dev/null && md5sum $W/cnet2_m9xj320wr1_fromstudio.npz || echo FROMSTUDIO-MISSING
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320_done_dgx.sh
echo "dgx_rc=$?"
echo "M320DONE2240A L20_END"
