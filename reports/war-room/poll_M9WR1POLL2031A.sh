echo "M9WR1POLL2031A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_poll_studio.sh <<'STEOF'
echo "M9WR1POLL2031A STUDIO $(date '+%T')"
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
if [ -d ~/junyang/carbon/data/ds_m9xj_w ]; then
  echo DS_W_PRESENT
  ls -la ~/junyang/carbon/data/ds_m9xj_w/ | head -25
  cat ~/junyang/carbon/data/ds_m9xj_w/meta.json 2>/dev/null || echo NO-META
else
  echo DS_W_ABSENT
fi
ls -lt ~/junyang/carbon/data/ | head -6
ls ~/junyang/carbon/data/ | grep -i -E 'm9xj|md5|manifest' || echo NO-M9XJ-FILES
STEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_poll_studio.sh
echo "studio_rc=$?"
cat > /tmp/m9wr1_poll_dgx.sh <<'DGEOF'
echo "M9WR1POLL2031A DGX $(date '+%T')"
cd /home/arm/junyang/carbon-repo && git log --oneline -1
ls /home/arm/junyang/carbon-repo/data/m9xj_corpus/ | wc -l
python3 -c "import json;print('matches:',len(json.load(open('/home/arm/junyang/carbon-repo/data/m9xj_matches.json'))))" 2>/dev/null || echo MATCHES-FAIL
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
ls -d /home/arm/junyang/carbon/data/ds_m9xj* 2>/dev/null || echo NO-DS-M9XJ
for d in /home/arm/junyang/carbon/data/ds_m9xj /home/arm/junyang/carbon/data/ds_m9xj_w; do
  if [ -d "$d" ]; then echo "== $d =="; ls "$d" | head; cat "$d/meta.json" 2>/dev/null; fi
done
ls -lt /home/arm/junyang/carbon/data/*.log 2>/dev/null | head -5
ls /home/arm/junyang/carbon/data/weights/ | grep -i m9xj || echo NO-M9XJ-WEIGHTS
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m9wr1_poll_dgx.sh
echo "dgx_rc=$?"
echo "M9WR1POLL2031A L20_END"
