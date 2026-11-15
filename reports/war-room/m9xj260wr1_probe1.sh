echo "M9WR1PROBE1 L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_studio.sh <<'STEOF'
echo "M9WR1PROBE1 STUDIO_BEGIN $(hostname) $(date '+%F %T %Z')"
echo '--- trainer procs ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
echo '--- load ---'
uptime
echo '--- workdir ---'
ls ~/junyang/carbon/ 2>/dev/null
echo '--- data dir m9xj ---'
ls -la ~/junyang/carbon/data/ 2>/dev/null | grep -i m9xj || echo NO-M9XJ-IN-DATA
echo '--- all ds dirs ---'
ls -d ~/junyang/carbon/data/ds_* 2>/dev/null
echo '--- weights m9xj ---'
ls -lt ~/junyang/carbon/data/weights/ 2>/dev/null | grep -i m9xj || echo NO-M9XJ-WEIGHTS
echo '--- repo clones ---'
ls -d ~/junyang/carbon-repo ~/junyang/carbon/repo ~/carbon-league 2>/dev/null || echo NO-OBVIOUS-REPO
find ~/junyang -maxdepth 3 -name 'm9xj_matches.json' 2>/dev/null | head
find ~/junyang -maxdepth 4 -name '*m9xj*' 2>/dev/null | head -20
echo '--- disk ---'
df -h ~/junyang | tail -1
echo '--- recent files in data (top 12 by mtime) ---'
ls -lt ~/junyang/carbon/data/ 2>/dev/null | head -13
echo "M9WR1PROBE1 STUDIO_END"
STEOF
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_studio.sh
echo "studio_ssh_rc=$?"
cat > /tmp/m9wr1_dgx.sh <<'DGEOF'
echo "M9WR1PROBE1 DGX_BEGIN $(hostname) $(date '+%F %T %Z')"
echo '--- trainer procs ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || true
echo '--- carbon data m9xj ---'
ls -la /home/arm/junyang/carbon/data/ 2>/dev/null | grep -i m9xj || echo NO-M9XJ-IN-DATA
ls -d /home/arm/junyang/carbon/data/ds_m9xj* 2>/dev/null
echo '--- weights m9xj ---'
ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i m9xj | head
echo '--- repo state ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null && git log --oneline -3 && git rev-parse --abbrev-ref HEAD
ls /home/arm/junyang/carbon-repo/data/m9xj_corpus/ 2>/dev/null | wc -l
python3 -c "import json;d=json.load(open('/home/arm/junyang/carbon-repo/data/m9xj_matches.json'));print('matches entries:',len(d))" 2>/dev/null || echo MATCHES-READ-FAIL
echo '--- studio push artifacts ---'
ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -13
grep -rl "m9xj" /home/arm/junyang/carbon/data/*.log 2>/dev/null | head
echo "M9WR1PROBE1 DGX_END"
DGEOF
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m9wr1_dgx.sh
echo "dgx_ssh_rc=$?"
echo "M9WR1PROBE1 L20_END"
