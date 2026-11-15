echo "K4STUPRB1 STUDIO_BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,etime,command | grep -E 'cnet2_train|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SHIP ---'
ps -ax -o pid,etime,command | grep -E 'ship_m3s2wr1' | grep -v grep || echo NO-SHIP
cat ~/junyang/carbon/data/m3s2wr1_ship.log 2>/dev/null | tail -3
echo '--- WR1-LOG-TAIL ---'
ls -l --time-style=+%F_%T ~/junyang/carbon/data/m3s2wr1.log 2>/dev/null || ls -l ~/junyang/carbon/data/m3s2wr1.log
tail -4 ~/junyang/carbon/data/m3s2wr1.log 2>/dev/null
grep -h '"ep": 19' ~/junyang/carbon/data/m3s2wr1.log 2>/dev/null | tail -1
grep -h '^best' ~/junyang/carbon/data/m3s2wr1.log 2>/dev/null | tail -1
echo '--- WEIGHTS-WR1 ---'
ls -lt ~/junyang/carbon/data/weights/ | head -8
echo '--- CODE ---'
ls ~/junyang/carbon/code/ | grep -E 'cnet2|build_top|set_uw|ladder' 
echo '--- DATA-DIRS ---'
ls ~/junyang/carbon/data/ | grep -iE 'k4v2|ds_|corpus' | head -30
echo '--- DGX-REPO-K4V2-VIA-LAN ---'
timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=20 arm@10.188.0.23 'cd /home/arm/junyang/carbon-repo && git rev-parse --short HEAD; ls data/k4v2_corpus 2>/dev/null | wc -l; ls data/k4v2_corpus 2>/dev/null | head -5; python3 -c "import json;d=json.load(open(\"data/k4v2_matches.json\"));print(\"matches\",len(d));print(d[0])" 2>&1 | head -5; ls /home/arm/junyang/carbon/data/k4v2_corpus 2>/dev/null; ls /home/arm/junyang/carbon/data/ds_k4v2 2>/dev/null || echo NO-DS-K4V2-ON-DGX' < /dev/null
echo "K4STUPRB1 STUDIO_END"
