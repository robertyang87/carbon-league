echo "K4STUPRB2 STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
echo '--- PROCS ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
pgrep -fl "ship_k4v2" || echo NO-SHIP-WATCHER
echo '--- EP0 LINES ---'
grep -h '"ep": 0, "val"' $D/k4v2r1.log | head -1
grep -h '"ep": 0, "val"' $D/k4v2g1.log | head -1
echo '--- LATEST ---'
tail -2 $D/k4v2r1.log; tail -2 $D/k4v2g1.log
ls -lt $D/weights/ | grep -i k4v2 | head -6
echo "K4STUPRB2 STUDIO_END"
