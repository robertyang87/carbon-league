#!/bin/bash
echo "ACCEPT_BEGIN"
date "+%Y-%m-%d %H:%M:%S %Z"
echo "--- trainer proc ---"
pgrep -fl "cnet2_train.py" || echo "NO_TRAINER"
ps -Ao pid,%cpu,etime,command | grep "[c]net2_train" | head -3
echo "--- watcher proc ---"
pgrep -fl "ship_m3s2wr1" || echo "NO_WATCHER"
echo "--- log tail ---"
tail -6 ~/junyang/carbon/data/m3s2wr1.log
echo "--- log stat ---"
stat -f "mtime=%Sm size=%z" -t "%H:%M:%S" ~/junyang/carbon/data/m3s2wr1.log
echo "--- weights out ---"
ls -la ~/junyang/carbon/data/weights/cnet2_m3s2wr1* 2>/dev/null || echo "NO_EP_YET"
echo "--- ship log ---"
cat ~/junyang/carbon/data/m3s2wr1_ship.log 2>/dev/null || echo "NO_SHIP_LOG_YET"
echo "ACCEPT_END"
