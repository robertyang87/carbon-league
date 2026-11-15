#!/bin/bash
echo "N1RT1A7K_BEGIN"
echo "== host =="; hostname; date "+%F %T %Z"; nproc
echo "== baseline field BEFORE (do not touch) =="
wc -l /root/carbon/data/bigeval_live/e0_m2_base.jsonl 2>&1
ps -p 119669 -o pid,etime,args --no-headers 2>&1 | head -2
echo "== locate n1r pack =="
ls -la /root/carbon/data/packs/ 2>&1 | head -30
find /root/carbon -name "*n1r*" -not -path "*/node_modules/*" 2>/dev/null | head -20
for f in /root/carbon/data/packs/n1r_int8.py /root/carbon/code/exp/n1r_int8.py; do
  if [ -f "$f" ]; then echo "FOUND $f"; md5sum "$f"; ls -la "$f"; fi
done
echo "== exp dir =="
ls /root/carbon/code/exp/ 2>&1 | head -60
echo "== code dir =="
ls /root/carbon/code/ 2>&1 | head -80
echo "== bigeval.py head/imports =="
grep -n "^import\|^from\|importlib\|def main\|def play\|def run_game" /root/carbon/code/bigeval.py 2>&1 | head -40
echo "== pack interface =="
P=$(find /root/carbon -name "n1r_int8.py" 2>/dev/null | head -1)
echo "pack=$P"
if [ -n "$P" ]; then
  grep -n "^class \|^def \|CONF\s*=\|def act\|def __init__\|def reset" "$P" | head -30
  echo "-- pack head --"; head -40 "$P"
fi
echo "== echo_e0.py head =="
head -30 /root/carbon/code/exp/echo_e0.py 2>&1
echo "== constraint mechanisms =="
which systemd-run taskset nice 2>&1
ls -d /sys/fs/cgroup 2>&1; cat /sys/fs/cgroup/cgroup.controllers 2>&1
systemd-run --version 2>&1 | head -1
echo "== venv python =="
/root/carbon/.venv/bin/python -c "import sys, numpy; print(sys.version.split()[0], 'numpy', numpy.__version__)" 2>&1
free -m | head -2
echo "N1RT1A7K_END"
