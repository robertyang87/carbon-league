#!/bin/bash
echo "N1RT2B8Q_BEGIN"
P=/root/carbon/data/packs/n1r_int8.py
echo "== bigeval.py size =="; wc -l /root/carbon/code/bigeval.py
echo "== bigeval.py full =="; cat /root/carbon/code/bigeval.py
echo "== pack entry points =="
grep -n "^def agent\|^agent \|^agent=\|def make_agent\|^class \|return _AG\|_AG(" "$P" | head -20
echo "== pack last 50 lines =="; tail -50 "$P"
echo "== env/engine modules =="
ls /root/carbon/code/ | grep -iE "env|game|engine|sim" | head -20
echo "N1RT2B8Q_END"
