echo K4PROBE2003_BEGIN
date "+%F %T %Z"
echo "--- K4R1_ON STAT + CONF ---"
ls -l --time-style=full-iso /root/carbon/code/exp/k4r1_on.py /root/carbon/code/exp/k4r1_off.py /root/carbon/code/exp/pk_k4r1.py
head -40 /root/carbon/code/exp/k4r1_on.py
echo "--- K4V2 PRESENT? ---"
ls -l /root/carbon/code/exp/*k4v2* 2>/dev/null || echo NO-K4V2-WRAPPERS
ls -l /root/carbon/data/weights/*k4v2* 2>/dev/null || echo NO-K4V2-WEIGHTS
echo "--- IDLE + RECENT LAUNCHERS ---"
ps -eo pid,args | grep bigeval | grep -v grep || echo NO-BIGEVAL-PROC
ls -lt /root/carbon/data/*.launcher.log 2>/dev/null | head -5
ls -lt /root/carbon/data/bigeval_live/ 2>/dev/null | head -8
cat /proc/loadavg
echo K4PROBE2003_END
