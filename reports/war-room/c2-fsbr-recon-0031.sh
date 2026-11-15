echo C2FSBR0031_BEGIN
hostname; uptime; date "+%F %T %Z"
echo "--- layout ---"
ls /root/carbon/code/ 2>/dev/null | head -40 || echo NO-ROOT-CARBON
echo "--- fastsim locations ---"
find /root/carbon -maxdepth 3 -name fastsim.py 2>/dev/null
echo "--- md5 gate ---"
md5sum /root/carbon/code/fastsim.py /root/carbon/code/packs/fastsim.py /root/carbon/code/bigeval.py /root/carbon/code/eval_v53.py 2>&1
echo "--- code version ---"
cat /root/carbon/code/CODE_VERSION 2>/dev/null || echo NO-CODE-VERSION-FILE
echo "--- exp snap ---"
ls /root/carbon/code/exp/ 2>/dev/null | grep -i snap || echo NO-SNAP-EXP
ls /root/carbon/code/exp/ 2>/dev/null | head -25
echo "--- procs ---"
ps -eo pid,etime,args | grep bigeval | grep -v grep || echo NO-BIGEVAL-PROC
echo "--- old live + packs ---"
ls -l /root/carbon/data/bigeval_live/ 2>/dev/null | grep -i snap
wc -l /root/carbon/data/bigeval_live/snap_70_echo2.jsonl 2>/dev/null || echo NO-OLD-LIVE
find /root/carbon -maxdepth 4 -name "*snap_70*" 2>/dev/null | head
echo C2FSBR0031_END
