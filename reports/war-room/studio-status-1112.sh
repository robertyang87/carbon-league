echo "MARK STUDIOSTAT1112"
date '+%Y-%m-%d %H:%M:%S %Z'
hostname; uptime
echo "--- mem/disk ---"
vm_stat | head -8; df -h / | tail -1
echo "--- training procs ---"
ps aux | grep -iE "python|sp-probe|train|smoke|bigeval|selfplay|carbon" | grep -v grep | head -20
echo "--- probe script / weights / datasets ---"
ls -la ~/junyang/carbon/ 2>/dev/null | head -30
ls ~/junyang/carbon/*.sh ~/junyang/carbon/**/sp-probe* 2>/dev/null
ls -la ~/junyang/carbon/data/ 2>/dev/null | head -40
md5 ~/junyang/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null || echo "m3s2r1L.npz ABSENT"
ls ~/junyang/carbon/data/ds_m3s2 2>/dev/null | head -5; ls ~/junyang/carbon/data/ds_fable 2>/dev/null | head -5
echo "--- fable logs ---"
tail -3 ~/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -3 ~/junyang/carbon/data/fable_g1.log 2>/dev/null
echo "--- hakimi2 ep19 ---"
ls -la --time-style=long-iso ~/junyang/carbon/data/*hakimi* 2>/dev/null | head; find ~/junyang/carbon -name "*hakimi2*ep19*" -o -name "*ep19.npz" 2>/dev/null | head -10
ls -lt ~/junyang/carbon/data/ 2>/dev/null | head -8
echo "MARK STUDIOSTAT1112 END"
