echo MARK_DGXSTATUS2243B
cd /home/arm/junyang/carbon-repo && timeout 280 git fetch origin main 2>&1 | tail -2; git rev-parse origin/main; git rev-list --count HEAD..origin/main; git log --oneline -3 origin/main
for d in m3s2_corpus carago_corpus fable_corpus; do echo "== $d"; ls /home/arm/junyang/carbon/data/$d/ | head -5; ls /home/arm/junyang/carbon/data/$d/ | wc -l; find /home/arm/junyang/carbon/data/$d -maxdepth 2 -type f | wc -l; done
wc -l /home/arm/junyang/carbon/data/bigeval_live/it74_master2.jsonl /home/arm/junyang/carbon/data/bigeval_live/it74_echo2.jsonl
ls -lt /home/arm/junyang/carbon/data/bigeval_live/ /home/arm/junyang/carbon/ 2>/dev/null | head -20
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 3 -name '*paired_du*' -o -maxdepth 3 -name '*it74*' -newermt '2026-11-14 19:00' 2>/dev/null | head -20
echo END_DGXSTATUS2243B
