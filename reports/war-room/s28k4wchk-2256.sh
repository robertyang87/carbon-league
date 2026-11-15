echo S28K4WCHK2256_BEGIN
date "+%F %T %Z"
ls -l /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz | grep -q "^344cfde939debb0de7f8834af069ea16 " && echo K4R1-FULL-MD5-OK || echo K4R1-MISMATCH
md5sum /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz | grep -q "^a594efd6cdb865d978e511c66d299206 " && echo K4G1-FULL-MD5-OK || echo K4G1-MISMATCH
echo "--- CHAIN ---"
cat /root/carbon/data/k4probe.status 2>/dev/null || echo NO-STATUS
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null || echo NO-LIVE-YET
echo S28K4WCHK2256_END
