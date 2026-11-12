cd /Users/baggioustb/dev/carbon/carbon-league/code
K=$HOME/.ssh/carbon_aliyun_ed25519
while ssh -i $K root@139.196.142.134 'pgrep -f "[b]igeval.py exp/cnet2_.*opp_1cz_v9" >/dev/null'; do sleep 20; done
rsync -az -e "ssh -i $K" "root@139.196.142.134:carbon/data/bigeval_live/c_*_v9.jsonl" ../data/bigeval_live/
python3 ../data/paired.py c_v6t5g_v9 c_v5t7g_v9 | cut -c1-260
ssh -i $K root@139.196.142.134 'cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=24; for L in 240 200 270; do for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"; ../.venv/bin/python bigeval.py exp/cnet2_v6s57_$L.py exp/$f.py $s 200 c_v6s57_${L}_$n > /dev/null 2>&1; done; done'
rsync -az -e "ssh -i $K" "root@139.196.142.134:carbon/data/bigeval_live/c_v6s57_*" ../data/bigeval_live/
for L in 240 200 270; do for n in v13 wb532; do python3 ../data/paired.py c_v6s57_${L}_$n c_v5t7g_$n | cut -c1-260; done; done
echo S57_DONE
