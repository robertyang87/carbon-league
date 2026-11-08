cd /Users/baggioustb/dev/carbon/carbon-league/code
until grep -q TRAIN_DONE ../data/v6_train.log; do sleep 30; done
K=$HOME/.ssh/carbon_aliyun_ed25519
rsync -az -e "ssh -i $K" ../data/weights/cnet2_b128v6.npz root@106.15.11.63:carbon/data/weights/
rsync -az -e "ssh -i $K" exp/cnet2_v6t*g.py root@106.15.11.63:carbon/code/exp/
ssh -i $K root@106.15.11.63 'cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=32; for t in 7 3 5 6; do for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"; ../.venv/bin/python bigeval.py exp/cnet2_v6t${t}g.py exp/$f.py $s 200 c_v6t${t}g_$n > /dev/null 2>&1; done; done'
rsync -az -e "ssh -i $K" "root@106.15.11.63:carbon/data/bigeval_live/c_v6t*" ../data/bigeval_live/
for t in 7 3 5 6; do for n in v13 wb532; do python3 ../data/paired.py c_v6t${t}g_$n c_v5t7g_$n; done; done
echo V6_SCREEN_DONE
