cd /Users/baggioustb/dev/carbon/carbon-league/code
until grep -q TRAIN_DONE ../data/v7_train.log; do sleep 30; done
K=$HOME/.ssh/carbon_aliyun_ed25519
rsync -az -e "ssh -i $K" ../data/weights/cnet2_b128v7.npz root@139.196.142.134:carbon/data/weights/
rsync -az -e "ssh -i $K" exp/cnet2_v7*.py root@139.196.142.134:carbon/code/exp/
cat > /tmp/claude-501/v7scr.sh <<'R'
#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
for c in v7s57_240 v7t5g v7t7g; do
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/submission_cnet2_v13.py 992000 200 c_${c}_v13 > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/submission_cnet2_v13.py 993000 200 c_${c}_v13b > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/opp_wb532.py 995000 200 c_${c}_wb532 > /dev/null 2>&1
  ../.venv/bin/python bigeval.py exp/cnet2_$c.py exp/cnet2_m2sstand.py 996000 200 c_${c}_m2ss > /dev/null 2>&1
done
echo DONE > ../data/v7scr.done
R
scp -i $K /tmp/claude-501/v7scr.sh root@139.196.142.134:carbon/code/cloud/v7scr.sh
ssh -i $K root@139.196.142.134 'rm -f ~/carbon/data/v7scr.done; cd ~/carbon/code; nohup bash cloud/v7scr.sh > /dev/null 2>&1 < /dev/null &'
until ssh -i $K root@139.196.142.134 '[ -f ~/carbon/data/v7scr.done ]'; do sleep 20; done
rsync -az -e "ssh -i $K" "root@139.196.142.134:carbon/data/bigeval_live/c_v7*" ../data/bigeval_live/
for c in v7s57_240 v7t5g v7t7g; do
  python3 ../data/paired.py c_${c}_v13+c_${c}_v13b c_v15pk_v13+c_v15pk_v13b | cut -c1-230
  python3 ../data/paired.py c_${c}_wb532 c_v15pk_wb532 | cut -c1-230
  python3 ../data/paired.py c_${c}_m2ss c_v15pk_m2ss | cut -c1-230
done
echo V7_SCREEN_DONE
