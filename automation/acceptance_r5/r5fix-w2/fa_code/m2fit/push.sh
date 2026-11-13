#!/bin/bash
# copy m2_bot and everything it imports to host 4
K=$HOME/.ssh/carbon_aliyun_ed25519
cd /Users/baggioustb/dev/carbon/carbon-league/code
ssh -i $K root@47.117.170.99 "mkdir -p /root/carbon/code/m2fit /root/carbon/code/mfit /root/carbon/code/exp"
scp -q -i $K m2_bot.py m2_bot_ctrees.json m2_bot_ptrees.json m2_bot_otrees.json root@47.117.170.99:/root/carbon/code/
scp -q -i $K m2fit/__init__.py m2fit/feats.py m2fit/stage.py m2fit/policy.py root@47.117.170.99:/root/carbon/code/m2fit/
ssh -i $K root@47.117.170.99 "test -f /root/carbon/code/mfit/extract_tgt.py" || scp -q -i $K mfit/extract.py mfit/extract_tgt.py root@47.117.170.99:/root/carbon/code/mfit/
scp -q -i $K exp/m2bot.py root@47.117.170.99:/root/carbon/code/exp/
