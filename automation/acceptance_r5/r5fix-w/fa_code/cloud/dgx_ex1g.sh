#!/bin/bash
# DGX: round ex1g = all complete + cut-short ex1 corpora (3 opponents, ~1030 games); train B (0.3/100/3) and W05 (0.5/50/4),
# then pure-net paired exam of both on seeds 994000+100 x 3 opponents (control = ex1ex_cnet2_b128m2t3_*, same machine).
cd ~/junyang/carbon/data
R="ssh -i $HOME/.ssh/id_ed25519_carbon_relay -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
mkdir -p ex1g_in/h16 ex1g_in/h17 ex1g_in/h18
cp -r ex1f_in/h16/ex1_m2s ex1g_in/h16/; cp -r gen_store/h16/ex1_m2so ex1g_in/h16/
cp -r ex1f_in/h17/ex1_m2s ex1f_in/h17/ex1_m2so ex1g_in/h17/; cp -r gen_store/h17/ex1_scout1 ex1g_in/h17/
cp -r ex1f_in/h18/ex1_m2s ex1g_in/h18/
timeout 300 rsync -az -e "$R" root@121.196.97.62:/root/carbon/data/ex1_m2so ex1g_in/h18/
for d in ex1g_in/*/ex1_*; do echo $d $(ls $d/replays | grep -c json.gz) $(test -f $d/index.json && echo IDX); done > ex1g_inputs.txt
bash ~/junyang/dgx_sweep.sh ex1g 0:0:100:1 B:0.3:100:3 W05:0.5:50:4
bash ~/junyang/dgx_ex1x.sh ex1ex cnet2_b128ex1gB cnet2_b128ex1gW05
echo G_DONE > ex1g.done
