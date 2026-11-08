#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=62 SB=0.28 SH=0.41
B=../.venv/bin/python
$B bigeval.py exp/cnet2_v7s_mfxc.py exp/submission_cnet2_v13.py 992000 200 c_v7s_mfxc_v13 > ../data/cap_v13.out 2>&1
$B bigeval.py exp/cnet2_v7s_mfxc.py exp/opp_wb532.py 995000 200 c_v7s_mfxc_wb532 > ../data/cap_wb.out 2>&1
$B bigeval.py exp/cnet2_v7s_mfxc.py exp/cnet2_m2sstand.py 996000 200 c_v7s_mfxc_m2ss > ../data/cap_m2.out 2>&1
echo DONE > ../data/cap.done
