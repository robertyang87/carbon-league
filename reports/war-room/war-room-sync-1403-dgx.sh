#!/bin/bash
# War-room sync v144 trip 14:03, 2026-11-15 — targeted DGX distill-cell-4 completion check (read-only).
# Cell lr0.0002_s400_b1_seed7 was update 383/400 at 13:59, est done ~14:02-14:03.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "WARROOMSYNC1403 $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-WARROOMSYNC1403; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader; echo ---DISTILL-PROC---; ps -eo pid,etime,args | grep -E "k0_distill|k0_calib_driver" | grep -v grep || echo NO_DISTILL_PROC; echo ---CELL4-NPZ---; ls -la --time-style=long-iso /home/arm/junyang/k0_calib/cells/lr0.0002_s400_b1_seed7.npz 2>&1; echo ---TRAIN-LOG-TAIL---; tail -5 /home/arm/junyang/k0_calib/train_lr0.0002_s400_b1_seed7.log 2>&1; echo ---GATE2---; ps -eo pid,etime,args | grep -E "gate2_driver" | grep -v grep | wc -l; ls -lt /home/arm/junyang/katago_probe_scratch/gatesA/gate2_log_*.txt 2>/dev/null | head -3; echo ---DRIVER-LOG-TAIL---; tail -3 /home/arm/junyang/k0_calib/driver.log 2>&1; echo ---END---' 2>&1
echo "rc=$?"
