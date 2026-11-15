#!/bin/bash
# Studio status direct read, run 2026-11-13 11:27 CST, marker STUDIOSTAT1127
# Read-only. Tailscale direct SSH (cloud-assistant dead, retired).
# NOTE: ProxyCommand must be passed literally on the ssh command line, not via a variable (quoting breaks).
MARK=STUDIOSTAT1127
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 "echo MARKER=$MARK; hostname; uptime; df -h /; ps aux | grep -iE 'python|train|sp-probe|selfplay|stage0|bigeval|cnet2' | grep -v grep | head -20; ls -la ~/junyang/carbon/sp_probe_studio.sh ~/junyang/carbon/sp-probe-studio.sh 2>&1; ls -lt ~/junyang/carbon/data/weights/ | head -8; md5 -q ~/junyang/carbon/data/weights/cnet2_m3s2r1L.npz 2>/dev/null; ls ~/junyang/carbon/data/ | head -20; ls -lt ~/junyang/carbon/data/*.log | head -8; tail -3 ~/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -3 ~/junyang/carbon/data/fable_g1.log 2>/dev/null; ls /Users/arm/junyang/carbon/data/selfplay_probe/studio/replays/ 2>/dev/null | wc -l; tail -5 /tmp/sprobe_studio.log 2>/dev/null; echo END-MARKER=$MARK"
echo "EXIT=$?"
