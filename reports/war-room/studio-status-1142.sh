#!/bin/bash
# Studio status direct read, run 2026-11-13 11:42 CST, marker STUDIOSTAT1142
# Read-only. Tailscale direct SSH (cloud-assistant dead, retired).
# NOTE: ProxyCommand must be passed literally on the ssh command line, not via a variable (quoting breaks).
MARK=STUDIOSTAT1142
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 "echo MARKER=$MARK; hostname; uptime; df -h /; ps aux | grep -iE 'python|train|sp-probe|selfplay|stage0|bigeval|cnet2' | grep -v grep | head -20; echo ---PS-END---; ls -lt ~/junyang/carbon/data/*.log 2>/dev/null | head -10; tail -2 ~/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -2 ~/junyang/carbon/data/fable_g1.log 2>/dev/null; echo ---NEW-FILES---; find ~/junyang/carbon/data -newermt '2026-11-13 11:33:00' -type f 2>/dev/null | head -20; echo ---PROBE---; ls /Users/arm/junyang/carbon/data/selfplay_probe/studio/replays/ 2>/dev/null | wc -l; tail -3 /tmp/sprobe_studio.log 2>/dev/null; echo ---WEIGHTS---; ls -lt ~/junyang/carbon/data/weights/ | head -6; echo END-MARKER=$MARK"
echo "EXIT=$?"
