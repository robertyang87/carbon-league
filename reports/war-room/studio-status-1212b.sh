#!/bin/bash
# Studio status VERIFICATION read, run 2026-11-13 12:1x CST, marker STUDIOSTAT1212B
# Read-only. Re-checks: process table (ps aux form), full data dir listing, locate teacher/heads logs anywhere under ~/junyang/carbon, weights dir.
MARK=STUDIOSTAT1212B
timeout -k 30 150 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 "echo MARKER=$MARK; date; echo ---PS-AUX---; ps aux | grep -iE 'python|stage0|teacher|train|heads|spgen|selfplay|cnet2' | grep -v grep; echo ---PS-AUX-END---; echo ---DATA-LS---; ls -la ~/junyang/carbon/data/ | head -60; echo ---FIND-TEACHER-HEADS---; find ~/junyang/carbon /tmp -maxdepth 4 \( -iname '*teacher*' -o -iname '*heads*' \) 2>/dev/null | head -30; echo ---WEIGHTS---; ls -lt ~/junyang/carbon/data/weights/ | head -10; echo ---FABLE-MTIME---; ls -l ~/junyang/carbon/data/fable_r1.log ~/junyang/carbon/data/fable_g1.log; echo ---ANY-PYTHON---; pgrep -fl python3; echo END-MARKER=$MARK"
echo "EXIT=$?"
