#!/bin/bash
# s26stage verify-2 1950: correct-path recheck of exp wrappers (they live under code/exp/,
# per s26xfer-dgx.sh which rsyncs h24 /root/carbon/code/ -> $ST/code/ and md5s
# $ST/code/exp/echo2.py) + git pull in /home/arm/junyang/carbon-repo.
# Read-only except the explicitly authorized git pull. Marker: S26VFY21950X
MARK=S26VFY21950X
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
S=/home/arm/junyang/s26stage
echo "S26VFY21950X DGX-BEGIN $(date '+%F %T %Z')"
echo '--- EXP-PATH-CHECK ---'
ls -d "$S/exp" 2>&1; ls -d "$S/code/exp" 2>&1
for f in code/exp/echo2.py code/exp/echo_e0.py; do
  if [ -f "$S/$f" ]; then echo -n "$f: "; md5sum "$S/$f" | awk '{print $1}'; else echo "$f: MISSING"; fi
done
echo '--- PYCOMPILE-CORRECTED ---'
"$S/.venv/bin/python" -m py_compile "$S/code/bigeval.py" "$S/code/eval_v53.py" "$S/code/exp/echo2.py" "$S/code/exp/echo_e0.py" && echo PYCOMPILE-OK
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- GIT-PULL ---'
cd /home/arm/junyang/carbon-repo 2>/dev/null || { echo REPO-DIR-MISSING; exit 1; }
git rev-parse --abbrev-ref HEAD
echo -n "HEAD-BEFORE: "; git rev-parse HEAD
echo -n "BEHIND-BEFORE: "; git rev-list --count HEAD..@{u} 2>&1
timeout 60 git pull 2>&1 | tail -5
echo "PULL-RC=$?"
echo -n "HEAD-AFTER: "; git rev-parse HEAD
echo -n "BEHIND-AFTER: "; git rev-list --count HEAD..@{u} 2>&1
git log -1 --format='%ci %s'
echo "S26VFY21950X DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
