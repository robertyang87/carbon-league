#!/bin/bash
# DGX status probe trip 12:43 - follow-up C: resolve HEAD. Marker DGXSTATUS1243C. Read-only.
echo "DGXSTATUS1243C $(date '+%H:%M:%S %Z')"
cd /home/arm/junyang/carbon-repo
echo "== BRANCH/STATUS =="
git branch --show-current 2>&1
git rev-parse HEAD 2>&1
git log --oneline -5 2>&1
git status --porcelain=v1 --branch 2>&1 | head -10
echo "== REMOTE NOW =="
timeout 30 git ls-remote origin HEAD 2>&1
echo "== AHEAD/BEHIND vs remote HEAD a9287a4 =="
git rev-list --count a9287a4ca3adf871651c74c6e05f350c07bcf1c7..HEAD 2>&1
git rev-list --count HEAD..a9287a4ca3adf871651c74c6e05f350c07bcf1c7 2>&1
echo "== RECENT REFLOG (last 5) =="
git reflog -5 2>&1
echo "== ANY PULL/FETCH RUNNING NOW =="
ps -eo pid,etime,cmd | grep -E 'git (fetch|pull|checkout|merge|rebase)' | grep -v grep || echo NO_GIT_PROC
echo "DGXSTATUS1243C-END $(date '+%H:%M:%S %Z')"
