#!/bin/bash
# War-room sync 19:33 follow-up: C2 correct-path progress + DGX archive file list.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1933-probe2.log
{
echo "WARROOMSYNC1933B PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== C2 spot31 correct paths ==="
timeout -k 30 120 ssh -F /home/hatch/.ssh/carbon_config spot31 'echo MARKER-C2K01933B; ls /root/carbon/data/k0_annot_c2/ 2>&1 | head; echo ---PROGRESS---; python3 -c "import json;d=json.load(open(\"/root/carbon/data/k0_annot_c2/progress.json\"));print(\"games-done:\",len(d.get(\"games\",{})));print(\"identity:\",d.get(\"identity\",{}).get(\"config_sha256\"));print(\"weights:\",d.get(\"identity\",{}).get(\"weights_md5\",\"-\"))" 2>&1; echo ---SHARDS---; ls /root/carbon/data/k0_annot_c2/shards/ 2>/dev/null | wc -l; ls -lat /root/carbon/data/k0_annot_c2/shards/ 2>/dev/null | head -5; echo ---END-C2B---' 2>&1
echo "C2 SSH-RC=$?"
echo "=== DGX archive listing ==="
timeout -k 30 120 ssh -F /home/hatch/.ssh/carbon_config dgx 'echo MARKER-DGX1933B; find /home/arm/junyang/l20_ppo_u_archive_20261114 -type f -printf "%s %p\n" 2>/dev/null | sort -k2; echo ---MANIFEST---; ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/*.md5 /home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST* 2>&1; echo ---END-DGXB---' 2>&1
echo "DGX SSH-RC=$?"
echo "WARROOMSYNC1933B PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
