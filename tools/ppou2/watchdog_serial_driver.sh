#!/bin/bash
# ppou2 serial-driver watchdog (cron, every 10 min).
# The driver is stateless (it derives schedule state from L20 on every poll),
# so a relaunch after a VM replacement resumes correctly. Guards ONLY the
# serial driver: the anchor / first-round watchers are supposed to exit once
# their event fires, and must NOT be relaunched.
cd /home/hatch/workspace/carbon-live/ppou2 || exit 1
if pgrep -f 'serial_drive[r].py' > /dev/null 2>&1; then
  exit 0
fi
echo "$(date '+%F %T %Z') watchdog: serial driver missing, relaunching" >> watchdog.log
TZ=Asia/Shanghai nohup python3 serial_driver.py > /dev/null 2>&1 < /dev/null &
