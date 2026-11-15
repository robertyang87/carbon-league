#!/bin/bash
# Point apt at the PUBLIC Aliyun mirror (the internal cloud mirror endpoint
# 100.100.2.148 is unreachable from this spot instance) and install the
# NVIDIA 580 server driver for real. Runs ON the spot machine, detached.
set -u
LOG=/root/driver_install3.log
exec >> "$LOG" 2>&1
echo "=== $(date '+%F %T') APT DRIVER FIX START"
grep -rl "mirrors.cloud.aliyuncs.com" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | while read -r f; do
  sed -i 's/mirrors\.cloud\.aliyuncs\.com/mirrors.aliyun.com/g' "$f"
  echo "patched $f"
done
apt-get update -y
echo "--- update rc=$?"
DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-580-server
echo "--- install rc=$?"
dpkg -l | grep -E "nvidia-driver|nvidia-utils" | head -4
modprobe nvidia && echo MODPROBE_OK || echo MODPROBE_FAIL
nvidia-smi || echo NO_SMI_YET
echo "=== $(date '+%F %T') DONE"
