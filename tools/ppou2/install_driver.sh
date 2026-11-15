#!/bin/bash
# NVIDIA driver install for the bare-image spot L20 (Ubuntu 24.04).
# Runs ON the target. Matches gpu-l20-6: driver 580.126.09 (580 branch).
# Prints DRIVER_OK at the end, or DRIVER_NEEDS_REBOOT if modprobe fails.
set -ux
export DEBIAN_FRONTEND=noninteractive
echo "blacklist nouveau" > /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" >> /etc/modprobe.d/blacklist-nouveau.conf
apt-get update -y
apt-get install -y linux-headers-$(uname -r) build-essential dkms
# Prefer the server (headless) flavour; fall back to the desktop metapackage.
apt-get install -y nvidia-driver-580-server || apt-get install -y nvidia-driver-580
update-initramfs -u
modprobe -r nouveau 2>/dev/null || true
if modprobe nvidia && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; then
  echo DRIVER_OK
else
  echo DRIVER_NEEDS_REBOOT
fi
