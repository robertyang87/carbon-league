#!/bin/bash
# Fix DNS on the spot machine and restart the driver install.
# Runs ON the spot machine. Patterns are built by concatenation so this
# script's own process never matches them.
set -u
P1="apt-get upd""ate"
P2="install_dri""ver.sh"
pkill -f "$P1" 2>/dev/null
pkill -f "$P2" 2>/dev/null
sleep 2
pkill -9 -f "$P1" 2>/dev/null
rm -f /var/lib/apt/lists/lock /var/cache/apt/archives/lock /var/lib/dpkg/lock-frontend 2>/dev/null
resolvectl dns eth0 223.5.5.5 223.6.6.6
sleep 1
if getent hosts mirrors.aliyun.com > /dev/null 2>&1; then
  echo DNS_OK
  getent hosts mirrors.aliyun.com | head -1
else
  echo DNS_STILL_BROKEN
  resolvectl status | head -30
  exit 1
fi
setsid nohup bash /root/install_driver.sh > /root/driver_install2.log 2>&1 &
echo DRIVER_INSTALL_RELAUNCHED
