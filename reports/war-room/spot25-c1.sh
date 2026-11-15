#!/bin/bash
# spot25 step C1: read own host key fingerprint from the box itself.
MARK="SPOT25C1K4"
echo "MARKER $MARK START"
date '+%F %T %Z'
hostname
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
ls -la /root/carbon 2>/dev/null || echo "NO-CARBON-DIR"
echo "MARKER $MARK END"
