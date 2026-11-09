#!/bin/bash
# zt_join.sh HOST -- make a fresh cloud host the jump to the DGX again:
# install ZeroTier from its official apt repository, join network 633e31d8a27bf450, print the node id the user
# must authorise in ZeroTier Central, then point the Mac's ssh alias carbon-spot3 at HOST.
set -euo pipefail
H=$1; K=~/.ssh/carbon_aliyun_ed25519; S="ssh -i $K -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 root@$H"
$S 'bash -s' <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
if ! command -v zerotier-cli >/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/zerotier/ZeroTierOne/main/doc/contact%40zerotier.com.gpg | gpg --dearmor > /usr/share/keyrings/zerotier.gpg
  . /etc/os-release
  echo "deb [signed-by=/usr/share/keyrings/zerotier.gpg] https://download.zerotier.com/debian/$VERSION_CODENAME $VERSION_CODENAME main" > /etc/apt/sources.list.d/zerotier.list
  apt-get update -qq && apt-get install -y -qq zerotier-one
fi
systemctl enable --now zerotier-one
sleep 3; zerotier-cli join 633e31d8a27bf450 >/dev/null; sleep 2
echo "NODE_ID $(zerotier-cli info | awk '{print $3}')"; zerotier-cli listnetworks | tail -1
REMOTE
sed -i '' "/^Host carbon-spot3$/,/^$/ s/^  HostName .*/  HostName $H/" ~/.ssh/config
grep -A2 "^Host carbon-spot3" ~/.ssh/config | head -3
