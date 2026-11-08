#!/bin/bash
# deploy.sh HOST [USER]  -- push code + needed weights to the cloud box and build the venv there.
# Never copies ladder cookies/passwords, 1C-Zero/WB sources, or other authors' reference bots.
set -euo pipefail
H=$1; U=${2:-root}; K=~/.ssh/carbon_aliyun_ed25519
SSH="ssh -i $K -o StrictHostKeyChecking=accept-new $U@$H"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
$SSH 'mkdir -p ~/carbon/code ~/carbon/data/weights ~/carbon/data/rl2 ~/carbon/data/bigeval_live'
rsync -az --delete -e "ssh -i $K" \
  --exclude '__pycache__' --exclude 'exp/opp_1cz_*' --exclude 'exp/*1cz*' --exclude 'exp/*1czero*' \
  --exclude 'reference_*' --exclude 'exp/cnet2_m2stand.py' --exclude '*.log' \
  "$ROOT/code/" $U@$H:carbon/code/
rsync -az -e "ssh -i $K" "$ROOT/data/weights/"{cnet2_b128.npz,cnet2_b128_q8.npz,cnet2_s64.npz,cnet2_s64_q8.npz,cnet2_b128v5.npz,cnet2_b128v3.npz} \
  $U@$H:carbon/data/weights/ 2>/dev/null || rsync -az -e "ssh -i $K" "$ROOT/data/weights/"cnet2_b128*.npz "$ROOT/data/weights/"cnet2_s64*.npz $U@$H:carbon/data/weights/
$SSH 'bash -s' <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync)
dpkg -s python3.12-venv >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq python3.12-venv python3-pip); [ -x .venv/bin/pip ] || rm -rf ~/carbon/.venv
cd ~/carbon
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip -q install --upgrade pip
.venv/bin/pip -q install "numpy==2.*" scipy
.venv/bin/pip -q install torch --index-url https://download.pytorch.org/whl/cpu || .venv/bin/pip -q install torch
.venv/bin/python -c 'import sys,numpy,torch,os;print("ok",sys.version.split()[0],numpy.__version__,torch.__version__,os.cpu_count(),"cpus")'
REMOTE
