#!/bin/bash
# ppou2 split: environment copy to the spot L20. Run on the Muse VM.
# Usage: prep_copy.sh <target_tailnet_ip> <check|copy-direct|copy-relay|verify>
# Source machine is fixed: gpu-l20-6 = root@100.79.251.87.
set -u
TGT_IP="${1:?target tailnet ip}"
MODE="${2:?mode}"
KEY=/home/hatch/.ssh/id_ed25519
KH=/home/hatch/.ssh/known_hosts
PROXY='nc -X connect -x hatch-egress-proxy:3130 %h %p'
SRC=root@100.79.251.87
DST=root@${TGT_IP}
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KH -o ProxyCommand=$PROXY"
SSH_TGT="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=accept-new"

case "$MODE" in
check)
  # Target prerequisites: GPU driver, tailscale, disk, os, python.
  $SSH_TGT "$DST" 'hostname; nvidia-smi --query-gpu=name,memory.total --format=csv; df -h / | tail -1; nproc; free -g | head -2; . /etc/os-release; echo $PRETTY_NAME; which python3; tailscale status --self 2>/dev/null | head -2' 
  ;;
copy-direct)
  # Preferred: gpu-l20-6 -> target over tailnet, driven from the source.
  # Auth depends on staging having authorized a key gpu-l20-6 holds;
  # if this fails, fall back to copy-relay (VM key must be authorized).
  $SSH "$SRC" "rsync -a --partial --info=progress2 \
    -e 'ssh -o StrictHostKeyChecking=accept-new' \
    /root/carbon/.venv /root/carbon/code /root/carbon/data/weights \
    /root/carbon/data/rl3/ppou2_c \
    /root/carbon/launch_ppou2_v5.py /root/carbon/remote_edit_kl.py \
    /root/carbon/remote_serial_check.py /root/carbon/remote_kl_check.py \
    /root/carbon/FILING.md \
    root@${TGT_IP}:/root/carbon/"
  ;;
copy-relay)
  # Fallback: pull to this VM then push to target (slow, ~1.3 MB/s per leg).
  mkdir -p /home/hatch/workspace/carbon-live/ppou2/relay
  rsync -a --partial -e "$SSH" $SRC:/root/carbon/data/rl3/ppou2_c /home/hatch/workspace/carbon-live/ppou2/relay/
  rsync -a --partial -e "$SSH" $SRC:/root/carbon/data/weights /home/hatch/workspace/carbon-live/ppou2/relay/
  rsync -a --partial -e "$SSH_TGT" /home/hatch/workspace/carbon-live/ppou2/relay/ppou2_c $DST:/root/carbon/data/rl3/
  rsync -a --partial -e "$SSH_TGT" /home/hatch/workspace/carbon-live/ppou2/relay/weights $DST:/root/carbon/data/
  ;;
verify)
  # md5 reconciliation of the files that decide correctness.
  FILES="data/rl3/ppou2_c/state_last.pt data/rl3/ppou2_c/state_anchor_frozen.pt data/rl3/ppou2_c/train.jsonl data/rl3/ppou2_c/league.json data/weights/cnet2_m3s2r1L.npz code/rl2_ppo_train.py launch_ppou2_v5.py"
  for f in $FILES; do
    a=$($SSH "$SRC" "md5sum /root/carbon/$f" | cut -d' ' -f1)
    b=$($SSH_TGT "$DST" "md5sum /root/carbon/$f" | cut -d' ' -f1)
    [ "$a" = "$b" ] && echo "OK  $f $a" || echo "MISMATCH $f src=$a dst=$b"
  done
  ;;
esac
