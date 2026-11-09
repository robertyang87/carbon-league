#!/bin/bash
# Mac relay: copies finished training runs (marked by the last epoch checkpoint) DGX -> Mac -> host 4, so the Mac and
# host-4 backlogs can start as soon as a run ends. usage: weights_relay.sh HOST4_IP RUN [RUN ...]   (RUN e.g. dg6:2)
K=$HOME/.ssh/carbon_aliyun_ed25519; H4=$1; shift; W=/Users/baggioustb/dev/carbon/carbon-league/data/weights
left="$*"
while [ -n "$left" ]; do
  next=""
  for r in $left; do n=${r%%:*}; e=${r##*:}
    if ssh -o ConnectTimeout=30 dgx-via-spot3 "test -f junyang/carbon/data/weights/cnet2_b128${n}_ep${e}.npz" </dev/null; then
      sleep 30   # the plain .npz (best epoch) is written in the same save
      rsync -az "dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128${n}.npz" "dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128${n}_ep${e}.npz" "$W/" </dev/null \
        && scp -q -i "$K" "$W/cnet2_b128${n}.npz" "$W/cnet2_b128${n}_ep${e}.npz" "root@${H4}:/root/carbon/data/weights/" </dev/null \
        && echo "$(date +%T) relayed $n" || next="$next $r"
    else next="$next $r"; fi
  done
  left="${next# }"; [ -n "$left" ] && sleep 120
done
echo "$(date +%T) RELAY_DONE"
