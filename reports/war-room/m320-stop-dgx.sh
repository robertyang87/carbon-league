#!/bin/bash
# M320 STOP (runs ON DGX): report chain state; kill ONLY this chain's processes
# (85-game build/train must not proceed). Surgical matching, print everything.
DATA=/home/arm/junyang/carbon/data
LOGD=$DATA/dq_logs
echo "STOP-BEGIN $(date '+%F %T %Z')"
echo '--- chain state file ---'
cat "$DATA/m9xj320_chain.state" 2>/dev/null || echo NO-STATE-FILE
echo '--- relevant processes before ---'
ps -eo pid,ppid,etime,args | grep -E 'cnet2_train|replica_chain|build_top|set_uw|build_m9xj|bash -s' | grep -v grep || echo NONE
echo '--- datasets ---'
ls -ld "$DATA/ds_m9xj320" "$DATA/ds_m9xj320_w" 2>/dev/null || echo NO-DS-YET
ls -l "$DATA/M9XJ320_READY.txt" 2>/dev/null || echo NO-READY-FILE
echo '--- kills ---'
for p in $(pgrep -f 'cnet2_train[.]py'); do
  if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'ds_m9xj320'; then echo "KILL train pid $p"; kill $p 2>/dev/null; fi
done
for p in $(pgrep -f 'replica_chain[.]sh'); do echo "KILL replica pid $p"; kill $p 2>/dev/null; done
for p in $(pgrep -f 'build_top[.]py|set_uw[.]py'); do echo "KILL build pid $p"; kill $p 2>/dev/null; done
sleep 2
# outer chain bash: 'bash -s' sessions older than 60s sitting in the carbon tree
for p in $(pgrep -f '^bash -s$'); do
  cwd=$(readlink /proc/$p/cwd 2>/dev/null)
  case "$cwd" in
    /home/arm/junyang/carbon*|/home/arm/junyang/carbon-repo*)
      echo "KILL outer-chain pid $p cwd=$cwd"; kill $p 2>/dev/null ;;
  esac
done
sleep 3
for p in $(pgrep -f 'cnet2_train[.]py'); do echo "FORCE-KILL train pid $p"; kill -9 $p 2>/dev/null; done
for p in $(pgrep -f 'replica_chain[.]sh'); do echo "FORCE-KILL replica pid $p"; kill -9 $p 2>/dev/null; done
echo '--- processes after ---'
ps -eo pid,ppid,etime,args | grep -E 'cnet2_train|replica_chain|build_top|set_uw' | grep -v grep || echo NONE-RUNNING
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- scrap the 85-game build if present (rename, never train on it) ---'
[ -d "$DATA/ds_m9xj320" ] && mv "$DATA/ds_m9xj320" "$DATA/ds_m9xj320_85scrapped" && echo SCRAPPED-DS
[ -d "$DATA/ds_m9xj320_w" ] && mv "$DATA/ds_m9xj320_w" "$DATA/ds_m9xj320_w_85scrapped" && echo SCRAPPED-DSW
[ -f "$DATA/M9XJ320_READY.txt" ] && mv "$DATA/M9XJ320_READY.txt" "$DATA/M9XJ320_READY_85scrapped.txt" && echo SCRAPPED-READY
[ -d "$DATA/m9xj320_corpus" ] && cp -r "$DATA/m9xj320_corpus" "$DATA/m9xj320_corpus_85snapshot" && echo CORPUS-85-SNAPSHOT-SAVED
echo '--- replica log tail ---'
tail -6 "$LOGD/replica_chain_m9xj320.log" 2>/dev/null || echo NO-REPLICA-LOG
echo "STOP-END $(date '+%F %T %Z')"
