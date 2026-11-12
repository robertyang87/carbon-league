#!/bin/bash
# host 14: keep only the multi-car (SEQ_KEYS=3) exam; the rest moved to host 15
for p in $(ps -eo pid,args | awk '/bash \/root\/q_h14[cd].sh/ && !/awk/ {print $1}'); do kill $p; done
sleep 2; ps -eo args | awk '/q_h14[cd]/ && !/awk/' | wc -l
setsid nohup bash -c 'while ps -eo args | grep -q "[f]x_m2t3S20q3"; do sleep 30; done; echo Q3_DONE > /root/carbon/data/q3.done' > /dev/null 2>&1 < /dev/null &
