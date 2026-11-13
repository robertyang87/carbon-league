#!/bin/bash
for p in $(ps -eo pid,args | awk '/bash \/root\/q_h14b.sh/ && !/awk/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '/fx_m2t3S20q3/ && !/awk/ {print $1}'); do kill $p; done
sleep 3; ps -eo args | awk '/q_h14b|fx_m2t3S20q3/ && !/awk/' | wc -l
