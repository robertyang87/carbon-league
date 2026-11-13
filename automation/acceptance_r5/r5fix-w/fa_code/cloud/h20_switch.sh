#!/bin/bash
for p in $(ps -eo pid,args | awk '/bash \/root\/q_ex2/ && !/awk/ {print $1}'); do kill $p; done
for p in $(ps -eo pid,args | awk '/awr_gen.py ..\/data\/ex2_/ && !/awk/ {print $1}'); do kill $p; done
sleep 3; ps -eo args | awk '/awr_gen/ && !/awk/' | wc -l
setsid nohup bash /root/q_bline1.sh > /root/q_bline1.log 2>&1 < /dev/null &
