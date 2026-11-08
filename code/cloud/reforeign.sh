#!/bin/bash
pkill -f "/root/foreign.sh"; sleep 1; rm -f /root/carbon/data/foreign.done; nohup bash /root/foreign.sh >/dev/null 2>&1 </dev/null &
