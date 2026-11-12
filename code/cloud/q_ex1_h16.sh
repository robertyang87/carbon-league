#!/bin/bash
until [ -f /root/carbon/data/s50b.done ]; do sleep 30; done
bash /root/q_ex1.sh 5100000 167
