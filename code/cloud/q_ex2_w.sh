#!/bin/bash
until [ -f /root/carbon/data/ex1_5300000.done ]; do sleep 60; done
bash /root/q_ex2.sh 5310000 500
