#!/bin/bash
until [ -f /root/carbon/data/cb.done ] && [ -f /root/carbon/data/fkcar.done ]; do sleep 30; done
bash /root/q_ex1.sh 5000000 167
