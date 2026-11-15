#!/bin/bash
echo "N1RT4D9W_BEGIN"
cd /root/carbon/code
wc -l eval_v53.py
grep -n "^def \|^class \|^import\|^from" eval_v53.py | head -40
echo "== play_game body =="
awk '/^def play_game/,/^def [a-zA-Z_]+\(/' eval_v53.py | head -120
echo "N1RT4D9W_END"
