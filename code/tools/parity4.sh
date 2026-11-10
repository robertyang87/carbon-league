# parity4.sh NEW REF SEED0 N -- run parity in 4 parallel chunks on the Mac, print OK/MISMATCH per chunk
cd /Users/baggioustb/dev/carbon/carbon-league/code; export OMP_NUM_THREADS=1
n=$(( $4 / 4 ))
for i in 0 1 2 3; do ../.venv/bin/python cnet2_search_parity.py $1 $2 exp/cnet2_m2s2stand.py $(( $3 + i * n )) $n > /tmp/claude_par_$i.txt 2>&1 & done; wait
cat /tmp/claude_par_*.txt | tail -24; rm -f /tmp/claude_par_*.txt
