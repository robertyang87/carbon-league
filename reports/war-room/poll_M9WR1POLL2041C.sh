echo "M9WR1POLL2041C L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_poll_studio.sh <<'STEOF'
echo "M9WR1POLL2041C STUDIO $(date '+%T')"
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
D=~/junyang/carbon/data
if [ -d $D/ds_m9xj_w ]; then
  echo DS_W_PRESENT
  ls $D/ds_m9xj_w/ | head -20
  ~/junyang/carbon/.venv/bin/python -c "
import json
m=json.load(open('$D/ds_m9xj_w/meta.json'.replace(chr(126),'$HOME')))
print('meta keys:', sorted(m.keys()))
print('total:', m.get('total'))
print('weighted:', m.get('weighted'))
" 2>/dev/null || echo META-READ-FAIL
  du -sh $D/ds_m9xj_w
else
  echo DS_W_ABSENT
fi
ls -lt $D/ | head -6
ls $D/ | grep -i -E 'm9xj|md5|manifest' || echo NO-M9XJ-FILES
STEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_poll_studio.sh
echo "studio_rc=$?"
cat > /tmp/m9wr1_poll_dgx.sh <<'DGEOF'
echo "M9WR1POLL2041C DGX $(date '+%T')"
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
D=/home/arm/junyang/carbon/data
for d in ds_m9xj ds_m9xj_w; do
  if [ -d $D/$d ]; then
    echo "== $d =="
    python3 -c "
import json
m=json.load(open('$D/$d/meta.json'))
print('total:', m.get('total'), 'weighted:', m.get('weighted'))
" 2>/dev/null || echo META-FAIL-$d
    du -sh $D/$d | cut -f1
  else
    echo "== $d ABSENT =="
  fi
done
ls -lt $D/*.log 2>/dev/null | head -6
ls -lt $D/weights/ | grep -i m9xj | head -5 || echo NO-M9XJ-WEIGHTS
ls $D/ | grep -i -E 'ship|push' | head
tail -3 $D/m9xj*_ship.log 2>/dev/null || true
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m9wr1_poll_dgx.sh
echo "dgx_rc=$?"
echo "M9WR1POLL2041C L20_END"
