echo "M320POLL2125A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_poll_dgx.sh <<'DGEOF'
echo "M320POLL2125A DGX $(date '+%F %T %Z') host=$(hostname)"
D=/home/arm/junyang/carbon/data
echo '--- data dir (recent 40) ---'
ls -lt $D | head -40
echo '--- 320/ready/manifest files ---'
ls $D | grep -iE '320|ready|manifest' || echo NONE-FOUND
for f in $D/*320* $D/*READY* $D/*ready* $D/*manifest*; do
  [ -f "$f" ] || continue
  case "$f" in *.json.gz|*.npz|*.log) continue;; esac
  echo "== $f =="; head -c 3000 "$f"; echo
done
echo '--- chain states ---'
for f in $D/*chain*.state $D/*.state; do [ -f "$f" ] && echo "$f: $(cat $f)"; done
echo '--- datasets ---'
for d in ds_m9xj320 ds_m9xj320_w; do
  if [ -d $D/$d ]; then
    echo "== $d =="
    python3 -c "
import json
m=json.load(open('$D/$d/meta.json'))
for k in sorted(m.keys()):
    if k!='games': print(k,':',m[k])
g=m.get('games',[])
print('games:',len(g),'val games:',sum(1 for x in g if x.get('val')))
print('sum n:',sum(x.get('n',0) for x in g))
" 2>/dev/null || echo META-FAIL-$d
    du -sh $D/$d | cut -f1
  else echo "== $d ABSENT =="; fi
done
echo '--- matches/corpus ---'
python3 -c "import json;print('m9xj320 matches:',len(json.load(open('$D/m9xj320_matches.json'))))" 2>/dev/null || echo NO-320-MATCHES-FILE
ls $D/m9xj320_corpus/replays/*.json.gz 2>/dev/null | wc -l
ls $D/m9xj320_corpus/*.json.gz 2>/dev/null | wc -l
echo '--- trainers on DGX (observe only) ---'
pgrep -af 'cnet2_train.py' || echo NO-TRAINER
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320_poll_dgx.sh
echo "dgx_rc=$?"
cat > /tmp/m320_poll_studio.sh <<'STEOF'
echo "M320POLL2125A STUDIO $(date '+%F %T %Z')"
pgrep -fl 'cnet2_train.py' || echo NO-TRAINER
D=~/junyang/carbon/data
ls $D | grep -iE '320' || echo NO-320-FILES-ON-STUDIO
df -h $HOME | tail -1
STEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m320_poll_studio.sh
echo "studio_rc=$?"
echo "M320POLL2125A L20_END"
