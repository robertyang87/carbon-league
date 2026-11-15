echo "M320P1343B L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_poll_dgx.sh <<'DGEOF'
echo "M320P1343B DGX $(date '+%F %T %Z') host=$(hostname)"
D=/home/arm/junyang/carbon/data
echo '--- 320/ready/manifest files ---'
ls $D | grep -iE '320|ready|manifest' || echo NONE-FOUND
READY=NO
for f in $D/*320*READY* $D/*READY*320* $D/*320*ready* $D/m9xj320_manifest.json $D/*320*manifest* $D/*manifest*320*; do
  [ -f "$f" ] || continue
  READY=YES
  echo "== MANIFEST/READY: $f =="; head -c 6000 "$f"; echo
done
echo "READY320=$READY"
echo '--- 320 chain states ---'
for f in $D/*320*.state $D/*320*chain*; do [ -f "$f" ] && echo "$f: $(head -c 300 $f)"; done
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
echo "corpus replays dir: $(ls $D/m9xj320_corpus/replays/*.json.gz 2>/dev/null | wc -l)"
echo "corpus flat dir: $(ls $D/m9xj320_corpus/*.json.gz 2>/dev/null | wc -l)"
echo '--- recent data dir (top 12) ---'
ls -lt $D | head -12
echo '--- trainers on DGX (observe only) ---'
pgrep -af 'cnet2_train.py' || echo NO-TRAINER
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m320_poll_dgx.sh
echo "dgx_rc=$?"
echo "M320P1343B L20_END"
