echo "__MARK__ L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_chk_dgx.sh <<'DGEOF'
echo "__MARK__ DGX $(date '+%T')"
D=/home/arm/junyang/carbon/data
echo '--- ds_m9xj_w files ---'
ls $D/ds_m9xj_w/
echo '--- ds_m9xj_w meta summary ---'
python3 - <<'PYEOF'
import json
m = json.load(open('/home/arm/junyang/carbon/data/ds_m9xj_w/meta.json'))
for k in sorted(m.keys()):
    if k != 'games':
        print(k, ':', m[k])
g = m.get('games', [])
print('games:', len(g), 'val games:', sum(1 for x in g if x.get('val')))
print('sum n:', sum(x.get('n', 0) for x in g))
ms = json.load(open('/home/arm/junyang/carbon/data/ds_m9xj/meta.json'))
gm = {x['key']: x for x in ms.get('games', [])}
same = all(gm.get(x['key'], {}).get('n') == x.get('n') and gm.get(x['key'], {}).get('val') == x.get('val') for x in g if x.get('val'))
print('val games identical to ds_m9xj (n+val flags):', same)
PYEOF
echo '--- set_uw evidence ---'
grep -o '"uw[^,]*' $D/ds_m9xj_w/meta.json | head -3 || echo NO-UW-IN-META
ls $D/ds_m9xj_w/ | grep -i uw || echo NO-UW-FILE
echo '--- chain/push logs ---'
ls -lt $D/ | head -8
ls /home/arm/junyang/carbon/*.log /home/arm/junyang/carbon/data/*m9xj* 2>/dev/null | head
grep -l 'studio\|10.211.43.68\|10.188' /home/arm/junyang/carbon/data/*.log 2>/dev/null | head -3
echo '--- trainers ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAINER
for f in $D/weights/cnet2_m9xj260*_log.jsonl; do echo "$f: $(tail -1 $f 2>/dev/null | cut -c1-160)"; done
DGEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' < /tmp/m9wr1_chk_dgx.sh
echo "dgx_rc=$?"
cat > /tmp/m9wr1_chk_studio.sh <<'STEOF'
echo "__MARK__ STUDIO $(date '+%T')"
D=~/junyang/carbon/data
[ -d $D/ds_m9xj_w ] && echo DS_W_PRESENT || echo DS_W_ABSENT
echo '--- ds_m3s2_w (known-good) files ---'
ls $D/ds_m3s2_w/
echo '--- studio reach DGX LAN test ---'
timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 'echo DGX-LAN-OK; ls /home/arm/junyang/carbon/data/ds_m9xj_w/ | head -3' 2>&1 | head -5
STEOF
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_chk_studio.sh
echo "studio_rc=$?"
echo "__MARK__ L20_END"
