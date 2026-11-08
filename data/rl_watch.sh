#!/bin/bash
# emits: each learner iteration summary, each cloud eval result (paired vs v14 (c_v5t7g_*)), errors
cd /Users/baggioustb/dev/carbon/carbon-league/code
OUT=../data/rl2/split2
(tail -n 0 -F ../data/rl2/split2.log | grep --line-buffered -E '^\{|Traceback|Error' | python3 -u -c "
import sys,json
for l in sys.stdin:
    try: d=json.loads(l)
    except Exception: print('LEARNER', l.strip()[:200]); continue
    m=d.get('margin',{})
    print(f\"IT {d['it']} games {d['games']} upd {d['upd_s']}s wr {d['winrate']} kl {d['kl']} clip {d['clipfrac']} vs v14 {m.get('exp/submission_cnet2_v14.py')}\")
" &)
seen=""
while true; do
  for f in ../data/cloud/bigeval_live/rl_*_wb532.jsonl; do
    [ -f "$f" ] || continue; it=$(basename $f | cut -d_ -f2)
    [ $(wc -l < $f) -ge 200 ] || continue
    case " $seen " in *" $it "*) continue;; esac
    [ $(wc -l < ../data/cloud/bigeval_live/rl_${it}_v13.jsonl) -ge 200 ] || continue
    cp ../data/cloud/bigeval_live/rl_${it}_*.jsonl ../data/bigeval_live/
    echo "EVAL it=$it | $(python3 ../data/paired.py rl_${it}_v13 c_v5t7g_v13 | cut -c1-120) | $(python3 ../data/paired.py rl_${it}_wb532 c_v5t7g_wb532 | cut -c1-120)"
    seen="$seen $it"
  done
  sleep 60
done
