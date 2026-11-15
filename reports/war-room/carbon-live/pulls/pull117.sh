#!/bin/bash
# usage: pull117.sh <strategy_id> <outfile>
SID="$1"; OUT="$2"; OFF=0; : > "$OUT.tmp"
while true; do
  curl -skL "https://117.72.210.134/api/matches?compact=1&limit=100&offset=$OFF&strategy=$SID" -o "$OUT.page"
  python3 - "$OUT.page" "$OUT.tmp" <<'PY'
import json,sys
page=json.load(open(sys.argv[1]))
tmp=open(sys.argv[2],'a')
ms=page.get('matches', page if isinstance(page,list) else [])
for m in ms: tmp.write(json.dumps(m,ensure_ascii=False)+"\n")
tmp.close()
nxt=page.get('next_offset') if isinstance(page,dict) else None
print(f"offset_got={len(ms)} next={nxt}")
open('/tmp/pull117_next.txt','w').write(str(nxt) if nxt is not None else '')
PY
  NXT=$(cat /tmp/pull117_next.txt)
  if [ -z "$NXT" ] || [ "$NXT" = "None" ]; then break; fi
  OFF=$NXT
done
python3 - "$OUT.tmp" "$OUT" <<'PY'
import json,sys
ms=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
json.dump(ms,open(sys.argv[2],'w'),ensure_ascii=False)
print(f"total={len(ms)}")
PY
rm -f "$OUT.tmp" "$OUT.page"
