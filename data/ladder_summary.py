import json, gzip, os, collections, subprocess
import sys
me = sys.argv[1] if len(sys.argv) > 1 else 'claude-opus55-cloud-v16b'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'v16b'
TOP = ('claude-AlphaCarbon-Master2s', 'claude-AlphaCarbon-Master2', '无始6.0', '无始7.0')
ms = [m for m in json.load(open(f'../data/{TAG}_matches.json')) if m.get('scores')]
got = lost = n = 0; gself = [0.0]; lself = [0.0]; top = []; w = 0; d150 = []
for m in ms:
    a = m['name_a'] == me; s = m['scores']; d = (s['a'] - s['b']) * (1 if a else -1); o = m['name_b' if a else 'name_a']
    w += d > 0
    if o in TOP: top.append((o.replace('claude-AlphaCarbon-', ''), round(d)))
    f = f"../data/{TAG}_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
    if not os.path.exists(f): continue
    order = m.get('order') or ['a', 'b']; pm = order.index('a' if a else 'b'); po = 1 - pm
    stp = json.load(gzip.open(f))['steps']; prev = None; n += 1
    for t in range(len(stp)):
        P = stp[t][0]['observation']['players']
        if t == 150: d150.append(P[pm][0] - P[po][0])
        if prev:
            for p, acc in ((po, 'g'), (pm, 'l')):
                q = 1 - p if p in (0, 1) else None
                gain = [v[1] - prev[q][k][1] for k, v in P[q][2].items() if k in prev[q]]   # the other side's cargo growth
                for k, w0 in prev[p].items():
                    if k not in P[p][2] and w0[2] == 'COLLECTOR' and w0[1] > 0:
                        enemy = any(g >= 0.9 * w0[1] for g in gain)   # seized by the other side, else an own merge
                        if acc == 'g': got += w0[1] if enemy else 0; gself[0] += 0 if enemy else w0[1]
                        else: lost += w0[1] if enemy else 0; lself[0] += 0 if enemy else w0[1]
        prev = {p: {k: list(v) for k, v in P[p][2].items()} for p in (pm, po)}
r = subprocess.run(['python3', 'ladder.py', 'mine'], capture_output=True, text=True).stdout
rat = [json.loads(l) for l in r.splitlines() if f'"{me}"' in l]
rt = f"{round(rat[0]['rating'])}±{round(rat[0]['rating_sd'])}" if rat else '?'
tw = sum(1 for _, x in top if x > 0)
print(f"{TAG.upper()} games {len(ms)} wins {w} rating {rt} | per game seized {got/max(1,n):.0f} lost {lost/max(1,n):.0f} (enemy-attributed; own merges: ours {lself[0]/max(1,n):.0f} theirs {gself[0]/max(1,n):.0f}) | "
      f"top4 {tw}-{len(top)-tw} {top} | d150 mean {sum(d150)/max(1,len(d150)):+.0f}")
