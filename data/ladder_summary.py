import json, gzip, os, collections, subprocess
import sys
me = sys.argv[1] if len(sys.argv) > 1 else 'claude-opus55-cloud-v16b'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'v16b'
TOP = ('claude-AlphaCarbon-Master2s', 'claude-AlphaCarbon-Master2', '无始6.0', '无始7.0')
ms = [m for m in json.load(open(f'../data/{TAG}_matches.json')) if m.get('scores')]
n = 0; tally = collections.Counter();
DIST = lambda a, b: min(abs(a // 15 - b // 15), 15 - abs(a // 15 - b // 15)) + min(abs(a % 15 - b % 15), 15 - abs(a % 15 - b % 15))
top = []; w = 0; d150 = []
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
                q = 1 - p
                gq = [v[1] - prev[q][k][1] for k, v in P[q][2].items() if k in prev[q]]   # other side's cargo growth
                gp = [v[1] - prev[p][k][1] for k, v in P[p][2].items() if k in prev[p]]   # own side's cargo growth
                cq = P[q][0] - pcash[q]; cp = P[p][0] - pcash[p]                          # cash growth (deposits)
                bq = [int(b) for b in P[q][1].values()]; bp = [int(b) for b in P[p][1].values()]
                for k, w0 in prev[p].items():
                    if k not in P[p][2] and w0[2] == 'COLLECTOR' and w0[1] > 0:
                        c = w0[1]; near = lambda bs: any(DIST(w0[0], b) <= 1 for b in bs)
                        # enemy: an enemy car grew by it, or it was seized next to an enemy base and deposited at once
                        if any(g >= 0.9 * c for g in gq) or (near(bq) and cq >= 0.9 * c): kind = 'e'
                        elif any(g >= 0.9 * c for g in gp) or (near(bp) and cp >= 0.9 * c): kind = 's'   # own merge
                        else: kind = 'u'
                        tally[acc + kind] += c
        prev = {p: {k: list(v) for k, v in P[p][2].items()} for p in (pm, po)}; pcash = {p: P[p][0] for p in (pm, po)}
r = subprocess.run(['python3', 'ladder.py', 'mine'], capture_output=True, text=True).stdout
rat = [json.loads(l) for l in r.splitlines() if f'"{me}"' in l]
rt = f"{round(rat[0]['rating'])}±{round(rat[0]['rating_sd'])}" if rat else '?'
tw = sum(1 for _, x in top if x > 0)
print(f"{TAG.upper()} games {len(ms)} wins {w} rating {rt} | per game seized {tally['ge']/max(1,n):.0f} lost {tally['le']/max(1,n):.0f} (enemy-attributed; own merges ours {tally['ls']/max(1,n):.0f} theirs {tally['gs']/max(1,n):.0f}; unknown ours {tally['lu']/max(1,n):.0f} theirs {tally['gu']/max(1,n):.0f}) | "
      f"top4 {tw}-{len(top)-tw} {top} | d150 mean {sum(d150)/max(1,len(d150)):+.0f}")
