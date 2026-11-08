import json, gzip, os, collections, subprocess
me = 'claude-opus55-cloud-v16b'
TOP = ('claude-AlphaCarbon-Master2s', 'claude-AlphaCarbon-Master2', '无始6.0', '无始7.0')
ms = [m for m in json.load(open('../data/v16b_matches.json')) if m.get('scores')]
got = lost = n = 0; top = []; w = 0; d150 = []
for m in ms:
    a = m['name_a'] == me; s = m['scores']; d = (s['a'] - s['b']) * (1 if a else -1); o = m['name_b' if a else 'name_a']
    w += d > 0
    if o in TOP: top.append((o.replace('claude-AlphaCarbon-', ''), round(d)))
    f = f"../data/v16b_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
    if not os.path.exists(f): continue
    order = m.get('order') or ['a', 'b']; pm = order.index('a' if a else 'b'); po = 1 - pm
    stp = json.load(gzip.open(f))['steps']; prev = None; n += 1
    for t in range(len(stp)):
        P = stp[t][0]['observation']['players']
        if t == 150: d150.append(P[pm][0] - P[po][0])
        if prev:
            for p, acc in ((po, 'g'), (pm, 'l')):
                for k, w0 in prev[p].items():
                    if k not in P[p][2] and w0[2] == 'COLLECTOR' and w0[1] > 0:
                        if acc == 'g': got += w0[1]
                        else: lost += w0[1]
        prev = {p: {k: list(v) for k, v in P[p][2].items()} for p in (pm, po)}
r = subprocess.run(['python3', 'ladder.py', 'mine'], capture_output=True, text=True).stdout
rat = [json.loads(l) for l in r.splitlines() if '"claude-opus55-cloud-v16b"' in l]
rt = f"{round(rat[0]['rating'])}±{round(rat[0]['rating_sd'])}" if rat else '?'
tw = sum(1 for _, x in top if x > 0)
print(f"V16B games {len(ms)} wins {w} rating {rt} | per game seized {got/max(1,n):.0f} lost {lost/max(1,n):.0f} (v16: 224/446) | "
      f"top4 {tw}-{len(top)-tw} {top} | d150 mean {sum(d150)/max(1,len(d150)):+.0f}")
