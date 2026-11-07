"""v9_study -- how 1C-Zero-v9 beats us, and how others beat 1C-Zero-v9."""
import gzip, json, os, sys, statistics as st, collections
sys.path.insert(0, '.')
import ledger_books as LB, replay_ledger as RL
def phases(stp, p):
    """cash at 50/100/150/200/250/299; cargo lost by phase; deposits by phase"""
    cash = {}; lost = collections.Counter(); dep = collections.Counter(); prev = None
    for t in range(len(stp)):
        o = stp[t][0]['observation']; pl = o['players'][p]; w = pl[2]; bases = set(pl[1].values())
        if t in (50, 100, 150, 200, 250, 299): cash[t] = pl[0]
        ph = min(t // 100, 2)
        if prev is not None:
            for k, w0 in prev.items():
                w1 = w.get(k)
                if w1 is None and w0[2] == 'COLLECTOR': lost[ph] += w0[1]
                elif w1 is not None and w0[2] == 'COLLECTOR' and w0[1] > 0 and w1[1] == 0 and w1[0] in bases: dep[ph] += w0[1]
        prev = {k: list(v) for k, v in w.items()}
    return cash, lost, dep
def side(stp, p):
    b = LB.books(stp, p); r = RL.side_stats(stp, p); c, l, d = phases(stp, p)
    return dict(dep=b['dep'], treenet=b['tree'] - b['plant_spend'], tree=b['tree'], plants=b['plants'], seizes=b['seizes'],
                rec=b['rec'], lost=r['lost'], col100=r['col100'], pl100=r['pl100'], peakc=r['peak_col'], cash=c, lostph=l, depph=d, end=b['cash'])
def show(title, pairs):
    print(f"\n== {title}: {len(pairs)} games")
    keys = ['dep', 'tree', 'treenet', 'plants', 'seizes', 'rec', 'lost', 'col100', 'pl100', 'peakc', 'end']
    for k in keys:
        a = st.mean(x[0][k] for x in pairs); b = st.mean(x[1][k] for x in pairs)
        print(f"  {k:8s} A {a:8.1f}   B {b:8.1f}   A-B {a-b:+8.1f}")
    for t in (50, 100, 150, 200, 250, 299):
        print(f"  cash@{t:<3d} A-B {st.mean(x[0]['cash'][t]-x[1]['cash'][t] for x in pairs):+8.0f}")
    for ph in range(3):
        print(f"  phase {ph*100:3d}-{ph*100+99}: deposits A-B {st.mean(x[0]['depph'][ph]-x[1]['depph'][ph] for x in pairs):+7.0f}   cargo lost A {st.mean(x[0]['lostph'][ph] for x in pairs):6.0f} B {st.mean(x[1]['lostph'][ph] for x in pairs):6.0f}")
# 1) us (A) vs 1C-Zero-v9 (B)
ms = json.load(open('../data/vs_1czv9.json')); pairs = []
for m in ms:
    fn = f"../data/vs_1czv9/{m['id']}-{m['game']}.json.gz"
    if not os.path.exists(fn): continue
    stp = json.load(gzip.open(fn))['steps']; me = 'a' if m['name_a'] == m['_us'] else 'b'; p = m['order'].index(me)
    pairs.append((side(stp, p), side(stp, 1 - p)))
    print(f"  {m['_us'][-3:]} {m['scores'][me]:7.0f} : {m['scores']['b' if me=='a' else 'a']:7.0f}")
show("A = us (v12/v13), B = 1C-Zero-v9", pairs)
# 2) games where 1C-Zero-v9 lost, from top corpus: A = winner, B = v9
idx = json.load(open('../data/top_corpus/index.json')); won = []; lostg = collections.Counter(); allv9 = collections.Counter()
for g in idx:
    names = {k: v['name'] for k, v in g['strategies'].items()}
    if '1C-Zero-v9' not in names.values(): continue
    v9 = [k for k, n in names.items() if n == '1C-Zero-v9'][0]; op = 'b' if v9 == 'a' else 'a'
    allv9[names[op]] += 1
    if g['scores'][v9] >= g['scores'][op]: continue
    lostg[names[op]] += 1
    fn = os.path.join('../data/top_corpus', g['file'])
    if not os.path.exists(fn): continue
    stp = json.load(gzip.open(fn))['steps']; pw = g['order'].index(op)
    won.append((side(stp, pw), side(stp, 1 - pw)))
print("\n1C-Zero-v9 losses by opponent (lost/played):", {k: f"{lostg[k]}/{allv9[k]}" for k in lostg})
show("A = opponent that BEAT 1C-Zero-v9, B = 1C-Zero-v9", won)
