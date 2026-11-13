"""cargo_study -- trip size at delivery and cargo at death (interceptions), per side."""
import gzip, json, os, sys, statistics as st, collections, glob
def side(stp, p):
    deliv = []; deaths = []; prev = None; maxc = []
    for t in range(len(stp)):
        o = stp[t][0]['observation']; pl = o['players'][p]; w = pl[2]; bases = set(pl[1].values())
        if prev is not None:
            for k, w0 in prev.items():
                w1 = w.get(k)
                if w0[2] != 'COLLECTOR': continue
                if w1 is None:
                    if w0[1] > 0: deaths.append((t, w0[1]))
                elif w0[1] > 0 and w1[1] == 0 and w1[0] in bases: deliv.append((t, w0[1]))
        prev = {k: list(v) for k, v in w.items()}
    return deliv, deaths
def summ(lbl, D):
    dl = [c for g in D for t, c in g[0]]; dd = [c for g in D for t, c in g[1]]; n = len(D)
    q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))] if xs else 0
    print(f"{lbl:34s} games {n:3d} | trips/g {len(dl)/n:5.1f} mean load {st.mean(dl):6.1f} median {q(dl,.5):6.1f} p90 {q(dl,.9):6.1f}"
          f" | lost/g {sum(dd)/n:6.1f} deaths-with-cargo/g {len(dd)/n:4.1f} mean cargo at death {st.mean(dd) if dd else 0:6.1f}")
    for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 1e9)):
        a = sum(1 for c in dl if lo <= c < hi); b = [c for c in dd if lo <= c < hi]
        print(f"      load {lo:>3}-{'' if hi > 1e8 else int(hi):<4}: delivered trips {a/n:5.1f}/g   lost {len(b)/n:4.2f}/g = {sum(b)/n:6.1f} cargo/g")
us = collections.defaultdict(list); them = collections.defaultdict(list)
for f, ours in (('../data/v12_matches.json', 'claude-opus55-cloud-v12'), ('../data/v13_matches.json', 'claude-opus55-cloud-v13')):
    for m in json.load(open(f)):
        fn = f"../data/v12_ladder/replays/{m['id']}-{m['game']}.json.gz"
        if not os.path.exists(fn): continue
        stp = json.load(gzip.open(fn))['steps']; me = 'a' if m['name_a'] == ours else 'b'; p = m['order'].index(me)
        op = m['name_b' if me == 'a' else 'name_a']; top = any(k in op for k in ('Master', '1C-Zero-v7', '1C-Zero-v8', '1C-Zero-v9', 'RyanX-AIPowercore-V11', 'Ultra v14'))
        g = 'vs top' if top else 'vs rest'
        us[g].append(side(stp, p)); them[g].append(side(stp, 1 - p))
for g in ('vs top', 'vs rest'):
    summ(f"v12 {g}", us[g]); summ(f"opponents ({g})", them[g])
# top bots among themselves
idx = json.load(open('../data/top_corpus/index.json')); tb = collections.defaultdict(list)
for gm in idx:
    fn = os.path.join('../data/top_corpus', gm['file'])
    if not os.path.exists(fn): continue
    names = {k: v['name'] for k, v in gm['strategies'].items()}
    for k, n in names.items():
        if n in ('claude-AlphaCarbon-Master', '1C-Zero-v9', '1C-Zero-v8-preview'):
            if len(tb[n]) >= 40: continue
            stp = json.load(gzip.open(fn))['steps']; tb[n].append(side(stp, gm['order'].index(k)))
for n, D in tb.items(): summ(n + " (all games)", D)
