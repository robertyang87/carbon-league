"""bot_study -- interception mechanics, opening and tree siting for one side of a replay."""
import gzip, json, os, sys, statistics as st, collections
NB8 = lambda c: [((c//15+dr)%15)*15+(c%15+dc)%15 for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc]
def tdist(a, b):
    r = abs(a//15-b//15); c = abs(a%15-b%15); return min(r,15-r)+min(c,15-c)
def side(stp, p):
    q = 1-p; kills = []; prev = None; out = {}
    rec_steps = []; trees_born = []; tree_seen = set(); seized_by_enemy = 0; my_trees_total = 0
    for t in range(len(stp)):
        o = stp[t][0]['observation']; me = o['players'][p]; en = o['players'][q]
        W = me[2]; E = en[2]
        if t in (30, 50, 100):
            out[f'units{t}'] = len(W); out[f'col{t}'] = sum(1 for w in W.values() if w[2]=='COLLECTOR'); out[f'cash{t}'] = me[0]
        if prev is not None:
            pW, pE = prev
            rec_steps += [t for k in W if k not in pW]
            pos_now = {w[0]: (k, w) for k, w in W.items()}
            for k, e0 in pE.items():
                if k in E or e0[2] != 'COLLECTOR' or e0[1] <= 0: continue
                # enemy loaded collector died; who of ours stands on its last cell or a neighbour now
                for kk, w in W.items():
                    if w[2] == 'COLLECTOR' and kk in pW and tdist(w[0], e0[0]) <= 1 and w[1] >= pW[kk][1] + e0[1]*0.5:
                        kills.append((t, e0[1], pW[kk][1], tdist(pW[kk][0], e0[0]))); break
        mine_trees = me[3] if len(me) > 3 else {}
        carbon = stp[t][p]['observation'].get('carbon')
        for k, v in mine_trees.items():
            if k.startswith(f'player-{p}-') and k not in tree_seen:
                tree_seen.add(k); my_trees_total += 1
                pos = v[0] if isinstance(v, (list, tuple)) else None
                if carbon and pos is not None: trees_born.append((t, sum(carbon[c] for c in NB8(int(pos)))))
        prev = ({k: list(v) for k, v in W.items()}, {k: list(v) for k, v in E.items()})
    out['kills'] = len(kills); out['kill_cargo'] = sum(k[1] for k in kills)
    out['hunter_empty'] = sum(1 for k in kills if k[2] == 0) / max(1, len(kills))
    out['rec_by50'] = sum(1 for s in rec_steps if s <= 50); out['rec_by100'] = sum(1 for s in rec_steps if s <= 100)
    out['tree_nb'] = st.mean(x[1] for x in trees_born) if trees_born else 0
    out['trees_by100'] = sum(1 for x in trees_born if x[0] <= 100); out['trees_late'] = sum(1 for x in trees_born if x[0] >= 230)
    out['kill_steps_late'] = sum(1 for k in kills if k[0] >= 200)
    return out
def collect(pairs):
    keys = ['cash30','cash50','cash100','units50','col50','units100','col100','rec_by50','rec_by100','kills','kill_cargo','hunter_empty','kill_steps_late','tree_nb','trees_by100','trees_late']
    return {k: st.mean(x[k] for x in pairs) for k in keys}
def show(label, rows):
    print(f"{label:28s} " + "  ".join(f"{k}={v:.2f}" if k=='hunter_empty' else f"{k}={v:.0f}" for k, v in rows.items()))
res = {}
ms = json.load(open('../data/v10z_matches.json')); A = []; O = []
for m in ms:
    fn = f"../data/v10z/{m['id']}-{m['game']}.json.gz"
    if not os.path.exists(fn) or m['scores']['a'] is None: continue
    stp = json.load(gzip.open(fn))['steps']; me = 'a' if m['name_a']=='1C-Zero-v10' else 'b'; p = m['order'].index(me)
    A.append(side(stp, p)); O.append(side(stp, 1-p))
show('1C-Zero-v10', collect(A)); show('  its opponents', collect(O))
idx = json.load(open('../data/top_corpus/index.json')); V9 = []; MA = []
for g in idx:
    nm = {k: v['name'] for k, v in g['strategies'].items()}; fn = os.path.join('../data/top_corpus', g['file'])
    if not os.path.exists(fn): continue
    for k, n in nm.items():
        if n == '1C-Zero-v9' and len(V9) < 40: V9.append(side(json.load(gzip.open(fn))['steps'], g['order'].index(k)))
        if n == 'claude-AlphaCarbon-Master' and len(MA) < 40: MA.append(side(json.load(gzip.open(fn))['steps'], g['order'].index(k)))
show('1C-Zero-v9', collect(V9)); show('Master', collect(MA))
V12 = []
for m in json.load(open('../data/v12_matches.json')):
    fn = f"../data/v12_ladder/replays/{m['id']}-{m['game']}.json.gz"
    if not os.path.exists(fn) or m['scores']['a'] is None: continue
    me = 'a' if m['name_a']=='claude-opus55-cloud-v12' else 'b'; V12.append(side(json.load(gzip.open(fn))['steps'], m['order'].index(me)))
show('v12', collect(V12))
