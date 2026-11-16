"""Echo4 243-game redo analysis. Joins parsed243 with matches file (validated),
computes win/loss aggregates, residual separation, rule-based mode classification,
opponent composition. Rules mirror the manual 26/51-game classifications."""
import json, statistics as st
from collections import Counter, defaultdict

P = '/home/hatch/workspace/echo4-analysis/'
parsed = json.load(open(P + 'echo4_parsed243.json'))
matches = json.load(open(P + 'echo4_pull243/echo4_matches.json'))
assert len(parsed) == len(matches) == 243

# --- validate join by index (parsed order == matches file order) ---
bad = 0
rows = []
for pr, m in zip(parsed, matches):
    slot = 'a' if m['name_a'] == 'Echo4' else 'b'
    opp_name = m['name_b'] if slot == 'a' else m['name_a']
    margin = m['scores'][slot] - m['scores']['b' if slot == 'a' else 'a']
    if pr['opp'] != opp_name or abs(pr['margin'] - margin) > 0.01 or pr['won'] != (margin > 0):
        bad += 1
        continue
    first = (m['order'][0] == slot)
    rows.append(dict(id=m['id'], opp=opp_name, margin=margin, won=margin > 0,
                     first=first, created=m['created'], t=pr, o=pr['opp_metrics']))
print('join mismatches:', bad, '| rows:', len(rows))

def cts(x):  # cash per tree-step
    return x['cash_final'] / x['tree_steps'] if x['tree_steps'] else 0.0

def agg(games, key):
    return {k: round(st.mean([g[key][k] for g in games]), 1)
            for k in ('plants', 'tree_steps', 'end_trees', 'end_cargo', 'peak_cargo',
                      'cash50', 'cash100', 'cash200', 'cash270', 'cash_final',
                      'last30', 'first_plant', 'recruits')}
    
W = [r for r in rows if r['won']]
L = [r for r in rows if not r['won']]
print(f"\nTOTAL {len(rows)} games: {len(W)}W-{len(L)}L")
for name, gs in (('wins', W), ('losses', L)):
    ms = [g['margin'] for g in gs]
    print(f"{name}: mean margin {st.mean(ms):+.1f} median {st.median(ms):+.1f}")
print('loss margin buckets:',
      {b: sum(1 for g in L if lo < -g['margin'] <= hi) for b, lo, hi in
       (('<=-1500', 1500, 1e9), ('-1500~-1000', 1000, 1500), ('-1000~-500', 500, 1000),
        ('-500~-200', 200, 500), ('-200~0', 0, 200))})

print('\nEcho4 agg wins  :', agg(W, 't'))
print('Echo4 agg losses:', agg(L, 't'))
print('Opp   agg in W  :', agg(W, 'o'))
print('Opp   agg in L  :', agg(L, 'o'))
print('Echo4 cash/tree_step: wins %.2f losses %.2f | opp: wins %.2f losses %.2f' % (
    st.mean([cts(g['t']) for g in W]), st.mean([cts(g['t']) for g in L]),
    st.mean([cts(g['o']) for g in W]), st.mean([cts(g['o']) for g in L])))

# --- residual separation (the strategy-group key question) ---
for label, k in (('end_trees', 'end_trees'), ('end_cargo', 'end_cargo')):
    wv = [g['t'][k] for g in W]; lv = [g['t'][k] for g in L]
    print(f"{label}: wins mean {st.mean(wv):.2f} median {st.median(wv):.1f} | "
          f"losses mean {st.mean(lv):.2f} median {st.median(lv):.1f}")
print('losses with end_trees>=5:', sum(1 for g in L if g['t']['end_trees'] >= 5),
      '| >=7:', sum(1 for g in L if g['t']['end_trees'] >= 7),
      '| wins >=7:', sum(1 for g in W if g['t']['end_trees'] >= 7))
print('losses with end_cargo>=200:', sum(1 for g in L if g['t']['end_cargo'] >= 200),
      '| wins >=200:', sum(1 for g in W if g['t']['end_cargo'] >= 200))
flip = [g for g in L if g['t']['end_cargo'] >= -g['margin']]
print('losses where end_cargo alone >= |margin| (cargo flips):', len(flip),
      [(g['opp'], round(g['margin']), round(g['t']['end_cargo'])) for g in flip])

# --- move order ---
for mv, gs in (('first', [r for r in rows if r['first']]), ('second', [r for r in rows if not r['first']])):
    w = sum(1 for g in gs if g['won'])
    print(f"move {mv}: {len(gs)} games {w}W-{len(gs)-w}L")

# --- placement-40 reconstruction (chronological first 40 = last 40 in file order) ---
p40 = rows[-40:]
print('placement40 check:', sum(1 for g in p40 if g['won']), 'W -',
      sum(1 for g in p40 if not g['won']), 'L, mean margin',
      round(st.mean([g['margin'] for g in p40]), 1))

# --- mode classification of losses ---
def classify(g):
    t, o = g['t'], g['o']
    tsr = t['tree_steps'] / o['tree_steps'] if o['tree_steps'] else 1
    cr = cts(t) / cts(o) if cts(o) else 1
    d100 = t['cash100'] - o['cash100']; d200 = t['cash200'] - o['cash200']
    d270 = t['cash270'] - o['cash270']; d30 = t['last30'] - o['last30']
    flags = []
    if t['end_cargo'] >= -g['margin']:
        flags.append('M1-cargo-decisive')
    if t['end_trees'] >= 7 or t['end_cargo'] >= 200:
        flags.append('M1-residual-heavy')
    if d200 > 0 and d270 < 0 and (d200 - d270) >= 800:
        flags.append('M5-window200-270')
    if d100 >= -200 and d200 <= -800:
        flags.append('W100-200-hole')
    if d270 >= -200 and d30 <= -500:
        flags.append('last30-fade')
    if t['first_plant'] - o['first_plant'] >= 8:
        flags.append('M4-late-first-plant')
    if 'M5-window200-270' in flags:
        prim = 'M5'
    elif 'M1-cargo-decisive' in flags:
        prim = 'M1-cargo'
    elif tsr <= 0.80:
        prim = 'M2-capacity'
    elif tsr >= 0.95 and cr <= 0.80:
        prim = 'M3-conversion'
    elif 'W100-200-hole' in flags:
        prim = 'W100-200'
    elif 'last30-fade' in flags:
        prim = 'last30'
    elif tsr < 0.95:
        prim = 'M2-lean'
    elif cr < 0.95:
        prim = 'M3-lean'
    else:
        prim = 'coinflip/other'
    return prim, flags, round(tsr, 2), round(cr, 2)

tally = Counter(); flagtally = Counter(); loss_rows = []
for g in L:
    prim, flags, tsr, cr = classify(g)
    tally[prim] += 1
    for f in flags: flagtally[f] += 1
    loss_rows.append((g['margin'], g['opp'], '先' if g['first'] else '后', prim, flags, tsr, cr,
                      g['t']['plants'], g['o']['plants'], g['t']['tree_steps'], g['o']['tree_steps'],
                      g['t']['end_trees'], g['t']['end_cargo'], g['id']))
print('\nMODE TALLY (primary):', dict(tally.most_common()))
print('FLAG TALLY:', dict(flagtally.most_common()))
loss_rows.sort()
print('\nWORST 25 LOSSES:')
for r in loss_rows[:25]:
    print(f"  {r[0]:+8.0f} vs {r[1]:28s} {r[2]} prim={r[3]:14s} ts {r[9]}/{r[10]} tsr={r[5]} cr={r[6]} "
          f"trees={r[11]} cargo={r[12]:.0f} flags={r[4]} id={r[13][:8]}")

# --- opponent composition ---
oc = Counter(g['opp'] for g in L); ow = Counter(g['opp'] for g in W)
print('\nLOSS OPPONENTS:', dict(oc.most_common()))
print('WIN  OPPONENTS:', dict(ow.most_common()))
json.dump([{**{k: r[k] for k in ('id', 'opp', 'margin', 'won', 'first')},
            'prim': classify(r)[0] if not r['won'] else None} for r in rows],
          open(P + 'echo4_rows243.json', 'w'), ensure_ascii=False)
print('\nsaved echo4_rows243.json')
