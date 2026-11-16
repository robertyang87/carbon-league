import glob, json, statistics as st

packs = {}
for path in sorted(glob.glob('/root/carbon/sig_adj1/pool_*.jsonl')):
    name = path.split('/')[-1].rsplit('_s', 1)[0]
    rows = [json.loads(l) for l in open(path)]
    packs.setdefault(name, []).extend(rows)

def agg(rows):
    n = len(rows)
    ts = [r['pol']['treesteps'] for r in rows]
    cash = [r['pol']['cash'] for r in rows]
    ratios = [c / t for c, t in zip(cash, ts) if t > 0]
    return {
        'n': n,
        'zero_ts': sum(1 for t in ts if t == 0),
        'treesteps': round(st.mean(ts), 1),
        'cash': round(st.mean(cash), 1),
        'cash_per_ts_pgr': round(st.mean(ratios), 3),
        'cash_per_ts_rom': round(st.mean(cash) / st.mean(ts), 3),
        'recruits': round(st.mean([r['pol']['pla_rec'] for r in rows]), 2),
        'opp_deaths': round(st.mean([r['opp']['pla_dead'] for r in rows]), 3),
        'opp_cash40': round(st.mean([r['pol']['e_cash40'] for r in rows]), 1),
        'opp_cash25': round(st.mean([r['pol']['e_cash25'] for r in rows]), 1),
        'winrate': round(st.mean([1.0 if r['win'] else 0.0 for r in rows]), 3),
    }

out = {name: agg(rows) for name, rows in sorted(packs.items())}
print(json.dumps(out, indent=1))
print()
print('%-28s %5s %8s %7s %7s %7s %8s %9s' % ('pack', 'n', 'treestep', 'c/tsPG', 'c/tsRM', 'recr', 'oppDeath', 'oppCash40'))
for name, a in out.items():
    print('%-28s %5d %8.1f %7.3f %7.3f %7.2f %8.3f %9.1f' % (name, a['n'], a['treesteps'], a['cash_per_ts_pgr'], a['cash_per_ts_rom'], a['recruits'], a['opp_deaths'], a['opp_cash40']))
print()
print('Truth lines: capacity >= 1593.0 treesteps; conversion >= 7.235 cash/ts (per-game-ratio)')
for name, a in out.items():
    print(name, 'CAP' if a['treesteps'] >= 1593.0 else '   ', 'CONV' if a['cash_per_ts_pgr'] >= 7.235 else '    ')
rank = sorted(out.items(), key=lambda kv: -kv[1]['opp_deaths'])
print('\nConfrontation ranking (opp deaths/game):')
for i, (name, a) in enumerate(rank, 1):
    print(i, name, a['opp_deaths'], 'oppCash40', a['opp_cash40'])
