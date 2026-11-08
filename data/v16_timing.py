"""platform per-step timing for a strategy from the non-compact matches API (result[j].action_timing[side])"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
import ladder
sid = sys.argv[1]; me = sys.argv[2]
ladder.session()
rows = []
for off in range(0, 400, 100):
    r = ladder.request(f'/api/matches?limit=100&offset={off}&strategy={sid}')
    ms = r if isinstance(r, list) else r.get('matches', r.get('items', []))
    if not ms: break
    for m in ms:
        side = 'a' if m.get('name_a') == me else 'b'
        for g in (m.get('result') or []):
            t = (g.get('action_timing') or {}).get(side) or (g.get('action_timing') or {}).get(me)
            if t: rows.append((m.get('created', 0), t.get('calls'), t.get('total_seconds'), t.get('last_seconds'), m.get('name_b' if side == 'a' else 'name_a')))
rows.sort()
for c, calls, tot, last, opp in rows[:12]:
    print(f"calls {calls} total {tot:.1f}s mean {1000*tot/max(1,calls):.0f} ms last {1000*(last or 0):.0f} ms vs {opp}")
if rows:
    mean = sum(1000 * r[2] / max(1, r[1]) for r in rows) / len(rows)
    print(f"games {len(rows)} mean ms/step {mean:.0f} max-total {max(r[2] for r in rows):.0f}s  k_est {(mean-25)/164:.2f}")
