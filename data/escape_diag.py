"""escape_diag -- why did the search not run away? Replays v17's non-Master losses; for every seizure of one of our
loaded collectors (cargo >= 150), re-runs our search (v17 config, float nets, LOG on) along our side's observations
and prints, for the 3 steps before the seizure: was the victim a key unit, was it searched, candidate moves and
their values vs the default, and what was chosen."""
import gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'env_src', 'carbon_challenge-master'))
import cnet2_search as S
import cnet2_feat as F
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
CONF = {'MODE': 'mcts', 'CLOCK': 'wall', 'BUDGET': 99.0, 'HARD': 99.0, 'ABORT_DL': None, 'START_DL': None, 'ROOT_S': 10,
        'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True,
        'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240, 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7),
        'ID_PRIOR': 0.0, 'LOG': True}
me_name = 'claude-opus55-cloud-v17'
MASTER = ('claude-AlphaCarbon-Master2s', 'claude-AlphaCarbon-Master2')
ms = [m for m in json.load(open('../data/v17_matches.json')) if m.get('scores')]
summary = {"events": 0, "victim_key": 0, "victim_searched": 0, "changed": 0, "not_key": 0}
for m in ms:
    a = m['name_a'] == me_name; s = m['scores']; d = (s['a'] - s['b']) * (1 if a else -1); o = m['name_b' if a else 'name_a']
    if d >= 0 or o in MASTER:
        continue
    f = f"../data/v17_ladder/replays/{m['id']}-{m.get('game', 1)}.json.gz"
    stp = json.load(gzip.open(f))['steps']; order = m.get('order') or ['a', 'b']; pm = order.index('a' if a else 'b')
    events = []
    for t in range(len(stp) - 1):
        P0 = stp[t][0]['observation']['players']; P1 = stp[t + 1][0]['observation']['players']
        for uid, w in P0[pm][2].items():
            if w[2] == 'COLLECTOR' and w[1] >= 150 and uid not in P1[pm][2]:
                events.append((t + 1, uid, w[1]))
    if not events:
        continue
    ag = S.make_search_agent(dict(CONF), weights=os.path.join(W, 'cnet2_b128v9.npz'), rweights=os.path.join(W, 'cnet2_s64v8.npz'))
    want = {}
    for (te, uid, c) in events:
        for k in (te - 3, te - 2, te - 1):
            want.setdefault(k, []).append((te, uid, c))
    print(f"\n=== {o} {d:+.0f}  seizures of >=150: {[(t, round(c)) for t, _, c in events]}")
    for t in range(0, max(want) + 1):
        o0 = stp[t][0]['observation']
        obs = {"step": o0["step"], "players": o0["players"], "carbon": stp[t][pm]['observation']['carbon'], "player": pm,
               "remainingOverageTime": 60}
        nlog = len(ag.log)
        act = ag(obs, None)
        if t not in want:
            continue
        keys = S.key_units(obs, ag.cfg)
        newlog = ag.log[nlog:]
        for (te, uid, c) in want[t]:
            summary["events"] += t == te - 1
            isk = uid in [u for _, u in keys]
            pos = obs["players"][pm][2][uid][0] if uid in obs["players"][pm][2] else None
            en = [(e[0], round(e[1])) for e in obs["players"][1 - pm][2].values() if e[2] == 'COLLECTOR' and pos is not None and F.DIST[pos, e[0]] <= 3]
            line = f"  t{t} (seized at t{te}, cargo {c:.0f}) victim key={isk} keys={[(round(k), u.split('-')[-1]) for k, u in keys][:3]} enemy collectors <=3: {en} | our move {act.get(uid)}"
            if newlog:
                st_, cmeta, vals, best = newlog[-1][:4]
                line += f" | searched: cand {cmeta} vals {vals} best {best}"
            print(line)
            if t == te - 1:
                summary["victim_key"] += isk; summary["victim_searched"] += bool(newlog); summary["not_key"] += not isk
print("\n", json.dumps(summary))
