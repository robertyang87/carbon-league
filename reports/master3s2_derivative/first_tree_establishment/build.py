"""Offer early establishment only on already-reached, payable and uncontested sites."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
PARENT = FAMILY/'ladder/prediction_feedback/package/candidate.py'
PARENT_SHA = '8b4c4bafba2b3b2887139163bd6397b3f7d1acf178a43ea1756d8f80306efe43'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


POLICY = '''def _opening_plant(obs, configuration, est, known, actions, st):
    if st.get('first_tree_seen') or obs['step'] > 25:
        return actions
    players = obs['players']
    me = obs['player']
    workers, trees, bases, cash = parse(players)
    if players[me][3]:
        st['first_tree_seen'] = True
        return actions
    if obs['step'] < 4:
        return actions
    # With no owned trees, the platform's protected first-tree price applies.
    price = float(configuration.get('plantCost', DEFAULT_CONFIG['plantCost']))
    recruits = sum(float(configuration.get(
        'recPlanterCost' if actions.get(bid) == 'RECPLANTER' else 'recCollectorCost', 30))
        for bid, _ in bases[me] if bid in actions)
    if cash[me] < recruits + price:
        return actions
    forbidden = {v[0] for v in trees.values()} | {p for team in bases for _, p in team}
    enemy = [v[0] for v in workers.values() if v[3] != me]
    team = {uid: v for uid, v in workers.items() if v[3] == me}
    planned = {uid: move(v[0], WIDX.get(actions.get(uid, 'NONE'), 0)) for uid, v in team.items()}
    offers = []
    for uid, (cell, _, role, _) in team.items():
        if role != 'P' or cell in forbidden or any(DIST[cell][p] <= 2 for p in enemy):
            continue
        if any(z == cell for u,z in planned.items() if u != uid):
            continue
        ring = [cell, *NB8[cell]]
        if sum(bool(known[p]) for p in ring) < 5:
            continue
        # Only consider sites whose known/estimated local carbon could cover
        # the first tree price within roughly five initial absorption turns.
        potential = sum(est[p] for p in ring)
        rate = float(configuration.get('cellAbsorptionRate', DEFAULT_CONFIG['cellAbsorptionRate']))
        if rate * potential * 5.0 >= price:
            offers.append((potential, uid))
    if not offers:
        return actions
    _, uid = max(offers)
    if uid not in actions:
        return actions
    result = actions.copy()
    result.pop(uid)
    return result


'''


def build():
    raw = PARENT.read_bytes()
    assert sha(raw) == PARENT_SHA
    boundary = raw.index(b'def _act(')
    old = b"    st['tr'].set_actions(players, act)\r\n"
    desc = """    if step == 0:
        st.pop('first_tree_seen', None)
    if configuration is not None and 0 < step <= 25:
        act = _opening_plant(obs, configuration, est, known, act, st)
"""
    assert raw.count(old) == 1
    result = raw[:boundary] + POLICY.replace('\n','\r\n').encode() + raw[boundary:]
    result = result.replace(old,desc.replace('\n','\r\n').encode()+old)
    dst = HERE/'candidate.py'
    compile(result, str(dst), 'exec')
    if dst.exists():
        assert dst.read_bytes() == result
    else:
        dst.write_bytes(result)
    return sha(result)


if __name__ == '__main__':
    print(build())
