"""Align only our neural rollout assignment with our public actor's existing endgame return policy."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'cold_start/candidate.py'
SHA='7cf5c6c91fc59b46b2a38924c4c5ab56143fb1242d082f9426303af7336ac3ed'
def build():
    raw=PARENT.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    src=raw.decode().replace('\r\n','\n')
    old='def _sample(players, me, lw, lb, rng, step, fixed=None):'
    new='def _sample(players, me, lw, lb, rng, step, fixed=None, home_x=None):'
    assert src.count(old)==1;src=src.replace(old,new)
    old="""    cand = []
    for wid, w in workers.items():
        if w[3] == me and wid not in (fixed or {}):
            p = _probs(lw[w[0]])
            a = rng.choice(5, p=p)
            cand.append((p[a], wid, w, p, a))"""
    new="""    cand = []
    route_home = home_x is not None and step >= LAST_TURN + 1 - (SIZE // 2 * 2) - _HOME[0]
    if route_home:
        our_bases = {pos for _, pos in bases[me]}
        their_bases = {pos for _, pos in bases[1 - me]}
    for wid, w in workers.items():
        if w[3] == me and wid not in (fixed or {}):
            p = _probs(lw[w[0]])
            a = rng.choice(5, p=p)
            if route_home and w[2] == 'C':
                k = go_home(w[0], w[1], p, home_x, our_bases, their_bases, step, _HOME)
                if k is not None:
                    cand.append((2.0, wid, w, p, k))
                    continue
            cand.append((p[a], wid, w, p, a))"""
    assert src.count(old)==1;src=src.replace(old,new)
    corrections=[
        ('lb[2 * i + 1]) for i in range(n))','lb[2 * i + 1], X[2 * i]) for i in range(n))'),
        ('lw2[r], lb2[r]) for r in range(len(part)))','lw2[r], lb2[r], mine[s0 + r][2]) for r in range(len(part)))'),
        ('lw2[2 * r + 1], lb2[2 * r + 1]) for r in range(len(part)))','lw2[2 * r + 1], lb2[2 * r + 1], mine[s0 + r][2]) for r in range(len(part)))'),
        ('for (lwm, lbm, lwp, lbp), b in zip(pol, live):','for (lwm, lbm, lwp, lbp, own_x), b in zip(pol, live):'),
        ("am = _sample(o['players'], me, lwm, lbm, r1, o['step'])","am = _sample(o['players'], me, lwm, lbm, r1, o['step'], home_x=own_x)")]
    for old,new in corrections:
        assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    return src.replace('\n','\r\n').encode()
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args()
    data=build();args.out.write_bytes(data);print(hashlib.sha256(data).hexdigest())
