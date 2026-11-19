"""Settle a threatened load when a home base is available this turn."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'intercept_response_dev'))
from forecast import _TO, _D, _DIRECTION

def choose(obs, orders):
    seat = obs['player']
    own, enemy = obs['players'][seat], obs['players'][1-seat]
    options = []
    for uid, (pos, load, kind) in own[2].items():
        if kind != 'COLLECTOR' or load < 80:
            continue
        old = orders.get(uid)
        old_cell = _TO[pos, _DIRECTION.index(old)]
        if old_cell in own[1].values():
            continue
        threats = [(int(_D[old_cell,w[0]]),k) for k,w in enemy[2].items()
                   if w[2]=='COLLECTOR' and w[1]<=load]
        if not threats or min(threats)[0] > 3:
            continue
        used = {_TO[w[0],_DIRECTION.index(orders.get(k))] for k,w in own[2].items() if k!=uid}
        for bid, base in own[1].items():
            if base in used or orders.get(bid) or _D[pos,base]!=1:
                continue
            action = _DIRECTION[list(_TO[pos]).index(base)]
            options.append(dict(worker=uid, t=obs['step'], target=min(threats)[1],
                                old=old or 'NONE',alternative=action, tree='',cargo=load,base=base))
    return max(options,key=lambda v:(v['cargo'],v['worker']),default=None)
