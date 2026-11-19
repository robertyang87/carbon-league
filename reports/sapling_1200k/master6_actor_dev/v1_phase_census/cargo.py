"""Follow cash from each load held entering the last 49 advances, including friendly relays."""
import gzip
import inspect
from contextlib import contextmanager
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0,str(HERE.parent / 'lot_settlement_dev'))
import lots


@contextmanager
def actual_harvest():
    from zerosum_env.envs.carbon import helpers
    body, first = inspect.getsourcelines(helpers.Board.next)
    lines = [first+i for i,line in enumerate(body) if line.strip() == 'worker._carbon += delta_carbon']
    if len(lines) != 1:
        raise ValueError('changed official harvest assignment')
    previous = helpers.Worker.__setattr__
    inherited = '__setattr__' not in helpers.Worker.__dict__
    events = []
    def capture(worker,key,value):
        if key == '_carbon' and hasattr(worker, key):
            frame = sys._getframe(1)
            if frame.f_code is helpers.Board.next.__code__ and frame.f_lineno == lines[0]:
                events.append((worker.player_id,worker.id,float(value-worker.carbon)))
        previous(worker,key,value)
    helpers.Worker.__setattr__ = capture
    try:
        yield events
    finally:
        if inherited:
            del helpers.Worker.__setattr__
        else:
            helpers.Worker.__setattr__ = previous


def follow(row, events):
    path = ROOT / row['sources'][0]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('different action tape')
    raw = json.loads(gzip.decompress(payload))
    sim = lots.FastCarbon(raw['game_configuration']).reset(seed=raw['seed'])
    for joint in raw['actions'][:250]:
        sim.step(*joint)
    boundary = sim.full_observation()
    if round(sim.cash[row['seat']]-sim.cash[1-row['seat']],2) != row['phases'][-2]['cash_lead']:
        raise ValueError('state 250 differs from cash census')
    harvest = [Counter(),Counter()]
    events.clear()
    stock = {}
    moves = []
    for i, joint in enumerate(raw['actions'][250:]):
        values, transitions, _ = lots.one_step(sim, joint, None, verify=True)
        stock.update({(i, uid):v for uid,v in values.items()})
        moves.append(transitions)
    for owner, uid, amount in events:
        if amount < 0:
            raise ValueError('negative official harvest')
        harvest[owner][uid] += amount
    if list(sim.cash) != raw['last_post_cash']:
        raise ValueError('terminal cash mismatch')
    ending = lots.positive_collectors(sim.full_observation(),None)
    stock.update({(len(moves), uid):v for uid,v in ending.items()})
    timeline = lots.Timeline(stock,moves,len(moves))
    seats = []
    for side, player in enumerate(boundary['players']):
        detail = []
        values = Counter()
        for uid,(pos, amount, role) in sorted(player[2].items()):
            if role != 'COLLECTOR' or amount<=0:continue
            events = timeline.events(0,uid)
            for e in events:
                e['step'] += 250
                if e['is_terminal_allocation']:
                    values[e['reason']] += e['amount']
            detail.append(dict(worker=uid,pos=pos,amount=amount,events=events))
        balance = sum(x['amount'] for x in detail)
        if abs(sum(values.values())-balance) > 1e-4:
            raise ValueError('lost opening principal')
        normal = row['phases'][-1]['accounts'][side].get('own_delivery',0)
        remaining = normal-values['own_bank']
        if remaining < -.005:
            raise ValueError('pre-existing principal exceeds normal deposits')
        seats.append(dict(gross_new_harvest=sum(harvest[side].values()),harvest_by_worker=dict(harvest[side]),opening_cargo=balance,starting_cargo_normally_banked=values['own_bank'],
                          later_other_principal_normally_banked=remaining, normal_deposits=normal,
                          normal_amount_by_harvest_identity='Residual can include new harvest, captured enemy freight or regathered dropped cargo.',
                          allocations=dict(values), workers=detail))
    return dict(seed=row['seed'],seat=row['seat'],terrain=row['map_sha256'],sha256=row['sha256'],by_seat=seats)


def main():
    src = json.loads(gzip.decompress((HERE/'games.json.gz').read_bytes()))
    with actual_harvest() as events:
        records = [follow(r,events) for r in src]
    if not all(sum(a['gross_new_harvest'] for a in r['by_seat'])>0 for r in records):
        raise ValueError('no recorded harvest')
    (HERE/'cargo250.json.gz').write_bytes(gzip.compress(json.dumps(records,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    for key in ['gross_new_harvest','opening_cargo','starting_cargo_normally_banked','later_other_principal_normally_banked','normal_deposits']:
        totals = [sum(r['by_seat'][r['seat']^j][key] for r in records)/len(records) for j in (0,1)]
        print(key, totals)


if __name__=='__main__':
    main()
