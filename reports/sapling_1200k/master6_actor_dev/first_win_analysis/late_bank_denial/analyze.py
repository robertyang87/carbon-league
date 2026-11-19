"""Frozen freight cover on all 28 late-game tapes. Opponent responses are never queried."""
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parents[1]
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ACTOR / 'freight_response'), str(ACTOR / 'freight_horizon'), str(BASE)]
from policy import choose_intervention, destination, geometry, home
from predictor import FreightPredictor
from turning_point_ledger import FastCarbon, CashEvents

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    blob = path.read_bytes()
    return json.loads(gzip.decompress(blob) if path.suffix == '.gz' else blob)

def tick(sim, pair):
    sim.cash = CashEvents(list(sim.cash))
    sim.step(*pair)
    return list(sim.cash.events)

def deposits(events):
    return [[side, uid, cell, value] for kind, side, value, *data in events
            if kind == 'own_delivery' for uid, cell in [data]]

def money(rows, side, uid=None):
    return sum(row[3] for row in rows if row[0] == side and (uid is None or row[1] == uid))

def test_change(state, cfg, orders, obs, offer, recorded):
    seat = obs['player']
    uid, target = offer['worker'], offer['target_worker']
    outputs = []
    for new in (False, True):
        sim = FastCarbon(cfg)
        sim.set_state(state)
        path = []
        pay = []
        for i, pair in enumerate(orders):
            p = [dict(x) for x in pair]
            if i == 0 and new:
                p[seat][uid] = offer['alternative']
            events = tick(sim,p)
            receipts = deposits(events)
            if not new:
                assert receipts == recorded[i]['deposits'] and list(sim.cash) == recorded[i]['cash'], 'baseline differs'
            pay.extend(receipts)
            w = sim.workers.get(uid)
            path.append({'h':i,'receipts':receipts,'bank': [money(pay,s) for s in (seat,1-seat)],
                'own_uid_bank':money(pay,seat,uid),'target_uid_bank':money(pay,1-seat,target),
                'own_alive':w is not None,'remaining_own_cargo':w.carbon if w else 0,
                'remaining_own_cell':w.pos if w else None,'cash_margin':sim.cash[seat]-sim.cash[1-seat],
                'target_alive':target in sim.workers})
        outputs.append(path)
    return outputs

def examine(game, model):
    tape = ROOT/game['raw']
    raw = read(tape)
    assert sha(tape)==game['raw_sha256'] and raw['pack_sha256']==game['pack_sha256'] and game['turns']==299
    seat = game['seat']
    cfg = raw['game_configuration']
    sim = FastCarbon(cfg).reset(seed=game['seed'])
    assert list(sim.carbon)==raw['initial_full_carbon']
    hist, steps, snapshots, picks = {}, {}, {}, {}
    lead269 = None
    for t,pair in enumerate(raw['actions']):
        if t>=256:
            obs = sim.obs(seat)
            hist[t] = obs
            if t==269:
                lead269=sim.cash[seat]-sim.cash[1-seat]
            if t>=261:
                snapshots[t]=sim.get_state()
                if t<=289:
                    q=model.predict(obs,[hist[j] for j in range(t-5,t)])
                    hit=choose_intervention(obs,cfg,seat,pair[seat],q)
                    if hit:
                        picks[t]=hit
        events=tick(sim,pair)
        if t>=261:
            steps[t]={'deposits':deposits(events),'cash':list(sim.cash)}
    assert list(sim.cash)==raw['last_post_cash'] and list(sim.cash)==game['official_scores']
    assert abs(sim.cash[seat]-sim.cash[1-seat]-game['official_margin'])<1e-5
    bank=[[t,*x] for t in range(269,299) for x in steps[t]['deposits']]
    sum_bank=[sum(x[4] for x in bank if x[1]==s) for s in (seat,1-seat)]
    challenges=[]
    for t,offer in picks.items():
        rem=min(9,299-t)
        paths=test_change(snapshots[t],cfg,raw['actions'][t:t+rem],hist[t],offer,[steps[j] for j in range(t,t+rem)])
        own=hist[t]['players'][seat]
        bases=own[1].values()
        od=home(destination(offer['own_cell'],offer['old'],15),bases)
        nd=home(offer['landing'],bases)
        origin=offer['own_cell']
        # Document where the actual parent headed versus the changed first step.
        history=[{'t':j,'own':hist[j]['players'][seat][2].get(offer['worker']),
                  'opponent':hist[j]['players'][1-seat][2].get(offer['target_worker'])}
                 for j in (t-5,t-3,t)]
        challenges.append(dict(t=t,offer=offer,history=history,home_distance=[home(origin,bases),od,nd],
                               arms={'recorded':paths[0],'changed':paths[1]}))
    credits=[]
    for t,s,uid,cell,amount in bank:
        if s==seat:continue
        prev=[]
        for lag in range(9):
            j=t-lag; obs=hist[j]
            worker=obs['players'][1-seat][2].get(uid)
            offers,_=geometry(obs, raw['actions'][j][seat],uid)
            current=picks.get(j)
            matched=bool(current and current['target_worker']==uid)
            affordable=[r for r in offers if r['own_load']<=35 and
                        home(r['landing'],obs['players'][seat][1].values())<=
                        home(destination(r['own_cell'],r['old'],15),obs['players'][seat][1].values())]
            prev.append(dict(lag=lag,t=j,enemy=worker,parent_enemy_cmd=raw['actions'][j][1-seat].get(uid,'NONE')
                             if worker else None,geometry_options=len(offers),cheaper_options=len(affordable),
                             selected_target=matched))
        credits.append(dict(t=t,uid=uid,cell=cell,amount=amount,back=prev))
    return {'seed':game['seed'],'map':game['map'],'seat':seat,'source':game['source'],'raw_sha256':game['raw_sha256'],
            'result':game['result'],'margin':game['official_margin'],'lead269':round(lead269,2),
            'bank_all':bank,'own_enemy_bank':sum_bank,'enemy_credits':credits,'changes':challenges}

def main():
    src=HERE.parent/'evidence.json.gz'
    data=read(src)
    assert len(data['games'])==28 and data['new_complete'] and len(set((g['map'],g['seat']) for g in data['games']))==28
    model=FreightPredictor(ACTOR/'freight_horizon/model.json')
    games=[examine(g,model) for g in data['games']]
    info=dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [src,HERE/'protocol.json',Path(__file__),
             ACTOR/'freight_response/policy.py',ACTOR/'freight_horizon/predictor.py',ACTOR/'freight_horizon/model.json']},
             games=games)
    dest=HERE/'by_game.json.gz'
    dest.write_bytes(gzip.compress(json.dumps(info,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    print(len(games),sum(len(x['changes']) for x in games),flush=True)
if __name__=='__main__':
    main()
