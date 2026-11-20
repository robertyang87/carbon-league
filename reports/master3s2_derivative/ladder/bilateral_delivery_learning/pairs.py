"""Materialize the registered actual ten-transition self-deliveries for paired team orders."""
from functools import lru_cache
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
D = HERE.parents[1]
EXTRA = D/'root_rotation_policy/terminal/extra.py'
TARGET = 'own_delivery'
BANK_KEYS = (TARGET,'opponent_worker_delivery_to_this_base','base_collision_cash')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verified(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and digest(raw) != expected:
        raise ValueError(f'Incorrect receipt hash {path}')
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same(x, y):
    if x != y:
        raise ValueError(f'Receipt identity differs: {x!r} versus {y!r}')


def pair_key(c):
    return [c[k] for k in ('seed','seat','step','group')]


def legacy():
    lock = verified(D/'parallel_responses/terminal_teacher/protocol.json')
    valid, missing = [], []
    for c in lock['tickets']:
        q = {k:c[k] for k in ('seed','terrain','seat','step','group','source_sha')}
        if c['status'] == 'skipped':
            missing.append(dict(**q,reason='native_root_not_paired',paid_transitions=0))
            continue
        seat, seed, t = c['seat'], c['seed'], c['step']
        source = verified(D/c['root_path'],c['root_sha'])
        stem = f'{seed}_{seat}_{t}'
        if c['group']=='close':
            folder = D/'return_response_learning/close_games'
            r = source['rows'][c['close_ordinal']]
            tmp = verified(folder/'cases'/f"{c['close_ordinal']}.json")
            same(tmp['root'], [seed,seat,t])
            order, players, pub = r['orders'],r['players'],r['public_sha']
            names=('A','B')
            refs=[folder/'evidence'/f'{stem}_{name}.json.gz' for name in names]
        else:
            folder = D/'action_rank_data'/c['group']
            tmp = verified(folder/f'{stem}_pair.json')
            same(tmp['root_sha256'],c['root_sha'])
            same(tmp['status'],'complete')
            order=[a['order'] for a in source['arms']]
            players=source['public_root']['players']
            pub=source['authority']['public_sha256_by_seat'][seat]
            names=('first','second')
            refs=[folder/f'{stem}_{name}.json.gz' for name in names]
        margins=[tmp['arms'][name]['margin'] for name in names]
        receipts=[(p,tmp['arms'][n]['evidence_sha256']) for p,n in zip(refs,names)]
        valid.append(dict(**q,orders=order,players=players,public_sha=pub,margins=margins,receipts=receipts))
    return valid,missing


def mean_flow(card,outs):
    """Recompute the single long mean pair from physical orders; its source tape has no cash-event fields."""
    module=load_module('bilateral_mean_cash',D.parent/'sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/native_actor_corpus/league_policy/natural_original/economy.py')
    c=card['context']
    engine=module.FastCarbon(module.config()).reset(seed=c['seed'])
    same(digest(bytes(map(int,engine.carbon))),c['terrain'])
    with zipfile.ZipFile(D/c['source_path']) as z:
        turns=z.read(c['source_arm']+'/turns.jsonl').splitlines()
        for raw in turns[:c['step']]:
            r=json.loads(raw)
            engine.step(*(v['response']['action'] for v in r['calls']))
    same(engine.obs(c['seat'])['players'],card['players'])
    all_sides=[]
    for tape in outs:
        sim=copy.deepcopy(engine);sim.cash=module.CashEvents(sim.cash);values=[{k:[] for k in BANK_KEYS} for _ in (0,1)]
        for i,turn in enumerate(tape):
            sim.cash.events.clear();sim.step(*turn['orders']);same(list(sim.cash),turn['end_cash'])
            if i<10:
                for k,s,value,*details in sim.cash.events:
                    if k in BANK_KEYS:values[s][k].append(value)
        all_sides.append([{k:math.fsum(values[s][k]) for k in BANK_KEYS} for s in (c['seat'],1-c['seat'])])
    return all_sides


def settle(c):
    s,t=c['seat'],c['step']
    amounts=[];tapes=[];enemy=[];requests=[]
    for idx,(p,h) in enumerate(c['receipts']):
        ev=verified(p,h);tape=ev.get('rows',ev.get('tape'))
        steps=[v.get('t',v.get('step')) for v in tape];same(steps,list(range(t,299)))
        tape_0=tape[0];commands=tape_0.get('orders',tape_0.get('actions'))
        same(commands[s],c['orders'][idx]);enemy.append(commands[1-s])
        request=tape_0.get('input_sha256',tape_0.get('public_sha256',tape_0.get('public_sha')))
        same(request[s],c['public_sha']);requests.append(request)
        outcome=ev.get('outcome',ev)
        scores=outcome['scores'];same(scores,tape[-1].get('end_cash',tape[-1].get('wallets')))
        assert abs((scores[s]-scores[1-s])-c['margins'][idx])<1e-7
        if 'flow' in tape_0:
            totals=[{k:math.fsum(r['flow'][i].get(k,0) for r in tape[:10]) for k in BANK_KEYS} for i in (s,1-s)]
            for n,i in zip(totals,(s,1-s)):
                assert all(abs(n[k]-outcome['first10'][i]['components'].get(k,0))<1e-7 for k in BANK_KEYS)
            amounts.append(totals)
        elif 'flows' in tape_0:
            amounts.append([{key:math.fsum(value for r in tape[:10] for k,j,value,*_ in r['flows'] if k==key and j==i) for key in BANK_KEYS} for i in (s,1-s)])
        tapes.append(tape)
    same(enemy[0],enemy[1]);same(requests[0],requests[1]);same(len(tapes[0]),len(tapes[1]))
    if c['orders'][0]==c['orders'][1]:raise ValueError('The physical orders must differ')
    if len(tapes[0])<10:
        return None,dict(group=c['group'],seed=c['seed'],seat=s,step=t,terrain=c['terrain'],reason='fewer_than_ten_transitions',paid_transitions=len(tapes[0]))
    if not amounts:amounts=mean_flow(c,tapes)
    same(len(amounts),2)
    hashes=[h for _,h in c['receipts']]
    res={k:c[k] for k in ('seed','terrain','seat','step','players','orders','margins','group','source_sha')}
    own=[[side[TARGET] for side in arm] for arm in amounts]
    bank=[[math.fsum(side.values()) for side in arm] for arm in amounts]
    return dict(**res,actual=own,all_bank_actual=bank,bank_components=amounts,branch_sha256=hashes,source_public_sha256=c['public_sha']),None


@lru_cache(maxsize=1)
def inventory():
    ext=load_module('bilateral_frozen_pairs',EXTRA)
    reg=verified(HERE/'protocol.json')
    cards,excluded=legacy()
    for e in ext.cards():
        ctx=ext.context(e)
        c={k:e[k] for k in ('seed','terrain','seat','step','group','orders','margins','source_sha')}
        refs=[(D/e['references'][a]['path'],e['references'][a]['sha256']) for a in e['arms']]
        cards.append(dict(**c,players=ctx[1]['players'],public_sha=e['own_observation_sha'],receipts=refs,context=e))
    same([pair_key(c) for c in cards],verified(D/'root_rotation_policy/terminal/protocol.json')['keys'])
    expected=set(reg['other_excluded']);train={s['terrain'] for s in reg['train']}
    same({c['terrain'] for c in cards},expected)
    if train&expected:raise ValueError('Physical map in both training data and response cases')
    out=[]
    for c in cards:
        r,missing=settle(c)
        if missing:excluded.append(missing)
        else:out.append(r)
    return out,excluded


def rows():
    return inventory()[0]


def excluded():
    return inventory()[1]


if __name__=='__main__':
    data,skipped=inventory();print(len(data),'pairs',len({x['terrain'] for x in data}),'maps',[(r['seed'],r['step'],r['reason']) for r in skipped])
