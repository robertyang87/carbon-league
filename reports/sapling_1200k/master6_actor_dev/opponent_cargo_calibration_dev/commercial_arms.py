"""Physical post-movement transfers and subsequent normal base deposits on sealed native response tapes."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACTOR = HERE.parent
ROOT = ACTOR.parents[2]
sys.path.insert(0, str(ACTOR/'lot_settlement_dev'))
import lots

V1 = '9c54df17eba7cfecf6ac2698eccbbadc2607665e16ab613925188131ce5514e0'
MASTER = '65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'


def _checked(ok, note):
    if not ok:
        raise ValueError(note)


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _bytes(path, sha):
    b = path.read_bytes()
    _checked(hashlib.sha256(b).hexdigest() == sha, f'changed signed arm {path}')
    return b


def _tape(row, arm):
    meta = row['arms'][arm]
    source = ROOT/meta['action_file']
    raw = _bytes(source,meta['action_sha256'])
    if row['arms'][arm]['receipt'] != meta['action_file']:
        _bytes(ROOT/meta['receipt'], meta['receipt_sha256'])
    else:
        _checked(meta['action_sha256'] == meta['receipt_sha256'], 'receipt seal')
    data = json.loads(gzip.decompress(raw) if source.suffix == '.gz' else raw)
    signed_cash = None
    if isinstance(data,list):
        tape = data
    elif 'joint0' in data and 'history' in data:
        tape = [data['joint0'], *[h['joint'] for h in data['history']]]
        signed_cash = data.get('cash')
    elif 'orders' in data:
        tape = data['orders']
        signed_cash = data.get('final_cash') or data.get('cash')
    elif row['family']=='endgame_queue':
        proof = data['arms']['v1' if arm=='parent' else 'candidate']
        tape = [proof['first_joint'],proof['last_joint']]
        signed_cash = proof['end']['cash']
    else:
        raise ValueError(f'unknown sealed action format {source}')
    expected = row[arm],row['root_opponent']
    s = row['seat']
    clean = lambda x:{k:v for k,v in x.items() if v and v!='NONE'}
    _checked(len(tape)==meta['calls']+1 and meta['terminal_step']==row['t']+len(tape)
             and [clean(tape[0][i]) for i in (s,1-s)]==list(expected),'signed root/orders differ')
    return tape,signed_cash


def _arrivals(sim,pair):
    out={}
    for s,p in enumerate(sim.full_observation()['players']):
        for uid,(pos,cargo,role) in p[2].items():
            move = pair[s].get(uid)
            moving = move in sim._move_nbr
            dest = sim._move_nbr[move][pos] if moving else pos
            out[uid]={'uid':uid,'seat':s,'role':role,'origin':pos,'cell':dest,'move':move or 'NONE',
                      'cargo_before_move':cargo,'cargo_at_collision':cargo*(1-sim.config['moveCost']) if moving else cargo}
    return out


def _root_contests(sim,joint,seat):
    parties = _arrivals(sim,joint)
    # Expired trees are removed before collision. Their square may then route to an underlying base.
    landmark={p:('base_collision_cash',s) for s,bases in sim.bases.items() for _,p in bases}
    landmark.update({tr.pos:('tree_collision_cash',tr.player) for tr in sim.trees.values()
                     if tr.age < sim.config['treeLifespan']})
    next_id = list(sim.worker_counters)
    lots.one_step(sim,joint,None,True)
    events = sim.cash.events
    for e in events:
        tag,s=e[:2]
        if tag in ('hire_collector','hire_planter'):
            uid=f'player-{s}-worker-{next_id[s]}'
            next_id[s]+=1
            parties[uid]={'uid':uid,'seat':s,'role':tag[5:].upper(),'origin':e[4],'cell':e[4],
                          'move':tag,'cargo_before_move':0.,'cargo_at_collision':0.}
    bycell=defaultdict(list)
    for p in parties.values():
        bycell[p['cell']].append(p)
    land={e[3]:e for e in events if e[0] in ('tree_collision_cash','base_collision_cash')}
    dead={uid:(p,v,b,w) for uid,p,v,b,w in sim.last_collisions}
    counts=Counter()
    contests=[]
    for cell,workers in sorted(bycell.items()):
        if len(workers)<2:
            continue
        survivors=[w for w in workers if w['uid'] not in dead]
        _checked(len(survivors)<=1,'two collision survivors')
        hunter=survivors[0] if survivors else None
        # Death in a later enemy-base settlement is intentionally distinct from losing this collision.
        transfers=[]
        for w in workers:
            uid=w['uid']
            if uid not in dead:continue
            owner,qty,beneficiary,collector=dead[uid]
            _checked(owner==w['seat'] and math.isclose(qty,w['cargo_at_collision'],abs_tol=1e-7),'postmove cargo')
            if collector is not None:
                _checked(hunter is not None and hunter['seat']==collector==beneficiary and hunter['role']=='COLLECTOR','winner identity')
                route='enemy_worker_capture' if collector!=owner else 'friendly_worker_relay'
                if route=='enemy_worker_capture':
                    counts[collector]+=qty
            elif uid in land:
                _checked(hunter is None and beneficiary==land[uid][1] and math.isclose(land[uid][2],qty,abs_tol=1e-7),'landmark cash')
                route=land[uid][0]
            elif beneficiary is not None:
                # CashEvents omits exact-zero writes. Prove the venue, seat and zero cargo independently.
                _checked(hunter is None and qty==0 and cell in landmark and
                         landmark[cell][1]==beneficiary,'unaccounted landmark transfer')
                route=landmark[cell][0]
            else:
                _checked(hunter is None and cell not in landmark,'missing recorded beneficiary')
                route='ground'
            transfers.append({'victim_uid':uid,'victim_seat':owner,'beneficiary_seat':beneficiary,
                              'hunter_uid':hunter['uid'] if hunter else None,'route':route,'post_move_quantity':qty})
        contests.append({'cell':cell,'both_seats_present':len({w['seat'] for w in workers})==2,'participants':workers,
                         'winner_uid':hunter['uid'] if hunter else None,'winner_seat':hunter['seat'] if hunter else None,
                         'root_end_winner_alive':(hunter['uid'] in sim.workers if hunter else None),'transfers':transfers})
    _checked(sum(len(v['transfers']) for v in contests)==len(dead),'unassigned collision')
    return contests,[counts[seat],counts[1-seat]]


def _extinct(view, cfg):
    floor=min(cfg['recCollectorCost'],cfg['recPlanterCost'])
    return [not (work or trees) and (not bases or cash<floor) for cash,bases,work,trees in view['players']]


@lru_cache(maxsize=8)
def _official_root(raw_name, raw_hash, seat, t):
    sys.path.insert(0,str(ACTOR.parent/'root_recovery_branches'))
    from branches import begin, public
    p=ROOT/raw_name
    payload=_bytes(p,raw_hash)
    raw=json.loads(gzip.decompress(payload) if p.suffix=='.gz' else payload)
    import random
    import numpy as np
    py_state=random.getstate()
    np_state=np.random.get_state()
    try:
        env=begin(raw)
        for joint in raw['actions'][:t]:
            _checked(not env.done,'root follows terminal')
            env.step(joint)
        return public(env,seat)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _verify_root(row, board, official):
    view={k:board[k] for k in ('player','step','players','trees','carbon')}
    _checked(view['step']==row['t'] and view['player']==row['seat'],'root time or seat')
    if official is None and _digest(view)==row['public_root_sha256']:
        return 'exact_serialized'
    if official is None:
        official=_official_root(row['source'],row['source_sha256'],row['seat'],row['t'])
    original={k:official[k] for k in view}
    _checked(_digest(original)==row['public_root_sha256'],'official source hash changed')
    _checked(all(_digest(view[k])==_digest(original[k]) for k in ('player','step','players','trees')),
             'entity/actor or tree state differs from signed public view')
    a,b=view['carbon'],original['carbon']
    _checked(len(a)==len(b) and all(type(x) in (int,float) and type(y) in (int,float) and
             math.isfinite(x) and math.isfinite(y) and x==y for x,y in zip(a,b)),
             'visible carbon values differ from exact official board')
    return 'exact_serialized' if _digest(view)==row['public_root_sha256'] else 'identical_carbon_numbers_different_JSON_types'


def _evaluate(row,name,snapshot,config,expected_public):
    orders,cash_proof=_tape(row,name)
    s=row['seat'];sim=lots.FastCarbon(config);sim.set_state(snapshot)
    # set_state casts board carbon to float. Restore exact element types for the official JSON hash.
    sim.carbon[:] = snapshot['carbon']
    before=sim.obs(s); public={k:v for k,v in before.items() if k!='remainingOverageTime'}
    verification=_verify_root(row,public,expected_public)
    _checked(not any(_extinct(before,sim.config)) and sim.step_count<sim.config['episodeSteps']-1,'terminal root')
    start=list(sim.cash)
    contests,gains=_root_contests(sim,orders[0],s)
    root_deposit=Counter();future=Counter();cashflow=Counter();crossbase=Counter()
    for j,pair in enumerate(orders):
        if j:
            before=sim.obs(s)
            _checked(not any(_extinct(before,sim.config)) and sim.step_count<sim.config['episodeSteps']-1,'extra command after terminal')
            sim.cash.events.clear()
            sim.step(*pair)
        for e in sim.cash.events:
            tag,side,amount=e[:3]
            cashflow[side]+=amount
            if tag=='own_delivery':
                (future if j else root_deposit)[side]+=amount
            elif tag=='opponent_worker_delivery_to_this_base' and j:
                crossbase[side]+=amount
    post=sim.obs(s);ended=_extinct(post,sim.config);endcash=[v[0] for v in post['players']]
    _checked(sim.step_count==row['arms'][name]['terminal_step'] and (any(ended) or sim.step_count==sim.config['episodeSteps']-1),'short suffix')
    _checked(all(math.isclose(start[i]+cashflow[i],sim.cash[i],rel_tol=1e-10,abs_tol=1e-6) for i in (0,1)),'physical cash conservation')
    scores=[(sim.step_count-sim.config['episodeSteps']-1 if ended[i] else before['players'][i][0] if any(ended) else endcash[i]) for i in (0,1)]
    _checked(scores==row[name+'_scores'],'official score/extinction mismatch')
    if cash_proof is not None:
        _checked(len(cash_proof)==2 and all(abs(x-y)<=0.00500001 for x,y in zip(cash_proof,endcash)),'stored physical cash differs')
    return {'root_worker_capture_own':gains[0],'root_worker_capture_enemy':gains[1],'root_contests':contests,
            'root_normal_deposit_own':root_deposit[s],'root_normal_deposit_enemy':root_deposit[1-s],
            'suffix_normal_deposit_own':future[s],'suffix_normal_deposit_enemy':future[1-s],
            'suffix_foreign_worker_deposit_to_own_base':crossbase[s],'suffix_foreign_worker_deposit_to_enemy_base':crossbase[1-s],
            'terminal_physical_cash':endcash,'terminal_unrounded_treasury':list(sim.cash),'official_scores':scores,
            'terminal_extinction':ended,'terminal_step':sim.step_count,'normal_deposit_suffix_steps':len(orders)-1,
            'signed_terminal_physical_cash_available':cash_proof is not None,'action_file_sha256':row['arms'][name]['action_sha256'],'root_public_check':verification}


def audit_arms(row, snapshot, config, expected_public=None):
    """Use a complete get_state() snapshot before the shared signed root. Coordinates are physical seat unless named own/enemy."""
    _checked(row['parent_sha256']==V1 and row['examiner_sha256']==MASTER,'unexpected parent or examiner')
    _checked('tid_counter' in snapshot and 'wid_counter' in snapshot,'complete simulator state required')
    # Board and the official outer replay share a process-level counter table.
    # Preserve the outer interpreter across both diagnostic branches and exception paths.
    ids=lots.idgen._id_dict.copy()
    try:
        return {name:_evaluate(row,name,snapshot,config,expected_public) for name in ('parent','alternative')}
    finally:
        lots.idgen._id_dict.clear()
        lots.idgen._id_dict.update(ids)
