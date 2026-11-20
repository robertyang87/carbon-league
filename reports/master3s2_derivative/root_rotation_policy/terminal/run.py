"""Evaluate a frozen root prior on archived complete-command terminal comparisons."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
DER = HERE.parents[1]
LEGACY = DER / 'parallel_responses/terminal_teacher'
sys.path.insert(0, str(DER / 'opponent_policy'))
import original_policy

def sha(buf):
    return hashlib.sha256(buf).hexdigest()

def obs_sha(obs):
    return sha((json.dumps(obs, separators=(',', ':'), allow_nan=False)+'\n').encode())

def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)

def model(path, name, expected):
    assert sha(path.read_bytes()) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def source(card):
    seat, step = card['seat'], card['step']
    p = DER / card['root_path']
    assert sha(p.read_bytes()) == card['root_sha']
    raw = read(p)
    history = []
    if card['group'] == 'close':
        menu = raw['rows'][card['close_ordinal']]
        original = menu['ticket']
        path = DER / original['path']
        assert sha(path.read_bytes()) == original['sha256'] == card['source_sha']
        arm = original['arm']
        with zipfile.ZipFile(path) as archive, archive.open(arm + '/observations.jsonl.gz') as compressed, gzip.GzipFile(fileobj=compressed) as frames, archive.open(arm+'/turns.jsonl') as turns:
            for t in range(step+1):
                frame = json.loads(next(frames))
                obs = frame['observations'][seat]
                assert obs_sha(obs) == frame['public_observation_sha256'][seat]
                if t == step:
                    assert obs_sha(obs) == original['public_sha']
                    return history, obs, menu['orders']
                row = json.loads(next(turns))
                assert row['calls'][seat]['status'] == 'ok' and row['public_observation_sha256'][seat] == obs_sha(obs)
                history.append((obs, row['calls'][seat]['response']['action']))
    else:
        card_src = next(r for r in original_policy.PROTOCOL['games'] if r['sha256'] == card['source_sha'])
        assert card_src['expert_seat'] == seat
        for t, obs, _, issued in original_policy.free_entry.public_history(card_src):
            if t == step:
                assert all(obs[k] == raw['public_root'][k] for k in ('players', 'step', 'player', 'trees', 'carbon'))
                obs = dict(obs,remainingOverageTime=raw['public_root']['remainingOverageTime'])
                assert obs_sha(obs) == raw['authority']['public_sha256_by_seat'][seat]
                return history, obs, [arm['order'] for arm in raw['arms']]
            history.append((obs, issued))
    raise ValueError('Missing public root')

class RootBelief:
    def __init__(self, est, known):
        self.values = est, known
    def step(self, *args):
        return self.values
    def set_actions(self, *args):
        pass

def normalized(l):
    v = np.asarray(l, dtype=np.float64)
    return v - np.logaddexp.reduce(v, axis=1)[:,None]

def physical(order):
    return {k:v for k,v in order.items() if v and v!='NONE'}

def examine(mods, card, history, obs, options, margins):
    seat, step, seed = [card[k] for k in ('seat','step','seed')]
    assert len(history) == step and (obs['step'],obs['player']) == (step,seat)
    old,new = mods
    tracker = old.BeliefTracker(old._Bel, seat)
    prev = None
    for t,(ob,issue) in enumerate(history):
        assert (ob['step'],ob['player']) == (t,seat)
        tracker.step(ob['players'],t,ob['carbon'])
        tracker.set_actions(ob['players'],issue)
        prev = ob['players']
    est,kn = tracker.step(obs['players'],step,obs['carbon'])
    belief = RootBelief(est,kn)
    raw = old.encode(obs['players'],step,seat,est,kn,prev)
    assert np.array_equal(raw, new.encode(obs['players'],step,seat,est,kn,prev))
    x = np.clip(np.rint(raw * 100),0,255) * .01
    olds = old._POLICY.forward(x,old._SID,own_root=True)
    news = new._root_rotation_policy(x,new._SID)
    assert all(np.array_equal(a,b) for a,b in zip(olds,new._POLICY.forward(x,new._SID,own_root=True)))
    odds = [[normalized(l) for l in ls] for ls in (olds, news)]
    options = [physical(cmd) for cmd in options]
    assert options[0] != options[1] and len(margins)==2
    workers,_,bases,_ = old.parse(obs['players'])
    bpos = dict(bases[seat])
    features = []
    for uid in sorted(set(options[0])|set(options[1])):
        acts = [cmd.get(uid,'NONE') for cmd in options]
        if acts[0] == acts[1]:
            continue
        if uid in bpos:
            pos,load,role,index,actions = bpos[uid],0.,'base',1,old.BACT
        else:
            pos,load,role,who = workers[uid]
            assert who == seat
            role = ('loaded_collector' if load > 0 else 'empty_collector') if role=='C' else 'planter'
            index,actions = 0,old.WACT
        ai,bi = [actions.index(a) for a in acts]
        vals = [o[index][pos,[ai,bi]] for o in odds]
        features.append(dict(uid=uid,role=role,pos=int(pos),cargo=float(load),A=acts[0],B=acts[1],
                             probability=[np.exp(v).tolist() for v in vals],
                             log_B_over_A=[float(v[1]-v[0]) for v in vals]))
    score = np.array([v['log_B_over_A'] for v in features]).sum(axis=0)
    seeds = [int(sha(f'180pn/sample/{seed}/{seat}/{step}/{i}'.encode())[:8],16) % 2**31 for i in range(128)]
    def sample(mod, s):
        mod._ST = dict(tr=belief,prev=prev,rng=np.random.RandomState(s))
        return physical(mod._act(obs,None))
    originals = [sample(m,seeds[0]) for m in mods]
    old_enc,new_enc,old_net,new_net = old.encode,new.encode,old._POLICY.forward,new._root_rotation_policy
    counts = [[0]*3 for _ in mods]
    pairs = np.zeros((3,3),dtype=int)
    different = 0
    try:
        old.encode = new.encode = lambda *args: raw
        old._POLICY.forward = lambda *args,**kwargs: olds
        new._root_rotation_policy = lambda *args,**kwargs: news
        for ix,s in enumerate(seeds):
            issued = [sample(m,s) for m in mods]
            if ix == 0:
                assert issued == originals
            bins = [options.index(y) if y in options else 2 for y in issued]
            for arm,category in enumerate(bins):
                counts[arm][category] += 1
            pairs[tuple(bins)] += 1
            different += issued[0] != issued[1]
    finally:
        old.encode,new.encode,old._POLICY.forward,new._root_rotation_policy = old_enc,new_enc,old_net,new_net
    direction = int(np.sign(margins[1] - margins[0]))
    choose = [int(v>0) for v in score]
    return {k:card[k] for k in ('group','seed','seat','step','terrain','source_sha')} | dict(
        public_obs_sha256=obs_sha(obs),x_sha256=sha(x.tobytes()),options=options,margins=margins,
        wdl_differ=np.sign(margins[0]) != np.sign(margins[1]),features=features,log_B_over_A=score.tolist(),
        better_B=(True if direction>0 else False if direction<0 else None),choice=choose,
        true_margin_change=margins[choose[1]]-margins[choose[0]],
        oriented_log_gain=float(direction*(score[1]-score[0])),counts=counts,paired_bins=pairs.tolist(),
        n_decoded_changed=int(different))

def all_rows():
    protocol=read(HERE/'protocol.json')
    for path, digest in protocol['source_sha256'].items():
        assert sha((DER/path).read_bytes()) == digest
    old_cards = {(r['seed'],r['seat'],r['step']):r for r in read(LEGACY/'protocol.json')['tickets'] if r['status']!='skipped'}
    actual = {(r['seed'],r['seat'],r['step']):r for r in read(LEGACY/'score.json')['records'] if r['status']=='complete'}
    mods=[model(DER/'advantage_policy_update/v2_bc.py','terminal_original',protocol['parent_sha']),
          model(HERE.parent/'submission.py','terminal_rotated',protocol['child_sha'])]
    import torch;torch.set_num_threads(1)
    records=[]
    extras={}
    for seed,seat,t,group in protocol['keys']:
        key=(seed,seat,t)
        if key in old_cards:
            card=old_cards[key]
            history,obs,orders=source(card)
            margins=[r['margin'] for r in actual[key]['outcomes']]
        else:
            import extra
            if not extras: extras={(r['seed'],r['seat'],r['step']):r for r in extra.cards()}
            card=extras[key]|{'group':group}
            history,obs,_=extra.context(extras[key])
            orders,margins=card['orders'],card['margins']
        row=examine(mods,card,history,obs,orders,margins)
        print(group,seed,seat,t,row['choice'],row['oriented_log_gain'],flush=True)
        records.append(row)
    assert len(records)==25
    return records

if __name__=='__main__':
    records=all_rows()
    out=HERE/'results.json';tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(protocol_sha256=sha((HERE/'protocol.json').read_bytes()),records=records),default=lambda x:x.item(),indent=2,allow_nan=False)+'\n')
    tmp.replace(out)
