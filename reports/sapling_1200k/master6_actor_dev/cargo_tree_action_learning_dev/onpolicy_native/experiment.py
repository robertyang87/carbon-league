"""Train on eight existing development map/seats, with independent archived controls."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from build import HERE, PARENT, PARENT_SHA, build, sha

FIT = HERE.parent
ACTOR = FIT.parent
ROOT = ACTOR.parents[2]
SEEDS = (621758345, 621758346, 621758348, 621758349)
STEPS = (12,48,84,120,156,192,228,264)
FULL = ACTOR / 'deployed_mix_policy_gradient_dev/full_episode'
HELD = ((621758343, FULL / 'direct_original'), (621758344, FULL / 'direct_native_confirmation'))
ORIGINAL = "65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5"
sys.path.insert(0, str(ACTOR / 'first_win_dev'))
import native
sys.path.insert(0, str(ROOT / 'code'))
from fastsim import FastCarbon

def read(p):
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())

def write(path, doc):
    native.save(path,doc)

def source(seed):
    return FIT / ('direct_original' if seed in SEEDS[:2] else 'replicate_original')

def model(path):
    spec=importlib.util.spec_from_file_location("onpolicy_own",path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def manifest():
    def card(seed,seat,folder):
        p = folder / 'games/candidate' / f'raw_{seed}_{seat}.json.gz'
        receipt= p.with_name(f'receipt_{seed}_{seat}.json')
        r=read(receipt)
        if r['game']['raw_sha256']!=sha(p) or r['binding']['original'] != ORIGINAL:
            raise ValueError('unsigned raw tape')
        return dict(seed=seed,seat=seat,raw=str(p.relative_to(ROOT)),sha256=sha(p),
                receipt_sha256=sha(receipt),terrain=r['official']['map_sha256'],
                scores=r['official']['official_reward_by_seat'],candidate=r['binding']['candidate'])
    train=[card(s,p,source(s)) for s in SEEDS for p in (0,1)]
    held=[card(s,p,f) for s,f in HELD for p in (0,1)]
    if any(c['candidate']!=PARENT_SHA for c in train) or len({c['terrain'] for c in train+held})!=6:
        raise ValueError('baseline or board identity')
    protocol=dict(parent=PARENT_SHA, weights=sha(FIT/'candidate.npz'), original=ORIGINAL,
        seeds=list(SEEDS), steps=list(STEPS), historical_controls=train, held_command_audit=held,
        affected_network='one direct on-policy likelihood-ratio step for 5x96 worker readout and 5 biases in the deployed Sapling (25% logits), with it50 and safety decoder frozen',
        sampling='At eight scheduled turns sort worker ties by top-two legal logit gap <=1; choose first two complete commands that really differ, preserve candidate safety route/prefix/hidden and create no added own conflict. Sample softmax over that two-command support, private deterministic per-ticket PRNG. All other turns are ordinary parent decisions. Natural original search/temperature/clock from step zero. One trajectory per physical map x seat; keep all failures and early extinction.',
        objective='For each episode G=0.5*WDL+0.5*sigmoid(official terminal score difference/1000). Baseline is one independently sampled, already completed greedy 8da4 game on the identical terrain and seat. Apply G-b to the SUM of the actual supported log-probability gradients for this entire game, average over all 8 scheduled games including episodes with zero eligible decisions. Historical baselines reduce no guaranteed amount of variance. No discount, shaped reward, clipping of outcomes, or resampling.',
        update='One preconditioned score-function step. Divide the observed episode gradient componentwise by the within-policy expected diagonal Fisher plus 0.025. Rescale the resulting direction to a maximum L2 parameter step of 0.6 and maximum supported paired training-logit difference change of 0.45; apply the smaller scale with cap 1.0. No post-hoc learning rate or validation checkpoint selection. The behavior menu is frozen locally at data collection; final greedy deployment has a distinct policy law.',
        gates='Need >=24 signed supported samples overall and both sampled alternatives per physical seat across 8 games. Reject failure of quantized identity, raw-vs-issued or role attribution. Greedy package must alter physical commands in both seats on 343/344 held tapes, and two leave-one-map-out gradients must point in same hemisphere as full gradient (diagnostic stability only). A fresh development test additionally requires actual-response evidence against original on previously exposed held roots, and meaningful training payoff evidence. Formal 100 maps reserved. Seats, all draws and defeats counted. No proxy or behavior match counted as a win for a greedy student.',
        equivalence='The training build is disposable. Final pack is reconstructed by replacing only the existing Sapling tensors in the greedy parent. No training sampler, I/O or added dependency in the competition package; package size and separate clock/RSS checks required.',
        procedure={s:sha(HERE/s) for s in ('build.py','sampler.py','experiment.py','fit.py')}, training_pack=sha(build()))
    write(HERE/'protocol.json',protocol)
    return protocol

def replay(card, newer=None, smoke=False):
    raw=read(ROOT/card['raw']); seat=card['seat']; seed=card['seed']
    module=model(build() if smoke else newer)
    base=model(PARENT)
    world=FastCarbon(raw['game_configuration']).reset(seed=seed)
    os.environ['SAPLING_NATIVE_STREAM']=str(seed)
    os.environ.pop('SAPLING_NATIVE_TRACE',None)
    rows=[]
    for t,joint in enumerate(raw['actions']):
        obs=world.obs(seat)
        if t==0 or (t in STEPS) or (not smoke and t%5==0):
            reference=base.agent(obs,world.config)
            public=module.agent(obs,world.config)
            if smoke and reference != joint[seat]:
                raise ValueError('starting agent does not reconstruct recorded parent decision')
            event=module.last_training_event if smoke else None
            if smoke:
                if event is None:
                    if public!=reference:raise ValueError('changed non-sampling step')
                elif event.get('supported'):
                    if event['options'][0]!=reference or event['issued']!=public or not np.isclose(sum(event['probabilities']),1.):
                        raise ValueError('sample and physical command differ')
            rows.append(dict(t=t,changed=public!=reference,supported=bool(event and event.get('supported')),
                    cmd=public,parent=reference,
                    probability=event['probabilities'] if event and event.get('supported') else None))
        else:
            base._FEAT.observe(obs);module._FEAT.observe(obs)
        world.step(*joint)
    return dict(seed=seed,seat=seat,rows=rows)

def smoke():
    p=manifest()
    sample=[replay(c,smoke=True) for c in p['historical_controls'][:2]]
    by_seat={r['seat']:dict(supported=sum(v['supported'] for v in r['rows']),changed=sum(v['changed'] for v in r['rows'])) for r in sample}
    if min(r['changed'] for r in by_seat.values())<1:raise ValueError('no physical explorations issued')
    write(HERE/'preflight.json',dict(protocol=sha(HERE/'protocol.json'),by_seat=by_seat,demonstrations=sample))
    print(by_seat,flush=True)

def initialize():
    native.HERE=HERE;native.OUT=HERE/'games'
    d=native.driver()
    d.CANDIDATE=HERE/'training_policy.py'
    if sha(d.arena.prior.SOURCE)!=ORIGINAL:raise ValueError('original examiner drift')
    return d

def one(job):
    arm,seed,seat,cpu,board,auth=job
    os.environ['SAPLING_NATIVE_STREAM']=str(seed)
    folder=HERE/'traces'/f'{seed}_{seat}'
    folder.mkdir(parents=True,exist_ok=True)
    os.environ['SAPLING_NATIVE_TRACE']=str(folder)
    d=initialize()
    native.clock.run_one(d,job)
    wall=native.clock.verify_completed(d,job)
    dest=native.OUT/arm
    receipt=read(dest/f'receipt_{seed}_{seat}.json')
    official=d.official.check(dest,receipt['game'])
    raw=read(dest/receipt['game']['raw_file'])
    if official!=receipt['official'] or official['map_sha256']!=board or official['statuses']!=['DONE','DONE']:
        raise ValueError('bad original result')
    n=official['steps']
    found=sorted(int(f.stem) for f in folder.glob('*.json'))
    if found != [t for t in STEPS if t<n]:raise ValueError('missing, duplicated or extra behavior turn')
    events=[]
    for t in found:
        path=folder / f'{t:03d}.json';v=read(path)
        if v['seat']!=seat or v['step']!=t or v['issued']!=raw['actions'][t][seat]:
            raise ValueError('behavior receipt not bound to original game')
        events.append(dict(t=t,sha256=sha(path),supported=v['supported']))
    return dict(seed=seed,seat=seat,steps=n,score=official['official_reward_by_seat'],events=events,
                raw=sha(dest/receipt['game']['raw_file']),original_calls=len(wall['calls']),wall=wall['summary'])

def train_games():
    p=read(HERE/'protocol.json')
    if (p['parent']!=sha(PARENT) or p['training_pack']!=sha(build()) or
        any(sha(HERE/k)!=v for k,v in p['procedure'].items()) or read(HERE/'preflight.json')['protocol']!=sha(HERE/'protocol.json')):
        raise ValueError('registration/preflight mismatch')
    binding=dict(protocol=sha(HERE/'protocol.json'),preflight=sha(HERE/'preflight.json'),candidate=p['training_pack'],parent=PARENT_SHA,original=ORIGINAL)
    results=[]
    with native.SHARED.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for j,seed in enumerate(SEEDS):
            cards=[r for r in p['historical_controls'] if r['seed']==seed]
            if len(cards)!=2 or any(sha(ROOT/c['raw'])!=c['sha256'] for c in cards):raise ValueError('changed training registration')
            jobs=[('candidate',seed,c['seat'],(j+c['seat'])%2,c['terrain'],binding) for c in cards]
            with ProcessPoolExecutor(2) as pool:
                for row in pool.map(one,jobs):
                    write(HERE/f"complete_{seed}_{row['seat']}.json",row);results.append(row)
                    print(seed,row['seat'],row['score'],row['steps'],flush=True)
    write(HERE/'scores.json',results)

if __name__=='__main__':
    {'smoke':smoke,'games':train_games}[sys.argv[1]]()
