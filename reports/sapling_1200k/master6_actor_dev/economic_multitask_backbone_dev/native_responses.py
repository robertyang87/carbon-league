"""Only for qualified learned policies: share opening, beliefs and search warmup, then compare to terminal."""
from concurrent.futures import ProcessPoolExecutor
import fcntl,hashlib,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;ROOT=ACTOR.parents[2];BC=ACTOR/'broader_elite_backbone_dev'
sys.path.insert(0,str(ACTOR/'elite_role_mix_dev/conditional'))
import responses as r
PACK={'bc':BC/'candidate.py','economy':HERE/'candidate.py'}
PROTO=r.load(HERE/'native_conditions.json')
def source(c):
    p=ROOT/c['path'];record=r.load(p)
    if r.sha(p)!=c['source_sha'] or record['pack_sha256']!=c['source_pack'] or record['initial_terrain_u8_sha256']!=c['terrain']:raise ValueError("archive changed")
    receipt=r.load(p.with_name(p.name.replace('raw_','receipt_').replace('.json.gz','.json')))
    if receipt['game']['raw_sha256']!=c['source_sha'] or receipt['official']['statuses']!=['DONE','DONE']:raise ValueError('official source receipt')
    return record
def choice(c):
    raw=source(c);s=c['seat'];cfg=raw['game_configuration'];sim=r.FastCarbon(cfg).reset(seed=c['seed'])
    models={a:r.live.module(p,'econ_choose_'+a) for a,p in PACK.items()}
    for t,pair in enumerate(raw['actions'][:200]):
        obs=sim.obs(s)
        if t==0 or t>=100:
            act={k:r.live.clean(mod.agent(obs,cfg)) for k,mod in models.items()}
            if t>=100 and act['bc']!=act['economy']:
                return dict(**c,t=t,root=act,opponent=pair[1-s])
        else:
            for mod in models.values():mod._FEAT.observe(obs)
        sim.step(*pair)
    return dict(**c,t=None)

def play(job):
    case,arm,cpu,warm=job;os.sched_setaffinity(0,{cpu});raw=source(case)
    s=case['seat'];opp=1-s;t0=case['t'];cfg=raw['game_configuration'];env=r.live.begin(raw)
    own=r.live.module(PACK[arm],'econ_live_own');master=r.live.module(r.live.SOURCE,'econ_live_opp')
    if master._SEED is not None or master._TEMP!=1 or not master._SEARCH_OK or master._S_ABORT!=.58:raise ValueError('original settings')
    tracker=master.BeliefTracker(master._Bel,opp)
    for t,pair in enumerate(raw['actions'][:t0+1]):
        o=r.live.public(env,opp);x=r.live.public(env,s)
        tracker.step(o['players'],t,o['carbon']);tracker.set_actions(o['players'],pair[opp])
        if t==0 or t==t0:act=r.live.clean(own.agent(x,cfg))
        else:own._FEAT.observe(x)
        if t<t0:env.step(pair)
    if act!=case['root'][arm] or pair[opp]!=case['opponent']:raise ValueError('root drift')
    opening=list(pair);opening[s]=act;seed=int(hashlib.sha256(f"economic-v1:{case['seed']}:{s}:{t0}".encode()).hexdigest()[:8],16)
    master._S_COST[:]=warm;master._ST.clear();master._ST.update(tr=tracker,prev=o['players'],rng=np.random.RandomState(seed))
    recorder=r.native_clock.Recorder(master);opponent=recorder.wrap(master.agent)
    tape=[opening];env.step(opening)
    while not env.done:
        pair=[None,None];pair[s]=own.agent(r.live.public(env,s),cfg);pair[opp]=opponent(r.live.public(env,opp),cfg);env.step(pair);tape.append(pair)
    statuses=[seat.status for seat in env.states];scores=[seat.reward for seat in env.states];cash=[pl[0] for pl in env.states[0].observation.players]
    if statuses!=['DONE','DONE'] or recorder.errors or len(recorder.calls)!=len(tape)-1:raise ValueError('incomplete native run')
    margin=scores[s]-scores[opp];u=.5*((margin>0)+.5*(margin==0))+.5/(1+np.exp(-max(-60,min(60,margin/1000))))
    result=dict(case=case,arm=arm,policy_sha256=r.sha(PACK[arm]),opponent_sha256=r.sha(r.live.SOURCE),rng=seed,warm=warm,
                tape=tape,calls=recorder.calls,clock=r.native_clock.summarize(recorder.calls,master._S_ABORT,master._S_D2),
                official_score=scores,margin=margin,U=u,statuses=statuses,cash=r.cash_summary(raw,s,t0,tape,cash))
    r.write(HERE/f"response_{case['seed']}_{s}_{arm}.json.gz",result)
    return result

def run():
    decision=r.load(HERE/'decision.json')
    if not decision['native_gate']:raise ValueError("registered offline gate failed")
    panel=r.load(BC/'panel.json')['games']+r.load(BC/'opponent_holdout/panel.json')['games']
    for c in PROTO['maps']:
        if any(d['seed']==c['seed'] or d['terrain_sha256']==c['terrain'] for d in panel):raise ValueError('overlap')
    cases=[choice(c) for c in PROTO['maps']]
    r.write(HERE/'native_selection.json',dict(cases=cases,evaluation=r.sha(HERE/'decision.json'),plan=r.sha(HERE/'native_conditions.json')))
    if not all(c['t'] is not None for c in cases):return
    results=[]
    with r.LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for c in cases:
            with ProcessPoolExecutor(1) as pool:warm=pool.submit(r.native_warm).result()
            with ProcessPoolExecutor(2) as pool:
                ctl,trt=pool.map(play,[(c,'bc',c['seat'],warm),(c,'economy',1-c['seat'],warm)])
            results.append(dict(seed=c['seed'],seat=c['seat'],t=c['t'],control=ctl['margin'],treatment=trt['margin'],
                                delta_margin=trt['margin']-ctl['margin'],delta_U=trt['U']-ctl['U'],
                                distinct_replies=sum(a[1-c['seat']]!=b[1-c['seat']] for a,b in zip(ctl['tape'],trt['tape']))))
    r.write(HERE/'native_results.json',results)
if __name__=='__main__':run()
