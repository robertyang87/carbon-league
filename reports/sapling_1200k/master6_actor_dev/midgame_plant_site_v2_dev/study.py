"""Freeze source identities, check whole-tape public bounds, and compare the frozen first edits live."""
import fcntl
import hashlib
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ACTOR=HERE.parent;V1=ACTOR/'midgame_plant_funnel_dev'
sys.path.insert(0,str(V1))
v1=runpy.run_path(str(V1/'trial.py'));sr=v1['sr'];runner=v1['runner'];live=v1['live'];MODEL=v1['MODEL'];W=v1['W']
geom=(ACTOR/'intercept_response_dev/forecast.py').read_text().split('def carriers(')[0]
upstream=(MODEL.with_name('decision.py')).read_text().split('def consider(')[0].replace('from guard_tree import _D,_TO,_DIRECTION','')
route=(V1/'route.py').read_text().replace('from decision import _D, _DIRECTION, _TO, predicted, quote, values','')
code=geom+upstream+route+(HERE/'policy_v2.py').read_text()
policy={};exec(code,policy)
PACK=ACTOR/'actor_partner_dev';plan=sr.load(HERE/'site_protocol.json')
def digest(x):return hashlib.sha256(x.encode()).hexdigest()

def prepare():
    records=[];checks=[];chosen=[]
    if sr.sha(PACK/'candidate.py')!=plan['source_hash']:raise ValueError('parent identity')
    for seed,folder in zip(plan['holdout_seeds'],plan['source_directories']):
        for s in (0,1):
            p=ACTOR/folder/f'raw_{seed}_{s}.json.gz';raw=sr.load(p);receipt=sr.load(p.with_name(f'receipt_{seed}_{s}.json'))
            if sr.sha(p)!=receipt['game']['raw_sha256'] or raw['pack_sha256']!=plan['source_hash']:
                raise ValueError("mismatch in source tape")
            mod=live.module(PACK/'candidate.py','v2_floor_observer')
            cfg=raw['game_configuration'];sim=sr.FastCarbon(cfg).reset(seed=seed);mem={};hits=[];cells=0
            for t,pair in enumerate(raw['actions']):
                o=sim.obs(s);policy['record_floor'](o,mem,cfg)
                curr=mem['current_floor']
                if max(curr-sim.carbon)>.000001:raise ValueError(('current bound',seed,t,max(curr-sim.carbon)))
                if mod.agent(o,cfg)!=pair[s]:raise ValueError(('full prefix mismatch',seed,s,t))
                root=policy['choose'](o,mem,pair[s],cfg,W)
                if root:hits.append(root)
                floor=policy['next_floor'](o,curr,cfg,pair[s]);sim.step(*pair)
                if max(floor-sim.carbon)>.000001:raise ValueError(('following bound',seed,t,max(floor-sim.carbon)))
                cells+=len(curr)
            checks.append(dict(seed=seed,seat=s,parent_matching_turns=len(raw['actions']),public_floor_cell_transitions=cells))
            records.append(dict(seed=seed,seat=s,source=str(p.relative_to(sr.ROOT)),source_sha256=sr.sha(p),roots=hits))
    sr.retain(HERE/'verified_prefixes_and_bounds.json',checks);sr.retain(HERE/'survey.json.gz',records)
    used=set()
    for s in (0,1):
        g=next((r for r in records if r['seat']==s and r['seed'] not in used and r['roots']),None)
        if not g:break
        root=g['roots'][0];used.add(g['seed']);p=sr.ROOT/g['source'];receipt=sr.load(p.with_name(f'receipt_{g["seed"]}_{s}.json'))
        out=HERE/f'seat{s}';out.mkdir(exist_ok=True)
        sr.retain(out/'official_replay.json',dict(runs={out.name:dict(games=[receipt['official']])}))
        record={k:g[k] for k in ('seed','seat','source','source_sha256')}
        q=dict(seed=g['seed'],seat=s,root_t=root['t'],family='guaranteed_adjacent_soil',source_dir=str(out.relative_to(sr.ROOT)),
               record=record,root=root,pack_sha256=plan['source_hash'],policy_sha256=digest(code),
               scorer_weights_sha256=sr.sha(MODEL),gate_sha256=sr.sha(HERE/'site_protocol.json'))
        sr.retain(out/'live_protocol.json',q);live.prepare(out);chosen.append(q)
    sr.retain(HERE/'selected_sites.json',chosen)
    print('density',[(g['seed'],g['seat'],len(g['roots'])) for g in records],'selected',[(q['seed'],q['seat'],q['root_t']) for q in chosen],flush=True)

def play(s,arm):
    q=sr.load(HERE/f'seat{s}/live_protocol.json');root=q['root'];r=q['record']
    if (q['policy_sha256']!=digest(code) or q['scorer_weights_sha256']!=sr.sha(MODEL)
            or q['gate_sha256']!=sr.sha(HERE/'site_protocol.json')):raise ValueError("changed candidate")
    raw=sr.load(sr.ROOT/r['source']);cfg=raw['game_configuration'];sim=sr.FastCarbon(cfg).reset(seed=q['seed']);mem={}
    for t,pair in enumerate(raw['actions'][:root['t']+1]):
        obs=sim.obs(s);policy['record_floor'](obs,mem,cfg)
        if t==root['t'] and policy['choose'](obs,mem,pair[s],cfg,W)!=root:raise ValueError("changed root")
        sim.step(*pair)
    original=live.module
    def loadmod(path,name):
        mod=original(path,name)
        if name=='recovery_live_student' and arm=='approach':
            base=mod.agent
            def agent(obs,cfg):
                baseline=base(obs,cfg)
                if obs['step']==root['t']+1:
                    policy['record_floor'](obs,mem,cfg)
                    actions,reason=policy['pay'](obs,mem,baseline,cfg,root,W)
                    sr.retain(HERE/f'seat{s}/follow.json',dict(reason=reason,parent=baseline,executed=actions,t=obs['step']))
                    return actions
                return baseline
            mod.agent=agent
        return mod
    runner.HERE=HERE;runner.PRIOR=PACK;live.module=loadmod;runner.play(s,arm)

if __name__=='__main__':
    if sys.argv[1:]==['prepare']:prepare()
    elif len(sys.argv)>1:play(int(sys.argv[1]),sys.argv[2])
    else:
        qs=sr.load(HERE/'selected_sites.json')
        if len(qs)<2:print('Insufficient distinct roots. Stopped before native tests.',flush=True);sys.exit(0)
        for q in qs:
            s=q['seat']
            with (runner.TREE/'.continuation.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                jobs=[subprocess.Popen(['taskset','-c',str((s+i)%2),sys.executable,__file__,str(s),arm]) for i,arm in enumerate(('parent','approach'))]
                if any([j.wait() for j in jobs]):raise RuntimeError('native continuation failure')
