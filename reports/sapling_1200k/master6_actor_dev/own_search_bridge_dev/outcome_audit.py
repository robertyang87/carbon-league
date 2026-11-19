"""Reprice all true branch tapes after a separately blinded, public-input-only search rerun."""
from collections import Counter, defaultdict
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import numpy as np
import bridge

HERE=Path(__file__).resolve().parent
ROOT=bridge.ROOT
MASTER='65f268b2919ce30fe19355115a928eada032a6beaa018d801662d434c083f2b5'
PARTS=('collector','planter','base')
POLICIES=('parent','h1','h8')
read=bridge.load
sha=bridge.sha


def public_input(registrations):
    """One legal observation per turn, plus two interventions only. No native arm/tape/score access."""
    grouped=defaultdict(list)
    for relative_folder in registrations:
        folder=ROOT/relative_folder
        p=read(folder/'live_protocol.json');grouped[p['record']['source']].append((folder,p))
    with np.load(HERE/'rollout.npz',allow_pickle=False) as z:
        weights={k:z[k] for k in z.files}
    out={}
    for source,items in grouped.items():
        raw=read(ROOT/source)
        times={p['root_t']:(f,p) for f,p in items};seat=raw['candidate_seat']
        assert len(times)==len(items) and all(p['record']['source_sha256']==sha(ROOT/source) for f,p in items)
        # Use only the historical prefix completed before the last registered root.
        prefix=raw['actions'][:max(times)]
        cfg=raw['game_configuration'];sim=bridge.FastCarbon(cfg).reset(seed=raw['seed'])
        views=[bridge.F.Feat(i) for i in (0,1)]
        for t in range(max(times)+1):
            public=sim.obs(seat);views[seat].observe(public)
            enemy_x=views[1-seat].observe(bridge.public_enemy(public,views[seat].est))
            if t in times:
                folder,reg=times[t];data=read(folder/'root.json.gz')
                assert all(data['public_at_root'][k]==public[k] for k in ('step','player','players','carbon','trees'))
                candidates=[data['v1'],data['alternative']]
                res=bridge.play_public_pair(cfg,public,views,enemy_x,candidates,weights)
                out[str(folder.relative_to(ROOT))]=dict(samples=res['samples'],mean_delta=res['mean_delta'])
            if t<max(times):
                sim.step(*prefix[t])
    return out


def native_terminal(sim,actions,official):
    for n,pair in enumerate(actions):
        pre=list(sim.cash);sim.step(*pair)
        # Same extinction as the official evaluator. No copies of hidden observations
        # are supplied to the rollout model at this evaluation stage.
        hire_cost=min(sim.config['recCollectorCost'],sim.config['recPlanterCost'])
        e=[not sim.player_workers[s] and not sim.player_trees[s] and (not sim.bases[s] or sim.cash[s]<hire_cost) for s in (0,1)]
        if any(e) or sim.step_count>=sim.config['episodeSteps']-1:
            assert n+1==len(actions)
            scores=[sim.step_count-sim.config['episodeSteps']-1 if e[s] else (pre[s] if any(e) else sim.cash[s]) for s in (0,1)]
            if max(abs(a-b) for a,b in zip(scores,official))>.005001:raise ValueError('native terminal cash differs')
            return scores
    raise ValueError('no native terminal')


def match_root(slot_roles,actual,pred):
    if any(k not in slot_roles for k in actual.keys()|pred.keys()):
        raise ValueError('foreign UID in enemy command')
    roles={}
    for role in PARTS:
        inds=[k for k,v in slot_roles.items() if v==role]
        c=Counter()
        for uid in inds:
            a=actual.get(uid,'NONE');b=pred.get(uid,'NONE')
            c.update(slots=1,equal=int(a==b),actual_active=int(a!='NONE'),pred_active=int(b!='NONE'),
                     equal_on_actual_active=int(a!='NONE' and a==b),always_wait_correct=int(a=='NONE'))
        roles[role]=c
    return dict(roles=roles,whole=int(all(actual.get(k,'NONE')==pred.get(k,'NONE') for k in slot_roles)),
                worker_joint=int(all(actual.get(k,'NONE')==pred.get(k,'NONE') for k,v in slot_roles.items() if v!='base')))


def intervals(d):
    x=np.array(d,float);n=len(x);mu=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n))
    rng=np.random.default_rng(84102)
    ix=np.array(list(itertools.product(range(n),repeat=n))) if n<=4 else rng.integers(n,size=(30000,n))
    quant=np.quantile(x[ix].mean(axis=1),[.025,.975]).tolist()
    crit={4:3.182446305,12:2.20098516,16:2.131449546}[n]
    return dict(equal_map_mean=mu,maps=n,paired_se=se,percentile_map_bootstrap95=quant,
                reference_t95=[mu-crit*se,mu+crit*se],positive_maps=int((x>1e-7).sum()),negative_maps=int((x< -1e-7).sum()))


def summarize(rows):
    terrains=sorted({r['terrain'] for r in rows})
    results={}
    for a,b in [('h1','parent'),('h8','parent'),('h8','h1')]:
        deltas=[[r['policies'][a]['margin']-r['policies'][b]['margin'] for r in rows if r['terrain']==g] for g in terrains]
        table=intervals([statistics.mean(t) for t in deltas])
        table['map_values']=dict(zip(terrains,[statistics.mean(t) for t in deltas]))
        keyed=sorted([(r['policies'][a]['margin']-r['policies'][b]['margin'],r['case']) for r in rows])
        table['most_negative']=next((x for x in keyed if x[0]<-1e-7),None)
        table['most_positive']=next((x for x in reversed(keyed) if x[0]>1e-7),None)
        results[a+'_minus_'+b]=table
    wdl={k:{'wdl':dict(Counter(r['policies'][k]['wdl'] for r in rows)), 'switched':sum(r['policies'][k]['arm']=='alternative' for r in rows)} for k in POLICIES}
    forecasts={}
    for stream in range(2):
        q=[r['enemy_match'][stream] for r in rows]
        forecasts[str(stream)]={'roots':len(q),'whole_joint_exact':sum(i['whole'] for i in q),
                    'worker_joint_exact':sum(i['worker_joint'] for i in q),
                    'roles':{role:dict(sum((Counter(i['roles'][role]) for i in q),Counter())) for role in PARTS}}
    return dict(roots=len(rows),map_count=len(terrains),wdl=wdl,comparisons=results,enemy_prediction=forecasts)


def run():
    if os.sched_getaffinity(0)!={2} or os.getpriority(os.PRIO_PROCESS,0)<19:raise ValueError('only low-priority CPU2')
    bound=read(HERE/'screen.json')
    signed=read(HERE/'root_scores.json.gz')
    if len(signed)!=64 or sha(HERE/'root_scores.json.gz')!=bound['evidence_sha256']:
        raise ValueError('changed signed prediction panel')
    registered=sorted(str(folder.relative_to(ROOT)) for cases in bridge.registered_cases().values() for _,_,_,folder in cases)
    if len(set(registered))!=64 or set(registered)!={z['case'] for z in signed}:raise ValueError('registered panel identities')
    blind=public_input(registered)
    seen={};rows=[]
    for z in signed:
        case=z['case'];folder=ROOT/case;reg=read(folder/'live_protocol.json');root=read(folder/'root.json.gz')
        if blind[case]!={k:z[k] for k in ('samples','mean_delta')}:raise ValueError('blinded rollout differs')
        seed,seat,t=z['seed'],z['seat'],z['action_t'];enemy=1-seat
        assert (seed,seat,t)==(reg['seed'],reg['seat'],reg['root_t']) and z['root_sha256']==sha(folder/'root.json.gz')
        src=ROOT/reg['record']['source']
        if src not in seen:
            raw=read(src);assert sha(src)==reg['record']['source_sha256']
            sim=bridge.FastCarbon(raw['game_configuration']).reset(seed=seed)
            target_ts={q['action_t'] for q in signed if (q['seed'],q['seat'])==(seed,seat)}
            snapshots={}
            for n in range(max(target_ts)+1):
                if n in target_ts:snapshots[n]=sim.get_state()
                if n<max(target_ts):sim.step(*raw['actions'][n])
            seen[src]=raw['game_configuration'],snapshots,raw['initial_terrain_u8_sha256']
        cfg,snap,terrain=seen[src]
        if terrain!=z['physical_terrain']:raise ValueError('terrain identity')
        enemy_info=root['public_at_root']['players'][enemy]
        roles={u:w[2].lower() for u,w in enemy_info[2].items()}|{u:'base' for u in enemy_info[1]}
        assert {v for v in roles.values()}<=set(PARTS)
        worker=root['public_at_root']['players'][seat][2][reg['root']['worker']]
        assert worker[2]=='PLANTER'
        pair=[];enemy_cmd=None;arms={}
        for i,label in enumerate(('parent','alternative')):
            ap=folder/f'live_{label}.json';a=read(ap);tp=folder/a['action_tape'];tape=read(tp)
            expected=[root['v1'],root['alternative']][i]
            assert a['arm']==label and a['source_sha256']==MASTER and a['original_trajectory_sha256']==reg['record']['source_sha256']
            assert a['protocol_sha256']==sha(folder/'live_protocol.json') and a['statuses']==['DONE','DONE']
            assert a['action_sha256']==sha(tp)==z['actual_native'][i]['tape_sha256']
            assert tape[0]==a['root_actions'] and tape[0][seat]==expected and t+len(tape)==a['terminal_t']
            if enemy_cmd is None:enemy_cmd=tape[0][enemy]
            elif enemy_cmd!=tape[0][enemy]:raise ValueError('root enemy command differs between arms')
            sim=bridge.FastCarbon(cfg);sim.set_state(snap[t]);native_terminal(sim,tape,a['score'])
            score=[float(x) for x in a['score']];margin=score[seat]-score[enemy]
            assert abs(margin-a['margin'])<1e-7 and abs(z['actual_native'][i]['margin']-margin)<1e-7
            arms[label]=dict(score=score,margin=margin,wdl='W' if margin>0 else 'L' if margin<0 else 'D',receipt_sha256=sha(ap),tape_sha256=sha(tp))
            pair.append(a)
        assert pair[0]['shared_init_sha256']==pair[1]['shared_init_sha256']
        q=blind[case]
        picks={'parent':'parent',**{h:'alternative' if mean>0. else 'parent' for h,mean in q['mean_delta'].items()}}
        rows.append(dict(case=case,seed=seed,seat=seat,t=t,terrain=terrain,batch=z['batch'],worker_id=reg['root']['worker'],
                         root_commands=[root['v1'],root['alternative']],enemy_roles=roles,actual_enemy=tape[0][enemy],
                         predicted_enemy=[s['root_enemy_command'] for s in q['samples']],
                         enemy_match=[match_root(roles,enemy_cmd,s['root_enemy_command']) for s in q['samples']],
                         forecast_delta=q['mean_delta'],arms=arms,policies={h:dict(arm=k,margin=arms[k]['margin'],wdl=arms[k]['wdl']) for h,k in picks.items()}))
    results={name:summarize(sub) for name,sub in [('discovery',[r for r in rows if r['batch']<=6]),('held',[r for r in rows if r['batch']>6]),('all',rows)]}
    out={'source_sha256':sha(Path(__file__)),'score_file_sha256':sha(HERE/'root_scores.json.gz'),'bridge_sha256':sha(HERE/'bridge.py'),
         'scope':'True separately played continuations from 64 planter-only roots; reused historical labels, four held maps. WDL is a branch-root resolution, not 64 independent fresh whole games.',
         'blind_rerun':'All 64 matched exactly. Search saw only acting-seat observations from historical actions strictly before each root, reconstructed enemy observations from the acting belief and public visibility, two own complete commands, weights and config. Replayed true enemy root command and terminal scores entered only afterward in the scoring phase.',
         'uncertainty':'Paired map clusters, equal weight within map then equal maps; exact empirical percentile bootstrap on four held maps, seeded Monte Carlo otherwise. Small historical sample, no deployment/general population coverage.',
         'sections':results,'roots':rows}
    bridge.save(HERE/'outcome_audit.json',out)
    print({k:(v['comparisons']['h8_minus_h1']['equal_map_mean'],v['wdl']) for k,v in results.items()},flush=True)


if __name__=='__main__':run()
