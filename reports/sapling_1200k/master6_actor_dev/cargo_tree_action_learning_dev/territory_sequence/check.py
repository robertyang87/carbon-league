"""Measure the commands and probabilities of the finalized normal public-entry binaries on held terrain."""
import collections
from pathlib import Path

import numpy as np

from data import HERE,EXP,RULE,ROOT,Recorded,FastCarbon,sha,read,write,load_game
from gather import policy
import importlib.util
_spec=importlib.util.spec_from_file_location("territory_fit",HERE/"train.py")
_fit=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fit)
standard_weights=_fit.standard_weights

def mapavg(events,key):
    by=collections.defaultdict(list)
    for r in events:
        by[r['map']].append(r)
    return {m:float(np.average([e[key] for e in v],weights=[e['w'] for e in v])) for m,v in by.items()}

def main():
    rows=read(HERE/'sequence.json.gz');pairs=read(HERE/'pairs.json.gz')
    with np.load(HERE/'sequence.npz') as f:
        weights=standard_weights(rows,f['y'])[:,1]
    by=collections.defaultdict(lambda:collections.defaultdict(list))
    for i,r in enumerate(rows):
        if r['split']=='test':by[r['game']][r['t']].append((r,float(weights[i])))
    paths={'parent':EXP/'candidate.py','control':HERE/'control.py','candidate':HERE/'candidate.py'}
    fit=read(HERE/'fit.json')
    models={arm:Recorded(f) for arm,f in paths.items()}
    plain={arm:policy.load(f) for arm,f in paths.items()}
    assert all(sha(paths[k])==v['package'] for k,v in fit['outputs'].items())
    saved=[]; checks=0
    sel=[c for c in read(HERE/'sources.json') if c['split']=='test']
    for ix,c in enumerate(sel):
        cfg,actions,_,full=load_game(c)
        sim=FastCarbon(cfg).reset(seed=c['seed'],full_carbon=full); s=c['seat']
        query=by[c['sha256']]
        verify=ix==0 or ix==len(sel)-1
        for t in range(max(query,default=0)+1):
            obs=sim.obs(s)
            if t==0 or t in query:
                branches={arm:model.base(obs,cfg) for arm,model in models.items()}
                if verify:
                    for arm,model in plain.items():
                        actual=model.agent(obs,cfg)
                        instrumented=models[arm].net._FEAT.to_actions(branches[arm][0])
                        assert actual==instrumented
                        checks+=1
                for r,w in query[t]:
                    out=dict(map=c['terrain'],source=c['source'],teacher=c['teacher'],seat=s,t=t,uid=r['uid'],slot=r['slot'],role=r['role'],w=w,truth=r['label'])
                    for arm,(sent,slots) in branches.items():
                        f=models[arm].net._FEAT;slot=r['slot'];assert f.own_uid[slot]==r['uid']
                        z=slots[slot][-1];mask=slots[slot][2];actual=int(sent[slot]);a=r['label']
                        if mask[a]:
                            z=z.astype(float);valid=z[mask];mx=valid.max();loss=float(np.log(np.exp(valid-mx).sum())+mx-z[a])
                        else:loss=-np.log(1e-8)
                        out.update({arm+'_cmd':actual,arm+'_nll':float(loss),arm+'_agree':int(actual==a),arm+'_available':bool(mask[a])})
                    assert out['parent_cmd']==r['sent']
                    saved.append(out)
            else:
                for model in models.values():model.net._FEAT.observe(obs)
                if verify:
                    for model in plain.values():model._FEAT.observe(obs)
            sim.step(*actions[t])
    assert len(saved)==sum(r['split']=='test' for r in rows)
    turns=collections.defaultdict(list)
    for r in pairs:
        if r['split']=='hold':turns[r['source'],r['seat']].append(r)
    paired=[]
    for (source,s),rs in sorted(turns.items()):
        raw=read(ROOT/source);cfg=raw['game_configuration'];sim=FastCarbon(cfg).reset(seed=raw['seed'])
        times=collections.defaultdict(list)
        for r in rs:times[r['t']].append(r)
        for t in range(max(times)+1):
            obs=sim.obs(s)
            if t==0 or t in times:
                snaps={arm:model.base(obs,cfg) for arm,model in models.items()}
                for r in times[t]:
                    row={k:r[k] for k in ('case','map','seat','uid','t')}
                    for arm,(sent,slots) in snaps.items():
                        f=models[arm].net._FEAT;cmd=f.to_actions(sent)
                        name=next((k for k,c in r['cmd'].items() if c==cmd),None)
                        observed=next((k for k,c in r['cmd'].items() if c.get(r['uid'])==cmd.get(r['uid'])),None)
                        row[arm]=dict(exact=name,focal_only=observed,margin=r['arms'][name]['official'] if name else None)
                    paired.append(row)
            else:
                for model in models.values():model.net._FEAT.observe(obs)
            if t<max(times):sim.step(*raw['actions'][t])
    metrics={}
    for group,pop in [('all',saved),('Master3s2',[r for r in saved if r['source']=='Master3s2']),('Top5',[r for r in saved if r['source']=='Top5'])]:
        metrics[group]={}
        for arm in paths:
            loss=mapavg(pop,arm+'_nll');agreement=mapavg(pop,arm+'_agree')
            metrics[group][arm]=dict(nll=float(np.mean(list(loss.values()))),agreement=float(np.mean(list(agreement.values()))),maps=loss,
                 rows=len(pop),terrains=len(loss),not_legal=sum(not r[arm+'_available'] for r in pop))
    changed=[r for r in saved if r['parent_cmd']!=r['candidate_cmd']]
    losses=metrics['all']
    qualifies=losses['candidate']['nll']<min(losses['control']['nll'],losses['parent']['nll']) and len(changed)>=4 and len({r['map'] for r in changed})>=2
    write(HERE/'held_actions.json.gz',saved);write(HERE/'held_native_pairs.json.gz',paired)
    write(HERE/'qualification.json',dict(parent=sha(paths['parent']),candidate=sha(paths['candidate']),control=sha(paths['control']),
        eligible_for_response=bool(qualifies),no_new_original_games_until_offline_pass=True,official_paired_return_is_observed_only_on_exact_joint=True,
        metrics=metrics,changed_rows=len(changed),changed_maps=len({r['map'] for r in changed}),plain_entry_crosschecks=checks,
        historical_held_pairs=len(paired),exact_matches={arm:sum(r[arm]['exact'] is not None for r in paired) for arm in paths},
        outcome_supported_changes=sum(r['parent']['exact']!=r['candidate']['exact'] and all(r[k]['exact'] for k in ('parent','candidate')) for r in paired),
        held_actions=sha(HERE/'held_actions.json.gz'),held_pairs=sha(HERE/'held_native_pairs.json.gz')))

if __name__=='__main__':
    main()
