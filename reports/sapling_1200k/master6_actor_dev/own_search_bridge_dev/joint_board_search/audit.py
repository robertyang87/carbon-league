"""Exercise free parent/child commands plus a separate fixed-action true-response ranking panel."""
from setup import *
import importlib.util,time,copy
from collections import Counter
import numpy as np
def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def run():
    p=bridge.load(HERE/'protocol.json');subset=[c for c in p['sources'] if c['split']=='held'];score_by_index={(r['seed'],r['seat'],r['action_t']):r for r in bridge.load(GROUP/'root_scores.json.gz')}
    calls=[];native=[];leaks=[];fallback=[]
    for card in subset:
        src=bridge.load(ROOT/card['path']);s=card['seat'];cfg=src['game_configuration'];sim=bridge.FastCarbon(cfg).reset(seed=card['seed'])
        pkg=module(HERE/'candidate.py','candidate_deployment');parent=module(PARENT,'parent_deployment')
        # Parent/candidate include byte-identical F/NP sources. Each separately owns its persistent weights/Feats.
        mroots={t:r for (seed,ss,t),r in score_by_index.items() if seed==card['seed'] and ss==s}
        selected={0,25,50,75,100,125,150,175,200,225,250,275,298}|set(mroots)
        views=[pkg.F.Feat(j) for j in (0,1)]
        for t,tape in enumerate(src['actions']):
            obs=sim.obs(s);x=views[s].observe(obs);oppobs=pkg.public_enemy(obs,views[s].est);ox=views[1-s].observe(oppobs)
            if t in selected:
                start=time.perf_counter();a=pkg.agent(obs,cfg);wall=time.perf_counter()-start;info=copy.deepcopy(pkg._JB_INFO)
                ref=parent.agent(obs,cfg);assert ref==info['base']
                assert info['status']!='error',info
                changed=[u for u in a.keys()|ref.keys() if a.get(u)!=ref.get(u)]
                if changed:assert info['status']=='complete' and a==info['candidates'][info['chosen']]
                details=[]
                for j,cmd in enumerate(info.get('candidates',[])):
                    ids=[u for u in cmd.keys()|ref.keys() if cmd.get(u)!=ref.get(u)]
                    role=[obs['players'][s][2].get(u,[0,0,'BASE'])[2] for u in ids]
                    details.append(dict(j=j,ids=ids,roles=role,distances=[int(pkg.F.DIST[obs['players'][s][2][u][0],obs['players'][s][2][v][0]]) for h,u in enumerate(ids) for v in ids[h+1:] if u in obs['players'][s][2] and v in obs['players'][s][2]]))
                calls.append(dict(seed=card['seed'],seat=s,t=t,wall=wall,status=info['status'],changed=changed,options=details,rounds=info['rounds'],chosen=info.get('chosen',0)))
                if t==150 and len(fallback)<2:
                    old=pkg._JB_PUBLIC;pkg._JB_PUBLIC=(obs,cfg,time.perf_counter()-2);late=pkg._mixed_actions(x,views[s].aux);assert pkg._FEAT.to_actions(late)==ref;pkg._JB_PUBLIC=old;fallback.append(card['seed'])
                if t==0:
                    # Re-initialize at identical public history and adversarial hidden diagnostics.
                    extra=dict(obs,full_carbon=[-1]*225,oracle_carbon=[0]*225,reveal={5:99},hidden_opponent_action={'hidden':'LEFT'})
                    assert pkg.agent(extra,cfg)==a
                    leaks.append(card['seed'])
                if t in mroots:
                    rr=mroots[t];q=bridge.load(ROOT/rr['case']/'root.json.gz')
                    winner,rounds,e=pkg._joint_search(cfg,obs,views,ox,[q['v1'],q['alternative']],time.perf_counter()+4.)
                    assert len(rounds)==3 and winner==rounds[-1]['best']
                    diff=rr['realized_alternative_minus_parent'];native.append(dict(seed=card['seed'],seat=s,t=t,rounds=rounds,sign_correct=(winner==int(diff>0)),selected_delta=diff*winner,pair_delta=diff))
            else:
                parent._FEAT.observe(obs);pkg._FEAT.observe(obs);pkg._JB_FEAT.observe(oppobs)
            sim.step(*tape)
    times=np.array([r['wall'] for r in calls]);byseed={}
    for seed in sorted({r['seed'] for r in native}):
        subset=[r for r in native if r['seed']==seed]
        byseed[seed]=dict(n=len(subset),matched=sum(r['sign_correct'] for r in subset),mean_delta=float(np.mean([r['selected_delta'] for r in subset])))
    proof=dict(held_source=bridge.sha(GROUP/'root_scores.json.gz'),binding=bridge.sha(HERE/'binding.json'),source=bridge.sha(__file__),normal_calls=len(calls),
            changed=sum(bool(r['changed']) for r in calls),multi_changed=sum(len(r['changed'])>=2 for r in calls),paired_within_two=sum(any(len(op['ids'])>=2 and any(d<=2 for d in op['distances']) for op in r['options']) for r in calls),
            timing=dict(median=np.median(times),p95=np.quantile(times,.95),max=times.max(),over_one=int((times>1).sum())),fallback=fallback,hidden=leaks,native=byseed)
    bridge.save(HERE/'entry.json.gz',dict(summary=proof,calls=calls,native=native));bridge.save(HERE/'summary.json',proof)
if __name__=='__main__':run()
