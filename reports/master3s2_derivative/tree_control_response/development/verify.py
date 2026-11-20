"""Replay a fixed pre-action panel, and continue up to two stable changed roots against original responses."""
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parents[1]
CANDIDATE='20d48bd206aae24277ac9a9870b03486250e8abd5ce23041aaad0e57bed3a290'
CONTRACT='c4193c63fd825908586d76ffba7f05b7aa9ed43814ac4d3f377cfd5fd5d2e210'
spec=importlib.util.spec_from_file_location('pair',FAMILY/'return_rank/development/root_responses.py')
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
base=pair.base


def call(m,tag,hist,tree_hist,obs,cfg,seed):
    if tag!='P':
        m._TREE_LAST=None
        m._TREE_HIST={obs['player']:copy.deepcopy(tree_hist)}
    result=pair.call(m,hist,obs,cfg,seed,[.001,.002,.001,.002],'P')
    result['mode']=tag
    result['trace']=copy.deepcopy(getattr(m,'_TREE_LAST',None)) if tag!='P' else None
    if result['trace'] is not None and result['menu'] is not None:
        assert result['trace']['after']==result['menu']
    return result


def work(q):
    return q['clocks']['K'],q['clocks']['H'],q['clocks']['rolls']


def choice(q):
    return q['action'],q['menu'],q['depths'],q['selected']


def score_targets(ticket,response,spot,frames,commands,seat):
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    sim=base.econ.FastCarbon(cfg).reset(seed=ticket['seed'])
    for row in commands[:ticket['step']]:
        samples=json.loads(row)['calls'];sim.step(*(s['response']['action'] for s in samples))
    sim.cash=base.econ.CashEvents(sim.cash)
    result={}
    for arm,entry in response.items():
        s=copy.deepcopy(sim);cumulative=[{},{}];early=[{},{}];owned=[0,0];target_income=[0.,0.];target_history=[]
        record=json.loads(gzip.decompress((HERE/entry['tape']).read_bytes()))
        for i,row in enumerate(record['rows']):
            s.cash.events.clear();s.step(*row['orders'])
            assert list(s.cash)==row['score_wallet']
            owner=next((q.player for q in s.trees.values() if q.pos==spot),None)
            if owner is not None:owned[owner]+=1
            if not target_history or target_history[-1]['owner']!=owner:target_history.append(dict(step=row['step']+1,owner=owner))
            events=base.econ.cash_flow(s.cash.events)
            for side in (0,1):
                for k,v in events[side].items():
                    cumulative[side][k]=cumulative[side].get(k,0.)+v
                    if i<10:early[side][k]=early[side].get(k,0.)+v
            for e in s.cash.events:
                if e[0]=='tree_absorption' and e[4]==spot:target_income[e[1]]+=e[2]
        result[arm]=dict(scores=record['official_scores'],margin=record['official_margin'],cash_sources=cumulative,first_ten_cash_sources=early,
                         target=spot,target_owner_steps=owned,target_ownership=target_history,target_gross_tree_income_by_seat=target_income)
    return result


def run():
    assert os.sched_getaffinity(0)=={2}
    os.environ['CARBON_DEBUG']='1'
    plan=json.loads((HERE/'roots.json').read_bytes());assert pair.sha((HERE/'roots.json').read_bytes())==CONTRACT
    files=[FAMILY/'cold_start/candidate.py',HERE.parent/'candidate.py',FAMILY.parents[1]/'sources/master3s2/master3s2.py']
    if not files[2].is_file():
        import subprocess
        frozen=subprocess.check_output(['git','show','HEAD:sources/master3s2/master3s2.py'],cwd=FAMILY)
        assert pair.sha(frozen)==plan['original_sha256']
        files[2]=FAMILY.parents[1]/'data/master3s2_source.py'
    assert [pair.sha(p.read_bytes()) for p in files]==[plan['parent_sha256'],CANDIDATE,plan['original_sha256']]
    p,e,o=[base.load('tree_original_%d'%i,v) for i,v in enumerate(files)]
    import torch
    torch.set_num_threads(1)
    for model in (p,e,o):
        assert model._TEMP==1 and model._SEED is None and model._S_ABORT==.58 and model._RX and model._SEARCH_OK
        model._net()
    cfg=base.econ.config();cfg.pop('randomSeed',None);rows=[];maps=[]
    pair.HERE=HERE;pair.PROTOCOL=CONTRACT
    out=HERE/'results.json.gz'
    if out.exists():raise ValueError('This registered panel is complete')
    risk=copy.deepcopy(e._TREE_RISK);cash=copy.deepcopy(e._TREE_CASH)
    for x in plan['roots']:
        r=dict(ticket=x)
        if 'step' not in x:
            r['status']='no_opportunity';rows.append(r);continue
        n,seat,t=x['ordinal'],x['seat'],x['step'];e._TREE_RISK=copy.deepcopy(risk);e._TREE_CASH=copy.deepcopy(cash)
        fs,cmds=pair.tape(x);obs=json.loads(fs[t])['observations'][seat];assert base.digest(obs)==x['public_observation_sha256']
        h=pair.historical_root(x,seat,p,cfg,fs,cmds)
        # The new menu owns a short history. Teacher-force the actual observed prefix and
        # reconstruct whether its hook ran on prior public calls under this native search.
        htree={}
        for k in range(max(0,t-5),t):
            prev=json.loads(fs[k])['observations'][seat]
            kh=dict(x,step=k);h0=pair.historical_root(kh,seat,e,cfg,fs,cmds)
            call(e,'warm',h0,htree,prev,cfg,(19073+7*k+x['seed']+seat)%2**32)
            htree=copy.deepcopy(e._TREE_HIST.get(seat,{}))
        seed=(x['seed']+19073+7*t+seat)%2**32;calls=[]
        for tag in (('P','E','E','P') if n%2==0 else ('E','P','P','E')):
            m=p if tag=='P' else e
            q=call(m,tag,h,htree,obs,cfg,seed);calls.append(q)
            if tag=='P':snap=pair.snapshot(p)
        p1,p2=[z for z in calls if z['mode']=='P'];e1,e2=[z for z in calls if z['mode']=='E']
        tr=e1['trace'] or {};new=tr.get('new',[]);opts=tr.get('plan',{}).get('roots',{})
        selected=[k for k in new if opts[k]==e1['action']]
        stable=choice(p1)==choice(p2) and choice(e1)==choice(e2)
        fair=len({work(z) for z in calls})==1
        qualifies=bool(selected and stable and fair and p1['action']!=e1['action'] and e1['action'] not in (p1['menu'] or [p1['action']]))
        r.update(status='verified',history_keys=sorted(htree),calls=calls,stable=stable,effort_equal=fair,
                 any_final_change=p1['action']!=e1['action'],added=new,selected_new=selected,qualified=qualifies)
        if qualifies:
            e._TREE_RISK['weights']=[0.]*len(risk['weights'])
            e._TREE_CASH['coef']=[[0.0,0.0] for v in cash['coef']]
            neutral=[call(e,'N',h,htree,obs,cfg,seed) for _ in range(2)]
            ablated_match=choice(neutral[0])==choice(neutral[1]) and all(work(z)==work(e1) for z in neutral)
            r.update(constant_heads=neutral,constant_heads_matched=ablated_match,learned_slopes_changed_final=bool(ablated_match and neutral[0]['action']!=e1['action']))
            if len(maps)<2 and x['seed'] not in maps:
                probe=dict(E=dict(e1,trace=dict(pivot=tr['plan']['worker'])),P=p1)
                resp=pair.continue_pair(x,probe,snap,p,o,cfg,fs,cmds,CANDIDATE,len(maps))
                r['responses']=resp;r['response_economy']=score_targets(x,resp,tr['plan']['target'],fs,cmds,seat);maps.append(x['seed'])
        rows.append(r)
        print('tree',n,x['seed'],seat,t,'added',new,'selected',selected,'qual',qualifies,flush=True)
        pair.save(HERE/'progress.json.gz',dict(protocol_sha256=CONTRACT,candidate_sha256=CANDIDATE,rows=rows))
    counts=dict(registered=len(rows),evaluated=sum(r['status']=='verified' for r in rows),added=sum(bool(r.get('added')) for r in rows),
                selected_new=sum(bool(r.get('selected_new')) for r in rows),qualified=sum(bool(r.get('qualified')) for r in rows),
                learned=sum(bool(r.get('learned_slopes_changed_final')) for r in rows),continued_maps=maps)
    pair.save(out,dict(protocol_sha256=CONTRACT,candidate_sha256=CANDIDATE,counts=counts,rows=rows,
                       cgroup={k:Path('/sys/fs/cgroup',k).read_text().strip() for k in ('cpu.max','memory.max')},
                       timing='Local original .58 s native search and process wall/CPU, one CPU affinity. No ladder certification.'))
    (HERE/'progress.json.gz').unlink(missing_ok=True)
    print(counts)


if __name__=='__main__':run()
