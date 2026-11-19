"""Inject a public legal candidate menu; read native per-roll values before pruning, and after complete extra rolls."""
from pathlib import Path
import sys,time,inspect,os,hashlib,json,copy
from collections import defaultdict,Counter
import numpy as np
HERE=Path(__file__).resolve().parent;J=HERE.parent;sys.path.insert(0,str(J))
from setup import ACTOR,GROUP,ROOT,bridge
sys.path.insert(0,str(GROUP/'teacher_window_dev'));import run as native
read=bridge.load;sha=bridge.sha
def instrument(m):
    original=inspect.getsource(m._search);a=original.index('    cands = [act0]');b=original.index('    if len(cands) < 2:',a)
    src=original[:a]+'''    cands = list(forced_menu)
    act0 = cands[0]
'''+original[b:]
    def at(needle,after):
        nonlocal src
        assert src.count(needle)==1
        src=src.replace(needle,needle+after)
    at('        depth = t','\n        retain(t,M,cands,dep,val,opp,alive)')
    at('        rolls += M','\n        retain(depth,rolls,cands,dep,val,opp,alive)')
    def retain(depth,rolls,menu,dep,val,opp,active):
        if tuple(active)!=tuple(range(len(menu))) or not all(dep.get(j)==depth for j in active):return
        assert set(range(rolls))<=set(opp)
        matrix=[[float(val[j,k]) for k in range(rolls)] for j in active]
        assert np.isfinite(matrix).all()
        log.append(dict(depth=depth,rolls=rolls,values=matrix,enemy_by_stream=[dict(opp[k]) for k in range(rolls)]))
    log=[];ns=dict(vars(m),retain=retain,forced_menu=None);exec(compile(src,'frozen_original::offline_restricted_search','exec'),ns)
    # _net() already warmed before injection, module mutable counters and cost list remain shared.
    m._search=ns['_search'];return ns,log
def source_cards():
    rr=read(GROUP/'root_scores.json.gz');batches=sorted({r['batch'] for r in rr})[-2:]
    cards=[]
    for r in rr:
        if r['batch'] not in batches:continue
        d=ROOT/r['case'];p=read(d/'live_protocol.json');q=read(d/'root.json.gz')
        assert sha(d/'root.json.gz')==r['root_sha256']
        cards.append(dict(seed=r['seed'],seat=r['seat'],t=r['action_t'],terrain=r['physical_terrain'],case=r['case'],source=p['record']['source'],srcsha=p['record']['source_sha256'],
                          cmds=[q['v1'],q['alternative']],margins=[z['margin'] for z in r['actual_native']]))
    assert len(cards)==16;return cards
def main():
    assert os.sched_getaffinity(0)=={2};os.environ['CARBON_DEBUG']='1'
    cards=source_cards()
    proto=dict(original=native.P['identity']['master'],test_ids=[c['case'] for c in cards],real_scores_source=sha(GROUP/'root_scores.json.gz'),
      label='Use genuine original .58s wall search with prescribed legal complete candidate set instead of its own generation. No change to belief, actor/value nets, CRN, rollout, native pruning or stopping logic. Select deepest last all-candidate completed equal depth, then maximum common fully completed rolls there. Require all active candidate x rollout cells, store raw matrix and shared modeled enemy command per stream. Never read _S_REC roll0 as aggregate. Use original _S_EPS tie threshold. If a candidate pruned, never mix it with survivors at deeper depths/larger rollout counts.',
      gate='16 registered held roots on four terrains. One invocation per root, include all results and abstentions; missing complete score: no alternative, counts as missed for accuracy. Require >8 correct of 16, complete 16/16, and positive mean official terminal gain across four maps (equal roots). No fresh matches. Only if passed, allow non-overlapping student-state label collection; no tuning.')
    bridge.save(HERE/'protocol.json',proto);by=defaultdict(list)
    for c in cards:by[c['source']].append(c)
    measured=[];cold=read(ROOT/native.P['cases'][0]['warm_source'])['warm_cost']
    for src,cc in by.items():
        raw=read(ROOT/src);assert sha(ROOT/src)==cc[0]['srcsha']
        s=cc[0]['seat'];sim=bridge.FastCarbon(raw['game_configuration']).reset(seed=cc[0]['seed']);cfg=sim.config
        teacher=native.install(native.live.SOURCE,s,472135+s,cold,'native_frozen_teacher');_ns,history=instrument(teacher)
        teacher._S_REC=[];_ns['_S_REC']=teacher._S_REC;take={c['t']:c for c in cc}
        for t in range(max(take)+1):
            obs=sim.obs(s);obs={k:v for k,v in obs.items() if k in ('player','step','players','trees','carbon','remainingOverageTime')}
            if t in take:
                c=take[t];d=ROOT/c['case'];rr=read(d/'root.json.gz');assert all(rr['public_at_root'][k]==obs[k] for k in ('player','step','players','trees','carbon'))
                _ns['forced_menu']=c['cmds'];history.clear();teacher._S_REC.clear()
                start=time.perf_counter();cpu=time.process_time();action=native.clean(teacher.agent(obs,cfg));cw=time.perf_counter()-start;ccpu=time.process_time()-cpu
                got=history[-1] if history else None
                if got:
                    means=np.mean(got['values'],axis=1);pick=int(means[1]>means[0]+teacher._S_EPS)
                    assert c['cmds'][pick]==action and teacher._S_REC[-1][-1]==pick
                else:means=None;pick=0
                measured.append(dict(case=c['case'],seed=c['seed'],seat=s,t=t,terrain=c['terrain'],complete=got is not None,all_complete_rounds=copy.deepcopy(history),mean=means.tolist() if got else None,picked=pick,wall=cw,cpu=ccpu,S_EPS=float(teacher._S_EPS),roll0_record=(teacher._S_REC[-1][4] if teacher._S_REC else None)))
                teacher._ST['tr'].set_actions(obs['players'],raw['actions'][t][s])
            else:native.historical(teacher,obs,raw['actions'][t][s])
            sim.step(*raw['actions'][t])
    outcomes={c['case']:c['margins'][1]-c['margins'][0] for c in cards}
    for r in measured:
        d=outcomes[r['case']];r.update(actual_pair_delta=d,correct=r['complete'] and r['picked']==int(d>0),selected_gain=d*r['picked'])
    groups={}
    for seed in sorted({r['seed'] for r in measured}):
        v=[r for r in measured if r['seed']==seed];groups[seed]=dict(n=len(v),correct=sum(int(r['correct']) for r in v),gain=float(np.mean([r['selected_gain'] for r in v])))
    ct=sum(int(r['complete']) for r in measured);hit=sum(int(r['correct']) for r in measured);gain=float(np.mean([v['gain'] for v in groups.values()]))
    bridge.save(HERE/'results.json.gz',dict(protocol=sha(HERE/'protocol.json'),source=sha(__file__),complete=ct,correct=hit,mean_gain=gain,gate=(ct==16 and hit>8 and gain>0),by_map=groups,rows=measured))
if __name__=='__main__':main()
