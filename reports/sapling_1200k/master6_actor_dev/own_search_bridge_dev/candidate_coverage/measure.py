"""Rank the two published joint interventions at the original player's actual public prefix."""
from pathlib import Path
import sys,json,inspect,gzip,os
from collections import defaultdict
from statistics import mean
import numpy as np
HERE=Path(__file__).resolve().parent;GROUP=HERE.parent;ACTOR=GROUP.parent
sys.path.insert(0,str(GROUP))
import bridge as b
sys.path.insert(0,str(ACTOR/'blended_policy_return_dev'))
import policy
PACK=b.ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
def norm(x):return {k:v for k,v in x.items() if v and v!='NONE'}
def ranked(r,k):return r[k] is not None and r[k]<=2
def summarize(rows):
    y=[r for r in rows if r['delta']>0]
    def rank(z,name='alt'):
        return {f'top{k}':sum(r[name+'_rank'] is not None and r[name+'_rank']<=k for r in z) for k in (1,2,3,5,9)}|{'n':len(z),'reachable':sum(r[name+'_rank'] is not None for r in z)}
    return dict(parent=rank(rows,'parent'),alternative=rank(rows),observed_positive_alt=rank(y),positive_total=len(y),
                both_top2=sum(all(r[n+'_rank'] is not None and r[n+'_rank']<=2 for n in ('parent','alt')) for r in rows),
                both_top3=sum(all(r[n+'_rank'] is not None and r[n+'_rank']<=3 for n in ('parent','alt')) for r in rows),
                parent_wins=sum(r['native_margins'][0]>0 for r in rows),alt_wins=sum(r['native_margins'][1]>0 for r in rows),
                oracle_wins=sum(max(r['native_margins'])>0 for r in rows),losing_to_winning=sum(r['native_margins'][0]<=0<r['native_margins'][1] for r in rows),
                oracle_mean_margin=mean(max(r['native_margins']) for r in rows),parent_mean_margin=mean(r['native_margins'][0] for r in rows))
def audit():
    assert os.sched_getaffinity(0)=={4}
    cases=b.load(GROUP/'root_scores.json.gz');grouped=defaultdict(dict);third={}
    for family in ('third_route','early_region'):
        for v in b.load(GROUP/family/'response.json'):
            third[(v['seed'],v['seat'],v['step'])]=(family,v)
    for v in cases:
        r=b.load(b.ROOT/v['case']/'live_protocol.json')
        grouped[r['record']['source']][r['root_t']]=(v,r)
    for (seed,seat,t),(family,pr) in third.items():
        if family!='early_region':continue
        loc=GROUP/family/f'seat{seat}';reg=b.load(loc/'live_protocol.json')
        case=dict(seed=seed,seat=seat,action_t=t,case=str(loc.relative_to(b.ROOT)),batch=0,
                  realized_alternative_minus_parent=pr['delta'],actual_native=[pr['arms'][a] for a in ('parent','third')])
        assert t not in grouped[reg['record']['source']]
        grouped[reg['record']['source']][t]=(case,reg)
    rows=[]
    for f,by_time in grouped.items():
        p=b.ROOT/f;raw=b.load(p);s=raw['candidate_seat'];cfg=raw['game_configuration'];sim=b.FastCarbon(cfg).reset(seed=raw['seed']);history=[]
        assert b.sha(PACK)==raw['pack_sha256']
        m=policy.load(PACK);F=sys.modules['cnet2_feat'];feat=F.Feat(s)
        src=inspect.getsource(m._return_decode)
        needle='action = int(np.argmax(np.where(legal, combined, -np.inf)))'
        assert src.count(needle)==1
        code=src.replace(needle,needle+'\n        capture(slot, combined, scores, legal)')
        captured={};fn=dict(vars(m),capture=lambda j,mix,heads,ok:captured.update({j:[mix.copy(),heads[0].copy(),heads[1].copy(),ok.copy()]}))
        exec(compile(code,'actual_parent_logits','exec'),fn)
        for t in range(max(by_time)+1):
            ob=sim.obs(s);x=feat.observe(ob);commands=raw['actions'][t]
            if t in by_time:
                v,reg=by_time[t];srcdir=b.ROOT/v['case'];assert b.sha(p)==reg['record']['source_sha256']
                if v['batch']==0:
                    uid=reg['root']['worker'];alt=b.load(srcdir/'live_third_actions.json.gz')[0][s];old=b.load(srcdir/'live_parent_actions.json.gz')[0][s]
                    q=dict(public_at_root=ob,v1=old,alternative=alt,true_prefix_slot=list(feat.own_uid).index(uid));v['physical_terrain']=raw['initial_terrain_u8_sha256']
                else:
                    q=b.load(srcdir/'root.json.gz');assert b.sha(srcdir/'root.json.gz')==v['root_sha256']
                assert all(ob[k]==q['public_at_root'][k] for k in ('step','players','player','carbon','trees'))
                old=norm(q['v1']);alt=norm(q['alternative']);uid=reg['root']['worker'];slot=q['true_prefix_slot'];assert uid==feat.own_uid[slot] and int(x['tok_kind'][slot])==1
                root=policy.Root(m,feat,x);base,_=root.decode(False);res,_=root.decode(True);y,rejected=policy.guard(m,x,base,res)
                assert np.array_equal(y,m._mixed_actions(x,feat.aux)) and old==feat.to_actions(y)==norm(commands[s])
                # Actual entry uses exactly one Feat history. Independently check old commands with public wrapper throughout the roots.
                assert norm(m.agent(ob,cfg))==old
                actual_logits=fn['_return_decode'](x,feat.aux,root.enc,not rejected)
                assert np.array_equal(y,actual_logits)
                logits=captured[slot];support=root.distribution(slot,policy.ZERO)
                ordered=sorted(support,key=lambda k:(-support[k][0],k))
                def find(cmd):return ordered.index(policy.key(cmd))+1 if policy.key(cmd) in ordered else None
                def physical_head(z):
                    return {cmd:float(z[int(F.CANON_MOVE[i]) if s else i]) for i,cmd in enumerate(('NONE','UP','RIGHT','DOWN','LEFT')) if logits[3][int(F.CANON_MOVE[i]) if s else i]}
                def describe(cmd):
                    direction=cmd.get(uid,'NONE');h=physical_head(logits[0]);return sorted(h,key=lambda j:-h[j]).index(direction)+1
                rec=dict(case=v['case'],seed=v['seed'],terrain=v['physical_terrain'],seat=s,t=t,worker=uid,slot=slot,role='planter',batch=v['batch'],
                    old=old.get(uid,'NONE'),alternative=alt.get(uid,'NONE'),parent_rank=find(old),alt_rank=find(alt),old_slot_rank=describe(old),alt_slot_rank=describe(alt),
                    complete_commands=[dict(cmd=json.loads(k),mass=support[k][0]) for k in ordered],logits={k:physical_head(lg) for k,lg in zip(('mixed','partner','sapling'),logits[:3])},
                    delta=v['realized_alternative_minus_parent'],native_margins=[it['margin'] for it in v['actual_native']],digest=b.sha(p),
                    prehistory=history[-10:],trace_windows={str(j):[t-j,t-1] for j in (3,5,10)})
                tr=third.pop((v['seed'],s,t),None)
                if tr:
                    fam,pr=tr;nc=dict(old);nc[uid]=pr['new_route'];nc=norm(nc)
                    taped=b.load(GROUP/fam/f'seat{s}'/'live_third_actions.json.gz')[0][s]
                    assert nc==norm(taped)
                    rec['third']=dict(family=fam,action=pr['new_route'],rank=find(nc),slot_rank=describe(nc),margin_delta=pr['delta'],source=str((GROUP/fam/'response.json').relative_to(b.ROOT)))
                rows.append(rec)
            elif t==0:m.agent(ob,cfg)
            else:m._FEAT.observe(ob)
            history.append(dict(t=t,seat_commands=commands,cash=[p[0] for p in ob['players']],workers=[p[2] for p in ob['players']],trees=[p[3] for p in ob['players']]))
            sim.step(*commands)
    assert len(rows)==66 and not third
    extras=[r for r in rows if r["batch"]==0];rows=[r for r in rows if r["batch"]!=0]
    batches=sorted(set(r['batch'] for r in rows));held=batches[-2:]
    put=dict(definition='After the logged prefix, move only the named planter; draw both actual component decoders using a common uniform number, greedily re-decode every subsequent slot; apply original guard, deduplicate full executed command dictionaries and sum interval probabilities. Ranks use descending joint mass with lexical ties. Actual production call is greedy; this finite candidate expansion is counterfactual. Source option (economic heuristic) selection prefilters isolated one-worker changes. No unseen alternatives have terminal labels.',
             separate_early_roots=extras,pack=b.sha(PACK),measurer=b.sha(__file__),sources={k:b.sha(GROUP/k) for k in ('root_scores.json.gz','third_route/response.json','early_region/response.json')},
             held_batches=held,all=summarize(rows),held=summarize([r for r in rows if r['batch'] in held]),
             by_map={k:summarize([r for r in rows if r['seed']==k]) for k in sorted(set(r['seed'] for r in rows))},
             by_role={'planter':summarize(rows),'collector':{'n':0},'base':{'n':0}},roots=rows)
    b.save(HERE/'coverage.json.gz',put)
if __name__=='__main__':audit()
